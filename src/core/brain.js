'use strict';
/**
 * Mine AI - brain. Model + memory ko jodta hai aur jawab stream karta hai.
 *
 * Modes:
 *   auto   - agar memory ka match bohat pakka ho to wahi, warna neural model
 *   neural - hamesha model se generate (apne training ka asli test)
 *   memory - sirf retrieval
 */

const fs = require('fs');
const path = require('path');
const { loadModel } = require('./checkpoint');
const { Memory } = require('./memory');
const { parseCorpus, buildPrompt } = require('./dataset');
const { generate } = require('./sampler');

const ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_MODEL = path.join(ROOT, 'models', 'mine-ai.json');
const DEFAULT_CORPUS_DIR = path.join(ROOT, 'data');

const FALLBACKS = [
  'Ye baat mujhe abhi nahi aati. Mujhe corpus mein sikha dijiye to agli baar jawab doonga.',
  'Mujhe is ka jawab nahi pata. Aap mujhe train kar ke ye sikha sakte hain.',
  'Is par mera training data khali hai. Thoda aur asaan lafzon mein poochh ke dekhiye.',
];

/** Model ke tootay phootay output ko saaf karta hai. */
function cleanReply(text) {
  let out = text.replace(/<\|[a-z]+\|>/g, ' ');
  out = out.replace(/\s+([,.!?])/g, '$1');
  out = out.replace(/\s+/g, ' ').trim();
  if (!out) return '';
  return out.charAt(0).toUpperCase() + out.slice(1);
}

/** Kya jawab itna kharab hai ke memory par jana behtar hai? */
function looksDegenerate(text) {
  const clean = text.trim();
  if (clean.length < 2) return true;
  const words = clean.toLowerCase().split(/\s+/);
  if (words.length >= 4) {
    const unique = new Set(words);
    if (unique.size <= Math.ceil(words.length / 3)) return true; // bohat repetition
  }
  for (let i = 0; i + 2 < words.length; i++) {
    if (words[i] === words[i + 1] && words[i + 1] === words[i + 2]) return true;
  }
  return false;
}

class MineBrain {
  constructor({ modelPath = DEFAULT_MODEL, corpusDir = DEFAULT_CORPUS_DIR } = {}) {
    if (!fs.existsSync(modelPath)) {
      throw new Error(
        `Model nahi mila: ${modelPath}\nPehle train karen:  npm run train`
      );
    }
    const loaded = loadModel(modelPath);
    this.model = loaded.model;
    this.tokenizer = loaded.tokenizer;
    this.meta = loaded.meta;
    this.config = loaded.config;
    this.modelPath = modelPath;

    const files = fs.existsSync(corpusDir)
      ? fs.readdirSync(corpusDir).filter((f) => f.endsWith('.txt')).sort()
      : [];
    const raw = files.map((f) => fs.readFileSync(path.join(corpusDir, f), 'utf8')).join('\n');
    this.memory = Memory.fromBlocks(raw ? parseCorpus(raw) : []);

    this.endToken = this.tokenizer.id('<|end|>');
    this.userToken = this.tokenizer.id('<|user|>');
    this.fallbackIndex = 0;
  }

  info() {
    return {
      name: 'Mine AI',
      params: this.model.numParams(),
      vocabSize: this.tokenizer.size,
      layers: this.config.nLayer,
      heads: this.config.nHead,
      embedding: this.config.nEmbd,
      blockSize: this.config.blockSize,
      memoryEntries: this.memory.size,
      trainedAt: this.meta.savedAt || null,
      valLoss: this.meta.valLoss ?? null,
      steps: this.meta.steps ?? null,
    };
  }

  nextFallback() {
    const msg = FALLBACKS[this.fallbackIndex % FALLBACKS.length];
    this.fallbackIndex++;
    return msg;
  }

  /** Pura jawab ek dafa mein. */
  respond(history, options = {}) {
    let text = '';
    let meta = null;
    for (const chunk of this.stream(history, options)) {
      if (chunk.type === 'token') text += chunk.text;
      if (chunk.type === 'done') meta = chunk;
    }
    return { text: meta ? meta.text : cleanReply(text), source: meta ? meta.source : 'neural', meta };
  }

  /**
   * Jawab ko token by token stream karta hai.
   * yield { type: 'token', text } ... phir { type: 'done', text, source, score }
   */
  *stream(history, options = {}) {
    const {
      mode = 'auto',
      temperature = 0.75,
      topK = 40,
      topP = 0.9,
      maxNewTokens = 60,
      memoryThreshold = 0.62,
      rescueThreshold = 0.34,
      seed = null,
    } = options;

    const lastUser = [...history].reverse().find((t) => t.role === 'user');
    const question = lastUser ? lastUser.text : '';
    const hit = question ? this.memory.search(question) : null;
    const score = hit ? hit.score : 0;

    if (mode === 'memory' || (mode === 'auto' && hit && score >= memoryThreshold)) {
      if (hit) {
        yield* this._streamText(hit.answer);
        return yield { type: 'done', text: hit.answer, source: 'memory', score, matched: hit.question };
      }
      if (mode === 'memory') {
        const msg = this.nextFallback();
        yield* this._streamText(msg);
        return yield { type: 'done', text: msg, source: 'fallback', score };
      }
    }

    // Neural generation
    const promptIds = buildPrompt(this.tokenizer, history, { blockSize: this.config.blockSize });
    const pieces = [];
    let emitted = '';
    for (const id of generate(this.model, promptIds, {
      maxNewTokens,
      temperature,
      topK,
      topP,
      repetitionPenalty: 1.18,
      stopTokens: [this.endToken, this.userToken],
      seed,
    })) {
      pieces.push(id);
      // Decode poore sequence ka, taake BPE pieces theek se jur ke aayen.
      const full = cleanReply(this.tokenizer.decode(pieces));
      if (full.length > emitted.length) {
        yield { type: 'token', text: full.slice(emitted.length) };
        emitted = full;
      }
    }

    const generated = cleanReply(this.tokenizer.decode(pieces));

    if (mode === 'auto' && looksDegenerate(generated)) {
      // Model bhatak gaya - memory se bacha lete hain.
      const rescue = hit && score >= rescueThreshold ? hit.answer : this.nextFallback();
      yield { type: 'reset' };
      yield* this._streamText(rescue);
      return yield {
        type: 'done',
        text: rescue,
        source: hit && score >= rescueThreshold ? 'memory-rescue' : 'fallback',
        score,
      };
    }

    if (!generated) {
      const msg = this.nextFallback();
      yield { type: 'reset' };
      yield* this._streamText(msg);
      return yield { type: 'done', text: msg, source: 'fallback', score };
    }

    return yield { type: 'done', text: generated, source: 'neural', score, tokens: pieces.length };
  }

  *_streamText(text) {
    const parts = text.split(/(\s+)/);
    for (const p of parts) {
      if (p) yield { type: 'token', text: p };
    }
  }
}

module.exports = { MineBrain, cleanReply, looksDegenerate, DEFAULT_MODEL };
