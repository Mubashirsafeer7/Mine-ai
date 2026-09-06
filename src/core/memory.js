'use strict';
/**
 * Mine AI - memory (retrieval) layer.
 *
 * Neural model chhota hai, is liye kuch jawab wo theek se nahi bana pata.
 * Ye layer corpus mein se sab se milta julta sawal dhoondti hai — TF-IDF
 * (words + character trigrams) aur cosine similarity se. Sab kuch scratch se,
 * koi search library nahi.
 */

function normalizeText(text) {
  return text
    .toLowerCase()
    .replace(/<\|[a-z]+\|>/g, ' ')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Words + char trigrams. Trigrams spelling ki chhoti galtiyan bardasht karte hain. */
function features(text) {
  const clean = normalizeText(text);
  if (!clean) return [];
  const out = [];
  const words = clean.split(' ');
  for (const w of words) {
    out.push('w:' + w);
    const padded = `^${w}$`;
    for (let i = 0; i + 3 <= padded.length; i++) out.push('c:' + padded.slice(i, i + 3));
  }
  return out;
}

function toCounts(list) {
  const map = new Map();
  for (const f of list) map.set(f, (map.get(f) || 0) + 1);
  return map;
}

class Memory {
  constructor() {
    this.entries = []; // { question, answer, vector: Map, norm }
    this.idf = new Map();
  }

  /** blocks: parseCorpus() ka output */
  static fromBlocks(blocks) {
    const memory = new Memory();
    const raw = [];
    for (const block of blocks) {
      const userTurns = block.turns.filter((t) => t.role === 'user' && t.text);
      const botTurns = block.turns.filter((t) => t.role === 'bot' && t.text);
      if (!userTurns.length || !botTurns.length) continue;
      raw.push({
        question: userTurns.map((t) => t.text).join(' '),
        answer: botTurns[botTurns.length - 1].text,
      });
    }

    const df = new Map();
    const docs = raw.map((r) => {
      const counts = toCounts(features(r.question));
      for (const f of counts.keys()) df.set(f, (df.get(f) || 0) + 1);
      return counts;
    });
    const N = Math.max(1, docs.length);
    for (const [f, count] of df) memory.idf.set(f, Math.log(1 + N / count));

    raw.forEach((r, i) => {
      const { vector, norm } = memory._vectorize(docs[i]);
      memory.entries.push({ question: r.question, answer: r.answer, vector, norm });
    });
    return memory;
  }

  _vectorize(counts) {
    const vector = new Map();
    let sum = 0;
    for (const [f, tf] of counts) {
      const idf = this.idf.get(f);
      if (idf === undefined) continue;
      const w = (1 + Math.log(tf)) * idf;
      vector.set(f, w);
      sum += w * w;
    }
    return { vector, norm: Math.sqrt(sum) || 1 };
  }

  /** Sab se milta julta entry. Return: { answer, question, score } ya null */
  search(query, { topN = 1 } = {}) {
    if (!this.entries.length) return topN === 1 ? null : [];
    const { vector, norm } = this._vectorize(toCounts(features(query)));
    if (vector.size === 0) return topN === 1 ? null : [];

    const scored = [];
    for (const entry of this.entries) {
      // Chhote vector par loop karna tez hai.
      const [small, big] = vector.size < entry.vector.size ? [vector, entry.vector] : [entry.vector, vector];
      let dot = 0;
      for (const [f, w] of small) {
        const other = big.get(f);
        if (other !== undefined) dot += w * other;
      }
      if (dot === 0) continue;
      scored.push({ question: entry.question, answer: entry.answer, score: dot / (norm * entry.norm) });
    }
    scored.sort((a, b) => b.score - a.score);
    return topN === 1 ? scored[0] || null : scored.slice(0, topN);
  }

  get size() {
    return this.entries.length;
  }
}

module.exports = { Memory, features, normalizeText };
