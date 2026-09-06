'use strict';
/**
 * Corpus parsing + training examples.
 *
 * Corpus format:
 *   <|user|> sawal
 *   <|bot|> jawab
 *   <|end|>
 *
 * Ek block mein ek se zyada turns bhi ho sakte hain (multi-turn chat).
 * Loss sirf bot ke tokens par lagta hai - user ke tokens sirf context hain.
 */

const fs = require('fs');
const { SPECIAL_TOKENS } = require('./tokenizer');

/** Raw text -> [{ turns: [{role, text}] }] */
function parseCorpus(text) {
  const blocks = [];
  let current = [];
  const lines = text.split('\n');
  let role = null;
  let buffer = [];

  const flushTurn = () => {
    if (role) {
      current.push({ role, text: buffer.join('\n').trim() });
      buffer = [];
      role = null;
    }
  };

  for (const line of lines) {
    const userMatch = line.match(/^<\|user\|>\s?(.*)$/);
    const botMatch = line.match(/^<\|bot\|>\s?(.*)$/);
    if (line.trim() === '<|end|>') {
      flushTurn();
      if (current.length) blocks.push({ turns: current });
      current = [];
    } else if (userMatch) {
      flushTurn();
      role = 'user';
      buffer.push(userMatch[1]);
    } else if (botMatch) {
      flushTurn();
      role = 'bot';
      buffer.push(botMatch[1]);
    } else if (role) {
      buffer.push(line);
    }
  }
  flushTurn();
  if (current.length) blocks.push({ turns: current });
  return blocks.filter((b) => b.turns.some((t) => t.role === 'bot' && t.text));
}

/**
 * Blocks -> training examples { ids, targets }.
 * targets[i] = ids[i+1] agar us position par bot bol raha ho, warna -1 (ignore).
 */
function buildExamples(blocks, tokenizer, blockSize) {
  const USER = tokenizer.id('<|user|>');
  const BOT = tokenizer.id('<|bot|>');
  const END = tokenizer.id('<|end|>');
  const examples = [];

  for (const block of blocks) {
    const ids = [];
    const supervised = []; // true = is token ko predict karna seekhna hai
    for (const turn of block.turns) {
      const body = tokenizer.encode(' ' + turn.text);
      if (turn.role === 'user') {
        ids.push(USER);
        supervised.push(false);
        for (const t of body) {
          ids.push(t);
          supervised.push(false);
        }
      } else {
        ids.push(BOT);
        supervised.push(false);
        for (const t of body) {
          ids.push(t);
          supervised.push(true);
        }
        ids.push(END);
        supervised.push(true);
      }
    }

    if (ids.length < 2) continue;
    const clipped = ids.length > blockSize ? ids.slice(0, blockSize) : ids;
    const sup = supervised.slice(0, clipped.length);

    const inputIds = clipped.slice(0, clipped.length - 1);
    const targets = new Int32Array(inputIds.length);
    for (let i = 0; i < inputIds.length; i++) {
      targets[i] = sup[i + 1] ? clipped[i + 1] : -1;
    }
    if (targets.some((t) => t >= 0)) examples.push({ ids: inputIds, targets });
  }
  return examples;
}

function loadCorpusFiles(paths) {
  return paths.map((p) => fs.readFileSync(p, 'utf8')).join('\n');
}

/** Prompt banane ke liye: chat history -> token ids (bot turn shuru kiya hua). */
function buildPrompt(tokenizer, history, { blockSize = 64 } = {}) {
  const USER = tokenizer.id('<|user|>');
  const BOT = tokenizer.id('<|bot|>');
  let ids = [];
  for (const turn of history) {
    ids.push(turn.role === 'user' ? USER : BOT);
    ids = ids.concat(tokenizer.encode(' ' + turn.text));
  }
  ids.push(BOT);
  // Context window se bara ho jaye to purani baat kaat dete hain.
  if (ids.length > blockSize - 1) ids = ids.slice(ids.length - (blockSize - 1));
  return ids;
}

module.exports = { parseCorpus, buildExamples, loadCorpusFiles, buildPrompt, SPECIAL_TOKENS };
