'use strict';
/**
 * Mine AI - BPE tokenizer, scratch se.
 *
 * Kaam kaise karta hai:
 *  1. Text ko special tokens par kaata jata hai (<|user|> waghera atomic rehte hain).
 *  2. Baaki text words mein toota hai, har word ke aage ki space "_" ban jati hai.
 *  3. Har word characters ki list hai. Sab se zyada aane wale adjacent pair ko
 *     baar baar merge karte hain jab tak vocab size poora na ho jaye.
 */

const SPECIAL_TOKENS = ['<|pad|>', '<|user|>', '<|bot|>', '<|end|>', '<|unk|>'];
const SPACE = '▁'; // leading space marker

class Tokenizer {
  constructor(data = null) {
    this.specials = SPECIAL_TOKENS.slice();
    this.vocab = []; // id -> string piece
    this.ids = new Map(); // piece -> id
    this.merges = new Map(); // "a b" -> rank
    this._cache = new Map();
    if (data) this.load(data);
  }

  get size() {
    return this.vocab.length;
  }

  id(piece) {
    const v = this.ids.get(piece);
    return v === undefined ? this.ids.get('<|unk|>') : v;
  }

  // ---- text -> words -------------------------------------------------------

  static splitSpecials(text) {
    const parts = [];
    const re = /<\|[a-z]+\|>/g;
    let last = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) parts.push({ special: false, text: text.slice(last, m.index) });
      parts.push({ special: true, text: m[0] });
      last = m.index + m[0].length;
    }
    if (last < text.length) parts.push({ special: false, text: text.slice(last) });
    return parts;
  }

  /** "salam kaise ho" -> ["▁salam", "▁kaise", "▁ho"] */
  static toWords(text) {
    const words = [];
    // Newline apna alag token, warna model formatting bhool jata hai.
    const chunks = text.split(/(\n)/);
    for (const chunk of chunks) {
      if (chunk === '') continue;
      if (chunk === '\n') {
        words.push('\n');
        continue;
      }
      for (const raw of chunk.split(/ +/)) {
        if (raw === '') continue;
        // Punctuation ko alag word banao taake vocab saaf rahe.
        const pieces = raw.match(/[\p{L}\p{N}]+|[^\p{L}\p{N}\s]/gu) || [];
        pieces.forEach((p, i) => words.push(i === 0 ? SPACE + p : p));
      }
    }
    return words;
  }

  // ---- training ------------------------------------------------------------

  train(text, vocabSize = 1024, { minFrequency = 2, verbose = false } = {}) {
    const wordFreq = new Map();
    for (const part of Tokenizer.splitSpecials(text)) {
      if (part.special) continue;
      for (const w of Tokenizer.toWords(part.text)) {
        wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
      }
    }

    // Base vocab: specials + har unique character.
    const pieces = new Set(this.specials);
    const words = [];
    for (const [word, freq] of wordFreq) {
      const chars = Array.from(word);
      for (const c of chars) pieces.add(c);
      words.push({ symbols: chars, freq });
    }

    this.vocab = Array.from(pieces);
    this.merges = new Map();

    let rank = 0;
    while (this.vocab.length < vocabSize) {
      const pairFreq = new Map();
      for (const w of words) {
        const s = w.symbols;
        for (let i = 0; i < s.length - 1; i++) {
          const key = s[i] + ' ' + s[i + 1];
          pairFreq.set(key, (pairFreq.get(key) || 0) + w.freq);
        }
      }
      let best = null;
      let bestCount = 0;
      for (const [key, count] of pairFreq) {
        if (count > bestCount) {
          bestCount = count;
          best = key;
        }
      }
      if (!best || bestCount < minFrequency) break;

      const [a, b] = best.split(' ');
      const merged = a + b;
      this.merges.set(best, rank++);
      this.vocab.push(merged);
      if (verbose && this.vocab.length % 100 === 0) {
        process.stdout.write(`  vocab ${this.vocab.length}/${vocabSize}\r`);
      }

      for (const w of words) {
        const s = w.symbols;
        if (s.length < 2) continue;
        const out = [];
        let i = 0;
        while (i < s.length) {
          if (i < s.length - 1 && s[i] === a && s[i + 1] === b) {
            out.push(merged);
            i += 2;
          } else {
            out.push(s[i]);
            i++;
          }
        }
        w.symbols = out;
      }
    }

    this.ids = new Map(this.vocab.map((p, i) => [p, i]));
    this._cache.clear();
    return this;
  }

  // ---- encode / decode -----------------------------------------------------

  _bpeWord(word) {
    const cached = this._cache.get(word);
    if (cached) return cached;
    let symbols = Array.from(word);
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = this.merges.get(symbols[i] + ' ' + symbols[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      symbols = [
        ...symbols.slice(0, bestIdx),
        symbols[bestIdx] + symbols[bestIdx + 1],
        ...symbols.slice(bestIdx + 2),
      ];
    }
    this._cache.set(word, symbols);
    return symbols;
  }

  encode(text) {
    const out = [];
    const unk = this.ids.get('<|unk|>');
    for (const part of Tokenizer.splitSpecials(text)) {
      if (part.special) {
        const id = this.ids.get(part.text);
        out.push(id === undefined ? unk : id);
        continue;
      }
      for (const word of Tokenizer.toWords(part.text)) {
        for (const sym of this._bpeWord(word)) {
          const id = this.ids.get(sym);
          if (id !== undefined) {
            out.push(id);
          } else {
            // Piece vocab mein nahi -> character by character.
            for (const ch of Array.from(sym)) {
              const cid = this.ids.get(ch);
              out.push(cid === undefined ? unk : cid);
            }
          }
        }
      }
    }
    return out;
  }

  decode(ids, { skipSpecial = true } = {}) {
    let text = '';
    for (const id of ids) {
      const piece = this.vocab[id];
      if (piece === undefined) continue;
      if (this.specials.includes(piece)) {
        if (skipSpecial) continue;
        text += piece;
        continue;
      }
      text += piece;
    }
    return text.split(SPACE).join(' ').replace(/^ /, '');
  }

  // ---- persistence ---------------------------------------------------------

  toJSON() {
    return {
      specials: this.specials,
      vocab: this.vocab,
      merges: Array.from(this.merges.entries()),
    };
  }

  load(data) {
    this.specials = data.specials || SPECIAL_TOKENS.slice();
    this.vocab = data.vocab;
    this.ids = new Map(this.vocab.map((p, i) => [p, i]));
    this.merges = new Map(data.merges);
    this._cache.clear();
    return this;
  }
}

module.exports = { Tokenizer, SPECIAL_TOKENS, SPACE };
