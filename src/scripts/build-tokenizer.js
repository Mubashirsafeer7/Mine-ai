'use strict';
/**
 * Tokenizer alag se bana kar dekhne ke liye (train.js khud bhi banata hai).
 *
 *   npm run tokenizer -- --vocab 900
 *   npm run tokenizer -- --encode "salam kaise ho"
 */

const fs = require('fs');
const path = require('path');
const { Tokenizer } = require('../core/tokenizer');
const { loadCorpusFiles } = require('../core/dataset');

const ROOT = path.resolve(__dirname, '..', '..');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue;
    const key = argv[i].slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) args[key] = true;
    else {
      args[key] = isNaN(Number(next)) ? next : Number(next);
      i++;
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const vocabSize = args.vocab || 900;
const dataDir = path.join(ROOT, 'data');
const files = fs.readdirSync(dataDir).filter((f) => f.endsWith('.txt')).map((f) => path.join(dataDir, f)).sort();
const raw = loadCorpusFiles(files);

const tokenizer = new Tokenizer();
tokenizer.train(raw, vocabSize, { verbose: true });

const outPath = path.join(ROOT, 'models', 'tokenizer.json');
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(tokenizer.toJSON(), null, 2));

console.log(`\nVocab size : ${tokenizer.size}`);
console.log(`Merges     : ${tokenizer.merges.size}`);
console.log(`Corpus     : ${raw.length.toLocaleString()} characters`);
const totalTokens = tokenizer.encode(raw).length;
console.log(`Tokens     : ${totalTokens.toLocaleString()} (${(raw.length / totalTokens).toFixed(2)} chars/token)`);
console.log(`Save       : ${path.relative(ROOT, outPath)}`);

const sample = typeof args.encode === 'string' ? args.encode : 'salam kaise ho, tumhara naam kya hai';
const ids = tokenizer.encode(sample);
console.log(`\nMisal: ${JSON.stringify(sample)}`);
console.log(`  ids    : [${ids.join(', ')}]`);
console.log(`  pieces : ${ids.map((id) => JSON.stringify(tokenizer.vocab[id])).join(' ')}`);
console.log(`  decode : ${JSON.stringify(tokenizer.decode(ids))}`);
