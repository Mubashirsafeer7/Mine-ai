'use strict';
/**
 * Model save/load. Weights base64 (Float32 raw) mein jate hain taake
 * file chhoti rahe aur JSON mein hi sab kuch aa jaye.
 */

const fs = require('fs');
const path = require('path');
const { MineAIModel } = require('./model');
const { Tokenizer } = require('./tokenizer');

function encodeWeights(params) {
  let total = 0;
  for (const p of params) total += p.size;
  const flat = new Float32Array(total);
  let off = 0;
  for (const p of params) {
    flat.set(p.data, off);
    off += p.size;
  }
  return Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength).toString('base64');
}

function decodeWeights(params, b64) {
  const buf = Buffer.from(b64, 'base64');
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  let off = 0;
  for (const p of params) {
    p.data.set(flat.subarray(off, off + p.size));
    off += p.size;
  }
  if (off !== flat.length) throw new Error(`checkpoint size mismatch: expected ${off}, got ${flat.length}`);
}

function saveModel(filePath, { model, tokenizer, meta = {} }) {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
  const payload = {
    format: 'mine-ai/v1',
    config: model.config,
    meta: { ...meta, savedAt: new Date().toISOString(), numParams: model.numParams() },
    tokenizer: tokenizer.toJSON(),
    weights: encodeWeights(model.parameters()),
  };
  fs.writeFileSync(filePath, JSON.stringify(payload));
  return filePath;
}

function loadModel(filePath) {
  const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  if (payload.format !== 'mine-ai/v1') throw new Error(`unknown checkpoint format: ${payload.format}`);
  const model = new MineAIModel(payload.config);
  decodeWeights(model.parameters(), payload.weights);
  const tokenizer = new Tokenizer(payload.tokenizer);
  return { model, tokenizer, meta: payload.meta || {}, config: payload.config };
}

module.exports = { saveModel, loadModel, encodeWeights, decodeWeights };
