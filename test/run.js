'use strict';
/**
 * Mine AI - tests. Koi test framework nahi, sirf plain node.
 *   npm test
 */

const assert = require('assert');
const T = require('../src/core/tensor');
const { Tensor } = T;
const { MineAIModel } = require('../src/core/model');
const { Tokenizer } = require('../src/core/tokenizer');
const { parseCorpus, buildExamples } = require('../src/core/dataset');
const { AdamW } = require('../src/core/optimizer');
const { encodeWeights, decodeWeights } = require('../src/core/checkpoint');
const { createRandom } = require('../src/core/random');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ok   ${name}`);
  } catch (err) {
    failed++;
    console.log(`  FAIL ${name}`);
    console.log(`       ${err.message}`);
  }
}

function tensorFrom(rows, cols, values, requiresGrad = true) {
  return new Tensor(rows, cols, Float32Array.from(values), requiresGrad);
}

// ---------------------------------------------------------------------------
console.log('\nmatrix primitives');

test('matmul matches hand-computed product', () => {
  const a = tensorFrom(2, 3, [1, 2, 3, 4, 5, 6], false);
  const b = tensorFrom(3, 2, [7, 8, 9, 10, 11, 12], false);
  const c = T.matmul(a, b);
  assert.deepStrictEqual(Array.from(c.data), [58, 64, 139, 154]);
});

test('matmulNT equals matmul with transposed operand', () => {
  const a = tensorFrom(2, 3, [1, 2, 3, 4, 5, 6], false);
  const bT = tensorFrom(2, 3, [7, 9, 11, 8, 10, 12], false); // b^T rows
  const c = T.matmulNT(a, bT);
  assert.deepStrictEqual(Array.from(c.data), [58, 64, 139, 154]);
});

// ---------------------------------------------------------------------------
console.log('\nautograd (finite differences)');

/**
 * f(): scalar Tensor banata hai. inputs: jin Tensors ka gradient check karna hai.
 * Har input ke kuch random elements par analytic vs numeric gradient compare.
 */
function gradCheck(name, buildLoss, inputs, { eps = 1e-3, tol = 2e-2, samples = 6 } = {}) {
  for (const t of inputs) {
    t.requiresGrad = true;
    t.ensureGrad();
    t.zeroGrad();
  }
  const loss = buildLoss();
  loss.backward();

  const rand = createRandom(7);
  for (const t of inputs) {
    const analytic = Float32Array.from(t.grad);
    const n = Math.min(samples, t.size);
    for (let s = 0; s < n; s++) {
      const idx = rand.int(t.size);
      const orig = t.data[idx];
      t.data[idx] = orig + eps;
      const plus = T.noGrad(() => buildLoss().data[0]);
      t.data[idx] = orig - eps;
      const minus = T.noGrad(() => buildLoss().data[0]);
      t.data[idx] = orig;
      const numeric = (plus - minus) / (2 * eps);
      const denom = Math.max(1e-3, Math.abs(numeric) + Math.abs(analytic[idx]));
      const relErr = Math.abs(numeric - analytic[idx]) / denom;
      assert.ok(
        relErr < tol,
        `${name}: grad mismatch at ${idx} — analytic ${analytic[idx].toFixed(6)}, numeric ${numeric.toFixed(6)} (rel ${relErr.toFixed(4)})`
      );
    }
  }
}

test('layerNorm gradients', () => {
  const rand = createRandom(3);
  const x = new Tensor(4, 6, null, true);
  for (let i = 0; i < x.size; i++) x.data[i] = rand.normal(0, 1);
  const gamma = new Tensor(1, 6, null, true);
  gamma.data.fill(1.2);
  const beta = new Tensor(1, 6, null, true);
  beta.data.fill(-0.3);
  const targets = Int32Array.from([1, 3, 0, 5]);
  gradCheck(
    'layerNorm',
    () => T.softmaxCrossEntropy(T.layerNorm(x, gamma, beta), targets),
    [x, gamma, beta]
  );
});

test('gelu gradients', () => {
  const rand = createRandom(11);
  const x = new Tensor(3, 5, null, true);
  for (let i = 0; i < x.size; i++) x.data[i] = rand.normal(0, 1.5);
  const targets = Int32Array.from([0, 4, 2]);
  gradCheck('gelu', () => T.softmaxCrossEntropy(T.gelu(x), targets), [x]);
});

test('causalSoftmax gradients and masking', () => {
  const rand = createRandom(5);
  const x = new Tensor(4, 4, null, true);
  for (let i = 0; i < x.size; i++) x.data[i] = rand.normal(0, 1);
  const probs = T.noGrad(() => T.causalSoftmax(x));
  // Row 0 sirf column 0 dekhta hai
  assert.ok(Math.abs(probs.data[0] - 1) < 1e-6, 'row 0 must attend only to itself');
  assert.strictEqual(probs.data[1], 0, 'future position must be masked');
  // Har row ka sum 1
  for (let i = 0; i < 4; i++) {
    let sum = 0;
    for (let j = 0; j < 4; j++) sum += probs.data[i * 4 + j];
    assert.ok(Math.abs(sum - 1) < 1e-5, `row ${i} sums to ${sum}`);
  }
  const v = new Tensor(4, 3, null, true);
  for (let i = 0; i < v.size; i++) v.data[i] = rand.normal(0, 1);
  const targets = Int32Array.from([0, 2, 1, 2]);
  gradCheck('causalSoftmax', () => T.softmaxCrossEntropy(T.matmul(T.causalSoftmax(x), v), targets), [x, v]);
});

test('embed / sliceCols / concatCols gradients', () => {
  const rand = createRandom(19);
  const w = new Tensor(7, 4, null, true);
  for (let i = 0; i < w.size; i++) w.data[i] = rand.normal(0, 1);
  const ids = [2, 5, 2, 0];
  const targets = Int32Array.from([1, 0, 3, 2]);
  gradCheck(
    'embed+slice+concat',
    () => {
      const e = T.embed(w, ids);
      const a = T.sliceCols(e, 0, 2);
      const b = T.sliceCols(e, 2, 4);
      return T.softmaxCrossEntropy(T.concatCols([b, a]), targets);
    },
    [w]
  );
});

test('full model gradients', () => {
  const model = new MineAIModel({ vocabSize: 11, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 8, seed: 42 });
  const ids = [1, 4, 7, 2, 9];
  const targets = Int32Array.from([-1, 3, 5, -1, 2]);
  const params = model.parameters();
  // Har tarah ka param cover karne ke liye kuch chun lete hain.
  const checked = [params[0], params[1], params[3], params[params.length - 1], params[params.length - 4]];
  gradCheck('model', () => model.loss(ids, targets), checked, { eps: 2e-3, tol: 3e-2, samples: 4 });
});

// ---------------------------------------------------------------------------
console.log('\ntokenizer');

test('round-trips text it was trained on', () => {
  const text = 'salam kaise ho\nmain theek hoon shukriya\nsalam kaise ho bhai\n';
  const tok = new Tokenizer().train(text, 200);
  const sentence = 'salam kaise ho bhai';
  assert.strictEqual(tok.decode(tok.encode(sentence)), sentence);
});

test('keeps special tokens atomic', () => {
  const text = '<|user|> salam\n<|bot|> walaikum salam\n<|end|>\n'.repeat(3);
  const tok = new Tokenizer().train(text, 150);
  const ids = tok.encode('<|user|> salam');
  assert.strictEqual(ids[0], tok.id('<|user|>'));
  assert.strictEqual(tok.decode(ids, { skipSpecial: false }), '<|user|> salam');
});

test('handles unseen characters without crashing', () => {
  const tok = new Tokenizer().train('salam kaise ho salam kaise ho\n', 100);
  const ids = tok.encode('zzz### 12345');
  assert.ok(ids.length > 0);
  assert.ok(ids.every((id) => id >= 0 && id < tok.size));
});

// ---------------------------------------------------------------------------
console.log('\ndataset');

test('parses corpus blocks and masks user tokens', () => {
  const corpus = '<|user|> salam\n<|bot|> walaikum salam\n<|end|>\n<|user|> kaise ho\n<|bot|> theek hoon\n<|end|>\n';
  const blocks = parseCorpus(corpus);
  assert.strictEqual(blocks.length, 2);
  assert.strictEqual(blocks[0].turns[0].role, 'user');
  assert.strictEqual(blocks[0].turns[1].text, 'walaikum salam');

  const tok = new Tokenizer().train(corpus, 150);
  const examples = buildExamples(blocks, tok, 64);
  assert.strictEqual(examples.length, 2);
  const ex = examples[0];
  assert.strictEqual(ex.ids.length, ex.targets.length);
  assert.ok(ex.targets[0] === -1, 'user ke pehle token par loss nahi hona chahiye');
  assert.ok(Array.from(ex.targets).some((t) => t >= 0), 'bot tokens supervised hone chahiye');
  // Aakhri supervised target <|end|> hona chahiye
  assert.strictEqual(ex.targets[ex.targets.length - 1], tok.id('<|end|>'));
});

// ---------------------------------------------------------------------------
console.log('\ntraining');

test('model can overfit a tiny dataset', () => {
  const model = new MineAIModel({ vocabSize: 12, blockSize: 10, nLayer: 2, nHead: 2, nEmbd: 16, seed: 7 });
  const data = [
    { ids: [1, 2, 3, 4], targets: Int32Array.from([-1, 5, 6, 7]) },
    { ids: [1, 8, 9, 4], targets: Int32Array.from([-1, 10, 11, 7]) },
  ];
  const opt = new AdamW(model.parameters(), { lr: 5e-2, weightDecay: 0 });
  const lossOf = () => T.noGrad(() => data.reduce((s, d) => s + model.loss(d.ids, d.targets).data[0], 0) / data.length);
  const before = lossOf();
  for (let step = 0; step < 60; step++) {
    opt.zeroGrad();
    for (const d of data) T.scale(model.loss(d.ids, d.targets), 1 / data.length).backward();
    opt.clipGradients(1.0);
    opt.step();
  }
  const after = lossOf();
  assert.ok(after < before * 0.25, `loss should collapse: ${before.toFixed(3)} -> ${after.toFixed(3)}`);
  assert.ok(after < 0.2, `expected near-zero loss, got ${after.toFixed(3)}`);
});

// ---------------------------------------------------------------------------
console.log('\ncheckpoint');

test('weights survive encode/decode round trip', () => {
  const a = new MineAIModel({ vocabSize: 9, blockSize: 6, nLayer: 1, nHead: 2, nEmbd: 8, seed: 1 });
  const b = new MineAIModel({ vocabSize: 9, blockSize: 6, nLayer: 1, nHead: 2, nEmbd: 8, seed: 99 });
  decodeWeights(b.parameters(), encodeWeights(a.parameters()));
  const pa = a.parameters();
  const pb = b.parameters();
  for (let i = 0; i < pa.length; i++) {
    assert.deepStrictEqual(Array.from(pb[i].data), Array.from(pa[i].data), `param ${i} mismatch`);
  }
  const ids = [1, 2, 3];
  const la = T.noGrad(() => a.forward(ids));
  const lb = T.noGrad(() => b.forward(ids));
  assert.deepStrictEqual(Array.from(lb.data), Array.from(la.data));
});

// ---------------------------------------------------------------------------
console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
