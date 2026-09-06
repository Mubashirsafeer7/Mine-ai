'use strict';
/**
 * Mine AI - Transformer (decoder-only), scratch se.
 *
 * Architecture GPT jaisa hai lekin chhota:
 *   token embedding + learned positional embedding
 *   N x [ LayerNorm -> Multi-Head Causal Self-Attention -> residual
 *         LayerNorm -> MLP (GELU) -> residual ]
 *   final LayerNorm -> tied lm_head (token embedding ka transpose)
 */

const T = require('./tensor');
const { Tensor } = T;
const { createRandom } = require('./random');

class Linear {
  constructor(inDim, outDim, rand, std = 0.02, useBias = true) {
    this.weight = new Tensor(outDim, inDim, null, true); // [out x in], matmulNT ke liye
    for (let i = 0; i < this.weight.data.length; i++) this.weight.data[i] = rand.normal(0, std);
    this.bias = useBias ? new Tensor(1, outDim, null, true) : null;
  }
  forward(x) {
    const y = T.matmulNT(x, this.weight);
    return this.bias ? T.addBias(y, this.bias) : y;
  }
  parameters() {
    return this.bias ? [this.weight, this.bias] : [this.weight];
  }
}

class LayerNormLayer {
  constructor(dim) {
    this.gamma = new Tensor(1, dim, null, true);
    this.beta = new Tensor(1, dim, null, true);
    this.gamma.data.fill(1);
  }
  forward(x) {
    return T.layerNorm(x, this.gamma, this.beta);
  }
  parameters() {
    return [this.gamma, this.beta];
  }
}

class SelfAttention {
  constructor(cfg, rand) {
    this.nHead = cfg.nHead;
    this.headDim = cfg.nEmbd / cfg.nHead;
    if (!Number.isInteger(this.headDim)) throw new Error('nEmbd must be divisible by nHead');
    this.scaleFactor = 1 / Math.sqrt(this.headDim);
    const projStd = 0.02 / Math.sqrt(2 * cfg.nLayer); // deep net ke liye chhota init
    this.q = new Linear(cfg.nEmbd, cfg.nEmbd, rand);
    this.k = new Linear(cfg.nEmbd, cfg.nEmbd, rand);
    this.v = new Linear(cfg.nEmbd, cfg.nEmbd, rand);
    this.proj = new Linear(cfg.nEmbd, cfg.nEmbd, rand, projStd);
  }
  forward(x) {
    const q = this.q.forward(x);
    const k = this.k.forward(x);
    const v = this.v.forward(x);
    const heads = [];
    for (let h = 0; h < this.nHead; h++) {
      const a = h * this.headDim;
      const b = a + this.headDim;
      const qh = T.sliceCols(q, a, b);
      const kh = T.sliceCols(k, a, b);
      const vh = T.sliceCols(v, a, b);
      const scores = T.scale(T.matmulNT(qh, kh), this.scaleFactor);
      const attn = T.causalSoftmax(scores);
      heads.push(T.matmul(attn, vh));
    }
    const merged = this.nHead === 1 ? heads[0] : T.concatCols(heads);
    return this.proj.forward(merged);
  }
  parameters() {
    return [...this.q.parameters(), ...this.k.parameters(), ...this.v.parameters(), ...this.proj.parameters()];
  }
}

class MLP {
  constructor(cfg, rand) {
    const hidden = cfg.nEmbd * 4;
    const projStd = 0.02 / Math.sqrt(2 * cfg.nLayer);
    this.fc = new Linear(cfg.nEmbd, hidden, rand);
    this.proj = new Linear(hidden, cfg.nEmbd, rand, projStd);
  }
  forward(x) {
    return this.proj.forward(T.gelu(this.fc.forward(x)));
  }
  parameters() {
    return [...this.fc.parameters(), ...this.proj.parameters()];
  }
}

class Block {
  constructor(cfg, rand) {
    this.ln1 = new LayerNormLayer(cfg.nEmbd);
    this.attn = new SelfAttention(cfg, rand);
    this.ln2 = new LayerNormLayer(cfg.nEmbd);
    this.mlp = new MLP(cfg, rand);
  }
  forward(x) {
    let h = T.add(x, this.attn.forward(this.ln1.forward(x)));
    h = T.add(h, this.mlp.forward(this.ln2.forward(h)));
    return h;
  }
  parameters() {
    return [...this.ln1.parameters(), ...this.attn.parameters(), ...this.ln2.parameters(), ...this.mlp.parameters()];
  }
}

const DEFAULT_CONFIG = {
  vocabSize: 1024,
  blockSize: 64,
  nLayer: 4,
  nHead: 4,
  nEmbd: 128,
  seed: 1337,
};

class MineAIModel {
  constructor(config = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    const cfg = this.config;
    const rand = createRandom(cfg.seed);
    this.tokenEmbedding = new Tensor(cfg.vocabSize, cfg.nEmbd, null, true);
    for (let i = 0; i < this.tokenEmbedding.data.length; i++) this.tokenEmbedding.data[i] = rand.normal(0, 0.02);
    this.posEmbedding = new Tensor(cfg.blockSize, cfg.nEmbd, null, true);
    for (let i = 0; i < this.posEmbedding.data.length; i++) this.posEmbedding.data[i] = rand.normal(0, 0.01);
    this.blocks = [];
    for (let i = 0; i < cfg.nLayer; i++) this.blocks.push(new Block(cfg, rand));
    this.lnFinal = new LayerNormLayer(cfg.nEmbd);
  }

  parameters() {
    const params = [this.tokenEmbedding, this.posEmbedding];
    for (const b of this.blocks) params.push(...b.parameters());
    params.push(...this.lnFinal.parameters());
    return params;
  }

  numParams() {
    return this.parameters().reduce((n, p) => n + p.size, 0);
  }

  /** ids: number[] / Int32Array (length <= blockSize) -> logits [T x vocab] */
  forward(ids) {
    const t = ids.length;
    if (t > this.config.blockSize) throw new Error(`sequence ${t} > blockSize ${this.config.blockSize}`);
    const posIds = new Int32Array(t);
    for (let i = 0; i < t; i++) posIds[i] = i;
    let x = T.add(T.embed(this.tokenEmbedding, ids), T.embed(this.posEmbedding, posIds));
    for (const block of this.blocks) x = block.forward(x);
    x = this.lnFinal.forward(x);
    return T.matmulNT(x, this.tokenEmbedding); // weight tying
  }

  /** Ek sequence ka loss. targets mein -1 = ignore. */
  loss(ids, targets) {
    return T.softmaxCrossEntropy(this.forward(ids), targets);
  }

  zeroGrad() {
    for (const p of this.parameters()) p.zeroGrad();
  }
}

module.exports = { MineAIModel, Linear, LayerNormLayer, Block, SelfAttention, MLP, DEFAULT_CONFIG };
