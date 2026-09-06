'use strict';
/** AdamW optimizer + global gradient clipping. Scratch se. */

class AdamW {
  constructor(params, { lr = 3e-3, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weightDecay = 0.01 } = {}) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.t = 0;
    this.m = params.map((p) => new Float32Array(p.size));
    this.v = params.map((p) => new Float32Array(p.size));
  }

  /** Gradients ka global L2 norm maxNorm tak seemit karo. Return: original norm. */
  clipGradients(maxNorm = 1.0) {
    let total = 0;
    for (const p of this.params) {
      if (!p.grad) continue;
      for (let i = 0; i < p.grad.length; i++) total += p.grad[i] * p.grad[i];
    }
    const norm = Math.sqrt(total);
    if (norm > maxNorm && norm > 0) {
      const s = maxNorm / norm;
      for (const p of this.params) {
        if (!p.grad) continue;
        for (let i = 0; i < p.grad.length; i++) p.grad[i] *= s;
      }
    }
    return norm;
  }

  step(lr = this.lr) {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      if (!p.grad) continue;
      const m = this.m[pi];
      const v = this.v[pi];
      const isVector = p.rows === 1; // LayerNorm gamma/beta aur bias par decay nahi
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        let update = mHat / (Math.sqrt(vHat) + this.eps);
        if (!isVector && this.weightDecay > 0) update += this.weightDecay * p.data[i];
        p.data[i] -= lr * update;
      }
    }
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
}

/** Warmup + cosine decay schedule. */
function cosineLR(step, totalSteps, { baseLR = 3e-3, warmup = 50, minLR = 3e-4 } = {}) {
  if (step < warmup) return (baseLR * (step + 1)) / warmup;
  const progress = Math.min(1, (step - warmup) / Math.max(1, totalSteps - warmup));
  return minLR + 0.5 * (baseLR - minLR) * (1 + Math.cos(Math.PI * progress));
}

module.exports = { AdamW, cosineLR };
