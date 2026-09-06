'use strict';
/**
 * Mine AI - Tensor + reverse-mode autograd.
 *
 * Sab kuch scratch se: koi numpy nahi, koi torch nahi.
 * Tensor hamesha 2D hai (rows x cols), row-major Float32Array mein.
 * Batch ko hum loop se handle karte hain, is liye 3D ki zaroorat nahi padti.
 */

let GRAD_ENABLED = true;

function noGrad(fn) {
  const prev = GRAD_ENABLED;
  GRAD_ENABLED = false;
  try {
    return fn();
  } finally {
    GRAD_ENABLED = prev;
  }
}

class Tensor {
  constructor(rows, cols, data = null, requiresGrad = false) {
    this.rows = rows;
    this.cols = cols;
    this.data = data || new Float32Array(rows * cols);
    this.requiresGrad = requiresGrad;
    this.grad = requiresGrad ? new Float32Array(rows * cols) : null;
    this._parents = [];
    this._backward = null;
  }

  get size() {
    return this.rows * this.cols;
  }

  ensureGrad() {
    if (!this.grad) this.grad = new Float32Array(this.rows * this.cols);
    return this.grad;
  }

  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }

  clone() {
    return new Tensor(this.rows, this.cols, this.data.slice(), false);
  }

  /** Scalar tensor se poore graph par gradients chalao. */
  backward() {
    const topo = [];
    const seen = new Set();
    (function visit(t) {
      if (seen.has(t)) return;
      seen.add(t);
      for (const p of t._parents) visit(p);
      topo.push(t);
    })(this);

    this.ensureGrad();
    if (this.size === 1) this.grad[0] = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (t._backward) t._backward();
    }
  }
}

// ---------------------------------------------------------------------------
// Matrix primitives. Yehi sab se zyada time khate hain, is liye seedhe loops.
// ---------------------------------------------------------------------------

/** C[MxN] = A[MxK] @ B[KxN]  (overwrite) */
function gemm(A, M, K, B, N, C) {
  C.fill(0);
  for (let i = 0; i < M; i++) {
    const aOff = i * K;
    const cOff = i * N;
    for (let k = 0; k < K; k++) {
      const a = A[aOff + k];
      if (a === 0) continue;
      const bOff = k * N;
      for (let j = 0; j < N; j++) C[cOff + j] += a * B[bOff + j];
    }
  }
}

/** C[MxN] += A[MxK] @ B[NxK]^T */
function gemmABT(A, M, K, B, N, C) {
  for (let i = 0; i < M; i++) {
    const aOff = i * K;
    const cOff = i * N;
    for (let j = 0; j < N; j++) {
      const bOff = j * K;
      let s = 0;
      for (let k = 0; k < K; k++) s += A[aOff + k] * B[bOff + k];
      C[cOff + j] += s;
    }
  }
}

/** C[KxN] += A[MxK]^T @ B[MxN] */
function gemmATB(A, M, K, B, N, C) {
  for (let m = 0; m < M; m++) {
    const aOff = m * K;
    const bOff = m * N;
    for (let k = 0; k < K; k++) {
      const a = A[aOff + k];
      if (a === 0) continue;
      const cOff = k * N;
      for (let j = 0; j < N; j++) C[cOff + j] += a * B[bOff + j];
    }
  }
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

function mkOut(rows, cols, parents) {
  const rg = GRAD_ENABLED && parents.some((p) => p.requiresGrad);
  const out = new Tensor(rows, cols, null, rg);
  if (rg) out._parents = parents;
  return out;
}

/** a[MxK] @ b[KxN] */
function matmul(a, b) {
  if (a.cols !== b.rows) throw new Error(`matmul shape mismatch ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
  const out = mkOut(a.rows, b.cols, [a, b]);
  gemm(a.data, a.rows, a.cols, b.data, b.cols, out.data);
  if (out.requiresGrad) {
    out._backward = () => {
      const g = out.grad;
      if (a.requiresGrad) gemmABT(g, a.rows, b.cols, b.data, a.cols, a.ensureGrad());
      if (b.requiresGrad) gemmATB(a.data, a.rows, a.cols, g, b.cols, b.ensureGrad());
    };
  }
  return out;
}

/** a[MxK] @ b[NxK]^T  -> [MxN]. Attention scores aur tied lm-head ke liye. */
function matmulNT(a, b) {
  if (a.cols !== b.cols) throw new Error(`matmulNT shape mismatch ${a.rows}x${a.cols} , ${b.rows}x${b.cols}`);
  const out = mkOut(a.rows, b.rows, [a, b]);
  out.data.fill(0);
  gemmABT(a.data, a.rows, a.cols, b.data, b.rows, out.data);
  if (out.requiresGrad) {
    out._backward = () => {
      const g = out.grad; // [M x N]
      // dA[MxK] += g[MxN] @ B[NxK]
      if (a.requiresGrad) {
        const dA = a.ensureGrad();
        const M = a.rows, N = b.rows, K = a.cols;
        for (let i = 0; i < M; i++) {
          const gOff = i * N, aOff = i * K;
          for (let n = 0; n < N; n++) {
            const gv = g[gOff + n];
            if (gv === 0) continue;
            const bOff = n * K;
            for (let k = 0; k < K; k++) dA[aOff + k] += gv * b.data[bOff + k];
          }
        }
      }
      // dB[NxK] += g^T[NxM] @ A[MxK]
      if (b.requiresGrad) gemmATB(g, a.rows, b.rows, a.data, a.cols, b.ensureGrad());
    };
  }
  return out;
}

/** x[MxN] + bias[1xN] (broadcast) */
function addBias(x, bias) {
  const out = mkOut(x.rows, x.cols, [x, bias]);
  const N = x.cols;
  for (let i = 0; i < x.rows; i++) {
    const off = i * N;
    for (let j = 0; j < N; j++) out.data[off + j] = x.data[off + j] + bias.data[j];
  }
  if (out.requiresGrad) {
    out._backward = () => {
      const g = out.grad;
      if (x.requiresGrad) {
        const dx = x.ensureGrad();
        for (let i = 0; i < g.length; i++) dx[i] += g[i];
      }
      if (bias.requiresGrad) {
        const db = bias.ensureGrad();
        for (let i = 0; i < x.rows; i++) {
          const off = i * N;
          for (let j = 0; j < N; j++) db[j] += g[off + j];
        }
      }
    };
  }
  return out;
}

/** element-wise a + b (same shape) */
function add(a, b) {
  if (a.rows !== b.rows || a.cols !== b.cols) throw new Error('add shape mismatch');
  const out = mkOut(a.rows, a.cols, [a, b]);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  if (out.requiresGrad) {
    out._backward = () => {
      const g = out.grad;
      if (a.requiresGrad) {
        const d = a.ensureGrad();
        for (let i = 0; i < g.length; i++) d[i] += g[i];
      }
      if (b.requiresGrad) {
        const d = b.ensureGrad();
        for (let i = 0; i < g.length; i++) d[i] += g[i];
      }
    };
  }
  return out;
}

/** x * k (scalar) */
function scale(x, k) {
  const out = mkOut(x.rows, x.cols, [x]);
  for (let i = 0; i < x.data.length; i++) out.data[i] = x.data[i] * k;
  if (out.requiresGrad) {
    out._backward = () => {
      const d = x.ensureGrad();
      for (let i = 0; i < out.grad.length; i++) d[i] += out.grad[i] * k;
    };
  }
  return out;
}

const GELU_C = Math.sqrt(2 / Math.PI);

/** GELU (tanh approximation) */
function gelu(x) {
  const out = mkOut(x.rows, x.cols, [x]);
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i];
    const inner = GELU_C * (v + 0.044715 * v * v * v);
    out.data[i] = 0.5 * v * (1 + Math.tanh(inner));
  }
  if (out.requiresGrad) {
    out._backward = () => {
      const d = x.ensureGrad();
      for (let i = 0; i < x.data.length; i++) {
        const v = x.data[i];
        const inner = GELU_C * (v + 0.044715 * v * v * v);
        const t = Math.tanh(inner);
        const dInner = GELU_C * (1 + 3 * 0.044715 * v * v);
        const dv = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * dInner;
        d[i] += out.grad[i] * dv;
      }
    };
  }
  return out;
}

/** Row-wise LayerNorm: gamma/beta [1 x N] */
function layerNorm(x, gamma, beta, eps = 1e-5) {
  const out = mkOut(x.rows, x.cols, [x, gamma, beta]);
  const N = x.cols;
  const xhat = new Float32Array(x.data.length);
  const invStd = new Float32Array(x.rows);
  for (let i = 0; i < x.rows; i++) {
    const off = i * N;
    let mean = 0;
    for (let j = 0; j < N; j++) mean += x.data[off + j];
    mean /= N;
    let variance = 0;
    for (let j = 0; j < N; j++) {
      const d = x.data[off + j] - mean;
      variance += d * d;
    }
    variance /= N;
    const inv = 1 / Math.sqrt(variance + eps);
    invStd[i] = inv;
    for (let j = 0; j < N; j++) {
      const h = (x.data[off + j] - mean) * inv;
      xhat[off + j] = h;
      out.data[off + j] = h * gamma.data[j] + beta.data[j];
    }
  }
  if (out.requiresGrad) {
    out._backward = () => {
      const g = out.grad;
      const dx = x.requiresGrad ? x.ensureGrad() : null;
      const dg = gamma.requiresGrad ? gamma.ensureGrad() : null;
      const db = beta.requiresGrad ? beta.ensureGrad() : null;
      for (let i = 0; i < x.rows; i++) {
        const off = i * N;
        let sumDy = 0;
        let sumDyXhat = 0;
        for (let j = 0; j < N; j++) {
          const dy = g[off + j] * gamma.data[j];
          sumDy += dy;
          sumDyXhat += dy * xhat[off + j];
          if (dg) dg[j] += g[off + j] * xhat[off + j];
          if (db) db[j] += g[off + j];
        }
        if (dx) {
          const inv = invStd[i];
          for (let j = 0; j < N; j++) {
            const dy = g[off + j] * gamma.data[j];
            dx[off + j] += inv * (dy - sumDy / N - xhat[off + j] * sumDyXhat / N);
          }
        }
      }
    };
  }
  return out;
}

/**
 * Causal masked row-wise softmax.
 * Row i sirf columns 0..i tak dekh sakta hai (future tokens blocked).
 */
function causalSoftmax(x) {
  const out = mkOut(x.rows, x.cols, [x]);
  const N = x.cols;
  for (let i = 0; i < x.rows; i++) {
    const off = i * N;
    const limit = Math.min(i, N - 1);
    let max = -Infinity;
    for (let j = 0; j <= limit; j++) if (x.data[off + j] > max) max = x.data[off + j];
    let sum = 0;
    for (let j = 0; j <= limit; j++) {
      const e = Math.exp(x.data[off + j] - max);
      out.data[off + j] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let j = 0; j <= limit; j++) out.data[off + j] *= inv;
    for (let j = limit + 1; j < N; j++) out.data[off + j] = 0;
  }
  if (out.requiresGrad) {
    out._backward = () => {
      const d = x.ensureGrad();
      for (let i = 0; i < x.rows; i++) {
        const off = i * N;
        const limit = Math.min(i, N - 1);
        let dot = 0;
        for (let j = 0; j <= limit; j++) dot += out.grad[off + j] * out.data[off + j];
        for (let j = 0; j <= limit; j++) {
          d[off + j] += out.data[off + j] * (out.grad[off + j] - dot);
        }
      }
    };
  }
  return out;
}

/** x[:, start:end] */
function sliceCols(x, start, end) {
  const width = end - start;
  const out = mkOut(x.rows, width, [x]);
  for (let i = 0; i < x.rows; i++) {
    const src = i * x.cols + start;
    const dst = i * width;
    for (let j = 0; j < width; j++) out.data[dst + j] = x.data[src + j];
  }
  if (out.requiresGrad) {
    out._backward = () => {
      const d = x.ensureGrad();
      for (let i = 0; i < x.rows; i++) {
        const src = i * x.cols + start;
        const dst = i * width;
        for (let j = 0; j < width; j++) d[src + j] += out.grad[dst + j];
      }
    };
  }
  return out;
}

/** Column-wise concat (attention heads jodne ke liye) */
function concatCols(parts) {
  const rows = parts[0].rows;
  let cols = 0;
  for (const p of parts) cols += p.cols;
  const out = mkOut(rows, cols, parts);
  let colOff = 0;
  for (const p of parts) {
    for (let i = 0; i < rows; i++) {
      const src = i * p.cols;
      const dst = i * cols + colOff;
      for (let j = 0; j < p.cols; j++) out.data[dst + j] = p.data[src + j];
    }
    colOff += p.cols;
  }
  if (out.requiresGrad) {
    out._backward = () => {
      let off = 0;
      for (const p of parts) {
        if (p.requiresGrad) {
          const d = p.ensureGrad();
          for (let i = 0; i < rows; i++) {
            const src = i * cols + off;
            const dst = i * p.cols;
            for (let j = 0; j < p.cols; j++) d[dst + j] += out.grad[src + j];
          }
        }
        off += p.cols;
      }
    };
  }
  return out;
}

/** Embedding lookup: weight[V x D], ids -> [T x D] */
function embed(weight, ids) {
  const D = weight.cols;
  const out = mkOut(ids.length, D, [weight]);
  for (let i = 0; i < ids.length; i++) {
    const src = ids[i] * D;
    const dst = i * D;
    for (let j = 0; j < D; j++) out.data[dst + j] = weight.data[src + j];
  }
  if (out.requiresGrad) {
    out._backward = () => {
      const d = weight.ensureGrad();
      for (let i = 0; i < ids.length; i++) {
        const dst = ids[i] * D;
        const src = i * D;
        for (let j = 0; j < D; j++) d[dst + j] += out.grad[src + j];
      }
    };
  }
  return out;
}

/**
 * Softmax + cross entropy ek saath (numerically stable).
 * logits [T x V], targets Int32Array[T]; -1 target ko ignore kiya jata hai.
 * Return: scalar tensor (mean loss over counted tokens).
 */
function softmaxCrossEntropy(logits, targets) {
  const T = logits.rows;
  const V = logits.cols;
  const probs = new Float32Array(T * V);
  let loss = 0;
  let count = 0;
  for (let i = 0; i < T; i++) {
    const off = i * V;
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (logits.data[off + j] > max) max = logits.data[off + j];
    let sum = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp(logits.data[off + j] - max);
      probs[off + j] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let j = 0; j < V; j++) probs[off + j] *= inv;
    const t = targets[i];
    if (t >= 0) {
      loss += -Math.log(Math.max(probs[off + t], 1e-12));
      count++;
    }
  }
  const denom = count || 1;
  const out = mkOut(1, 1, [logits]);
  out.data[0] = loss / denom;
  if (out.requiresGrad) {
    out._backward = () => {
      const d = logits.ensureGrad();
      const g = out.grad[0] / denom;
      for (let i = 0; i < T; i++) {
        const t = targets[i];
        if (t < 0) continue;
        const off = i * V;
        for (let j = 0; j < V; j++) d[off + j] += g * probs[off + j];
        d[off + t] -= g;
      }
    };
  }
  return out;
}

module.exports = {
  Tensor,
  noGrad,
  matmul,
  matmulNT,
  addBias,
  add,
  scale,
  gelu,
  layerNorm,
  causalSoftmax,
  sliceCols,
  concatCols,
  embed,
  softmaxCrossEntropy,
  gemm,
  gemmABT,
  gemmATB,
};
