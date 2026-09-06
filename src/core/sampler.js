'use strict';
/** Token sampling: temperature, top-k, top-p (nucleus), repetition penalty. */

const T = require('./tensor');
const { createRandom } = require('./random');

function softmaxInPlace(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.exp(logits[i] - max);
    sum += logits[i];
  }
  for (let i = 0; i < logits.length; i++) logits[i] /= sum;
  return logits;
}

function sampleFromLogits(rawLogits, opts) {
  const {
    temperature = 0.8,
    topK = 40,
    topP = 0.9,
    repetitionPenalty = 1.15,
    recentTokens = [],
    banned = [],
    rand = Math.random,
  } = opts;

  const logits = Float32Array.from(rawLogits);

  for (const id of banned) logits[id] = -Infinity;

  if (repetitionPenalty !== 1) {
    const seen = new Set(recentTokens);
    for (const id of seen) {
      if (logits[id] > 0) logits[id] /= repetitionPenalty;
      else logits[id] *= repetitionPenalty;
    }
  }

  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  for (let i = 0; i < logits.length; i++) logits[i] /= temperature;

  const probs = softmaxInPlace(logits);
  let candidates = [];
  for (let i = 0; i < probs.length; i++) if (probs[i] > 0) candidates.push([i, probs[i]]);
  candidates.sort((a, b) => b[1] - a[1]);

  if (topK > 0 && candidates.length > topK) candidates = candidates.slice(0, topK);

  if (topP > 0 && topP < 1) {
    let cum = 0;
    const kept = [];
    for (const c of candidates) {
      kept.push(c);
      cum += c[1];
      if (cum >= topP) break;
    }
    candidates = kept;
  }

  let total = 0;
  for (const c of candidates) total += c[1];
  let r = rand() * total;
  for (const c of candidates) {
    r -= c[1];
    if (r <= 0) return c[0];
  }
  return candidates[candidates.length - 1][0];
}

/**
 * Prompt ids se aage tokens banao. Generator hai, is liye server
 * har token turant stream kar sakta hai.
 */
function* generate(model, promptIds, options = {}) {
  const {
    maxNewTokens = 80,
    temperature = 0.8,
    topK = 40,
    topP = 0.9,
    repetitionPenalty = 1.15,
    stopTokens = [],
    seed = null,
  } = options;

  const rand = seed === null ? Math.random : createRandom(seed);
  const blockSize = model.config.blockSize;
  const stop = new Set(stopTokens);
  let context = Array.from(promptIds);

  for (let n = 0; n < maxNewTokens; n++) {
    const window = context.slice(-blockSize);
    const logits = T.noGrad(() => model.forward(window));
    const last = logits.data.subarray((window.length - 1) * logits.cols, window.length * logits.cols);
    const next = sampleFromLogits(last, {
      temperature,
      topK,
      topP,
      repetitionPenalty,
      recentTokens: context.slice(-24),
      rand,
    });
    if (stop.has(next)) return;
    context.push(next);
    yield next;
  }
}

module.exports = { generate, sampleFromLogits, softmaxInPlace };
