'use strict';
/** Seeded RNG (mulberry32) taake training reproducible rahe. */

function createRandom(seed = 1337) {
  let a = seed >>> 0;
  const rand = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rand.normal = (mean = 0, std = 1) => {
    // Box-Muller
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  rand.int = (n) => Math.floor(rand() * n);
  rand.shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = rand.int(i + 1);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  };
  return rand;
}

module.exports = { createRandom };
