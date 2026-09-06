'use strict';
/**
 * Mine AI - training script.
 *
 *   npm run train -- --steps 1200 --batch 12
 *
 * Ye script corpus parhta hai, tokenizer banata hai, model train karta hai
 * aur models/mine-ai.json mein checkpoint save karta hai.
 */

const fs = require('fs');
const path = require('path');
const T = require('../core/tensor');
const { Tokenizer } = require('../core/tokenizer');
const { MineAIModel } = require('../core/model');
const { AdamW, cosineLR } = require('../core/optimizer');
const { parseCorpus, buildExamples, loadCorpusFiles } = require('../core/dataset');
const { saveModel } = require('../core/checkpoint');
const { createRandom } = require('../core/random');

const ROOT = path.resolve(__dirname, '..', '..');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = isNaN(Number(next)) ? next : Number(next);
      i++;
    }
  }
  return args;
}

function formatTime(ms) {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  const config = {
    vocabSize: args.vocab || 900,
    blockSize: args.block || 64,
    nLayer: args.layers || 4,
    nHead: args.heads || 4,
    nEmbd: args.embd || 128,
    seed: args.seed || 1337,
  };
  const steps = args.steps || 1500;
  const batchSize = args.batch || 12;
  const baseLR = args.lr || 3e-3;
  const outPath = path.resolve(ROOT, args.out || 'models/mine-ai.json');
  const corpusDir = path.resolve(ROOT, 'data');

  const corpusFiles = fs
    .readdirSync(corpusDir)
    .filter((f) => f.endsWith('.txt'))
    .map((f) => path.join(corpusDir, f))
    .sort();
  if (!corpusFiles.length) {
    console.error('data/ mein koi .txt corpus nahi mila.');
    process.exit(1);
  }

  console.log('Mine AI - training\n');
  console.log('Corpus files:', corpusFiles.map((f) => path.basename(f)).join(', '));

  const raw = loadCorpusFiles(corpusFiles);
  const blocks = parseCorpus(raw);
  console.log(`Conversations : ${blocks.length}`);

  console.log('Tokenizer train ho raha hai...');
  const tokenizer = new Tokenizer();
  tokenizer.train(raw, config.vocabSize, { verbose: true });
  config.vocabSize = tokenizer.size; // asal vocab chhota ho sakta hai
  console.log(`Vocab size    : ${tokenizer.size}`);

  const examples = buildExamples(blocks, tokenizer, config.blockSize);
  if (!examples.length) {
    console.error('Koi training example nahi bana. Corpus format check karen.');
    process.exit(1);
  }

  const rand = createRandom(config.seed);
  const shuffled = rand.shuffle(examples.slice());
  const valCount = Math.max(1, Math.floor(shuffled.length * 0.05));
  const valSet = shuffled.slice(0, valCount);
  const trainSet = shuffled.slice(valCount);

  const avgLen = trainSet.reduce((n, e) => n + e.ids.length, 0) / trainSet.length;
  console.log(`Examples      : ${trainSet.length} train / ${valSet.length} val (avg ${avgLen.toFixed(1)} tokens)`);

  const model = new MineAIModel(config);
  console.log(`Model         : ${config.nLayer}L ${config.nHead}H ${config.nEmbd}D, ${model.numParams().toLocaleString()} params`);
  console.log(`Training      : ${steps} steps, batch ${batchSize}, lr ${baseLR}\n`);

  const optimizer = new AdamW(model.parameters(), { lr: baseLR, weightDecay: 0.01 });
  const started = Date.now();
  let order = rand.shuffle(trainSet.slice());
  let cursor = 0;
  let bestVal = Infinity;

  const evalLoss = (set) =>
    T.noGrad(() => {
      let total = 0;
      for (const ex of set) total += model.loss(ex.ids, ex.targets).data[0];
      return total / set.length;
    });

  for (let step = 0; step < steps; step++) {
    optimizer.zeroGrad();
    let running = 0;
    for (let b = 0; b < batchSize; b++) {
      if (cursor >= order.length) {
        order = rand.shuffle(trainSet.slice());
        cursor = 0;
      }
      const ex = order[cursor++];
      const loss = model.loss(ex.ids, ex.targets);
      running += loss.data[0];
      T.scale(loss, 1 / batchSize).backward();
    }
    const gradNorm = optimizer.clipGradients(1.0);
    const lr = cosineLR(step, steps, { baseLR, warmup: Math.min(60, Math.floor(steps * 0.05)), minLR: baseLR * 0.1 });
    optimizer.step(lr);

    if (step % 25 === 0 || step === steps - 1) {
      const elapsed = Date.now() - started;
      const eta = step > 0 ? ((elapsed / (step + 1)) * (steps - step - 1)) : 0;
      const trainLoss = running / batchSize;
      let line = `step ${String(step + 1).padStart(5)}/${steps}  loss ${trainLoss.toFixed(4)}  lr ${lr.toExponential(2)}  |g| ${gradNorm.toFixed(2)}  eta ${formatTime(eta)}`;
      if (step % 100 === 0 || step === steps - 1) {
        const vl = evalLoss(valSet);
        line += `  val ${vl.toFixed(4)}`;
        if (vl < bestVal) bestVal = vl;
      }
      console.log(line);
    }
  }

  const finalTrain = evalLoss(trainSet.slice(0, Math.min(64, trainSet.length)));
  const finalVal = evalLoss(valSet);
  console.log(`\nFinal loss: train ${finalTrain.toFixed(4)}  val ${finalVal.toFixed(4)}`);

  saveModel(outPath, {
    model,
    tokenizer,
    meta: {
      steps,
      batchSize,
      baseLR,
      trainLoss: finalTrain,
      valLoss: finalVal,
      examples: trainSet.length,
      corpus: corpusFiles.map((f) => path.basename(f)),
      trainingTimeMs: Date.now() - started,
    },
  });
  const sizeMB = (fs.statSync(outPath).size / 1024 / 1024).toFixed(2);
  console.log(`Model save ho gaya: ${path.relative(ROOT, outPath)} (${sizeMB} MB)`);
  console.log(`Total waqt: ${formatTime(Date.now() - started)}`);
}

main();
