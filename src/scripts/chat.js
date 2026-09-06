'use strict';
/**
 * Mine AI - terminal chat.
 *
 *   npm run chat
 *   npm run chat -- --mode neural --temp 0.9
 *
 * Commands: /reset  /mode <auto|neural|memory>  /temp <0-1.4>  /info  /exit
 */

const readline = require('readline');
const path = require('path');
const { MineBrain } = require('../core/brain');

const ROOT = path.resolve(__dirname, '..', '..');

const C = {
  dim: (s) => `\x1b[2m${s}\x1b[0m`,
  bold: (s) => `\x1b[1m${s}\x1b[0m`,
  blue: (s) => `\x1b[34m${s}\x1b[0m`,
  cyan: (s) => `\x1b[36m${s}\x1b[0m`,
  green: (s) => `\x1b[32m${s}\x1b[0m`,
  yellow: (s) => `\x1b[33m${s}\x1b[0m`,
  red: (s) => `\x1b[31m${s}\x1b[0m`,
};

const SOURCE_COLOR = {
  neural: C.cyan,
  memory: C.green,
  'memory-rescue': C.yellow,
  fallback: C.red,
};

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

function main() {
  const args = parseArgs(process.argv.slice(2));
  let brain;
  try {
    brain = new MineBrain({ modelPath: args.model ? path.resolve(ROOT, args.model) : undefined });
  } catch (err) {
    console.error(C.red(err.message));
    process.exit(1);
  }

  let mode = ['auto', 'neural', 'memory'].includes(args.mode) ? args.mode : 'auto';
  let temperature = typeof args.temp === 'number' ? args.temp : 0.75;
  let history = [];

  const info = brain.info();
  console.log(C.bold('\n  Mine AI') + C.dim(' — scratch se bana chat AI\n'));
  console.log(
    C.dim(`  ${info.layers}L ${info.heads}H ${info.embedding}D · ${info.params.toLocaleString()} params · vocab ${info.vocabSize} · memory ${info.memoryEntries}`)
  );
  console.log(C.dim('  /reset  /mode <auto|neural|memory>  /temp <0-1.4>  /info  /exit\n'));

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const prompt = () => {
    rl.setPrompt(C.blue('you ') + C.dim('› '));
    rl.prompt();
  };

  prompt();

  rl.on('line', (line) => {
    const text = line.trim();
    if (!text) return prompt();

    if (text.startsWith('/')) {
      const [cmd, value] = text.slice(1).split(/\s+/);
      if (cmd === 'exit' || cmd === 'quit') return rl.close();
      if (cmd === 'reset') {
        history = [];
        console.log(C.dim('  chat reset ho gayi\n'));
        return prompt();
      }
      if (cmd === 'mode') {
        if (['auto', 'neural', 'memory'].includes(value)) {
          mode = value;
          console.log(C.dim(`  mode: ${mode}\n`));
        } else console.log(C.red('  mode: auto | neural | memory\n'));
        return prompt();
      }
      if (cmd === 'temp') {
        const t = Number(value);
        if (!isNaN(t) && t >= 0 && t <= 1.4) {
          temperature = t;
          console.log(C.dim(`  temperature: ${temperature}\n`));
        } else console.log(C.red('  temp 0 se 1.4 ke darmiyan\n'));
        return prompt();
      }
      if (cmd === 'info') {
        console.log(C.dim('  ' + JSON.stringify(brain.info(), null, 2).split('\n').join('\n  ') + '\n'));
        return prompt();
      }
      console.log(C.red(`  anjaan command: /${cmd}\n`));
      return prompt();
    }

    history.push({ role: 'user', text });
    process.stdout.write(C.cyan('ai  ') + C.dim('› '));

    const started = Date.now();
    let printed = '';
    let final = null;
    for (const chunk of brain.stream(history, { mode, temperature })) {
      if (chunk.type === 'token') {
        process.stdout.write(chunk.text);
        printed += chunk.text;
      } else if (chunk.type === 'reset') {
        // Model bhatak gaya - line saaf kar ke dobara likhte hain.
        readline.clearLine(process.stdout, 0);
        readline.cursorTo(process.stdout, 0);
        process.stdout.write(C.cyan('ai  ') + C.dim('› '));
        printed = '';
      } else if (chunk.type === 'done') {
        final = chunk;
      }
    }

    if (final) {
      history.push({ role: 'bot', text: final.text });
      const color = SOURCE_COLOR[final.source] || C.dim;
      const extra = final.source !== 'neural' && final.score ? ` ${final.score.toFixed(2)}` : '';
      console.log('\n' + C.dim('    ') + color(`[${final.source}${extra}]`) + C.dim(` ${Date.now() - started}ms\n`));
    } else {
      console.log('\n');
    }
    prompt();
  });

  rl.on('close', () => {
    console.log(C.dim('\n  Allah hafiz!\n'));
    process.exit(0);
  });
}

main();
