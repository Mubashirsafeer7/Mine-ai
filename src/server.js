'use strict';
/**
 * Mine AI - web server. Sirf Node ke built-in modules, koi package nahi.
 *
 *   npm start           -> http://localhost:3000
 *   PORT=8080 npm start
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { MineBrain } = require('./core/brain');

const ROOT = path.resolve(__dirname, '..');
const WEB_DIR = path.join(ROOT, 'web');
const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '0.0.0.0';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.png': 'image/png',
  '.webmanifest': 'application/manifest+json',
};

let brain = null;
let brainError = null;
try {
  brain = new MineBrain();
} catch (err) {
  brainError = err.message;
}

function sendJSON(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(data),
    'Cache-Control': 'no-store',
  });
  res.end(data);
}

function serveStatic(req, res, urlPath) {
  const rel = urlPath === '/' ? 'index.html' : urlPath.replace(/^\/+/, '');
  const filePath = path.join(WEB_DIR, rel);
  // Directory traversal band.
  if (!filePath.startsWith(WEB_DIR + path.sep) && filePath !== path.join(WEB_DIR, 'index.html')) {
    res.writeHead(403).end('Forbidden');
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('404 - not found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  });
}

function readBody(req, limit = 256 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error('body too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

/** Har token ke baad event loop ko sans dena, warna stream ek sath aata hai. */
const breathe = () => new Promise((resolve) => setImmediate(resolve));

function sanitizeHistory(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter((m) => m && typeof m.text === 'string' && m.text.trim())
    .map((m) => ({
      role: m.role === 'bot' || m.role === 'assistant' ? 'bot' : 'user',
      text: m.text.trim().slice(0, 600),
    }))
    .slice(-12);
}

async function handleChat(req, res) {
  if (!brain) {
    sendJSON(res, 503, { error: brainError || 'model load nahi hua' });
    return;
  }

  let payload;
  try {
    payload = JSON.parse(await readBody(req));
  } catch (err) {
    sendJSON(res, 400, { error: 'invalid json' });
    return;
  }

  const history = sanitizeHistory(payload.messages);
  if (!history.length) {
    sendJSON(res, 400, { error: 'messages khali hain' });
    return;
  }

  const options = {
    mode: ['auto', 'neural', 'memory'].includes(payload.mode) ? payload.mode : 'auto',
    temperature: Math.min(1.5, Math.max(0, Number(payload.temperature) || 0.75)),
    topK: Math.min(200, Math.max(0, Number(payload.topK) || 40)),
    topP: Math.min(1, Math.max(0.05, Number(payload.topP) || 0.9)),
    maxNewTokens: Math.min(200, Math.max(8, Number(payload.maxNewTokens) || 60)),
  };

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  let closed = false;
  req.on('close', () => {
    closed = true;
  });

  const startedAt = Date.now();
  try {
    for (const chunk of brain.stream(history, options)) {
      if (closed) return;
      if (chunk.type === 'token') send('token', { text: chunk.text });
      else if (chunk.type === 'reset') send('reset', {});
      else if (chunk.type === 'done') {
        send('done', {
          text: chunk.text,
          source: chunk.source,
          score: Number((chunk.score || 0).toFixed(3)),
          ms: Date.now() - startedAt,
        });
      }
      await breathe();
    }
  } catch (err) {
    send('error', { message: err.message });
  }
  res.end();
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);

  if (url.pathname === '/api/chat' && req.method === 'POST') {
    await handleChat(req, res);
    return;
  }
  if (url.pathname === '/api/info' && req.method === 'GET') {
    sendJSON(res, brain ? 200 : 503, brain ? brain.info() : { error: brainError });
    return;
  }
  if (req.method !== 'GET') {
    res.writeHead(405).end('Method not allowed');
    return;
  }
  serveStatic(req, res, url.pathname);
});

server.listen(PORT, HOST, () => {
  console.log('\n  Mine AI');
  if (brain) {
    const info = brain.info();
    console.log(`  model: ${info.layers}L ${info.heads}H ${info.embedding}D, ${info.params.toLocaleString()} params`);
    console.log(`  memory: ${info.memoryEntries} conversations`);
  } else {
    console.log(`  ⚠ model load nahi hua: ${brainError}`);
  }
  console.log(`  http://localhost:${PORT}\n`);
});
