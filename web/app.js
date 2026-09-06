'use strict';
/** Mine AI - frontend. Koi framework nahi, plain JS. */

const chatEl = document.getElementById('chat');
const formEl = document.getElementById('composer');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const modeEl = document.getElementById('mode');
const tempEl = document.getElementById('temperature');
const tempValueEl = document.getElementById('tempValue');
const modelLineEl = document.getElementById('modelLine');
const statusLineEl = document.getElementById('statusLine');
const clearBtn = document.getElementById('clearBtn');
const themeBtn = document.getElementById('themeBtn');
const welcomeTemplate = document.getElementById('welcome').cloneNode(true);

let history = [];
let busy = false;

// ---------------------------------------------------------------- theme

const savedTheme = (() => {
  try {
    return localStorage.getItem('mine-ai-theme');
  } catch (err) {
    return null;
  }
})();
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

themeBtn.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem('mine-ai-theme', next);
  } catch (err) {
    /* private mode - koi baat nahi */
  }
});

// ---------------------------------------------------------------- helpers

function scrollToEnd() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.remove();
}

function addRow(role) {
  const row = document.createElement('div');
  row.className = `row ${role}`;

  const bubbleWrap = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubbleWrap.appendChild(bubble);

  if (role === 'bot') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '◈';
    row.appendChild(avatar);
  }
  row.appendChild(bubbleWrap);
  chatEl.appendChild(row);
  scrollToEnd();
  return { row, bubble, bubbleWrap };
}

function setTyping(bubble) {
  bubble.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
}

function setMeta(wrap, { source, score, ms }) {
  const meta = document.createElement('div');
  meta.className = 'meta';

  const badge = document.createElement('span');
  badge.className = `badge ${source}`;
  badge.textContent = source.replace('-', ' ');
  meta.appendChild(badge);

  const bits = [];
  if (typeof ms === 'number') bits.push(`${ms} ms`);
  if (source !== 'neural' && typeof score === 'number' && score > 0) bits.push(`match ${score.toFixed(2)}`);
  if (bits.length) {
    const info = document.createElement('span');
    info.textContent = bits.join(' · ');
    meta.appendChild(info);
  }
  wrap.appendChild(meta);
}

function autoGrow() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}

// ---------------------------------------------------------------- info

async function loadInfo() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    if (!res.ok) throw new Error(info.error || 'model load nahi hua');
    modelLineEl.textContent =
      `${info.layers}L · ${info.heads}H · ${info.embedding}D · ${info.params.toLocaleString()} params · vocab ${info.vocabSize}`;
    const val = typeof info.valLoss === 'number' ? ` · val loss ${info.valLoss.toFixed(3)}` : '';
    statusLineEl.textContent = `Sab kuch local — koi API nahi. ${info.memoryEntries} conversations memory mein${val}.`;
  } catch (err) {
    modelLineEl.textContent = 'model nahi mila';
    statusLineEl.textContent = `${err.message} — pehle chalayen:  npm run train`;
  }
}

// ---------------------------------------------------------------- chat

async function send(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  hideWelcome();

  const userText = text.trim();
  history.push({ role: 'user', text: userText });
  addRow('user').bubble.textContent = userText;

  const { bubble, bubbleWrap } = addRow('bot');
  setTyping(bubble);

  let answer = '';
  let started = false;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: history,
        mode: modeEl.value,
        temperature: Number(tempEl.value),
      }),
    });

    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `server error ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const lines = raw.split('\n');
        let event = 'message';
        let data = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) event = line.slice(7).trim();
          else if (line.startsWith('data: ')) data += line.slice(6);
        }
        if (!data) continue;
        const payload = JSON.parse(data);

        if (event === 'token') {
          if (!started) {
            bubble.textContent = '';
            started = true;
          }
          answer += payload.text;
          bubble.textContent = answer;
          scrollToEnd();
        } else if (event === 'reset') {
          answer = '';
          bubble.textContent = '';
        } else if (event === 'done') {
          answer = payload.text;
          bubble.textContent = answer;
          setMeta(bubbleWrap, payload);
          history.push({ role: 'bot', text: answer });
          scrollToEnd();
        } else if (event === 'error') {
          throw new Error(payload.message);
        }
      }
    }
  } catch (err) {
    bubble.textContent = `Masla aa gaya: ${err.message}`;
    setMeta(bubbleWrap, { source: 'error' });
  } finally {
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// ---------------------------------------------------------------- events

formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = inputEl.value;
  inputEl.value = '';
  autoGrow();
  send(text);
});

inputEl.addEventListener('input', autoGrow);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

tempEl.addEventListener('input', () => {
  tempValueEl.textContent = Number(tempEl.value).toFixed(2);
});

clearBtn.addEventListener('click', () => {
  history = [];
  chatEl.innerHTML = '';
  chatEl.appendChild(welcomeTemplate.cloneNode(true));
  bindChips();
  inputEl.focus();
});

function bindChips() {
  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => send(chip.textContent.trim()));
  });
}

bindChips();
loadInfo();
inputEl.focus();
