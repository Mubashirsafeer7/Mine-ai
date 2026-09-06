'use strict';
/* =============================================================================
   Mine AI — interface logic. No framework. Chats live in the browser.
   ========================================================================== */

const $ = (id) => document.getElementById(id);

const el = {
  app: $('app'),
  scrim: $('scrim'),
  history: $('history'),
  search: $('searchInput'),
  newChat: $('newChatBtn'),
  collapse: $('collapseBtn'),
  expand: $('expandBtn'),
  menu: $('menuBtn'),
  theme: $('themeBtn'),
  title: $('topbarTitle'),
  stream: $('stream'),
  messages: $('messages'),
  empty: $('empty'),
  chips: $('chips'),
  jump: $('jumpBtn'),
  composer: $('composer'),
  input: $('input'),
  send: $('sendBtn'),
  modeGroup: $('modeGroup'),
  tuneBtn: $('tuneBtn'),
  tunePop: $('tunePop'),
  temperature: $('temperature'),
  tempOut: $('tempOut'),
  maxTokens: $('maxTokens'),
  lenOut: $('lenOut'),
  rail: $('rail'),
  railBtn: $('railBtn'),
  railClose: $('railClose'),
  statGrid: $('statGrid'),
  ctxFill: $('ctxFill'),
  ctxNote: $('ctxNote'),
  modelChipText: $('modelChipText'),
  modelPulse: $('modelPulse'),
  toast: $('toast'),
};

const KEYS = { chats: 'mine-ai/v1/chats', prefs: 'mine-ai/v1/prefs' };

const state = {
  chats: [],
  currentId: null,
  mode: 'auto',
  temperature: 0.75,
  maxTokens: 60,
  theme: 'light',
  busy: false,
  controller: null,
  info: null,
};

/* ------------------------------- storage -------------------------------- */

function readStore(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (err) {
    return fallback;
  }
}

function writeStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (err) {
    /* storage blocked — chats stay for this session only */
  }
}

const saveChats = () => writeStore(KEYS.chats, state.chats.slice(0, 80));
const savePrefs = () =>
  writeStore(KEYS.prefs, {
    mode: state.mode,
    temperature: state.temperature,
    maxTokens: state.maxTokens,
    theme: state.theme,
    sideClosed: el.app.classList.contains('side-closed'),
  });

/* -------------------------------- theme --------------------------------- */

function applyTheme(theme) {
  state.theme = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = state.theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = state.theme === 'dark' ? '#131316' : '#fbfaf8';
}

/* -------------------------------- toast --------------------------------- */

let toastTimer = null;
function toast(message) {
  el.toast.textContent = message;
  el.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove('show'), 1800);
}

/* ------------------------------- helpers -------------------------------- */

function titleFrom(text) {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length > 40 ? clean.slice(0, 40) + '…' : clean || 'New chat';
}

function dayBucket(ts) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (ts >= today) return 'Today';
  if (ts >= today - 86400000) return 'Yesterday';
  if (ts >= today - 7 * 86400000) return 'Previous 7 days';
  return 'Older';
}

/** Rough client-side estimate, only used for the context meter. */
const estimateTokens = (text) => Math.ceil(text.trim().length / 3.4);

const nearBottom = (px = 120) =>
  el.stream.scrollHeight - el.stream.scrollTop - el.stream.clientHeight < px;

function scrollToEnd(force = false) {
  if (force || nearBottom(200)) el.stream.scrollTop = el.stream.scrollHeight;
}

const icon = (paths) => `<svg viewBox="0 0 20 20" aria-hidden="true">${paths}</svg>`;

const ICONS = {
  copy: '<rect x="7" y="7" width="9" height="9" rx="2"/><path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4H5.5A1.5 1.5 0 0 0 4 5.5v6A1.5 1.5 0 0 0 5.5 13H7"/>',
  redo: '<path d="M15.5 8.5A6 6 0 1 0 16 12"/><path d="M16 4v4.5h-4.5"/>',
  close: '<path d="M5.5 5.5l9 9M14.5 5.5l-9 9"/>',
};

/* ------------------------------ chat model ------------------------------ */

const currentChat = () => state.chats.find((c) => c.id === state.currentId) || null;

function newChat({ silent = false } = {}) {
  stop();
  state.currentId = null;
  el.messages.innerHTML = '';
  el.empty.hidden = false;
  el.title.textContent = 'New chat';
  renderHistory();
  updateContextMeter();
  if (!silent) el.input.focus();
}

function ensureChat(firstMessage) {
  const existing = currentChat();
  if (existing) return existing;
  const chat = {
    id: 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    title: titleFrom(firstMessage),
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  state.chats.unshift(chat);
  state.currentId = chat.id;
  el.title.textContent = chat.title;
  return chat;
}

function openChat(id) {
  const chat = state.chats.find((c) => c.id === id);
  if (!chat) return;
  stop();
  state.currentId = id;
  el.title.textContent = chat.title;
  el.empty.hidden = chat.messages.length > 0;
  el.messages.innerHTML = '';
  for (const m of chat.messages) {
    if (m.role === 'user') addUserMessage(m.text);
    else addBotMessage(m.text, m.meta || {});
  }
  renderHistory();
  updateContextMeter();
  closeMobileSidebar();
  requestAnimationFrame(() => scrollToEnd(true));
}

function deleteChat(id, event) {
  event.stopPropagation();
  state.chats = state.chats.filter((c) => c.id !== id);
  saveChats();
  if (state.currentId === id) newChat({ silent: true });
  else renderHistory();
  toast('Chat deleted');
}

/* ------------------------------ rendering ------------------------------- */

function renderHistory() {
  const query = el.search.value.trim().toLowerCase();
  const list = query
    ? state.chats.filter(
        (c) =>
          c.title.toLowerCase().includes(query) ||
          c.messages.some((m) => m.text.toLowerCase().includes(query))
      )
    : state.chats;

  el.history.innerHTML = '';

  if (!list.length) {
    const empty = document.createElement('p');
    empty.className = 'hist-empty';
    empty.textContent = query ? 'No matches' : 'No chats yet';
    el.history.appendChild(empty);
    return;
  }

  let lastBucket = null;
  for (const chat of list) {
    const bucket = dayBucket(chat.updatedAt || chat.createdAt);
    if (bucket !== lastBucket) {
      const head = document.createElement('div');
      head.className = 'hist-group';
      head.textContent = bucket;
      el.history.appendChild(head);
      lastBucket = bucket;
    }

    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'hist-item' + (chat.id === state.currentId ? ' active' : '');
    item.addEventListener('click', () => openChat(chat.id));

    const title = document.createElement('span');
    title.className = 'hist-title';
    title.textContent = chat.title;
    item.appendChild(title);

    const del = document.createElement('span');
    del.className = 'hist-del';
    del.setAttribute('role', 'button');
    del.setAttribute('aria-label', 'Delete chat');
    del.innerHTML = icon(ICONS.close);
    del.addEventListener('click', (e) => deleteChat(chat.id, e));
    item.appendChild(del);

    el.history.appendChild(item);
  }
}

function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg user';
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = text;
  row.appendChild(body);
  el.messages.appendChild(row);
  return row;
}

function makeBotShell() {
  const row = document.createElement('div');
  row.className = 'msg bot';
  row.innerHTML =
    '<svg class="msg-mark" viewBox="0 0 32 32" aria-hidden="true"><use href="#gemMark"/></svg>' +
    '<div class="body"><div class="text"></div></div>';
  el.messages.appendChild(row);
  return { row, body: row.querySelector('.body'), text: row.querySelector('.text') };
}

function metaNode(meta, getText, onRegenerate) {
  const wrap = document.createElement('div');
  wrap.className = 'meta';

  if (meta.source) {
    const src = document.createElement('span');
    src.className = 'src ' + meta.source;
    src.innerHTML = '<span class="dot"></span>' + meta.source.replace('-', ' ');
    wrap.appendChild(src);
  }

  const bits = [];
  if (meta.stopped) bits.push('stopped');
  if (typeof meta.ms === 'number') bits.push(`${meta.ms} ms`);

  bits.forEach((bit) => {
    const sep = document.createElement('span');
    sep.className = 'meta-sep';
    sep.textContent = '·';
    const s = document.createElement('span');
    s.textContent = bit;
    wrap.append(sep, s);
  });

  const tools = document.createElement('div');
  tools.className = 'tools';

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.className = 'tool';
  copyBtn.title = 'Copy';
  copyBtn.setAttribute('aria-label', 'Copy reply');
  copyBtn.innerHTML = icon(ICONS.copy);
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(getText());
      toast('Copied');
    } catch (err) {
      toast('Could not copy');
    }
  });
  tools.appendChild(copyBtn);

  if (onRegenerate) {
    const redo = document.createElement('button');
    redo.type = 'button';
    redo.className = 'tool';
    redo.title = 'Retry';
    redo.setAttribute('aria-label', 'Generate again');
    redo.innerHTML = icon(ICONS.redo);
    redo.addEventListener('click', onRegenerate);
    tools.appendChild(redo);
  }

  wrap.appendChild(tools);
  return wrap;
}

function addBotMessage(text, meta) {
  const shell = makeBotShell();
  shell.text.textContent = text;
  shell.body.appendChild(metaNode(meta, () => text, regenerate));
  return shell;
}

/* -------------------------------- rail ---------------------------------- */

function statCard(label, value, wide = false) {
  const dl = document.createElement('dl');
  dl.className = 'stat' + (wide ? ' wide' : '');
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value;
  dl.append(dt, dd);
  return dl;
}

function renderStats(info) {
  el.statGrid.innerHTML = '';
  const rows = [
    ['Parameters', info.params.toLocaleString()],
    ['Vocab', info.vocabSize.toLocaleString()],
    ['Layers', String(info.layers)],
    ['Heads', String(info.heads)],
    ['Embedding', info.embedding + 'D'],
    ['Context', info.blockSize + ' tok'],
  ];
  for (const [label, value] of rows) el.statGrid.appendChild(statCard(label, value));
  if (typeof info.valLoss === 'number') {
    el.statGrid.appendChild(statCard('Validation loss', info.valLoss.toFixed(4), true));
  }
}

function updateContextMeter() {
  const chat = currentChat();
  const block = state.info ? state.info.blockSize : 64;
  let used = estimateTokens(el.input.value);
  if (chat) {
    for (let i = chat.messages.length - 1; i >= 0 && used < block; i--) {
      used += estimateTokens(chat.messages[i].text) + 1;
    }
  }
  el.ctxFill.style.width = Math.min(100, Math.round((used / block) * 100)) + '%';
  el.ctxNote.textContent = `${Math.min(used, block)} / ${block} tokens`;
}

async function loadInfo() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    if (!res.ok) throw new Error(info.error || 'model not loaded');
    state.info = info;
    renderStats(info);
    el.modelPulse.className = 'pulse live';
    el.modelChipText.textContent =
      `${info.layers}L · ${info.heads}H · ${info.embedding}D · ${Math.round(info.params / 1000)}k`;
    updateContextMeter();
  } catch (err) {
    el.modelPulse.className = 'pulse dead';
    el.modelChipText.textContent = 'no model';
    el.statGrid.innerHTML = '';
    el.statGrid.appendChild(statCard('Model', err.message, true));
    el.statGrid.appendChild(statCard('Fix', 'npm run train', true));
  }
}

/* ------------------------------- sending -------------------------------- */

function setBusy(busy) {
  state.busy = busy;
  el.composer.classList.toggle('busy', busy);
  el.send.disabled = !busy && !el.input.value.trim();
}

function stop() {
  if (state.controller) {
    state.controller.abort();
    state.controller = null;
  }
}

function regenerate() {
  const chat = currentChat();
  if (!chat || state.busy) return;
  const last = chat.messages[chat.messages.length - 1];
  if (!last || last.role !== 'bot') return;
  chat.messages.pop();
  saveChats();
  const rows = el.messages.querySelectorAll('.msg.bot');
  if (rows.length) rows[rows.length - 1].remove();
  run(chat);
}

async function submit(text) {
  const clean = (text || '').trim();
  if (!clean || state.busy) return;

  const chat = ensureChat(clean);
  chat.messages.push({ role: 'user', text: clean });
  chat.updatedAt = Date.now();
  saveChats();

  el.empty.hidden = true;
  addUserMessage(clean);
  renderHistory();
  scrollToEnd(true);

  await run(chat);
}

async function run(chat) {
  setBusy(true);
  const shell = makeBotShell();
  shell.text.innerHTML = '<span class="thinking-line">Thinking…</span>';
  scrollToEnd(true);

  const controller = new AbortController();
  state.controller = controller;

  let answer = '';
  let started = false;
  let finalMeta = null;

  const beginStream = () => {
    if (started) return;
    started = true;
    shell.row.classList.add('streaming');
    shell.text.textContent = '';
  };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        messages: chat.messages,
        mode: state.mode,
        temperature: state.temperature,
        maxNewTokens: state.maxTokens,
      }),
    });

    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `server returned ${res.status}`);
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
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        let event = 'message';
        let data = '';
        for (const line of frame.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7).trim();
          else if (line.startsWith('data: ')) data += line.slice(6);
        }
        if (!data) continue;
        const payload = JSON.parse(data);

        if (event === 'token') {
          beginStream();
          answer += payload.text;
          shell.text.textContent = answer;
          scrollToEnd();
        } else if (event === 'reset') {
          answer = '';
          shell.text.textContent = '';
        } else if (event === 'done') {
          beginStream();
          answer = payload.text;
          shell.text.textContent = answer;
          finalMeta = payload;
        } else if (event === 'error') {
          throw new Error(payload.message);
        }
      }
    }

    if (!finalMeta && answer) finalMeta = { source: 'neural' };
  } catch (err) {
    if (err.name === 'AbortError') {
      if (answer) {
        finalMeta = { source: 'neural', stopped: true };
      } else {
        shell.row.remove();
        state.controller = null;
        setBusy(false);
        return;
      }
    } else {
      beginStream();
      answer = `Something went wrong — ${err.message}`;
      shell.text.textContent = answer;
      finalMeta = { source: 'error' };
    }
  }

  shell.row.classList.remove('streaming');
  state.controller = null;
  setBusy(false);

  const meta = finalMeta || { source: 'error' };
  const text = answer;
  shell.body.appendChild(metaNode(meta, () => text, regenerate));

  if (meta.source !== 'error') {
    chat.messages.push({ role: 'bot', text, meta });
    chat.updatedAt = Date.now();
    saveChats();
    renderHistory();
  }
  updateContextMeter();
  scrollToEnd();
}

/* -------------------------------- events -------------------------------- */

function autoGrow() {
  el.input.style.height = 'auto';
  el.input.style.height = Math.min(el.input.scrollHeight, 180) + 'px';
}

el.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  if (state.busy) {
    stop();
    return;
  }
  const text = el.input.value;
  el.input.value = '';
  autoGrow();
  submit(text);
});

el.input.addEventListener('input', () => {
  autoGrow();
  el.send.disabled = !state.busy && !el.input.value.trim();
  updateContextMeter();
});

el.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    el.composer.requestSubmit();
  }
});

el.chips.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (chip) submit(chip.dataset.q);
});

el.newChat.addEventListener('click', () => newChat());
el.search.addEventListener('input', renderHistory);

el.stream.addEventListener('scroll', () => {
  el.jump.hidden = nearBottom(160);
});
el.jump.addEventListener('click', () => scrollToEnd(true));

function setMode(mode) {
  state.mode = mode;
  el.modeGroup.querySelectorAll('.seg').forEach((b) => {
    b.setAttribute('aria-checked', b.dataset.mode === mode ? 'true' : 'false');
  });
  savePrefs();
}

el.modeGroup.addEventListener('click', (e) => {
  const seg = e.target.closest('.seg');
  if (seg) setMode(seg.dataset.mode);
});

const closeTune = () => (el.tunePop.hidden = true);

el.tuneBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  el.tunePop.hidden = !el.tunePop.hidden;
});
document.addEventListener('click', (e) => {
  if (!el.tunePop.hidden && !e.target.closest('.tune')) closeTune();
});

el.temperature.addEventListener('input', () => {
  state.temperature = Number(el.temperature.value);
  el.tempOut.textContent = state.temperature.toFixed(2);
  savePrefs();
});
el.maxTokens.addEventListener('input', () => {
  state.maxTokens = Number(el.maxTokens.value);
  el.lenOut.textContent = String(state.maxTokens);
  savePrefs();
});

function toggleRail(force) {
  const open = force === undefined ? !el.app.classList.contains('rail-open') : force;
  el.app.classList.toggle('rail-open', open);
  el.rail.hidden = !open;
  el.railBtn.classList.toggle('on', open);
}
el.railBtn.addEventListener('click', () => toggleRail());
el.railClose.addEventListener('click', () => toggleRail(false));

const isMobile = () => window.matchMedia('(max-width: 820px)').matches;

function openMobileSidebar() {
  el.app.classList.add('side-open');
  el.scrim.hidden = false;
  requestAnimationFrame(() => el.scrim.classList.add('show'));
}

function closeMobileSidebar() {
  el.app.classList.remove('side-open');
  el.scrim.classList.remove('show');
  setTimeout(() => {
    if (!el.app.classList.contains('side-open')) el.scrim.hidden = true;
  }, 260);
}

function toggleSidebar() {
  if (isMobile()) {
    if (el.app.classList.contains('side-open')) closeMobileSidebar();
    else openMobileSidebar();
  } else {
    el.app.classList.toggle('side-closed');
    savePrefs();
  }
}

el.menu.addEventListener('click', toggleSidebar);
el.collapse.addEventListener('click', toggleSidebar);
el.expand.addEventListener('click', toggleSidebar);
el.scrim.addEventListener('click', closeMobileSidebar);

el.theme.addEventListener('click', () => {
  applyTheme(state.theme === 'dark' ? 'light' : 'dark');
  savePrefs();
});

document.addEventListener('keydown', (e) => {
  const mod = e.metaKey || e.ctrlKey;
  const key = e.key.toLowerCase();

  if (mod && key === 'k') {
    e.preventDefault();
    newChat();
  } else if (mod && key === 'b') {
    e.preventDefault();
    toggleSidebar();
  } else if (e.key === 'Escape') {
    if (!el.tunePop.hidden) closeTune();
    else if (el.app.classList.contains('side-open')) closeMobileSidebar();
    else if (state.busy) stop();
  }
});

/* --------------------------------- boot --------------------------------- */

function boot() {
  const prefs = readStore(KEYS.prefs, {});
  applyTheme(prefs.theme === 'dark' ? 'dark' : 'light');

  state.temperature = typeof prefs.temperature === 'number' ? prefs.temperature : 0.75;
  state.maxTokens = typeof prefs.maxTokens === 'number' ? prefs.maxTokens : 60;
  el.temperature.value = String(state.temperature);
  el.tempOut.textContent = state.temperature.toFixed(2);
  el.maxTokens.value = String(state.maxTokens);
  el.lenOut.textContent = String(state.maxTokens);

  if (prefs.sideClosed && !isMobile()) el.app.classList.add('side-closed');

  const stored = readStore(KEYS.chats, []);
  state.chats = Array.isArray(stored)
    ? stored.filter((c) => c && typeof c.id === 'string' && Array.isArray(c.messages))
    : [];

  setMode(['auto', 'neural', 'memory'].includes(prefs.mode) ? prefs.mode : 'auto');
  renderHistory();
  setBusy(false);
  loadInfo();
  el.input.focus();
}

boot();
