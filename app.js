import { fetchManifest, unlock, loadIndex, retrieve, deepRetrieve, conceptRetrieve, isDesignAsk, warmReranker } from './lib/rag.js';
import { PROVIDERS, SYSTEM, buildContext, isLocal, checkCitations, stream, rewriteQuery, followUpQuery, llmRank, splitThink, describeError, buildUserMessage } from './lib/llm.js';
import * as db from './lib/db.js';

const $ = (id) => document.getElementById(id);
const ls = { get: (k, d = '') => { try { return localStorage.getItem(k) ?? d; } catch { return d; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* blocked */ } }, del: (k) => { try { localStorage.removeItem(k); } catch { /* blocked */ } } };
const ss = { get: (k) => { try { return sessionStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { sessionStorage.setItem(k, v); } catch { /* blocked */ } }, del: (k) => { try { sessionStorage.removeItem(k); } catch { /* blocked */ } } };
let man, ix, conv, convs = [], abort;

// ---------- unlock ----------
async function tryUnlock(pass) {
  $('gateMsg').textContent = 'Deriving key...';
  const key = await unlock(man, pass);
  if (!key) { $('gateMsg').textContent = 'Wrong passphrase.'; return false; }
  $('gateMsg').textContent = 'Decrypting library...';
  ix = await loadIndex(man, key);
  $('gate').hidden = true; $('demo').textContent = 'internal library'; $('status').textContent = `v${man.version} · ${ix.docs.length} chunks · ${ix.books.length} book(s)`;
  window.__ready = true;
  if (ls.get('rerank', '1') === '1') warmReranker(); // load the cross-encoder in the background
  return true;
}
$('gateForm').onsubmit = async (e) => {
  e.preventDefault(); const p = $('pass').value.trim();
  try { if (await tryUnlock(p) && $('remember').checked) ss.set('pp', p); } catch (err) { $('gateMsg').textContent = 'Error: ' + err.message; }
  $('pass').value = '';
};
$('btnLock').onclick = () => { ss.del('pp'); ix = null; location.reload(); };

// ---------- settings (one key per provider) ----------
const cfg = () => { const provider = ls.get('provider', 'anthropic'); return { provider, model: ls.get('model-' + provider, PROVIDERS[provider].model), key: ls.get('key-' + provider), base: provider === 'openai' ? ls.get('base') : '', timeoutMs: Math.max(5, Number(ls.get('llmTimeout', '30')) || 30) * 1000 }; };
function fillSettings() {
  const c = cfg(); $('prov').innerHTML = Object.entries(PROVIDERS).map(([k, v]) => `<option value="${k}" ${k === c.provider ? 'selected' : ''}>${v.label}</option>`).join('');
  $('model').value = c.model; $('key').value = c.key; $('base').value = ls.get('base'); $('timeout').value = ls.get('llmTimeout', '30');
  $('models').innerHTML = PROVIDERS[c.provider].models.map((m) => `<option value="${m}">`).join('');
  $('useRerank').checked = ls.get('rerank', '1') === '1'; $('llmRerank').checked = ls.get('llmRerank') === '1'; $('deep').checked = ls.get('deep') === '1';
}
$('btnSet').onclick = () => { fillSettings(); $('dlg').showModal(); };
$('prov').onchange = () => { ls.set('provider', $('prov').value); fillSettings(); };
$('dlg').onclose = () => { const p = $('prov').value; ls.set('provider', p); ls.set('model-' + p, $('model').value); ls.set('key-' + p, $('key').value); ls.set('base', $('base').value); ls.set('llmTimeout', String(Math.max(5, Number($('timeout').value) || 30))); ls.set('rerank', $('useRerank').checked ? '1' : '0'); ls.set('llmRerank', $('llmRerank').checked ? '1' : '0'); ls.set('deep', $('deep').checked ? '1' : '0'); };
$('forget').onclick = () => { ls.del('key-' + $('prov').value); $('key').value = ''; };

// ---------- conversations ----------
async function refresh() {
  convs = await db.listConvs();
  $('convList').innerHTML = '';
  for (const c of convs) {
    const d = document.createElement('div'); d.className = 'conv' + (c.id === conv?.id ? ' on' : '');
    d.innerHTML = '<span></span><button title="Rename">✎</button><button title="Delete">✕</button>'; d.firstChild.textContent = c.title;
    d.firstChild.onclick = () => { conv = c; show(); refresh(); };
    d.children[1].onclick = async () => { const t = prompt('Rename chat', c.title); if (t) { c.title = t; await db.putConv(c); refresh(); } };
    d.children[2].onclick = async () => { if (confirm('Delete this chat?')) { await db.delConv(c.id); if (conv?.id === c.id) conv = null; await start(); } };
    $('convList').append(d);
  }
}
async function start() { convs = await db.listConvs(); conv = conv && convs.find((c) => c.id === conv.id) || convs[0] || db.newConv(); show(); refresh(); }
$('newChat').onclick = async () => { conv = db.newConv(); await db.putConv(conv); show(); refresh(); };
$('exportAll').onclick = async () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([await db.exportConvs()], { type: 'application/json' })); a.download = 'nishkamtb-rag-chats.json'; a.click(); };
function add(role, text, extra = '') { const p = document.createElement('div'); p.className = 'm ' + role; p.textContent = text; if (extra) { const f = document.createElement('div'); f.className = 'flag'; f.textContent = extra; p.append(f); } $('msgs').append(p); $('msgs').scrollTop = 1e9; return p; }
// Assistant answers: status line (stages), search meta, thinking disclosure, answer text, citation flag.
function newAnswer() {
  const el = document.createElement('div'); el.className = 'm assistant';
  el.innerHTML = '<div class="status"></div><div class="meta muted"></div><details class="think" hidden><summary>Thinking</summary><pre></pre></details><div class="ans"></div><div class="flag"></div>';
  $('msgs').append(el); $('msgs').scrollTop = 1e9;
  const [status, meta, think, ans, flag] = el.children;
  return { el, status, meta, think, pre: think.lastElementChild, ans, flag };
}
function paintAnswer(v, visible, thinking = '', flags = '') {
  v.ans.textContent = visible; v.pre.textContent = thinking; v.think.hidden = !thinking; v.flag.textContent = flags;
}
// Stage line: finished stages stay as muted lines with their seconds; the running one ticks once a second.
function progress(el) {
  const done = [], p = { cur: null, t0: 0, timer: 0 };
  const draw = () => {
    el.replaceChildren(...done.map((t) => Object.assign(document.createElement('div'), { className: 'muted', textContent: t })));
    if (p.cur) el.append(Object.assign(document.createElement('div'), { className: 'live', textContent: `${p.cur} ${Math.round((performance.now() - p.t0) / 1000)} s` }));
  };
  const finish = () => {
    if (p.cur) done.push(`${p.cur} ${Math.round((performance.now() - p.t0) / 1000)} s`);
    p.cur = null; clearInterval(p.timer); p.timer = 0;
  };
  return {
    set(msg) { finish(); p.cur = msg; p.t0 = performance.now(); p.timer = setInterval(draw, 1000); draw(); },
    note(msg) { finish(); done.push(msg); draw(); },
    clear() { finish(); el.replaceChildren(); },
  };
}
function show() {
  $('msgs').innerHTML = '';
  for (const m of conv.messages) {
    if (m.role === 'assistant') paintAnswer(newAnswer(), m.content, m.think, m.flags);
    else add(m.role, m.content, m.flags);
  }
  showSources(conv.messages.findLast((m) => m.sources)?.sources || []);
}
function showSources(s) {
  $('srcList').innerHTML = s.length ? '' : 'No sources yet.';
  for (const x of s) { const d = document.createElement('div'); d.className = 'src'; d.innerHTML = '<b></b><div class="meta"></div><p></p>'; d.children[0].textContent = `S${x.n} · ${x.book_title}`; d.children[1].textContent = `${x.authors.join(', ')} · ch ${x.chapter} · ${x.section} · p ${x.page} · role ${x.pedagogical_role} · score ${x.score?.toFixed(3)}`; d.children[2].textContent = x.text; $('srcList').append(d); }
}

// ---------- ask ----------
$('stop').onclick = () => abort?.abort();
$('ask').onsubmit = async (e) => {
  e.preventDefault(); const question = $('q').value.trim(); if (!question || !ix || !$('stop').hidden) return;   // one answer at a time
  const c = cfg(); if (!c.key) { $('btnSet').click(); return; }
  $('q').value = ''; $('send').hidden = true; $('stop').hidden = false; abort = new AbortController();
  const history = conv.messages.map(({ role, content }) => ({ role, content }));
  conv.messages.push({ role: 'user', content: question }); add('user', question);
  if (conv.title === 'New chat') conv.title = question.slice(0, 40);
  const v = newAnswer(), prog = progress(v.status);
  let answer = '', think = '', raw = '', flags = '', started = false;
  try {
    const llm = { provider: c.provider, model: c.model, key: c.key, base: c.base, signal: abort.signal, timeoutMs: c.timeoutMs };
    const sq = history.length ? (prog.set('Rewriting question…'), await rewriteQuery(llm, history, question)) : question;
    const opts = { rerank: ls.get('rerank', '1') === '1', progress: (stage, n) => {
      if (stage === 'models') prog.set('Loading search models (first time only)…');
      else if (stage === 'search') prog.set('Searching the library…');
      else if (stage === 'rerank') prog.set(`Reranking ${n} passages…`);
    } };
    if (ls.get('llmRerank') === '1') opts.llmRank = (q, cands) => llmRank(llm, q, cands);
    let r = ls.get('deep') === '1'
      ? await deepRetrieve(ix, sq, opts, (s) => (prog.set(`Asking ${c.provider}/${c.model} for a follow-up search…`), followUpQuery(llm, question, s)))
      : await retrieve(ix, sq, opts);
    if (r.weak) { prog.note('Whole question matched weakly; searching each concept separately'); r = await conceptRetrieve(ix, sq, opts, r); }
    if (r.weak && isDesignAsk(question)) { r = { ...r, weak: false }; prog.note('Library coverage is thin: the answer will be mostly editorial'); }   // design work is grounded by the Nishkam brief too
    showSources(r.sources);   // sources are on screen before the model is called
    v.meta.textContent = `search: "${sq}" · ${JSON.stringify(r.T)} ms` + (r.rerankErr ? ` · ${r.rerankErr}, using RRF order` : '');
    prog.note(`Found ${r.sources.length} sources`);
    if (r.weak) { answer = 'Weak or no coverage in the library for this question. Not answering from outside knowledge.'; paintAnswer(v, answer); }
    else {
      prog.set(`Asking ${c.provider}/${c.model}…`);
      const local = isLocal(c.base), src = local ? r.sources.slice(0, r.concepts ? 6 : 4) : r.sources;   // local models: fewer, shorter sources
      const ctx = buildContext(src, local ? { maxChars: 6000, parentChars: 0 } : {});
      const msgs = [...history.slice(local ? -2 : -8), { role: 'user', content: buildUserMessage({ question, ctx, brief: ix.brief, local }) }];
      const onStage = (s) => { if (s === 'connected') prog.set('Thinking…'); };
      for await (const t of stream({ ...llm, system: SYSTEM, messages: msgs, onStage })) {
        if (!started) { started = true; prog.clear(); }
        raw += t; ({ answer, think } = splitThink(raw)); paintAnswer(v, answer, think); $('msgs').scrollTop = 1e9;
      }
      if (!started) prog.clear();
      ({ answer, think } = splitThink(raw, true)); paintAnswer(v, answer, think);
      const bad = checkCitations(answer, r.sources);   // citation check on the visible answer only
      if (bad.length) flags = 'Citation check: ' + bad.join('; ');
      v.flag.textContent = flags;
    }
    conv.messages.push({ role: 'assistant', content: answer, think, sources: r.sources, flags });
  } catch (err) {
    prog.clear();
    const msg = describeError(err, { provider: c.provider, base: c.base, timeoutMs: c.timeoutMs });
    if (answer) { v.flag.textContent = msg; conv.messages.push({ role: 'assistant', content: answer, think }); }   // keep partial text, show why it stopped
    else { paintAnswer(v, msg); v.ans.classList.add('warn'); }
  }
  await db.putConv(conv); refresh(); $('send').hidden = false; $('stop').hidden = true;
};
$('q').onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('ask').requestSubmit(); } };

// ---------- test hook (used by the headless check; no LLM) ----------
window.__retrieve = (q, o) => retrieve(ix, q, o);

(async () => {
  try { man = await fetchManifest(); } catch (e) { $('gateMsg').textContent = 'Cannot load index manifest.'; return; }
  await start();
  const p = ss.get('pp'); if (p) try { await tryUnlock(p); } catch { ss.del('pp'); }
})();
