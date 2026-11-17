import { fetchManifest, unlock, loadIndex, retrieve, deepRetrieve, warmReranker } from './lib/rag.js';
import { PROVIDERS, SYSTEM, buildContext, checkCitations, stream, rewriteQuery, followUpQuery, llmRank } from './lib/llm.js';
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
  e.preventDefault(); const p = $('pass').value;
  try { if (await tryUnlock(p) && $('remember').checked) ss.set('pp', p); } catch (err) { $('gateMsg').textContent = 'Error: ' + err.message; }
  $('pass').value = '';
};
$('btnLock').onclick = () => { ss.del('pp'); ix = null; location.reload(); };

// ---------- settings (one key per provider) ----------
const cfg = () => { const provider = ls.get('provider', 'anthropic'); return { provider, model: ls.get('model-' + provider, PROVIDERS[provider].model), key: ls.get('key-' + provider), base: provider === 'openai' ? ls.get('base') : '' }; };
function fillSettings() {
  const c = cfg(); $('prov').innerHTML = Object.entries(PROVIDERS).map(([k, v]) => `<option value="${k}" ${k === c.provider ? 'selected' : ''}>${v.label}</option>`).join('');
  $('model').value = c.model; $('key').value = c.key; $('base').value = ls.get('base');
  $('models').innerHTML = PROVIDERS[c.provider].models.map((m) => `<option value="${m}">`).join('');
  $('useRerank').checked = ls.get('rerank', '1') === '1'; $('llmRerank').checked = ls.get('llmRerank') === '1'; $('deep').checked = ls.get('deep') === '1';
}
$('btnSet').onclick = () => { fillSettings(); $('dlg').showModal(); };
$('prov').onchange = () => { ls.set('provider', $('prov').value); fillSettings(); };
$('dlg').onclose = () => { const p = $('prov').value; ls.set('provider', p); ls.set('model-' + p, $('model').value); ls.set('key-' + p, $('key').value); ls.set('base', $('base').value); ls.set('rerank', $('useRerank').checked ? '1' : '0'); ls.set('llmRerank', $('llmRerank').checked ? '1' : '0'); ls.set('deep', $('deep').checked ? '1' : '0'); };
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
function show() { $('msgs').innerHTML = ''; conv.messages.forEach((m) => add(m.role, m.content, m.flags)); showSources(conv.messages.findLast((m) => m.sources)?.sources || []); }
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
  const out = add('assistant', '...'); let answer = '', flags = '';
  try {
    const llm = { provider: c.provider, model: c.model, key: c.key, base: c.base, signal: abort.signal };
    const sq = history.length ? await rewriteQuery(llm, history, question) : question;
    const opts = { rerank: ls.get('rerank', '1') === '1' };
    if (ls.get('llmRerank') === '1') opts.llmRank = (q, cands) => llmRank(llm, q, cands);
    const r = ls.get('deep') === '1' ? await deepRetrieve(ix, sq, opts, (s) => followUpQuery(llm, question, s)) : await retrieve(ix, sq, opts);
    showSources(r.sources);
    out.textContent = `(search: "${sq}" · ${JSON.stringify(r.T)} ms)\n`;
    if (r.rerankErr) out.textContent += r.rerankErr + ', using RRF order\n';
    if (r.weak) { answer = 'Weak or no coverage in the library for this question. Not answering from outside knowledge.'; out.textContent += answer; }
    else {
      const msgs = [...history, { role: 'user', content: `${buildContext(r.sources)}\n\nQuestion: ${question}` }];
      for await (const t of stream({ ...llm, system: SYSTEM, messages: msgs })) { answer += t; out.textContent = answer; $('msgs').scrollTop = 1e9; }
      const bad = checkCitations(answer, r.sources);
      if (bad.length) { flags = 'Citation check: ' + bad.join('; '); const f = document.createElement('div'); f.className = 'flag'; f.textContent = flags; out.append(f); }
    }
    conv.messages.push({ role: 'assistant', content: answer, sources: r.sources, flags });
  } catch (err) { out.textContent = 'Error: ' + err.message; if (answer) conv.messages.push({ role: 'assistant', content: answer }); }
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
