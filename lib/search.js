// Pure retrieval maths (no DOM, no network): BM25 over shards with global stats, dense cosine, RRF, role boost.
export function tokenize(s, stop) { // keep identical to tools/build_index.py
  const st = new Set(stop), out = [];
  for (let w of s.toLowerCase().match(/[a-z0-9_]+/g) || []) {
    if (st.has(w)) continue;
    if (w.length > 3 && w.endsWith('s')) w = w.slice(0, -1);
    out.push(w);
  }
  return out;
}
// shards: [{offset, dl, postings}]; glob: {N, avgdl, df, stop}. Returns Map(globalIdx -> score)
export function bm25(query, shards, glob, k1 = 1.2, b = 0.75) {
  const sc = new Map();
  for (const t of new Set(tokenize(query, glob.stop))) {
    const df = Object.hasOwn(glob.df, t) ? glob.df[t] : 0; if (!df) continue;   // own keys only: 'constructor' must not hit Object.prototype
    const idf = Math.log(1 + (glob.N - df + 0.5) / (df + 0.5));
    for (const s of shards) for (const [d, tf] of (Object.hasOwn(s.postings, t) ? s.postings[t] : [])) {
      const v = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * s.dl[d] / glob.avgdl));
      sc.set(s.offset + d, (sc.get(s.offset + d) || 0) + v);
    }
  }
  return sc;
}
// vecs: [{offset, data:Int8Array, n}]; q: unit vector. Stored int8 = round(v*127).
export function dense(q, vecs, dim) {
  const sc = new Map();
  for (const s of vecs) for (let i = 0; i < s.n; i++) {
    let d = 0; for (let j = 0; j < dim; j++) d += s.data[i * dim + j] * q[j];
    sc.set(s.offset + i, d / 127);
  }
  return sc;
}
export const rank = (sc, k) => [...sc].filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).slice(0, k).map(([i]) => i);
export function rrf(lists, k = 60) { // lists of ids best-first -> Map(id -> fused score)
  const f = new Map();
  for (const l of lists) l.forEach((id, r) => f.set(id, (f.get(id) || 0) + 1 / (k + r + 1)));
  return f;
}
export function wantedRole(q) {
  q = q.toLowerCase();
  if (/exercise|practice problem|homework|quiz|assignment/.test(q)) return 'exercise';
  if (/pitfall|mistake|gotcha|common error|avoid|wrong/.test(q)) return 'pitfall';
  if (/example|show me|sample|demonstrat/.test(q)) return 'example';
  return null;
}
export function boost(fused, docs, role, factor = 1.5) {
  if (!role) return fused;
  const o = new Map(fused);
  for (const [i, v] of o) if (docs[i].pedagogical_role === role) o.set(i, v * factor);
  return o;
}
// Strongest first and last, weaker in the middle (lost-in-the-middle). Input best-first.
export function edgeOrder(items) {
  if (items.length < 3) return items;
  return [items[0], ...items.slice(2), items[1]];
}
// eval metrics: ranked ids vs relevant ids
export function metrics(ranked, rel, k = 10) {
  const top = ranked.slice(0, k), R = new Set(rel);
  const ri = top.findIndex((x) => R.has(x));
  const dcg = top.reduce((s, x, i) => s + (R.has(x) ? 1 / Math.log2(i + 2) : 0), 0);
  let idcg = 0; for (let i = 0; i < Math.min(R.size, k); i++) idcg += 1 / Math.log2(i + 2);
  return { hit: ri >= 0 ? 1 : 0, mrr: ri < 0 ? 0 : 1 / (ri + 1), recall: R.size ? top.filter((x) => R.has(x)).length / R.size : 0, ndcg: idcg ? dcg / idcg : 0 };
}
