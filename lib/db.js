// Chat history in IndexedDB: conversations {id,title,created,updated,messages:[{role,content,sources?}]}
const open = () => new Promise((ok, no) => {
  const r = indexedDB.open('nishkamtb-rag', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('convs', { keyPath: 'id' });
  r.onsuccess = () => ok(r.result); r.onerror = () => no(r.error);
});
const tx = async (mode, f) => {
  const db = await open();
  return new Promise((ok, no) => { const t = db.transaction('convs', mode); const rq = f(t.objectStore('convs')); t.oncomplete = () => ok(rq?.result); t.onerror = () => no(t.error); });
};
export const listConvs = async () => (await tx('readonly', (s) => s.getAll())).sort((a, b) => b.updated - a.updated);
export const putConv = (c) => { c.updated = Date.now(); return tx('readwrite', (s) => s.put(c)); };
export const delConv = (id) => tx('readwrite', (s) => s.delete(id));
export const newConv = () => ({ id: crypto.randomUUID(), title: 'New chat', created: Date.now(), updated: Date.now(), messages: [] });
export const exportConvs = async () => JSON.stringify(await listConvs(), null, 1);
