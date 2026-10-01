const ALGORITHMS = ['bubble','selection','insertion','merge','quick','heap','shell','comb','counting'];

function createSortSources({ fetcher = globalThis.fetch.bind(globalThis), baseURL, storage = null, delay = 500, timeout = 5000 } = {}) {
  const files = new Map(), listeners = new Set();
  let token = null, local = false, loading = null, disposed = false;
  const url = name => new URL(name, baseURL).href;
  const key = id => `sort-algo-demo:draft:v2:${id}`;
  const legacy = id => `sort-algo-demo:draft:v1:${id}`;
  function notify(file) { if (!disposed) listeners.forEach(callback => callback(file)); }
  function backup(file) {
    try {
      if (file.source === file.diskSource && !file.inFlight) storage?.removeItem(key(file.id));
      else storage?.setItem(key(file.id), JSON.stringify({ source:file.source, baseRevision:file.blocked ? file.draftRevision : file.revision, baseSource:file.blocked ? file.draftBase : file.diskSource }));
      if (storage) storage.removeItem(legacy(file.id));
    } catch { /* Editing remains available without browser storage. */ }
  }
  async function request(name, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetcher(url(name), { cache:'no-store', ...options, signal:controller.signal });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw Object.assign(new Error(data.message || `HTTP ${response.status}`), { status:response.status });
      }
      return response;
    } finally { clearTimeout(timer); }
  }
  const store = {
    ready:false,
    get local() { return local; },
    get(id) { return files.get(id); },
    subscribe(callback) { listeners.add(callback); return () => listeners.delete(callback); },
    load() {
      if (loading) return loading;
      loading = (async () => {
        // Pages has no API; static mode is expected, not a save error.
        try {
          const capability = await (await request('../api/capabilities')).json();
          local = capability.protocol === 'sort-source-v1' && capability.save === true && typeof capability.token === 'string';
          if (local) token = capability.token;
        } catch { local = false; }
        await Promise.all(ALGORITHMS.map(async id => {
          const data = local ? await (await request(`../api/algorithms/${id}`)).json()
            : { source:await (await request(`../algorithms/${id}_sort.cpp`)).text(), revision:null };
          if (typeof data.source !== 'string' || (local && !/^[a-f0-9]{64}$/.test(data.revision || ''))) throw new Error(`${id}_sort.cpp：源码响应不合法`);
          let draft = null, old = null;
          try { old = storage?.getItem(legacy(id)); draft = JSON.parse(storage?.getItem(key(id)) || 'null'); } catch { /* Malformed draft metadata does not replace disk source. */ }
          const restored = typeof draft?.source === 'string' ? draft.source : old ?? data.source;
          const protectedDraft = local && restored !== data.source && !(draft && draft.baseRevision === data.revision && draft.baseSource === data.source);
          const file = { id, source:restored, diskSource:data.source, revision:data.revision, state:protectedDraft ? 'conflict' : local ? restored === data.source ? 'saved' : 'pending' : 'draft', message:protectedDraft ? '恢复的草稿没有匹配的磁盘版本，不会自动写回。' : '', blocked:protectedDraft, draftRevision:draft?.baseRevision ?? null, draftBase:draft?.baseSource ?? null, timer:null, inFlight:false };
          files.set(id, file); notify(file);
        }));
        store.ready = true;
        return store;
      })();
      return loading;
    },
    edit(id, source) {
      const file = files.get(id);
      if (!file || disposed || source === file.source) return;
      file.source = source; backup(file);
      if (file.blocked) { notify(file); return; }
      clearTimeout(file.timer);
      file.state = local ? source === file.diskSource && !file.inFlight ? 'saved' : 'pending' : 'draft';
      file.message = '';
      if (local) file.timer = setTimeout(() => save(file), delay);
      notify(file);
    },
    retry(id) { const file = files.get(id); if (file && !file.blocked && local) { clearTimeout(file.timer); return save(file); } },
    async reload(id) {
      const file = files.get(id);
      if (!file || file.inFlight) throw new Error('请等待当前保存请求完成');
      clearTimeout(file.timer);
      const previous = file.source;
      const data = await (await request(`../api/algorithms/${id}`)).json();
      if (file.source !== previous || file.inFlight) throw new Error('重新载入期间又发生编辑或保存，已保留当前草稿，请重试');
      if (typeof data.source !== 'string' || !/^[a-f0-9]{64}$/.test(data.revision || '')) throw new Error('磁盘源码响应不合法');
      Object.assign(file, { source:data.source, diskSource:data.source, revision:data.revision, blocked:false, state:'saved', message:'' });
      backup(file); notify(file);
      return data.source;
    },
    dispose() { disposed = true; files.forEach(file => clearTimeout(file.timer)); listeners.clear(); }
  };
  async function save(file) {
    if (disposed || file.inFlight || file.blocked || !local) return;
    clearTimeout(file.timer);
    if (file.source === file.diskSource) { file.state = 'saved'; notify(file); return; }
    const source = file.source;
    file.inFlight = true; file.state = 'saving'; notify(file);
    try {
      const data = await (await request(`../api/algorithms/${file.id}`, {
        method:'PUT', headers:{ 'Content-Type':'application/json', 'X-Sort-Save-Token':token },
        body:JSON.stringify({ source, baseRevision:file.revision })
      })).json();
      if (!/^[a-f0-9]{64}$/.test(data.revision || '')) throw new Error('保存响应不合法');
      file.diskSource = source; file.revision = data.revision;
      file.state = file.source === source ? 'saved' : 'pending'; file.message = '';
      backup(file);
    } catch (error) {
      file.state = error.status === 409 ? 'conflict' : 'error';
      file.blocked = error.status === 409;
      if (file.blocked) { file.draftRevision = file.revision; file.draftBase = file.diskSource; }
      file.message = error.message; backup(file);
    } finally {
      file.inFlight = false; backup(file); notify(file);
      if (!disposed && file.state === 'pending') file.timer = setTimeout(() => save(file), delay);
    }
  }
  return store;
}

if (typeof module !== 'undefined') module.exports = { ALGORITHMS, createSortSources };
if (typeof window !== 'undefined') {
  window.createSortSources = createSortSources;
  let storage = null;
  try { storage = window.localStorage; } catch { /* Private browsing may disable storage. */ }
  window.SortSources = createSortSources({ baseURL:document.baseURI, storage });
}
