const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { createSortSources, ALGORITHMS } = require('../frontend/js/source-store.js');
const { createSortCodeEditor } = require('../frontend/js/code-editor.js');
const { createMonacoStub } = require('./editor.test.cjs');
const templates = require('./algorithms.cjs');
const hash = s => crypto.createHash('sha256').update(s).digest('hex');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate) { for(let i=0;i<150;i++){if(predicate())return;await pause(5);} throw Error('Timed out'); }
function fixture({ local = true, initial = [] } = {}) {
  const disk = new Map(Object.entries(templates)), writes = [], requests = [], drafts = new Map(initial);
  const storage = { getItem:k=>drafts.get(k) ?? null, setItem:(k,v)=>drafts.set(k,v), removeItem:k=>drafts.delete(k) };
  let gate = null, nextError = null;
  const fetcher = async (url, options) => {
    const path = new URL(url).pathname; requests.push(path);
    assert.ok(path.startsWith('/personal/sort-demo/'), 'Deployment subpath remains intact');
    if (path.endsWith('/api/capabilities')) return local ? Response.json({ protocol:'sort-source-v1', save:true, token:'test-token' }) : new Response('<html>404</html>', { status:404 });
    const id = path.match(/(?:algorithms\/)([a-z]+)(?:_sort.cpp)?$/)?.[1];
    assert.ok(ALGORITHMS.includes(id), path);
    if (options.method === 'PUT') {
      const data = JSON.parse(options.body); writes.push({ id, ...data });
      assert.equal(options.headers['X-Sort-Save-Token'],'test-token');
      if (gate) await gate;
      if (nextError) { const status=nextError;nextError=null;return Response.json({ message:'Save rejected' },{status}); }
      if (hash(disk.get(id)) !== data.baseRevision) return Response.json({ message:'Disk changed' }, { status:409 });
      disk.set(id,data.source);return Response.json({ revision:hash(data.source) });
    }
    return path.includes('/api/') ? Response.json({ source:disk.get(id), revision:hash(disk.get(id)) }) : new Response(disk.get(id));
  };
  const store = createSortSources({ fetcher, baseURL:'https://example.test/personal/sort-demo/frontend/index.html', storage, delay:30 });
  return { store, disk, drafts, storage, writes, requests, gate(p){gate=p;}, fail(status){nextError=status;} };
}
async function main() {
  const f=fixture(); await f.store.load();await pause(40);assert.equal(f.writes.length,0,'Loading never writes');
  const monaco=createMonacoStub(),tabs=ALGORITHMS.map(id=>({dataset:{algorithm:id},setAttribute(){}}));let changes=0;
  const editor=createSortCodeEditor(monaco, {}, { tabs,notice:{classList:{toggle(){}},textContent:''}, sources:f.store,onChange(){changes++;} });
  editor.showFile('bubble',f.disk.get('bubble'));
  editor.setExecutionSource(editor.getActiveFile().source);editor.highlight(5);assert.equal(f.writes.length,0);
  monaco.instance.model.setValue('first edit');await pause(10);monaco.instance.model.setValue('last edit');
  await pause(10);assert.equal(f.writes.length,0,'Debounce waits after final edit');
  editor.showFile('selection',f.disk.get('selection'));monaco.instance.model.setValue('selection edited');
  await until(()=>f.store.get('bubble').state==='saved' && f.store.get('selection').state==='saved');
  assert.equal(f.disk.get('bubble'),'last edit');assert.equal(f.disk.get('selection'),'selection edited');assert.equal(f.writes.length,2);
  assert.equal(changes,3,'Saving does not masquerade as editing');
  assert.ok(!f.drafts.has('sort-algo-demo:draft:v2:bubble'));
  let release;f.gate(new Promise(resolve=>release=resolve));f.store.edit('bubble','in flight');
  await until(()=>f.store.get('bubble').inFlight);f.store.edit('bubble','newer snapshot');
  assert.equal(f.store.get('bubble').source,'newer snapshot');release();f.gate(null);
  await until(()=>f.store.get('bubble').state==='saved');assert.equal(f.disk.get('bubble'),'newer snapshot');
  assert.equal(f.writes.at(-1).baseRevision,hash('in flight'),'Queued save uses acknowledged revision');
  f.fail(500);f.store.edit('bubble','failed source');await until(()=>f.store.get('bubble').state==='error');
  assert.equal(f.disk.get('bubble'),'newer snapshot');assert.ok(f.drafts.has('sort-algo-demo:draft:v2:bubble'));
  await f.store.retry('bubble');assert.equal(f.disk.get('bubble'),'failed source');
  f.disk.set('bubble','external editor');f.store.edit('bubble','conflicting draft');await until(()=>f.store.get('bubble').state==='conflict');
  const count=f.writes.length;f.store.edit('bubble','still conflicting');await pause(50);assert.equal(f.writes.length,count);assert.equal(f.disk.get('bubble'),'external editor');
  const restored=fixture({initial:[...f.drafts]});restored.disk.set('bubble','external editor');await restored.store.load();
  assert.equal(restored.store.get('bubble').state,'conflict');assert.equal(restored.store.get('bubble').source,'still conflicting');restored.store.dispose();
  const source=await f.store.reload('bubble');editor.reloadFile('bubble',source);await pause(50);
  assert.equal(editor.getActiveFile().source,'selection edited','Reloading inactive file does not switch models');
  editor.showFile('bubble',source);assert.equal(editor.getActiveFile().source,'external editor');assert.equal(f.writes.length,count,'Disk reload never autosaves');
  editor.dispose();f.store.dispose();

  const old=fixture({initial:[['sort-algo-demo:draft:v1:bubble','old browser draft']]});await old.store.load();
  assert.equal(old.store.get('bubble').state,'conflict');old.store.edit('bubble','edited old draft');await pause(50);assert.equal(old.writes.length,0);
  const oldAgain=fixture({initial:[...old.drafts]});await oldAgain.store.load();assert.equal(oldAgain.store.get('bubble').state,'conflict','Old draft cannot gain a new baseline by reloading');oldAgain.store.dispose();old.store.dispose();

  const matching=fixture({initial:[['sort-algo-demo:draft:v2:bubble',JSON.stringify({source:'restored valid draft',baseRevision:hash(templates.bubble),baseSource:templates.bubble})]]});
  await matching.store.load();await pause(50);assert.equal(matching.writes.length,0);await matching.store.retry('bubble');assert.equal(matching.disk.get('bubble'),'restored valid draft');matching.store.dispose();

  const staticMode=fixture({local:false});await staticMode.store.load();staticMode.store.edit('heap','static draft');await pause(50);
  assert.equal(staticMode.writes.length,0);assert.equal(staticMode.store.get('heap').state,'draft');assert.ok(staticMode.requests.includes('/personal/sort-demo/algorithms/heap_sort.cpp'));staticMode.store.dispose();
  const canceled=fixture();await canceled.store.load();canceled.store.edit('bubble','not sent');canceled.store.dispose();await pause(50);assert.equal(canceled.writes.length,0);
  console.log('SOURCE SAVE PASS: disk-source editor integration, debounce, cross-file saves, pending edits, errors/retry, conflicts, legacy recovery, static Pages paths and disposal.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
