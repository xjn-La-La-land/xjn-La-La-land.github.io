const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { createServer } = require('../scripts/serve.cjs');
const { createSortSources, ALGORITHMS } = require('../frontend/js/source-store.js');

async function main() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(),'sort-source-test-'));
  const root=path.join(directory,'project');await fs.mkdir(root);await fs.mkdir(path.join(root,'algorithms'));
  await fs.writeFile(path.join(directory,'outside.cpp'),'outside');
  for(const id of ALGORITHMS)await fs.copyFile(path.join(__dirname,'..','algorithms',`${id}_sort.cpp`),path.join(root,'algorithms',`${id}_sort.cpp`));
  const server=createServer({root});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin=`http://127.0.0.1:${server.address().port}`;
  const request=(url,options={})=>fetch(origin+url,options);
  const rawRequest=(url,headers)=>new Promise((resolve,reject)=>{
    const req=http.get(origin+url,{headers},response=>{response.resume();response.on('end',()=>resolve(response.statusCode));});req.on('error',reject);
  });
  try {
    const capability=await (await request('/api/capabilities')).json();assert.equal(capability.save,true);
    const headers={'Content-Type':'application/json',Origin:origin,'X-Sort-Save-Token':capability.token};
    const data=await (await request('/api/algorithms/bubble')).json();
    const put=(source,baseRevision=data.revision,extra={})=>request('/api/algorithms/bubble',{method:'PUT',headers:{...headers,...extra},body:JSON.stringify({source,baseRevision})});
    assert.equal((await put('changed')).status,200);assert.equal(await fs.readFile(path.join(root,'algorithms/bubble_sort.cpp'),'utf8'),'changed');
    assert.equal((await put('stale')).status,409);assert.equal(await fs.readFile(path.join(root,'algorithms/bubble_sort.cpp'),'utf8'),'changed');
    const current=await (await request('/api/algorithms/bubble')).json();
    assert.deepEqual((await Promise.all([put('winner 1',current.revision),put('winner 2',current.revision)])).map(r=>r.status).sort(),[200,409]);
    assert.equal((await put('unauthorized',current.revision,{Origin:'https://attacker.invalid'})).status,403);
    assert.equal((await put('unauthorized',current.revision,{'X-Sort-Save-Token':'bad'})).status,403);
    assert.equal(await rawRequest('/api/capabilities',{Host:'attacker.invalid'}),403);
    assert.equal(await rawRequest('/api/capabilities',{'Sec-Fetch-Site':'cross-site'}),403);
    assert.equal((await put('not JSON',current.revision,{'Content-Type':'text/plain'})).status,415);
    assert.equal((await put('x'.repeat(256*1024+1),current.revision)).status,413);
    assert.equal((await request('/api/algorithms/bubble',{method:'PUT',headers,body:'{bad'})).status,400);
    for(const name of ['/api/algorithms/unknown','/api/algorithms/%2e%2e%2foutside','/scripts/serve.cjs','/tests/server.test.cjs','/package.json','/algorithms/.bubble.tmp'])assert.equal((await request(name)).status,404,name);
    const linked=path.join(root,'algorithms/selection_sort.cpp');await fs.unlink(linked);await fs.symlink(path.join(directory,'outside.cpp'),linked);
    assert.equal((await request('/api/algorithms/selection')).status,403);assert.equal(await fs.readFile(path.join(directory,'outside.cpp'),'utf8'),'outside');
    await fs.unlink(linked);await fs.copyFile(path.join(__dirname,'../algorithms/selection_sort.cpp'),linked);
    const nodeFetch=(url,options)=>fetch(url,{...options,headers:{...options.headers,Origin:origin}});
    const store=createSortSources({fetcher:nodeFetch,baseURL:origin+'/frontend/index.html',delay:10});await store.load();
    store.edit('heap','// 已保存的中文源代码\nvoid heap_sort(int a[],int n) {}\n');
    await store.retry('heap');assert.equal(store.get('heap').state,'saved');assert.equal(await fs.readFile(path.join(root,'algorithms/heap_sort.cpp'),'utf8'),store.get('heap').source);store.dispose();
    assert.ok((await fs.readdir(path.join(root,'algorithms'))).every(n=>n.endsWith('.cpp')),'No temporary files remain');
    // Read-only checks against the actual relocated resources; never save here.
    const production=createServer();await new Promise(resolve=>production.listen(0,'127.0.0.1',resolve));
    try {
      const base=`http://127.0.0.1:${production.address().port}/`;
      const html=await (await fetch(base+'frontend/index.html')).text();
      const refs=[...html.matchAll(/(?:src|href)="([^"#]+)"/g)].map(m=>new URL(m[1],base+'frontend/index.html'));
      const worker=new URL('frontend/js/wasm-worker.js',base);
      const workerSource=await (await fetch(worker)).text();
      const imports=[...workerSource.matchAll(/importScripts\(([^)]+)\)/g)].flatMap(m=>[...m[1].matchAll(/'([^']+)'/g)].map(s=>new URL(s[1],worker)));
      const assets=['../assets/wasm-clang/clang','../assets/wasm-clang/lld','../assets/wasm-clang/memfs','../assets/wasm-clang/sysroot.tar','../runtime/trace-runtime.hpp'].map(name=>new URL(name,worker));
      for(const url of [base,...refs,...imports,...assets])assert.equal((await fetch(url,{method:'HEAD'})).status,200,String(url));
      assert.equal((await fetch(new URL('../assets/wasm-clang/clang',worker),{method:'HEAD'})).headers.get('content-type'),'application/wasm');
    } finally { await new Promise(resolve=>production.close(resolve)); }
    console.log('SERVER PASS: real atomic saves in temporary project, concurrent version conflict, token/Origin/Host checks, body bounds, path isolation, symlink rejection and frontend integration.');
  } finally { await new Promise(resolve=>server.close(resolve));await fs.rm(directory,{recursive:true,force:true}); }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
