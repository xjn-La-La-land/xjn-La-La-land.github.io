// Real production Clang/lld/program WASM. No browser/UI automation.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {Worker}=require('node:worker_threads');
const {createWasmRunner}=require('../frontend/js/compiler.js');
const {toAnimationTrace}=require('../frontend/js/trace-events.js');
const templates=require('./algorithms.cjs');
const bootstrap=String.raw`
const {parentPort,workerData}=require('node:worker_threads'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const env={console,TextEncoder,TextDecoder,WebAssembly,URL,Response,setTimeout,clearTimeout,performance};env.self=env;
env.location={href:'http://test.invalid/personal/sort-demo/frontend/js/wasm-worker.js'};
env.postMessage=(data,transfer)=>parentPort.postMessage(data,transfer);
const context=vm.createContext(env);
env.importScripts=(...names)=>{for(const name of names)vm.runInContext(fs.readFileSync(path.join(workerData.root,'frontend/js',name),'utf8'),context,{filename:name});};
env.fetch=async url=>{const prefix='/personal/sort-demo/',p=new URL(url).pathname;if(!p.startsWith(prefix))throw new Error('Bad project subpath');const file=p.slice(prefix.length);if(!/^(frontend\/assets\/wasm-clang\/(clang|lld|memfs|sysroot\.tar)|frontend\/runtime\/trace-runtime\.hpp)$/.test(file))throw new Error('Unexpected asset '+file);return new Response(fs.readFileSync(path.join(workerData.root,file)));};
vm.runInContext(fs.readFileSync(path.join(workerData.root,'frontend/js/wasm-worker.js'),'utf8'),context,{filename:'wasm-worker.js'});
parentPort.on('message',data=>env.onmessage({data}));`;
function createWorker(){const w=new Worker(bootstrap,{eval:true,workerData:{root:path.resolve(__dirname,'..')}});const api={postMessage:(d,t)=>w.postMessage(d,t),terminate:()=>w.terminate()};w.on('message',data=>api.onmessage?.({data}));w.on('error',error=>api.onerror?.({message:error.message,preventDefault(){}}));return api;}
let resolve,stdout='';
const runner=createWasmRunner({createWorker,onOutput:t=>stdout+=t,onFinish:r=>resolve({...r,stdout})});
function run(source,algorithm,values,trace=true){stdout='';return new Promise(done=>{resolve=done;runner.start({source,algorithm,values,trace});});}
const inputs=[[5,2,8,1,7,3,6,4],[9,1,9,2,2,99,1],[1,2,3,4,5],[5,4,3,2,1],[42],[7,7,7,7]];
function result(r){assert.equal(r.status,'done',r.message+'\n'+r.stdout);assert.equal(r.exitCode,0);const match=r.stdout.match(/排序结果: (\[[^\n]*\])/);assert.ok(match,r.stdout);return JSON.parse(match[1]);}
async function main(){
 let cases=0,events=0;const rendering=new Map();
 for(const [algorithm,source] of Object.entries(templates)) {
  for(const values of inputs) {
   const plain=await run(source,algorithm,values,false), traced=await run(source,algorithm,values,true);
   assert.deepEqual(result(traced),result(plain));assert.deepEqual(result(traced),values.slice().sort((a,b)=>a-b));
   assert.ok(traced.trace.length);const animation=toAnimationTrace(traced.trace,values,source.split('\n').length);
   assert.deepEqual(animation.at(-1).values,result(traced));assert.equal(animation.at(-1).action,'finish');
   assert.ok(traced.trace.some(e=>e.kind==='phase'),algorithm+' has recognized phase boundaries');
   const expected=values.slice().sort((a,b)=>a-b);
   for(const action of animation) {
    if(!action.state)continue;
    action.state.fixed.forEach(position=>assert.equal(action.values[position],expected[position],algorithm+' green slot is already final'));
    for(const range of action.state.ranges.filter(r=>['ordered','left','right'].includes(r.kind))) {
     const segment=action.values.slice(range.start,range.end);
     if(algorithm!=='quick')assert.deepEqual(segment,segment.slice().sort((a,b)=>a-b),algorithm+' marked segment really is ordered');
    }
   }
   if(algorithm==='bubble') {
    const ends=traced.trace.filter(e=>e.phase==='bubble.end');
    assert.ok(ends.length||values.length===1);
    ends.forEach(e=>assert.equal(e.variables.end,values.length-e.variables.pass));
   }
   if(algorithm==='bubble') {
    assert.equal(animation.filter(e=>e.action==='grip').length,traced.trace.filter(e=>e.kind==='compare').length,'Bubble exchanges must not grip again after comparison');
    animation.forEach((e,i)=>{if(e.action==='compare-keep')assert.equal(animation[i+1].action,e.result?'lift':'release');});
   }
   if(algorithm==='selection') {
    let comparison=0;
    traced.trace.forEach((e,i)=>{
     if(e.kind!=='compare')return;comparison++;
     const next=traced.trace[i+1];
     if(next?.kind!=='compare' || next.variables.minIndex!==e.variables.minIndex)return;
     const at=animation.findIndex(action=>action.action==='compare-keep' && action.comparisons===comparison);
     assert.ok(animation[at+1].retain.includes(e.ids[e.right]),'Actual unchanged minIndex never releases its gripper');
    });
   }
   events+=traced.trace.length;cases++;
   if(values===inputs[0])rendering.set(algorithm,traced.trace);
  }
  console.log('TRACE WASM PASS:',algorithm,'6 original/trace comparisons');
 }
 // AST-bound identifiers tolerate spelling/formatting edits, but changed logic
 // must not inherit invariants of the original algorithm.
 for(const [algorithm,source] of Object.entries(templates)) {
  const renamed='// 中文注释：语义不变\n'+source.replace(/\b(a|n|i|j|swapped|minIndex|width|left|mid|right|lo|hi|pivot|store|sift|heapSize|gap|count|out|value)\b/g,name=>'renamed_'+name);
  const traced=await run(renamed,algorithm,inputs[0]);
  assert.deepEqual(result(traced),inputs[0].slice().sort((a,b)=>a-b));
  assert.ok(traced.trace.some(e=>e.kind==='phase'),algorithm+' renamed declarations still recognized');
 }
 const descending=await run(templates.bubble.replace('a[j] > a[j+1]','a[j] < a[j+1]'),'bubble',[3,1,2]);
 assert.deepEqual(result(descending),[3,2,1]);assert.ok(descending.trace.some(e=>e.operator==='<'));
 assert.ok(!descending.trace.some(e=>e.kind==='phase'),'Changed comparison disables ascending stage contracts');
 assert.equal(toAnimationTrace(descending.trace,[3,1,2],templates.bubble.split('\n').length).at(-1).verifiedSorted,false);
 const changed=await run(templates.selection.replace('a[j] < a[minIndex]','a[j] > a[minIndex]'),'selection',[3,1,2]);assert.deepEqual(result(changed),[3,2,1]);
 const side=await run('void bubble_sort(int a[],size_t n){int store=0;int j=2;std::swap(a[store++],a[j]);std::printf("store=%d\\n",store);}','bubble',[5,2,8]);
 assert.deepEqual(result(side),[8,2,5]);assert.match(side.stdout,/store=1/);assert.equal(side.trace.find(e=>e.kind==='swap').left,0);
 const skipped=await run('void bubble_sort(int a[],size_t n){size_t j=n;if(j<n&&a[j]>a[0])std::swap(a[j],a[0]);}','bubble',[3,1,2]);assert.deepEqual(result(skipped),[3,1,2]);assert.ok(!skipped.trace.some(e=>e.kind==='compare'));
 const output=await run('void bubble_sort(int a[],size_t n){std::puts("TRACE {fake}");std::swap(a[0],a[1]);}','bubble',[2,1]);assert.deepEqual(result(output),[1,2]);assert.match(output.stdout,/TRACE \{fake\}/);
 // This syntax regression must not depend on whether the editable algorithm
 // currently uses a temporary variable or std::swap.
 const renamedSource='// 中文注释与 UTF-8 行列映射\nvoid bubble_sort(int a[],size_t n){\n for(size_t j=0;j+1<n;++j){\n  if (a[j]>a[j+1]) {int saved_value=a[j];a[j]=a[j+1];a[j+1]=saved_value;}\n }\n}';
 const renamed=await run(renamedSource,'bubble',[3,1,2]);assert.deepEqual(result(renamed),[1,2,3]);
 assert.equal(renamed.trace.find(e=>e.kind==='compare').line,renamedSource.split('\n').findIndex(line=>line.includes('if (a[j]'))+1);
 assert.ok(renamed.trace.some(e=>e.kind==='drop'&&'saved_value' in e.variables));
 const reversed=await run('void bubble_sort(int a[],size_t n){int saved=a[1];a[1]=a[0];a[0]=saved;}','bubble',[3,1,2]);assert.deepEqual(result(reversed),[1,3,2]);
 for(const source of ['void bubble_sort(int a[],size_t n){int*p=a;p[0]=1;}','void bubble_sort(int a[],size_t n){a[0]+=1;}','void bubble_sort(int a[],size_t n){std::sort(a,a+n);}','void bubble_sort(int a[],size_t n){std::swap(a[a[0]],a[1]);}']){const rejected=await run(source,'bubble',[2,1]);assert.equal(rejected.status,'error');assert.ok(!Array.isArray(rejected.trace));console.log('TRACE rejected:',rejected.message);}
 for(const [source,message] of [['void bubble_sort(int a[],size_t n){while(true){if(a[0]>a[1]){}}}','trace event limit'],['void bubble_sort(int a[],size_t n){std::swap(a[0],a[n]);}','outside registered range']]) {
  const failed=await run(source,'bubble',[2,1]);assert.equal(failed.status,'error');assert.ok(!Array.isArray(failed.trace));assert.match(failed.stdout,new RegExp(message));
 }
 const {createHarness}=require('./test.cjs');
 const {checkReusePhysics}=require('./gripper-reuse.test.cjs');
 const {gsap}=require('../frontend/assets/gsap.min.js');
 for(const algorithm of Object.keys(templates)) {
  const harness=createHarness(false,false,false,false,request=>rendering.get(request.algorithm));
  if(algorithm!=='bubble')harness.tabs.find(tab=>tab.dataset.algorithm===algorithm).dispatch('click');
  const timeline=harness.timeline;assert.ok(timeline,algorithm+' real trace timeline');
  const coordinates=()=>harness.get('#bars').children.map(bar=>Number(bar.attrs.transform.match(/-?\d+(?:\.\d+)?/)[0]));
  const start=coordinates(),pitch=start[1]-start[0];let intermediate=false;
  for(let time=0;time<timeline.duration();time+=.025){
   timeline.time(time,false);if(coordinates().some(x=>Math.abs((x-start[0])/pitch-Math.round((x-start[0])/pitch))>.01))intermediate=true;
   for(const bar of harness.get('#bars').children.filter(b=>b.classList.contains('finished'))) {
    const x=Number(bar.attrs.transform.match(/-?\d+(?:\.\d+)?/)[0]),position=Math.round((x-start[0])/pitch);
    assert.equal(Number(bar.dataset.value),rendering.get(algorithm).at(-1).values[position],algorithm+' rendered green bar has its final value and slot');
   }
  }
  if(algorithm!=='counting')assert.ok(intermediate,algorithm+' swaps travel through intermediate positions');
  timeline.time(timeline.duration(),false);
  const bars=harness.get('#bars').children;
  const displayed=bars.map(bar=>({value:Number(bar.dataset.value),x:Number(bar.attrs.transform.match(/-?\d+(?:\.\d+)?/)[0])})).sort((a,b)=>a.x-b.x).map(bar=>bar.value);
  assert.deepEqual(displayed,rendering.get(algorithm).at(-1).values,algorithm+' physical final slots');
  assert.equal(harness.get('#execution-state').textContent,'已完成');assert.ok(harness.editor.decorations.length);
  timeline.kill();console.log('TRACE RENDER PASS:',algorithm);
  if(algorithm!=='counting') {
   const reuse=checkReusePhysics(rendering.get(algorithm),inputs[0],algorithm);
   if(algorithm==='selection')assert.ok(reuse.partial>0 && reuse.crossings>0,'Actual selection minimum reuse and role changes');
   console.log('TRACE GRIP REUSE PASS:',algorithm,reuse);
  }
 }
 for(const values of [Array(12).fill(7),Array.from({length:12},(_,i)=>99-i)]) {
  const selection=await run(templates.selection,'selection',values);
  assert.deepEqual(result(selection),values.slice().sort((a,b)=>a-b));
  const reuse=checkReusePhysics(selection.trace,values,'selection');assert.ok(reuse.partial>0);
  console.log('TRACE DENSE SELECTION PASS:',reuse);
 }
 gsap.ticker.sleep();
 console.log('TRACE WASM PASS:',cases,'original/trace pairs;',events,'events; nine real rendering paths, edited operators, UTF-8, renamed/reversed manual exchange, pivot/lambda/rotate/buckets, side effects, short circuit, stdout isolation, limits and unsupported-write rejection.');
}
main().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>runner.dispose());
