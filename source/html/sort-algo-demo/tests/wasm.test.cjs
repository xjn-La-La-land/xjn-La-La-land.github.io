// Exercise the actual production WASM binaries, not a JavaScript sort stand-in.
// Node worker_threads supply the browser Worker shell; browser UI is tested separately.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { Worker } = require('node:worker_threads');
const { createWasmRunner } = require('../frontend/js/compiler.js');
const bootstrap = `
const { parentPort, workerData } = require('node:worker_threads');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
global.self = global;
self.location = { href: 'http://test.invalid/personal/sort-demo/frontend/js/wasm-worker.js' };
self.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);
global.importScripts = (...names) => { for (const name of names) vm.runInThisContext(fs.readFileSync(path.join(workerData.root, 'frontend/js', name), 'utf8'), { filename: name }); };
global.fetch = async url => {
  const prefix = '/personal/sort-demo/frontend/assets/wasm-clang/';
  const parsed = new URL(url);
  if (!parsed.pathname.startsWith(prefix)) throw new Error('Bad deployment-relative path: ' + parsed.pathname);
  const name = parsed.pathname.slice(prefix.length);
  if (!['clang', 'lld', 'memfs', 'sysroot.tar'].includes(name)) throw new Error('Unexpected asset');
  const bytes = fs.readFileSync(path.join(workerData.root, 'frontend/assets/wasm-clang', name));
  return new Response(bytes, { headers: { 'Content-Length': String(bytes.length) } });
};
vm.runInThisContext(fs.readFileSync(path.join(workerData.root, 'frontend/js/wasm-worker.js'), 'utf8'), { filename: 'wasm-worker.js' });
parentPort.on('message', data => self.onmessage({ data }));
`;
const workers = [];
function createWorker(mode) {
  const worker = new Worker(bootstrap, { eval: true, workerData: { root: require('node:path').resolve(__dirname, '..') } });
  const wrapper = {
    mode,
    postMessage: (data, transfer) => worker.postMessage(data, transfer),
    terminate() { this.terminated = true; return worker.terminate(); }
  };
  worker.on('message', data => wrapper.onmessage?.({ data }));
  worker.on('error', error => wrapper.onerror?.({ message: error.message, preventDefault() {} }));
  workers.push(wrapper);
  return wrapper;
}
let resolve, stdout = '', phases = [], stopAtRun = false;
const runner = createWasmRunner({
  createWorker,
  onOutput: text => { stdout += text; },
  onState: state => {
    phases.push(state.phase);
    if (state.phase === 'run' && stopAtRun) runner.stop();
  },
  onFinish: result => resolve({ ...result, stdout, phases: [...phases] })
});
function run(source, algorithm = 'bubble', stop = false) {
  stdout = ''; phases = []; stopAtRun = stop;
  return new Promise(done => { resolve = done; runner.start({ source, algorithm, values: [5, 2, 8, 1, 7, 3, 6, 4] }); });
}
async function main() {
  const templates = Object.entries(require('./algorithms.cjs'));
  assert.equal(templates.length, 9);
  for (const [algorithm, code] of templates) {
    const result = await run(code, algorithm);
    assert.equal(result.status, 'done', JSON.stringify(result));
    assert.equal(result.exitCode, 0);
    assert.match(result.stdout, /排序结果: \[1, 2, 3, 4, 5, 6, 7, 8\]/);
    console.log(`WASM PASS: ${algorithm} automatic main (${(result.elapsed / 1000).toFixed(2)}s)`);
    if (algorithm === 'bubble') {
      const edited = await run(code.replace('a[j] > a[j+1]', 'a[j] < a[j+1]'));
      assert.match(edited.stdout, /\[8, 7, 6, 5, 4, 3, 2, 1\]/);
    }
  }
  const all = await run(fs.readFileSync(__dirname + '/fixtures/all-algorithms.cpp', 'utf8'));
  assert.match(all.stdout, /TOTAL: 54\/54 PASS/);
  const utf8 = await run('#include <cstdio>\nint main(){ puts("你好 WASM 🧪 <script>"); return 0; }');
  assert.equal(utf8.status, 'done');
  assert.match(utf8.stdout, /你好 WASM 🧪 <script>/);
  const error = await run('int main(){ return unknown_name; }');
  assert.equal(error.status, 'error');
  assert.equal(error.exitCode, 1);
  assert.match(error.stdout, /bubble_sort.cpp:1:/);
  assert.ok(!error.phases.includes('run'), 'Do not execute stale object files after a compile error');
  const linkError = await run(fs.readFileSync(__dirname + '/fixtures/std-sort.cpp', 'utf8'));
  assert.equal(linkError.status, 'error');
  assert.match(linkError.stdout, /__lttf2/);
  const exit = await run('int main(){ return 7; }');
  assert.equal(exit.exitCode, 7);
  const flood = await run('#include <cstdio>\nint main(){for(;;) puts("0123456789");}');
  assert.equal(flood.status, 'error');
  assert.match(flood.message, /64 KiB/);
  const timeout = await run('int main(){volatile unsigned v=0; for(;;) ++v;}');
  assert.equal(timeout.status, 'timeout');
  assert.ok(timeout.phases.includes('run'));
  const stopped = await run('int main(){volatile unsigned v=0; for(;;) ++v;}', 'bubble', true);
  assert.equal(stopped.status, 'stopped');
  const recovery = await run('int main(){return 0;}');
  assert.equal(recovery.status, 'done');
  // All user executables were terminated; the compiler remains reusable.
  assert.equal(workers.filter(worker => worker.mode === 'compile').length, 1);
  assert.ok(workers.filter(worker => worker.mode === 'execute').every(worker => worker.terminated));
  console.log('WASM PASS: 54 fixture cases, edited code, UTF-8, syntax diagnostics, link errors, exit codes, output cap, real infinite-loop timeout, manual stop, recovery, warm reuse and project-subpath asset URLs.');
}
main().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => runner.dispose());
