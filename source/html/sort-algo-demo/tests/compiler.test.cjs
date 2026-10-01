const assert = require('node:assert/strict');
const { prepareSource, cleanOutput, createWasmRunner } = require('../frontend/js/compiler.js');

const prepared = prepareSource({ algorithm: 'bubble', source: 'void bubble_sort(int a[], size_t n) {}', values: [3, 1, 2] });
assert.equal(prepared.filename, 'bubble_sort.cpp');
assert.ok(prepared.usesDriver);
assert.match(prepared.source, /#line 1 "bubble_sort.cpp"/);
assert.match(prepared.source, /int a\[\] = \{3, 1, 2\}/);
assert.match(prepared.source, /bubble_sort\(a, n\)/);
assert.match(prepareSource({ algorithm: 'quick', source: 'void quick_sort(int*, int, int) {}', values: [2, 1] }).source, /quick_sort\(a, 0, static_cast<int>\(n\) - 1\)/);
for (const source of ['// main()\nvoid bubble_sort() {}', '/* int main(){} */\nvoid bubble_sort() {}', 'const char* s = "main()";', 'const char* s = R"tag(main())tag";']) {
  assert.ok(prepareSource({ algorithm: 'bubble', source, values: [2, 1] }).usesDriver);
}
const full = prepareSource({ algorithm: 'heap', source: '#include <cstdio>\nint main() { puts("hi"); return 7; }', values: [2, 1] });
assert.equal(full.usesDriver, false);
assert.equal((full.source.match(/int main/g) || []).length, 1);
assert.throws(() => prepareSource({ algorithm: '../evil', source: '', values: [1] }));
assert.throws(() => prepareSource({ algorithm: 'bubble', source: '', values: [NaN] }));
assert.throws(() => prepareSource({ algorithm: 'bubble', source: ' '.repeat(131073), values: [1] }), /128 KiB/);
assert.throws(() => prepareSource({ algorithm: 'bubble', source: 'int main(){}', values: [1], trace: true }), /自定义 main/);
assert.match(prepareSource({ ...prepared, source: 'void bubble_sort(int*, size_t){}', trace: true }).source, /sort_trace::bind/);
assert.equal(cleanOutput('\x1b[91merror\x1b[0m\r\n<img>\0'), 'error\n<img>');

class FakeWorker {
  static workers = [];
  constructor(mode) { this.mode = mode; this.sent = []; FakeWorker.workers.push(this); }
  postMessage(data) { this.sent.push(data); }
  terminate() { this.terminated = true; }
  emit(data) { this.onmessage({ data }); }
}
const results = [], phases = [], output = [];
const timers = new Map(); let timerId = 0;
const runner = createWasmRunner({
  createWorker: mode => new FakeWorker(mode),
  onState: state => phases.push(state), onOutput: text => output.push(text), onFinish: result => results.push(result),
  setTimer: (callback, ms) => { const id = ++timerId; timers.set(id, { callback, ms }); return id; },
  clearTimer: id => timers.delete(id), now: () => 1000
});
const input = { algorithm: 'bubble', source: 'void bubble_sort(int*, size_t){}', values: [2, 1] };
assert.equal(runner.start(input), true);
assert.equal(runner.start(input), false, 'Do not start concurrent jobs');
let compiler = FakeWorker.workers[0], id = compiler.sent[0].id;
compiler.emit({ id, type: 'phase', phase: 'compile' });
assert.equal([...timers.values()][0].ms, 30000);
compiler.emit({ id, type: 'output', text: '\x1b[91mwarning\x1b[0m' });
assert.equal(output.at(-1), 'warning');
compiler.emit({ id, type: 'compiled', wasm: new ArrayBuffer(8) });
let runtime = FakeWorker.workers.at(-1);
assert.equal(runtime.mode, 'execute');
runtime.emit({ id, type: 'phase', phase: 'run' });
assert.equal([...timers.values()][0].ms, 5000);
runtime.emit({ id, type: 'done', exitCode: 0 });
assert.equal(results.at(-1).status, 'done');
assert.ok(runtime.terminated);
assert.ok(!compiler.terminated, 'Keep the compiler warm');
assert.equal(timers.size, 0);
runner.start(input);
assert.equal(FakeWorker.workers.filter(worker => worker.mode === 'compile').length, 1);
const nextId = compiler.sent.at(-1).id;
compiler.emit({ id, type: 'done', exitCode: 0 });
assert.ok(runner.busy, 'Ignore previous job messages');
runner.stop();
assert.equal(results.at(-1).status, 'stopped');
assert.ok(compiler.terminated, 'Abort a compiling worker');
runner.start(input);
compiler = FakeWorker.workers.at(-1); id = compiler.sent.at(-1).id;
compiler.emit({ id, type: 'compiled', wasm: new ArrayBuffer(8) });
runtime = FakeWorker.workers.at(-1);
runner.stop();
assert.ok(runtime.terminated);
assert.ok(!compiler.terminated, 'Stopping a program preserves compiler modules');
runner.start(input);
[...timers.values()][0].callback();
assert.equal(results.at(-1).status, 'timeout');
assert.equal(timers.size, 0);
runner.start(input);
compiler = FakeWorker.workers.at(-1); id = compiler.sent.at(-1).id;
compiler.emit({ id, type: 'output', text: 'x'.repeat(65537) });
assert.equal(results.at(-1).status, 'error');
assert.match(results.at(-1).message, /输出/);
runner.dispose();
assert.equal(timers.size, 0);
runner.start({ ...input, trace: true });
compiler = FakeWorker.workers.at(-1); id = compiler.sent.at(-1).id;
assert.equal(compiler.sent.at(-1).originalSource, input.source);
compiler.emit({ id, type: 'phase', phase: 'analysis' });
assert.equal([...timers.values()][0].ms, 30000);
compiler.emit({ id, type: 'compiled', wasm: new ArrayBuffer(8) });
runtime = FakeWorker.workers.at(-1);
assert.equal(runtime.sent[0].trace, true);
runtime.emit({ id, type: 'done', exitCode: 0 });
assert.equal(results.at(-1).status, 'error', 'Refuse a trace job without complete trace data');
runner.start({ ...input, trace: true });
id = compiler.sent.at(-1).id;
compiler.emit({ id, type: 'compiled', wasm: new ArrayBuffer(8) });
runtime = FakeWorker.workers.at(-1);
runtime.emit({ id, type: 'done', exitCode: 0, trace: [{kind:'finish'}] });
assert.deepEqual(results.at(-1).trace, [{kind:'finish'}]);
runner.dispose();
console.log('Passed: source wrapping, main detection, line mapping, input limits, output sanitization, worker lifecycle, warm compiler, stale messages, stop, timeout and output cap.');
