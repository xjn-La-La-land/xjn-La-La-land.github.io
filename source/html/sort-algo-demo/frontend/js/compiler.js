const SORT_ALGORITHMS = new Set(['bubble', 'selection', 'insertion', 'merge', 'quick', 'heap', 'shell', 'comb', 'counting']);
const OUTPUT_LIMIT = 64 * 1024;

function prepareSource({ algorithm, source, values, trace = false }) {
  if (!SORT_ALGORITHMS.has(algorithm)) throw new Error('未知的排序算法。');
  if (typeof source !== 'string' || new TextEncoder().encode(source).length > 128 * 1024) throw new Error('源码不能超过 128 KiB。');
  if (!Array.isArray(values) || values.length < 1 || values.length > 12 || values.some(value => !Number.isInteger(value) || value < 1 || value > 99)) {
    throw new Error('运行数组需要 1 至 12 个 1 至 99 的整数。');
  }
  // Ignore comments and literals, including C++ raw strings, when detecting main.
  // This is deliberately not a C++ parser: macro-generated main requires explicit main.
  const code = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*|R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'/g, ' ');
  const usesDriver = !/\bmain\s*\(/.test(code);
  const filename = `${algorithm}_sort.cpp`;
  if (trace && !usesDriver) throw new Error('编译并演示使用界面数组与排序函数入口，暂不支持自定义 main；请使用普通编译运行。');
  if (!usesDriver) return { source: `#line 1 "${filename}"\n${source}\n`, filename, usesDriver, algorithm, values: [...values] };
  const call = algorithm === 'quick' ? 'quick_sort(a, 0, static_cast<int>(n) - 1)' : `${algorithm}_sort(a, n)`;
  return {
    filename, usesDriver, algorithm, values: [...values],
    source: `#include <algorithm>\n#include <cstddef>\n#include <cstdio>\n#include <cstdlib>\n#include <iostream>\n#include <vector>\n#include <string>\nusing std::size_t;\n#line 1 "${filename}"\n${source}\n
#line 1 "generated_driver.cpp"
int main() {
  int a[] = {${values.join(', ')}};
  const size_t n = sizeof(a) / sizeof(a[0]);
  ${trace ? 'sort_trace::bind(a, static_cast<int>(n));' : ''}
  ${call};
  ${trace ? 'sort_trace::complete();' : ''}
  std::printf("排序结果: [");
  for (size_t i = 0; i < n; ++i) std::printf("%s%d", i ? ", " : "", a[i]);
  std::puts("]");
  return 0;
}
`
  };
}

function cleanOutput(text) {
  return String(text).replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').replace(/\r\n?/g, '\n').replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
}

function createWasmRunner({
  createWorker = mode => new Worker(new URL('js/wasm-worker.js', document.baseURI), { name: `sort-${mode}` }),
  onState = () => {}, onOutput = () => {}, onFinish = () => {},
  setTimer = setTimeout, clearTimer = clearTimeout, now = () => performance.now()
} = {}) {
  let compiler = null, runtime = null, job = null, timer = null, serial = 0;
  const limits = { loading: 120000, analysis: 30000, compile: 30000, link: 20000, starting: 15000, run: 5000 };
  function discardCompiler() { compiler?.terminate(); compiler = null; }
  function finish(status, details = {}) {
    if (!job) return;
    const finished = { ...job, ...details, status, elapsed: now() - job.started };
    clearTimer(timer); timer = null;
    runtime?.terminate(); runtime = null;
    job = null;
    onState({ phase: status, busy: false });
    onFinish(finished);
  }
  function phase(next) {
    if (!job || !Object.hasOwn(limits, next)) return;
    job.phase = next;
    clearTimer(timer);
    timer = setTimer(() => {
      // A running program is independent of the warmed compiler worker.
      if (!runtime) discardCompiler();
      finish('timeout', { message: `${next === 'run' ? '程序运行' : '工具链加载或编译'}超时，已停止。` });
    }, limits[next]);
    onState({ phase: next, busy: true });
  }
  function fail(message, details = {}) {
    if (!runtime) discardCompiler();
    finish('error', { message, ...details });
  }
  function listen(worker) {
    worker.onmessage = ({ data }) => {
      if (!job || data?.id !== job.id || (worker !== compiler && worker !== runtime)) return;
      if (data.type === 'phase') phase(data.phase);
      else if (data.type === 'progress') onState({ phase: job.phase, busy: true, resource: data.resource, loaded: data.loaded, total: data.total });
      else if (data.type === 'output') {
        const text = cleanOutput(data.text);
        const bytes = new TextEncoder().encode(text).length;
        if (job.outputBytes + bytes > OUTPUT_LIMIT) { fail('输出超过 64 KiB，已停止。'); return; }
        job.outputBytes += bytes;
        onOutput(text);
      } else if (data.type === 'compiled' && worker === compiler && !runtime) {
        try {
          phase('starting');
          runtime = createWorker('execute');
          listen(runtime);
          runtime.postMessage({ type: 'execute', id: job.id, wasm: data.wasm, trace: job.trace, values: job.values, sourceLines: job.sourceLines }, [data.wasm]);
        } catch (error) { fail(error.message); }
      } else if (data.type === 'done' && worker === runtime) {
        if (job.trace && !Array.isArray(data.trace)) { fail('没有完整的 C++ 执行轨迹，已拒绝动画。'); return; }
        finish('done', { exitCode: data.exitCode ?? 0, trace: data.trace });
      }
      else if (data.type === 'error') {
        // C++ compile/link failures do not invalidate cached compiler modules.
        if (worker === compiler && data.exitCode != null) finish('error', { message: data.message, exitCode: data.exitCode });
        else fail(data.message || 'Worker 执行失败。', { exitCode: data.exitCode ?? null });
      }
    };
    worker.onerror = event => {
      event.preventDefault?.();
      if (job && (worker === compiler || worker === runtime)) fail(event.message || '无法加载 Worker；请用 HTTP 服务打开页面。');
    };
    worker.onmessageerror = () => { if (job && (worker === compiler || worker === runtime)) fail('Worker 消息无法读取。'); };
  }
  return {
    get busy() { return !!job; },
    start(input) {
      if (job) return false;
      const prepared = prepareSource(input);
      job = { ...prepared, trace: !!input.trace, sourceLines: input.source.split('\n').length, id: ++serial, started: now(), phase: 'loading', outputBytes: 0 };
      phase('loading');
      try {
        if (!compiler) { compiler = createWorker('compile'); listen(compiler); }
        compiler.postMessage({ type: 'compile', id: job.id, source: prepared.source, originalSource: input.source, filename: prepared.filename, algorithm: prepared.algorithm, values: prepared.values, trace: job.trace });
      } catch (error) { fail(error.message); }
      return true;
    },
    stop() {
      if (!job) return;
      if (!runtime) discardCompiler();
      finish('stopped', { message: '已停止。' });
    },
    dispose() { this.stop(); discardCompiler(); }
  };
}

if (typeof module !== 'undefined') module.exports = { prepareSource, cleanOutput, createWasmRunner };
if (typeof window !== 'undefined') Object.assign(window, { prepareSource, createWasmRunner });
