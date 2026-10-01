// The upstream WASI shim is kept unmodified; all product policy lives here.
importScripts('../assets/wasm-clang/shared.js');
importScripts('compiler.js', 'trace-source.js', 'trace-events.js');
const TOOLCHAIN_REVISION = '648c4a89997a351eef75cdaec3ef5b89d4937dec';
const moduleCache = new Map();
let sysroot = null, active = false;
let traceRuntime = null;
// Canvas is intentionally not attached to the application or its DOM.
self.canvas = null;
self.ctx2d = null;
self.requestAnimationFrame = () => { throw new Error('不支持 canvas_loop 异步程序。'); };

async function loadBuffer(filename, send) {
  const url = new URL(`../assets/wasm-clang/${filename}?v=${TOOLCHAIN_REVISION}`, self.location.href);
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`工具链资源 ${filename} 加载失败（HTTP ${response.status}）。`);
  if (!response.body) return response.arrayBuffer();
  const reader = response.body.getReader();
  const chunks = []; let loaded = 0, lastReport = 0;
  const total = Number(response.headers.get('Content-Length')) || 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    if (loaded - lastReport >= 1024 * 1024) { send({ type: 'progress', resource: filename, loaded, total }); lastReport = loaded; }
  }
  send({ type: 'progress', resource: filename, loaded, total });
  const result = new Uint8Array(loaded); let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result.buffer;
}

self.onmessage = async ({ data }) => {
  if (active || !['compile', 'execute'].includes(data?.type)) return;
  active = true;
  let outputBytes = 0, pending = '', firstWrite = true, astMode = false, astOutput = '';
  const decoder = new TextDecoder();
  const send = message => self.postMessage({ id: data.id, ...message });
  const flush = () => { if (pending) { send({ type: 'output', text: pending }); pending = ''; } };
  const hostWrite = latin1 => {
    if (astMode) {
      if (astOutput.length + latin1.length > 8 * 1024 * 1024) throw new Error('AST 超过 8 MiB，已停止。');
      astOutput += latin1; return;
    }
    // Upstream readStr represents WASI output bytes as Latin-1 JS characters.
    // Decode those bytes as UTF-8, retaining partial sequences across writes.
    outputBytes += latin1.length;
    if (outputBytes > 64 * 1024) throw new Error('输出超过 64 KiB，已停止。');
    pending += decoder.decode(Uint8Array.from(latin1, character => character.charCodeAt(0)), { stream: true });
    if (firstWrite || pending.length >= 4096) { firstWrite = false; flush(); }
  };
  const compileStreaming = filename => {
    if (!moduleCache.has(filename)) {
      const promise = loadBuffer(filename, send).then(buffer => WebAssembly.compile(buffer));
      moduleCache.set(filename, promise);
      promise.catch(() => moduleCache.delete(filename));
    }
    return moduleCache.get(filename);
  };
  const execute = data.type === 'execute';
  try {
    send({ type: 'phase', phase: execute ? 'starting' : 'loading' });
    const api = new API({
      hostWrite, compileStreaming,
      readBuffer: () => {
        if (execute) return Promise.resolve(new ArrayBuffer(512)); // Empty runtime FS, no headers or user files.
        if (!sysroot) sysroot = loadBuffer('sysroot.tar', send).catch(error => { sysroot = null; throw error; });
        return sysroot;
      }
    });
    // Suppress upstream shell-like command logs; retain actual stdout/stderr.
    api.hostLog = () => {};
    api.hostLogAsync = (_message, promise) => promise;
    api.clangCommonArgs.push('-std=c++17');
    await api.ready;
    if (execute) {
      const module = await WebAssembly.compile(data.wasm);
      send({ type: 'phase', phase: 'run' });
      const asynchronous = await api.run(module, 'program.wasm');
      if (asynchronous) throw new Error('不支持持续 canvas_loop；请使用普通 main 返回。');
      pending += decoder.decode(); flush();
      let trace;
      if (data.trace) {
        const bytes = api.memfs.getFileContents('sort-trace.jsonl');
        if (bytes.length > 8 * 1024 * 1024) throw new Error('执行轨迹超过 8 MiB。');
        trace = SortTraceEvents.parseTrace(new TextDecoder().decode(bytes), data.values, data.sourceLines);
      }
      send({ type: 'done', exitCode: 0, trace });
    } else {
      await Promise.all([api.getModule('clang'), api.getModule('lld')]);
      let source = data.source;
      if (data.trace) {
        send({ type: 'phase', phase: 'analysis' });
        const original = '#include <algorithm>\n#include <cstddef>\n#include <cstdio>\n#include <cstdlib>\n#include <iostream>\n#include <vector>\n#include <string>\nusing std::size_t;\n#line 1 "' + data.filename + '"\n' + data.originalSource;
        api.memfs.addFile('analysis.cc', new TextEncoder().encode(original));
        astMode = true;
        await api.run(await api.getModule('clang'), 'clang', '-cc1', ...api.clangCommonArgs, '-ast-dump', '-ast-dump-filter', `${data.algorithm}_sort`, '-x', 'c++', 'analysis.cc');
        astMode = false;
        const modified = SortTraceSource.instrumentTraceSource({ source: data.originalSource, ast: astOutput, name: `${data.algorithm}_sort`, filename: data.filename });
        if (!traceRuntime) {
          const response = await fetch(new URL('../runtime/trace-runtime.hpp', self.location.href));
          if (!response.ok) throw new Error('无法加载 trace-runtime.hpp');
          traceRuntime = await response.text();
        }
        source = '#line 1 "sort_trace_runtime.hpp"\n' + traceRuntime + '\n' + prepareSource({ algorithm: data.algorithm, source: modified.source, values: data.values, trace: true }).source;
      }
      send({ type: 'phase', phase: 'compile' });
      await api.compile({ input: 'editor.cc', obj: 'editor.o', contents: new TextEncoder().encode(source) });
      send({ type: 'phase', phase: 'link' });
      await api.link('editor.o', 'program.wasm');
      pending += decoder.decode(); flush();
      // Copy out of memfs before transferring; never detach memfs's memory.
      const wasm = api.memfs.getFileContents('program.wasm').slice().buffer;
      self.postMessage({ id: data.id, type: 'compiled', wasm }, [wasm]);
    }
  } catch (error) {
    if (astMode) { astMode = false; pending += astOutput.slice(0, 64 * 1024); }
    pending += decoder.decode(); flush();
    send({ type: 'error', message: error.message, exitCode: typeof error.code === 'number' ? error.code : null });
  } finally { active = false; }
};
