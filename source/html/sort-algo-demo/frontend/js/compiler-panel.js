function createCompilerPanel({ getFile, getValues, onTraceStart = () => {}, onTraceState = () => {}, onTraceFinish = () => {} }) {
  const get = selector => document.querySelector(selector);
  const run = get('#compile-run'), stop = get('#compile-stop'), state = get('#compile-state');
  const animate = get('#compile-animate');
  const windowElement = get('.code-window'), body = get('#output-body'), output = get('#compiler-output');
  const toggle = get('#output-toggle'), collapse = get('#output-collapse'), resize = get('#output-resize');
  const labels = { loading: '加载工具链', analysis: '分析与自动插桩', compile: '编译中', link: '链接中', starting: '准备执行', run: '运行中', done: '已完成', stopped: '已停止', timeout: '超时', error: '失败' };
  let lastPhase = '', text = '', expanded = true, pointer = null;
  let request = null;
  function append(value) {
    // Do not interpret program output as HTML or use an unbounded live log.
    text = (text + value).slice(-72 * 1024);
    const followsTail = body.scrollHeight - body.scrollTop - body.clientHeight < 36;
    output.textContent = text;
    if (followsTail) body.scrollTop = body.scrollHeight;
  }
  function setExpanded(next) {
    if (!next && (body.contains(document.activeElement) || resize === document.activeElement)) collapse.focus();
    expanded = next;
    windowElement.classList.toggle('output-collapsed', !expanded);
    body.hidden = resize.hidden = !expanded;
    toggle.setAttribute('aria-expanded', String(expanded));
    collapse.setAttribute('aria-expanded', String(expanded));
    collapse.setAttribute('aria-label', expanded ? '隐藏输出面板' : '显示输出面板');
    collapse.title = expanded ? '隐藏输出面板' : '显示输出面板';
    collapse.querySelector('.codicon').className = `codicon codicon-chevron-${expanded ? 'down' : 'up'}`;
  }
  const runner = window.createWasmRunner({
    onOutput: append,
    onState(next) {
      run.disabled = next.busy;
      if (animate) animate.disabled = next.busy;
      stop.disabled = !next.busy;
      state.dataset.busy = String(next.busy);
      state.dataset.kind = next.phase;
      let label = labels[next.phase] || next.phase;
      if (next.resource) label += ` · ${next.resource} ${(next.loaded / 1048576).toFixed(1)} MiB`;
      state.textContent = label;
      state.title = label;
      if (next.busy && lastPhase !== next.phase) { append(`[${labels[next.phase]}]\n`); lastPhase = next.phase; }
      if (request?.trace) onTraceState(next);
    },
    onFinish(result) {
      append(`\n[${labels[result.status]}] ${result.filename} · ${(result.elapsed / 1000).toFixed(2)} 秒${result.exitCode != null ? ` · 退出码 ${result.exitCode}` : ''}\n`);
      if (result.message) append(`${result.message}\n`);
      if (result.status === 'error' && /__lttf2/.test(text)) append('提示：当前旧版 sysroot 存在 std::sort 链接缺陷；不是排序动画的问题。\n');
      const finished = request; request = null;
      if (finished?.trace) onTraceFinish(result, finished);
      // Keep the status strip and retained log, giving the editor its height
      // back only after a successful run. Failures must remain visible.
      setExpanded(result.status !== 'done' || (result.exitCode ?? 0) !== 0);
    }
  });
  function start(trace = false, autoplay = true) {
    if (runner.busy) return false;
    setExpanded(true); text = ''; output.textContent = ''; lastPhase = '';
    try {
      if (location.protocol === 'file:') throw new Error('请通过 HTTP 打开页面：在项目目录运行 npm start，然后访问 http://127.0.0.1:3000。file:// 无法可靠加载编译 Worker。');
      const file = getFile();
      if (!file) throw new Error('Monaco 未就绪，无法读取当前代码。');
      const values = getValues();
      const prepared = window.prepareSource({ ...file, values, trace });
      get('#compile-file').textContent = file.filename;
      append(`[${trace ? '编译并演示' : '运行'}] ${file.filename}\n${prepared.usesDriver ? `输入数组: [${values.join(', ')}] · 自动生成 main` : '使用代码中的 main'}\n\n`);
      request = { ...file, values: values.slice(), trace, autoplay };
      if (trace) onTraceStart(request);
      return runner.start({ ...file, values, trace });
    } catch (error) {
      state.textContent = '无法运行'; state.dataset.kind = 'error'; append(`${error.message}\n`);
      const finished = request; request = null;
      if (trace) onTraceFinish({status:'error',message:error.message}, finished);
      return false;
    }
  }
  run.addEventListener('click', () => start());
  animate?.addEventListener('click', () => start(true));
  stop.addEventListener('click', () => runner.stop());
  get('#output-clear').addEventListener('click', () => { text = ''; output.textContent = ''; });
  toggle.addEventListener('click', () => setExpanded(!expanded));
  collapse.addEventListener('click', () => setExpanded(!expanded));
  // Use capture so Monaco's own Enter command does not consume the shortcut.
  windowElement.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); event.stopPropagation(); start(); }
  }, true);
  function setHeight(height) {
    const available = get('#code').clientHeight + (body.hidden ? 0 : body.clientHeight);
    const max = Math.min(420, available - 200);
    const next = Math.round(Math.min(max, Math.max(80, height)));
    windowElement.style.setProperty('--output-height', `${next}px`);
    resize.setAttribute('aria-valuenow', String(next));
    resize.setAttribute('aria-valuemax', String(max));
  }
  resize.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    pointer = { id: event.pointerId, y: event.clientY, height: body.clientHeight };
    resize.setPointerCapture(event.pointerId); resize.classList.add('dragging'); event.preventDefault();
  });
  resize.addEventListener('pointermove', event => { if (pointer?.id === event.pointerId) setHeight(pointer.height + pointer.y - event.clientY); });
  const endResize = () => { pointer = null; resize.classList.remove('dragging'); };
  resize.addEventListener('pointerup', endResize);
  resize.addEventListener('pointercancel', endResize);
  resize.addEventListener('lostpointercapture', endResize);
  resize.addEventListener('keydown', event => {
    if (['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      setHeight(event.key === 'Home' ? 80 : event.key === 'End' ? 420 : body.clientHeight + (event.key === 'ArrowUp' ? 20 : -20));
    }
  });
  window.addEventListener('resize', () => setHeight(body.hidden ? Number(resize.getAttribute('aria-valuenow')) : body.clientHeight));
  window.addEventListener('pagehide', () => runner.dispose());
  setHeight(180);
  return { runner, start, report: append };
}
if (typeof window !== 'undefined') window.createCompilerPanel = createCompilerPanel;
if (typeof module !== 'undefined') module.exports = { createCompilerPanel };
