"use strict";

function parseArray(text) {
  const normalized = text.trim().replaceAll("，", ",");
  if (!/^\d+(?:(?:\s*,\s*|\s+)\d+)*$/.test(normalized)) {
    throw new Error("请输入用逗号或空格分隔的整数。");
  }
  const values = normalized.split(/[,\s]+/).map(Number);
  if (values.length < 2 || values.length > 12) throw new Error("数组需要 2 至 12 个整数。");
  if (values.some(value => !Number.isInteger(value) || value < 1 || value > 99)) throw new Error("每个整数必须在 1 至 99 之间。");
  return values;
}

if (typeof module !== "undefined") module.exports = { parseArray };

function startSortDemo() {
  const sources = window.SortSources;
  let values = [5, 2, 8, 1, 7, 3, 6, 4];
  let algorithm = "bubble";
  let timeline = null;
  let mode = "running";
  let stepTarget = null;
  let stepEnds = [];
  let pendingTrace = null, cachedTrace = null, cachedKey = null;
  let resetGrippers = () => {};
  const toggle = document.querySelector("#play-toggle");
  const step = document.querySelector("#step");
  const restart = document.querySelector("#restart");
  const input = document.querySelector("#array-input");
  const error = document.querySelector("#array-error");
  const arrayForm = document.querySelector("#array-form");
  const code = document.querySelector("#code");
  const tabs = [...document.querySelectorAll(".code-tab")];
  const editorNotice = document.querySelector("#editor-notice");
  let draftStorage = null;
  try { draftStorage = window.localStorage; } catch { /* Editing also works without storage. */ }
  const codeEditor = window.monaco && window.createSortCodeEditor
    ? window.createSortCodeEditor(window.monaco, code, { notice: editorNotice, tabs, storage: draftStorage, sources, onChange: changed => { if (changed === algorithm) invalidateTrace('源码已修改，请编译并演示。'); } })
    : null;
  const compilerPanel = window.createCompilerPanel ? window.createCompilerPanel({
    getFile: () => codeEditor?.getActiveFile(),
    getValues: () => parseArray(input.value),
    onTraceStart(request) {
      values = request.values.slice(); cachedTrace = null; cachedKey = null;
      pendingTrace = { ...request, key: traceKey(request, request.values) };
      initialize(false, null, true);
    },
    onTraceState(state) {
      if (pendingTrace && state.busy) document.querySelector('#execution-state').textContent = { loading:'加载工具链', analysis:'分析与自动插桩', compile:'编译中', link:'链接中', starting:'准备采集', run:'采集 C++ 轨迹' }[state.phase] || state.phase;
    },
    onTraceFinish(result, request) {
      if (request && (!pendingTrace || pendingTrace.key !== traceKey(request, request.values) || pendingTrace.key !== currentTraceKey())) return;
      pendingTrace = null;
      try {
        if (result.status !== 'done' || result.exitCode !== 0) throw new Error(result.message || `轨迹采集${result.status === 'stopped' ? '已停止' : '失败'}。`);
        if (typeof window.SortTraceEvents?.toAnimationTrace !== 'function') throw new Error('轨迹转换模块未加载，请强制刷新页面并检查 trace-events.js 的加载状态。');
        cachedTrace = window.SortTraceEvents.toAnimationTrace(result.trace, request.values, request.source.split('\n').length);
        cachedKey = traceKey(request, request.values);
        initialize(request.autoplay, cachedTrace);
      } catch (exception) {
        timeline?.kill(); timeline = null; resetGrippers(); mode = 'error'; codeEditor?.setExecutionSource(null);
        document.querySelector('#status').textContent = exception.message;
        compilerPanel.report(`[动画未启动] ${exception.message}\n`); updateControls();
      }
    }
  }) : null;
  const explorerFiles = [...document.querySelectorAll(".explorer-file")];
  const workspace = document.querySelector(".workspace");
  const explorerSidebar = document.querySelector("#explorer-sidebar");
  const explorerToggle = document.querySelector("#explorer-toggle");
  const narrowLayout = window.matchMedia("(max-width: 720px)");
  function setExplorerExpanded(expanded) {
    if (!expanded && explorerSidebar.contains(document.activeElement)) explorerToggle.focus();
    workspace.classList.toggle("explorer-collapsed", !expanded);
    explorerSidebar.inert = !expanded;
    explorerSidebar.setAttribute("aria-hidden", String(!expanded));
    explorerToggle.setAttribute("aria-expanded", String(expanded));
    const label = expanded ? "隐藏资源管理器" : "显示资源管理器";
    explorerToggle.setAttribute("aria-label", label);
    explorerToggle.setAttribute("title", label);
  }
  setExplorerExpanded(!narrowLayout.matches);
  explorerToggle.addEventListener("click", () => {
    setExplorerExpanded(explorerToggle.getAttribute("aria-expanded") !== "true");
  });
  explorerSidebar.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      event.preventDefault();
      setExplorerExpanded(false);
    }
  });

  function renderCode() {
    const source = sources.get(algorithm).diskSource;
    if (codeEditor) codeEditor.showFile(algorithm, source);
    else {
      code.textContent = source;
      code.classList.add("code-fallback");
      editorNotice.textContent = "Monaco 未加载：请在项目目录运行 npm install 和 npm run build 后刷新。";
    }
    tabs.forEach(tab => {
      tab.setAttribute("aria-selected", String(tab.dataset.algorithm === algorithm));
      tab.setAttribute("aria-controls", "code");
    });
    code.setAttribute("aria-labelledby", `tab-${algorithm}`);
    explorerFiles.forEach(file => file.setAttribute("aria-current", String(file.dataset.algorithm === algorithm)));
    const activeTab = tabs.find(tab => tab.dataset.algorithm === algorithm);
    const tabStrip = document.querySelector(".code-tabs");
    const left = activeTab.offsetLeft - tabStrip.offsetLeft;
    const right = left + activeTab.offsetWidth;
    if (left < tabStrip.scrollLeft) tabStrip.scrollLeft = left;
    else if (right > tabStrip.scrollLeft + tabStrip.clientWidth) tabStrip.scrollLeft = right - tabStrip.clientWidth;
    const names = { bubble: ["冒泡排序", "bubble sort"], selection: ["选择排序", "selection sort"], insertion: ["插入排序", "insertion sort"], merge: ["归并排序", "merge sort"], quick: ["快速排序", "quick sort"], heap: ["堆排序", "heap sort"], shell: ["希尔排序", "shell sort"], comb: ["梳排序", "comb sort"], counting: ["计数排序", "counting sort"] };
    document.querySelector("h1").textContent = names[algorithm][0];
    document.querySelector(".subtitle").textContent = names[algorithm][1];
    document.querySelector("#explorer-current").textContent = names[algorithm][0];
    document.title = `${names[algorithm][0]} - 排序动画`;
    document.querySelector("main").setAttribute("aria-label", `${names[algorithm][0]}动画`);
    const descriptions = {
      bubble: "机械夹爪逐对比较相邻柱子，并将较大的值向右交换",
      selection: "机械夹爪扫描未排序区间，标记最小值并与区间起点交换",
      insertion: "机械夹爪将当前值向左移动，插入已排序区间",
      merge: "机械夹爪将右侧有序段中的值逐个插入左侧有序段",
      quick: "机械夹爪以枢轴划分数组并递归处理两侧",
      heap: "机械夹爪比较父子节点并调整最大堆，逐个将堆顶交换到末尾",
      shell: "机械夹爪按递减间隔比较和交换元素，完成交换式希尔排序",
      comb: "机械夹爪比较相隔 gap 的元素并交换逆序对，逐步缩小间隔",
      counting: "统计每个值出现的次数，再按值递增排列数组"
    };
    document.querySelector(".chart").setAttribute("aria-label", descriptions[algorithm]);
    const swappedVariable = document.querySelector("#swapped-variable");
    const minVariable = document.querySelector("#min-variable");
    if (swappedVariable.style) swappedVariable.style.display = ["bubble", "comb"].includes(algorithm) ? "" : "none";
    if (minVariable.style) minVariable.style.display = ["selection", "quick"].includes(algorithm) ? "" : "none";
    if (minVariable.firstChild) minVariable.firstChild.textContent = algorithm === "quick" ? "pivot = " : "minIndex = ";
    document.querySelector("#gap-variable").style.display = ["shell", "comb"].includes(algorithm) ? "" : "none";
    document.querySelector("#heap-variable").style.display = algorithm === "heap" ? "" : "none";
    document.querySelector("#tmp-variable").style.display = ["heap", "shell", "comb"].includes(algorithm) ? "none" : "";
    document.querySelector("#algorithm-note").hidden = !["heap", "shell", "comb"].includes(algorithm);
    document.querySelector("#primary-count-label").textContent = algorithm === "counting" ? "输入扫描" : "比较次数";
    document.querySelector("#secondary-count-label").textContent = algorithm === "counting" ? "数组写入" : "交换次数";
    updateSaveStatus();
  }

  function updateSaveStatus() {
    const file = sources.get(algorithm);
    const label = { saved:'已保存到', pending:'待保存到', saving:'正在保存到', error:'保存失败', conflict:'未同步草稿', draft:'浏览器草稿模式 · 不写入磁盘' }[file.state];
    document.querySelector('#source-save-status').textContent = `${label || ''}${['saved','pending','saving'].includes(file.state) ? ` algorithms/${algorithm}_sort.cpp` : ''}${file.message ? `：${file.message}` : ''}`;
    document.querySelector('.source-save-bar').setAttribute('data-state', file.state);
    document.querySelector('#source-save-retry').hidden = !['error','pending'].includes(file.state);
    document.querySelector('#source-export').hidden = !['conflict','error','draft'].includes(file.state);
    document.querySelector('#source-reload').hidden = file.state !== 'conflict';
    document.querySelector('#source-reload').disabled = file.inFlight;
  }
  sources.subscribe(file => { if (file.id === algorithm) updateSaveStatus(); });
  document.querySelector('#source-save-retry').addEventListener('click', () => sources.retry(algorithm));
  document.querySelector('#source-export').addEventListener('click', () => {
    const file = codeEditor?.getActiveFile();
    if (!file) return;
    const link = document.createElement('a'), url = URL.createObjectURL(new Blob([file.source], { type:'text/plain;charset=utf-8' }));
    link.href = url; link.download = file.filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.querySelector('#source-reload').addEventListener('click', async () => {
    if (!window.confirm('将丢弃当前文件的草稿并重新载入磁盘源码。需要保留的话，请先导出草稿。')) return;
    const id = algorithm;
    try { const source = await sources.reload(id); codeEditor?.reloadFile(id, source); }
    catch (exception) { document.querySelector('#source-save-status').textContent = `重新载入失败：${exception.message}`; }
  });

  function updateControls() {
    const running = mode === "running" || mode === "stepping";
    toggle.setAttribute("aria-label", running ? "暂停" : "继续");
    document.querySelector("#play-icon").setAttribute("src", running ? "icons/pause.svg" : "icons/play.svg");
    toggle.disabled = !timeline || mode === "complete";
    step.disabled = !timeline || mode === "complete" || mode === "stepping";
    document.querySelector("#execution-state").textContent = { running: "运行中", paused: "已暂停", stepping: "动作单步", complete: "已完成", error: "无法播放", preparing: "准备 C++ 轨迹", stale: "轨迹已失效" }[mode];
  }

  function traceKey(file, array) { return JSON.stringify([file.algorithm, file.source, array]); }
  function currentTraceKey() {
    try { const file = codeEditor?.getActiveFile(); return file ? traceKey(file, parseArray(input.value)) : null; } catch { return null; }
  }
  function invalidateTrace(message) {
    pendingTrace = null; cachedTrace = null; cachedKey = null;
    compilerPanel?.runner.stop(); timeline?.kill(); timeline = null; resetGrippers(); stepTarget = null; stepEnds = [];
    mode = 'stale'; codeEditor?.setExecutionSource(null);
    document.querySelector('#status').textContent = message; document.querySelector('#status').classList.remove('done'); updateControls();
  }

  function initialize(autoplay = false, execution = null, previewOnly = false) {
  if (timeline) timeline.kill();
  resetGrippers();
  timeline = null;
  stepTarget = null;
  stepEnds = [];
  mode = execution ? autoplay ? "running" : "paused" : "preparing";
  document.querySelector("#bars").replaceChildren();
  document.querySelector("#indices").replaceChildren();
  document.querySelector("#var-n").textContent = values.length;
  renderCode();
  const trace = execution || [{ line:1, action:'none', pair:[], order:values.map((_,id)=>id), values:values.slice(), i:null, j:null, minIndex:null, tmp:null, swapped:null, comparisons:0, swaps:0, scans:0, writes:0, sortedFrom:null }];
  const svgNS = "http://www.w3.org/2000/svg";
  // Two carried bars need separate lanes above the untouched bars when a swap
  // spans several slots. Keep the bubble/counting stage at its original size.
  const usesExchange = ["selection", "insertion", "merge", "quick", "heap", "shell", "comb"].includes(algorithm)
    || trace.some(event => event.action === 'lift' && Math.abs(event.order.indexOf(event.pair[0]) - event.order.indexOf(event.pair[1])) > 1);
  const baseline = usesExchange ? 380 : 290;
  const idleY = baseline - 130;
  const chart = document.querySelector(".chart");
  chart.setAttribute("viewBox", `0 0 484 ${baseline + 30}`);
  chart.style.aspectRatio = `484 / ${baseline + 30}`;
  const floor = document.querySelector("#array-baseline");
  floor.setAttribute("y1", baseline);
  floor.setAttribute("y2", baseline);
  const pitch = Math.min(57, 456 / values.length);
  const width = Math.min(40, pitch - 12);
  // Open jaws also need to fit beside a claw that remains closed in place.
  const openGap = Math.min(width / 2 + 11, pitch / 2 - 1);
  const origin = (484 - (values.length - 1) * pitch - width) / 2;
  const center = position => origin + position * pitch + width / 2;
  const max = trace.reduce((largest, event) => event.values.reduce((n, value) => Math.max(n, value), largest), Math.max(1, ...values));
  const heightFor = value => 22 + Math.max(0, value - 1) * 70 / Math.max(1, max - 1);
  const poses = values.map((value, id) => ({ x: origin + id * pitch, y: 0, height: heightFor(value) }));
  const bars = values.map((value, id) => {
    const group = document.createElementNS(svgNS, "g");
    group.classList.add("bar");
    group.dataset.id = id;
    group.dataset.value = value;
    const height = poses[id].height;
    const rect = document.createElementNS(svgNS, "rect");
    Object.entries({ x: 0, y: baseline - height, width, height, rx: 3 }).forEach(([key, val]) => rect.setAttribute(key, val));
    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("x", width / 2);
    text.setAttribute("style", `font-size:${Math.min(17, width * .56)}px`);
    text.setAttribute("y", baseline - height + Math.min(23, height - 4));
    text.textContent = value;
    group.append(rect, text);
    document.querySelector("#bars").append(group);
    const index = document.createElementNS(svgNS, "text");
    index.setAttribute("x", center(id));
    index.setAttribute("y", baseline + 20);
    index.textContent = id;
    document.querySelector("#indices").append(index);
    return group;
  });
  const pointer = document.querySelector("#comparator");
  const slider = document.querySelector("#slider");
  const machine = { x: (center(0) + center(1)) / 2 };
  const claws = [0, 1].map(index => ({
    x: center(index), y: idleY, gap: openGap, attached: null,
    element: document.querySelector(`#claw-${index}`),
    arm: document.querySelector(`#arm-${index}`),
    leftJaw: document.querySelector(`#claw-${index} .jaw-left`),
    rightJaw: document.querySelector(`#claw-${index} .jaw-right`)
  }));
  const relation = document.querySelector("#comparison");
  const status = document.querySelector("#status");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let selectedPair = [];
  let pairMode = "";

  function drawMachine() {
    poses.forEach((pose, id) => {
      bars[id].setAttribute("transform", `translate(${pose.x} ${pose.y})`);
      bars[id].children[0].setAttribute("y", baseline - pose.height);
      bars[id].children[0].setAttribute("height", pose.height);
      bars[id].children[1].setAttribute("y", baseline - pose.height + Math.min(23, pose.height - 4));
    });
    slider.setAttribute("transform", `translate(${machine.x} 0)`);
    claws.forEach(claw => {
      // A closed gripper follows the same pose as its bar, including lift and crossing.
      const pose = claw.attached === null ? null : poses[claw.attached];
      const x = pose ? pose.x + width / 2 : claw.x;
      const y = pose ? baseline - pose.height + pose.y - 14 : claw.y;
      claw.element.setAttribute("transform", `translate(${x} ${y})`);
      claw.arm.setAttribute("d", `M ${machine.x - 8} 37 L ${x - 8} ${y - 4} M ${machine.x + 8} 37 L ${x + 8} ${y - 4}`);
      claw.leftJaw.setAttribute("d", `M -12 0 H ${-claw.gap} V 25 H ${-claw.gap + 6}`);
      claw.rightJaw.setAttribute("d", `M 12 0 H ${claw.gap} V 25 H ${claw.gap - 6}`);
    });
  }

  function attach(grippers, pair) {
    grippers.forEach((claw, index) => { claw.attached = pair[index]; });
    drawMachine();
  }

  function detach(grippers = claws) {
    grippers.forEach(claw => {
      if (claw.attached === null) return;
      const pose = poses[claw.attached];
      claw.x = pose.x + width / 2;
      claw.y = baseline - pose.height + pose.y - 14;
      claw.attached = null;
    });
    drawMachine();
  }

  function colorMachine(mode = "") {
    pointer.classList.toggle("comparing", mode === "comparing");
    pointer.classList.toggle("exchanging", mode === "exchanging");
  }

  function clearPair() {
    selectedPair = [];
    pairMode = "";
    bars.forEach(bar => { bar.classList.remove("comparing", "exchanging"); });
    colorMachine();
    relation.textContent = "";
  }
  resetGrippers = () => {
    detach();
    claws.forEach(claw => { claw.gap = openGap; claw.y = idleY; });
    clearPair(); drawMachine();
  };

  function highlight(...lines) {
    codeEditor?.highlight(lines.at(-1) || 1);
  }

  function render(event) {
    if (["lift", "exchange"].includes(event.action)) {
      selectedPair = event.pair;
      pairMode = "exchanging";
    }
    document.querySelector("#comparisons").textContent = algorithm === "counting" ? event.scans || 0 : event.comparisons;
    document.querySelector("#swaps").textContent = algorithm === "counting" ? event.writes || 0 : event.swaps;
    document.querySelector("#var-i").textContent = event.i ?? "-";
    document.querySelector("#var-j").textContent = event.j ?? "-";
    document.querySelector("#var-swapped").textContent = event.swapped === null ? "-" : String(event.swapped);
    document.querySelector("#var-min").textContent = (algorithm === 'quick' ? event.variables?.pivot : event.minIndex) ?? "-";
    document.querySelector("#var-tmp").textContent = event.tmp ?? "-";
    document.querySelector("#var-gap").textContent = event.gap ?? "-";
    document.querySelector("#var-heap-size").textContent = event.heapSize ?? "-";
    document.querySelector("#algorithm-note").textContent = event.note || "";
    colorMachine(pairMode);
    bars.forEach((bar, id) => {
      const selected = selectedPair.includes(id);
      bar.classList.toggle("comparing", selected && pairMode === "comparing");
      bar.classList.toggle("exchanging", selected && pairMode === "exchanging");
      bar.classList.toggle("candidate", algorithm === "selection" && event.order?.[event.minIndex] === id);
      const position = Math.round((poses[id].x - origin) / pitch);
      const prefixSorted = ["selection", "insertion", "counting"].includes(algorithm);
      bar.classList.toggle("finished", event.action === "finish" || event.sortedFrom != null && (prefixSorted ? position < event.sortedFrom : ["bubble", "heap"].includes(algorithm) && position >= event.sortedFrom));
    });
    if (algorithm === "counting" && event.action === "count") relation.textContent = `count[${event.result}] = ${event.bucketCount}`;
    status.textContent = `a = [${event.values.join(", ")}]`;
    status.classList.toggle("done", event.action === "finish");
    highlight(event.line);
  }

  if (!execution) {
    codeEditor?.setExecutionSource(null); drawMachine(); render(trace[0]); updateControls();
    if (!previewOnly) {
      pendingTrace = null; compilerPanel?.runner.stop();
      if (!compilerPanel || !compilerPanel.start(true, autoplay)) {
        mode = 'error'; document.querySelector('#status').textContent = '无法准备动画，请检查输出面板并通过 HTTP 打开页面。'; updateControls();
      }
    }
    return;
  }
  codeEditor?.setExecutionSource(codeEditor.getActiveFile().source);
  if (!window.gsap) {
    status.textContent = "动画库加载失败，请检查 assets/gsap.min.js 的加载状态。";
    mode = "error";
  } else {
    const gsap = window.gsap;
    document.querySelector("#machine-rail").style.display = algorithm === "counting" ? "none" : "";
    gsap.set(pointer, { opacity: algorithm === "counting" ? 0 : 1 });
    drawMachine();
    render(trace[0]);
    if (reducedMotion) {
      const finalEvent = trace.at(-1);
      const finalOrder = finalEvent.order;
      finalOrder.forEach((id, position) => { poses[id].x = origin + position * pitch; });
      finalEvent.values.forEach((value, position) => {
        const id = finalOrder[position];
        const height = heightFor(value);
        poses[id].height = height;
        bars[id].dataset.value = value;
        bars[id].children[0].setAttribute("y", baseline - height);
        bars[id].children[0].setAttribute("height", height);
        bars[id].children[1].setAttribute("y", baseline - height + Math.min(23, height - 4));
        bars[id].children[1].textContent = value;
      });
      drawMachine();
      gsap.set(pointer, { opacity: 0 });
      render(finalEvent);
      mode = "complete";
    } else {
      timeline = gsap.timeline({ paused: !autoplay, delay: autoplay ? 1 : 0, onUpdate: drawMachine, onComplete: () => { mode = "complete"; stepTarget = null; updateControls(); } });
      let time = 0;
      // Plan the physical grippers' slots across exchanges. The timeline is built
      // before playback, so poses still contain the original coordinates here.
      const clawSlots = new Map(claws.map((claw, index) => [claw, index]));
      const clawHolds = new Map(claws.map(claw => [claw, null]));
      function release(at, retain = []) {
        const free = claws.filter(claw => !retain.includes(clawHolds.get(claw)));
        if (free.length) {
          timeline.to(free, { gap: openGap, duration: .14, ease: "sine.inOut" }, at);
          timeline.call(() => { detach(free); clearPair(); }, [], at + .14);
          timeline.to(free, { y: idleY, duration: .22, ease: "power2.inOut" }, at + .14);
        } else timeline.call(clearPair, [], at);
        free.forEach(claw => clawHolds.set(claw, null));
      }

      trace.forEach((event, index) => {
        const next = trace[index + 1];
        timeline.addLabel(`action-${index}`, time);
        timeline.call(() => render(event), [], time);
        if (event.action === "grip") {
          timeline.call(clearPair, [], time);
          const pair = [...event.pair].sort((a, b) => event.order.indexOf(a) - event.order.indexOf(b));
          const pairPositions = pair.map(id => event.order.indexOf(id));
          // Preserve the physical claw already holding each reused element,
          // even if that element becomes the other operand or changes slots.
          const grippers = pair.map(id => claws.find(claw => clawHolds.get(claw) === id));
          const free = claws.filter(claw => !grippers.includes(claw)).sort((a, b) => clawSlots.get(a) - clawSlots.get(b));
          grippers.forEach((claw, side) => { if (!claw) grippers[side] = free.shift(); });
          const moving = grippers.filter((claw, side) => clawHolds.get(claw) !== pair[side]);
          const retained = grippers.filter(claw => !moving.includes(claw));
          timeline.to(machine, { x: (center(pairPositions[0]) + center(pairPositions[1])) / 2, duration: .24, ease: "power2.inOut" }, time);
          // A free claw may pass from one side of a retained claw to the other.
          // First rise into a separate lane, then traverse, then descend. The
          // retained claw has no x/y/gap tween and stays attached throughout.
          const travelY = retained.length ? Math.min(idleY, ...retained.map(claw => {
            const id = clawHolds.get(claw), value = event.values[event.order.indexOf(id)];
            return baseline - heightFor(value) - 14 - 40;
          })) : idleY;
          const approach = retained.length && moving.length ? .18 : 0;
          grippers.forEach((claw, side) => {
            const value = event.values[pairPositions[side]];
            clawSlots.set(claw, pairPositions[side]);
            clawHolds.set(claw, pair[side]);
            if (!moving.includes(claw)) return;
            if (approach) timeline.to(claw, { y: travelY, duration: approach, ease: "power2.inOut" }, time);
            timeline.to(claw, { x: center(pairPositions[side]), duration: .24, ease: "power2.inOut" }, time + approach);
            timeline.to(claw, { y: baseline - heightFor(value) - 14, duration: .22, ease: "power2.inOut" }, time + approach + .24);
          });
          if (moving.length) {
            timeline.to(moving, { gap: width / 2 + 4, duration: .14, ease: "sine.inOut" }, time + approach + .46);
            const movingPair = moving.map(claw => pair[grippers.indexOf(claw)]);
            timeline.call(() => attach(moving, movingPair), [], time + approach + .60);
          }
          time += moving.length ? approach + .70 : .28;
        } else if (event.action.startsWith("compare-")) {
          timeline.call(() => {
            selectedPair = event.pair;
            pairMode = event.action === "compare-swap" && event.result ? "exchanging" : "comparing";
            render(event);
            const positions = event.pair.map(id => event.order.indexOf(id));
            const left = event.values[positions[0]], right = event.values[positions[1]];
            const operator = event.operator || (algorithm === "selection" || algorithm === "merge" ? (event.result ? ">" : "≤") : algorithm === "insertion" ? (event.result ? ">" : "≤") : algorithm === "quick" ? (event.result ? "<" : "≥") : (event.result ? ">" : "≤"));
            relation.textContent = event.relation || (event.pair.length > 1 ? `${left} ${operator} ${right}` : "");
          }, [], time);
          const keepGripped = ((algorithm === "bubble" || algorithm === "insertion") && event.result) || next?.action === "lift";
          const explicitRelease = next?.action === "release";
          if (!keepGripped && !explicitRelease) { release(time + .30); time += .72; }
          else time += .36;
        } else if (event.action === "lift") {
          const [left, right] = event.pair.map(id => poses[id]);
          const slots = Math.abs(event.order.indexOf(event.pair[0]) - event.order.indexOf(event.pair[1]));
          if (slots === 1) {
            timeline.to(left, { y: -(right.height + 38), duration: .28, ease: "power2.inOut" }, time);
          } else {
            const tallest = Math.max(...poses.map(pose => pose.height));
            // The lower lane clears every stationary bar; the upper lane also
            // clears the other carried bar and its gripper during crossing.
            timeline.to(right, { y: -(tallest + 24), duration: .28, ease: "power2.inOut" }, time);
            timeline.to(left, { y: -(tallest + right.height + 48), duration: .28, ease: "power2.inOut" }, time);
          }
          time += .36;
        } else if (event.action === "exchange") {
          const [left, right] = event.pair.map(id => poses[id]);
          const leftTarget = origin + event.order.indexOf(event.pair[0]) * pitch;
          const rightTarget = origin + event.order.indexOf(event.pair[1]) * pitch;
          const slots = Math.abs(event.order.indexOf(event.pair[0]) - event.order.indexOf(event.pair[1]));
          // event.order is already the post-swap order: each ID travels to its
          // own recorded destination, not to the other ID's destination.
          if (slots === 1) {
            // Match bubble sort: move the grounded right bar into the vacated
            // slot, then move the lifted left bar right and lower it.
            timeline.to(right, { x: rightTarget, duration: .32, ease: "power2.inOut" }, time);
            timeline.to(left, { x: leftTarget, duration: .32, ease: "power2.inOut" }, time + .40);
            timeline.to(left, { y: 0, duration: .26, ease: "power2.inOut" }, time + .72);
            time += 1.06;
          } else {
            const travel = .32 + .08 * slots;
            timeline.to(left, { x: leftTarget, duration: travel, ease: "power2.inOut" }, time);
            timeline.to(right, { x: rightTarget, duration: travel, ease: "power2.inOut" }, time);
            timeline.to([left, right], { y: 0, duration: .26, ease: "power2.inOut" }, time + travel);
            time += travel + .34;
          }
        } else if (event.action === "count") {
          timeline.call(() => {
            selectedPair = event.pair;
            pairMode = "comparing";
            render(event);
            relation.textContent = `count[${event.result}] = ${event.bucketCount}`;
          }, [], time);
          timeline.call(clearPair, [], time + .30);
          time += .40;
        } else if (event.action === "write") {
          const id = event.order[event.target ?? event.i];
          const pose = poses[id];
          const value = event.result;
          const height = heightFor(value);
          timeline.to(pose, { height, duration: .24, ease: "power2.inOut" }, time);
          timeline.call(() => {
            bars[id].dataset.value = value;
            bars[id].children[0].setAttribute("y", baseline - height);
            bars[id].children[0].setAttribute("height", height);
            bars[id].children[1].setAttribute("y", baseline - height + Math.min(23, height - 4));
            bars[id].children[1].textContent = value;
            relation.textContent = `a[${event.target ?? event.i}] = ${value}`;
            drawMachine();
          }, [], time + .24);
          time += .30;
        } else if (event.action === "shift") {
          timeline.to(poses[event.movingId ?? event.pair[1]], { x: origin + (event.target ?? event.j) * pitch, duration: .32, ease: "power2.inOut" }, time);
          time += .40;
        } else if (event.action === "drop") {
          const left = poses[event.movingId ?? event.pair[0]];
          timeline.to(left, { x: origin + (event.target ?? event.j + 1) * pitch, duration: .32, ease: "power2.inOut" }, time);
          timeline.to(left, { y: 0, duration: .26, ease: "power2.inOut" }, time + .32);
          timeline.to(event.pair.map(id => poses[id]), { y: 0, duration: .26, ease: "power2.inOut" }, time + .32);
          time += .66;
        } else if (event.action === "release") {
          claws.forEach(claw => { const id = clawHolds.get(claw); if (id !== null) clawSlots.set(claw, event.order.indexOf(id)); });
          release(time + .10, event.retain || []);
          time += event.retain?.length === 2 ? .28 : .52;
        } else if (event.action === "finish") {
          timeline.to(pointer, { opacity: 0, duration: .30 }, time);
          time += .30;
        } else time += .28;

        // Each boundary commits one statement and highlights the next executable line.
        if (next) {
          const boundary = time - .001;
          stepEnds.push(boundary);
          timeline.call(() => render(next), [], boundary);
        }
      });
      stepEnds.push(timeline.duration());
      stepEnds.slice(0, -1).forEach(boundary => {
        timeline.call(() => {
          if (mode !== "stepping" || Math.abs(stepTarget - boundary) > .001) return;
          timeline.pause(boundary, true);
          drawMachine();
          mode = "paused";
          stepTarget = null;
          updateControls();
        }, [], boundary);
      });
    }
  }
  updateControls();
  }

  toggle.addEventListener("click", () => {
    if (!timeline || mode === "complete") return;
    stepTarget = null;
    if (mode === "running" || mode === "stepping") { timeline.pause(); mode = "paused"; }
    else { mode = "running"; timeline.play(); }
    updateControls();
  });
  step.addEventListener("click", () => {
    if (!timeline || mode === "complete" || mode === "stepping") return;
    timeline.pause();
    stepTarget = stepEnds.find(boundary => boundary > timeline.time() + .001);
    if (stepTarget === undefined) return;
    mode = "stepping";
    timeline.play();
    updateControls();
  });
  restart.addEventListener("click", () => initialize(false, cachedKey && cachedKey === currentTraceKey() ? cachedTrace : null));
  input.addEventListener('input', () => invalidateTrace('输入已修改，请应用数组或编译并演示。'));
  document.querySelector("#randomize").addEventListener("click", () => {
    input.value = Array.from({ length: values.length }, () => Math.floor(Math.random() * 99) + 1).join(", ");
    arrayForm.requestSubmit();
  });
  arrayForm.addEventListener("submit", event => {
    event.preventDefault();
    try {
      const next = parseArray(input.value);
      values = next;
      input.setAttribute("aria-invalid", "false");
      error.textContent = "";
      initialize(false);
    } catch (exception) {
      input.setAttribute("aria-invalid", "true");
      error.textContent = exception.message;
    }
  });
  function selectAlgorithm(nextAlgorithm) {
    if (nextAlgorithm === algorithm) return;
    algorithm = nextAlgorithm;
    initialize(true);
  }
  tabs.forEach(tab => tab.addEventListener("click", () => selectAlgorithm(tab.dataset.algorithm)));
  explorerFiles.forEach(file => file.addEventListener("click", () => {
    selectAlgorithm(file.dataset.algorithm);
    if (narrowLayout.matches) setExplorerExpanded(false);
  }));
  initialize(true);
}

if (typeof document !== 'undefined') {
  if (window.SortSources?.ready) startSortDemo();
  else window.SortSources.load().then(startSortDemo).catch(exception => {
    document.querySelector('#editor-notice').textContent = `算法源码加载失败：${exception.message}。请检查服务并刷新页面。`;
    document.querySelector('#source-save-status').textContent = '源码未加载，编译和动画尚未启动。';
  });
}
