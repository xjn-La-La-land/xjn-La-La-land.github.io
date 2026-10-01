const assert = require("node:assert/strict");
const { buildTrace, buildExecutionTrace } = require("./fixtures/legacy-trace.cjs");
const { parseArray } = require("../frontend/js/demo.js");
const algorithms = ["bubble", "selection", "insertion", "merge", "quick", "heap", "shell", "comb", "counting"];
const extendedAlgorithms = algorithms.slice(1);

for (const algorithm of extendedAlgorithms) {
  for (const input of [[5, 2, 8, 1, 7, 3, 6, 4], [3, 1, 3, 2], [1, 2, 3], [99, 1]]) {
    const original = [...input];
    const trace = buildExecutionTrace(input, algorithm);
    assert.deepEqual(trace.at(-1).values, [...input].sort((a, b) => a - b));
    assert.deepEqual(input, original);
    assert.equal(trace.at(-1).action, "finish");
    assert.ok(trace.some(event => event.line > 0));
    assert.ok(trace.every(event => event.order.length === input.length && event.values.length === input.length));
  }
}

// Exercise the new traces independently of the renderer: equal keys, shrinking
// gaps, and heap extraction must preserve each original element and the suffix.
let seed = 0x12345678;
for (let length = 2; length <= 12; length++) for (let trial = 0; trial < 12; trial++) {
  const input = Array.from({ length }, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return 1 + seed % (trial % 3 === 0 ? 4 : 99);
  });
  if (trial % 3 === 1) input.sort((a, b) => b - a);
  if (trial % 3 === 2) input.sort((a, b) => a - b);
  for (const algorithm of ["heap", "shell", "comb"]) {
    const trace = buildExecutionTrace(input, algorithm);
    const sorted = [...input].sort((a, b) => a - b);
    assert.deepEqual(trace.at(-1).values, sorted);
    trace.forEach((event, index) => {
      assert.deepEqual(event.order.map(id => input[id]), event.values);
      assert.deepEqual([...event.order].sort((a, b) => a - b), input.map((_, id) => id));
      if (index) {
        assert.equal(event.swaps - trace[index - 1].swaps, Number(event.action === "exchange"));
        assert.equal(event.comparisons - trace[index - 1].comparisons, Number(event.action === "compare-select"));
      }
      if (algorithm === "heap") assert.deepEqual(event.values.slice(event.sortedFrom), sorted.slice(event.sortedFrom));
      if (event.action === "grip") {
        const slots = event.pair.map(id => event.order.indexOf(id));
        assert.ok(slots[0] >= 0 && slots[1] > slots[0]);
        if (algorithm !== "heap") assert.equal(slots[1] - slots[0], event.gap);
      }
    });
  }
}

assert.deepEqual(parseArray("5, 2 8，1"), [5, 2, 8, 1]);
assert.deepEqual(parseArray(" 1 1 99 "), [1, 1, 99]);
for (const input of ["", "1", "0 2", "100 2", "-1 2", "1.5 2", "2e1 2", "NaN 2", "1,,2", "1,2,", "1 2 x", Array(13).fill(1).join(" ")]) {
  assert.throws(() => parseArray(input));
}

for (const input of [[], [1], [1, 2, 3], [3, 2, 1], [2, 1, 2, 1], [5, 2, 8, 1, 7, 3, 6, 4]]) {
  const original = [...input];
  const events = buildTrace(input);
  const final = events.at(-1);
  assert.deepEqual(input, original);
  assert.deepEqual(final.values, [...input].sort((a, b) => a - b));
  assert.equal(final.swaps, input.reduce((total, value, i) => total + input.slice(i + 1).filter(next => next < value).length, 0));
  assert.equal(final.comparisons, events.filter(event => event.kind === "compare").length);
  let previous = events[0];
  for (const event of events.slice(1)) {
    assert.deepEqual(event.order.map(id => input[id]), event.values);
    assert.deepEqual([...event.order].sort((a, b) => a - b), input.map((_, i) => i));
    if (event.kind === "swap") {
      assert.equal(previous.kind, "compare");
      assert.ok(previous.values[event.j] > previous.values[event.j + 1]);
      const expected = [...previous.order];
      [expected[event.j], expected[event.j + 1]] = [expected[event.j + 1], expected[event.j]];
      assert.deepEqual(event.order, expected);
    } else {
      assert.deepEqual(event.order, previous.order);
    }
    const sorted = [...input].sort((a, b) => a - b);
    assert.deepEqual(event.values.slice(event.sortedFrom), sorted.slice(event.sortedFrom));
    previous = event;
  }
  for (let i = 1; i < final.order.length; i++) {
    if (final.values[i] === final.values[i - 1]) assert.ok(final.order[i] > final.order[i - 1]);
  }
}
const demo = buildTrace([5, 2, 8, 1, 7, 3, 6, 4]).at(-1);
assert.equal(demo.comparisons, 25);
assert.equal(demo.swaps, 14);
// Run the actual GSAP timeline against a small DOM stub, without a browser.
const fs = require("node:fs");
const vm = require("node:vm");
const { gsap } = require("../frontend/assets/gsap.min.js");
const { createSortCodeEditor } = require("../frontend/js/code-editor.js");
const { createMonacoStub } = require("./editor.test.cjs");
const html = fs.readFileSync(__dirname + "/../frontend/index.html", "utf8");
assert.deepEqual([...html.matchAll(/class="code-tab"[^>]+data-algorithm="([^"]+)"/g)].map(match => match[1]), algorithms);
assert.deepEqual([...html.matchAll(/class="explorer-file"[^>]+data-algorithm="([^"]+)"/g)].map(match => match[1]), algorithms);
function element() {
  const classes = new Set();
  const listeners = new Map();
  return {
    attrs: {}, dataset: {}, style: {}, children: [], textContent: "", innerHTML: "", opacity: 1, value: "", disabled: false,
    setAttribute(key, value) { this.attrs[key] = String(value); },
    getAttribute(key) { return this.attrs[key]; },
    contains(node) { return this.children.includes(node); },
    focus() { this.focused = true; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener(name, callback) { listeners.set(name, callback); },
    dispatch(name, event = {}) { if (!this.disabled) listeners.get(name)?.({ preventDefault() {}, ...event }); },
    classList: {
      add(...names) { names.forEach(name => classes.add(name)); },
      remove(...names) { names.forEach(name => classes.delete(name)); },
      toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); },
      contains(name) { return classes.has(name); }
    }
  };
}
function createHarness(reducedMotion = false, missingLibrary = false, narrowLayout = false, missingEditor = false, collect = null, deferred = false) {
  const elements = new Map();
  const monaco = createMonacoStub();
  const tabs = algorithms.map(algorithm => {
    const tab = element();
    tab.dataset.algorithm = algorithm;
    tab.offsetLeft = 43 + algorithms.indexOf(algorithm) * 146;
    tab.offsetWidth = 145;
    return tab;
  });
  const explorerFiles = algorithms.map(algorithm => {
    const file = element();
    file.dataset.algorithm = algorithm;
    return file;
  });
  const get = selector => {
    if (!elements.has(selector)) {
      const node = element();
      if (selector === ".code-tabs") Object.assign(node, { offsetLeft: 0, scrollLeft: 0, clientWidth: 300 });
      elements.set(selector, node);
    }
    return elements.get(selector);
  };
  get("#explorer-sidebar").children = explorerFiles;
  get("#array-input").value = "5, 2, 8, 1, 7, 3, 6, 4";
  const records = new Map(Object.entries(require('./algorithms.cjs')).map(([id, source]) => [id, { id, source, diskSource:source, state:'draft' }]));
  const sources = { ready:true, get:id=>records.get(id), subscribe:()=>()=>{}, edit(id, source){records.get(id).source=source;} };
  let timeline, panel;
  const documentStub = {
    activeElement: null,
    createElementNS: element, createElement: element, querySelector: get,
    querySelectorAll(selector) {
      if (selector === ".code-tab") return tabs;
      if (selector === ".explorer-file") return explorerFiles;
      return [];
    }
  };
  vm.runInNewContext(fs.readFileSync(__dirname + "/../frontend/js/demo.js", "utf8"), {
    document: documentStub,
    window: {
      monaco: missingEditor ? undefined : monaco, createSortCodeEditor, SortSources:sources,
      // The collector is a rendering fixture here, not a production fallback.
      // trace-wasm.test.cjs independently runs all nine actual C++ algorithms.
      SortTraceEvents: collect ? require('../frontend/js/trace-events.js') : { toAnimationTrace: record => {
        // The legacy bubble fixture records state before each statement. Production
        // animation events carry the state after their operation; align the
        // fixture contract while retaining the current action and source site.
        const events = buildExecutionTrace(record.values, record.algorithm);
        if (record.algorithm !== 'bubble') return events;
        return events.map((event, index) => ({ ...events[Math.min(index + 1, events.length - 1)],
          line: event.line, action: event.action, pair: event.pair, result: event.result,
          target: event.target, movingId: event.movingId }));
      } },
      createCompilerPanel(options) {
        panel = {
          requests: [],
          runner: { stop() {} }, report() {},
          deliver(request, result = { status: "done", exitCode: 0, trace: collect ? collect(request) : request }) {
            options.onTraceFinish(result, request);
          },
          start(trace, autoplay) {
            const file = options.getFile();
            if (!file) return false;
            const request = { ...file, values: options.getValues(), trace, autoplay };
            this.requests.push(request);
            options.onTraceStart(request);
            if (!deferred) this.deliver(request);
            return true;
          }
        };
        return panel;
      },
      matchMedia: query => ({ matches: query.includes("max-width") ? narrowLayout : reducedMotion }),
      gsap: missingLibrary ? undefined : {
        set: gsap.set,
        timeline(options) { timeline = gsap.timeline({ ...options, paused: true, delay: 0 }); return timeline; }
      }
    }
  });
  return { get, editor: monaco.instance, tabs, explorerFiles, document: documentStub, compiler: panel, get timeline() { return timeline; } };
}
const highlightedLine = harness => harness.editor.decorations[0]?.range.startLineNumber;
const numbers = value => value.match(/-?\d+(?:\.\d+)?/g).map(Number);

function checkGripTargets(input, algorithm = "bubble") {
  const harness = createHarness();
  harness.get("#array-input").value = input.join(", ");
  harness.get("#array-form").dispatch("submit");
  if (algorithm !== "bubble") harness.tabs.find(tab => tab.dataset.algorithm === algorithm).dispatch("click");
  const { get, timeline } = harness;
  const bars = get("#bars").children;
  const width = Number(bars[0].children[0].attrs.width);
  const slotCenters = get("#indices").children.map(index => Number(index.attrs.x));
  const grips = buildExecutionTrace(input, algorithm).filter(event => event.action === "grip");
  const closures = timeline.getChildren().filter(tween => tween.vars.gap === width / 2 + 4);
  assert.equal(closures.length, grips.length);
  let time = 0;
  closures.forEach((closure, index) => {
    const approachStart = closure.startTime() - .46;
    for (; time < approachStart; time += .02) timeline.time(Math.min(time, approachStart), false);
    timeline.time(approachStart, false);
    const initialX = [0, 1].map(side => numbers(get(`#claw-${side}`).attrs.transform)[0]);
    const leftSide = initialX[0] < initialX[1] ? 0 : 1;
    // Observe the approach BEFORE attachment can hide a bad target by snapping.
    const beforeAttach = closure.endTime() - .00001;
    for (time = approachStart; time < beforeAttach; time += .02) {
      timeline.time(time, false);
      const x = [0, 1].map(side => numbers(get(`#claw-${side}`).attrs.transform)[0]);
      assert.ok(x[1 - leftSide] - x[leftSide] >= 24 - .001,
        `Grip ${index}: retracted grippers must not cross or collide during approach`);
    }
    timeline.time(beforeAttach, false);
    const claws = [0, 1].map(side => numbers(get(`#claw-${side}`).attrs.transform));
    const expected = grips[index].pair.map(id => {
      const rect = bars[id].children[0].attrs;
      return [slotCenters[grips[index].order.indexOf(id)], Number(rect.y) - 14];
    }).sort((a, b) => a[0] - b[0]);
    const actual = [...claws].sort((a, b) => a[0] - b[0]);
    actual.forEach((pose, side) => {
      assert.ok(Math.abs(pose[0] - expected[side][0]) < .01 && Math.abs(pose[1] - expected[side][1]) < .01,
        `Grip ${index}: approach ${pose} should reach ${expected[side]} before attachment`);
    });
    const sliderX = numbers(get("#slider").attrs.transform)[0];
    assert.ok(Math.abs(sliderX - (expected[0][0] + expected[1][0]) / 2) < .01, `Grip ${index}: slider must center over the current pair`);
    timeline.time(closure.endTime() + .00001, false);
    claws.forEach((pose, side) => {
      const attached = numbers(get(`#claw-${side}`).attrs.transform);
      assert.ok(Math.hypot(attached[0] - pose[0], attached[1] - pose[1]) < .01, `Grip ${index}: attachment must not jump`);
    });
    time = closure.endTime() + .00001;
  });
  timeline.kill();
}
for (const input of [[5, 2, 8, 1, 7, 3, 6, 4], [2, 1, 2, 1], [99, 1], [1, 99], [5, 5, 5], [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88]]) checkGripTargets(input);
checkGripTargets([5, 2, 8, 1, 7, 3, 6, 4], "selection");
for (const algorithm of ["heap", "shell", "comb"]) checkGripTargets([5, 2, 8, 1, 7, 3, 6, 4], algorithm);

function exerciseTimeline(reducedMotion = false, missingLibrary = false, custom = null) {
  const harness = createHarness(reducedMotion, missingLibrary);
  const { get } = harness;
  if (custom) {
    get("#array-input").value = custom.join(", ");
    get("#array-form").dispatch("submit");
  }
  const timeline = harness.timeline;
  if (missingLibrary) {
    assert.match(get("#status").textContent, /gsap\.min\.js/);
    return;
  }
  const bars = get("#bars").children;
  let grippedFrames = 0, liftedFrames = 0;
  if (timeline) {
    for (let time = 0; time <= timeline.duration() + .02; time += .02) {
      timeline.time(Math.min(time, timeline.duration()), false);
      const poses = bars.map(bar => numbers(bar.attrs.transform));
      const boxes = poses.map(([x, offsetY], id) => {
        const rect = bars[id].children[0].attrs;
        return { x, y: Number(rect.y) + offsetY, height: Number(rect.height), width: Number(rect.width) };
      });
      for (let a = 0; a < boxes.length; a++) {
        for (let b = a + 1; b < boxes.length; b++) {
          const left = boxes[a], right = boxes[b];
          const overlapX = Math.min(left.x + left.width, right.x + right.width) - Math.max(left.x, right.x);
          const overlapY = Math.min(left.y + left.height, right.y + right.height) - Math.max(left.y, right.y);
          assert.ok(overlapX < .001 || overlapY < .001, `Bars overlap at ${time.toFixed(2)}s`);
        }
      }
      for (let side = 0; side < 2; side++) {
        const [x, y] = numbers(get(`#claw-${side}`).attrs.transform);
        const rod = numbers(get(`#arm-${side}`).attrs.d);
        assert.ok(Math.abs(rod[2] - (x - 8)) < .001);
        assert.ok(Math.abs(rod[3] - (y - 4)) < .001);
        assert.ok(Math.abs(rod[6] - (x + 8)) < .001);
        assert.ok(Math.abs(rod[7] - (y - 4)) < .001);
        assert.ok(y >= 37, "Gripper must remain below the rail");
        const gap = -numbers(get(`#claw-${side} .jaw-left`).attrs.d)[2];
        if (Math.abs(gap - (boxes[0].width / 2 + 4)) < .001) {
          assert.ok(boxes.some(box => Math.abs(box.x + box.width / 2 - x) < .001 && Math.abs(box.y - 14 - y) < .001), `Detached closed gripper at ${time.toFixed(2)}s`);
          grippedFrames++;
        }
      }
      if (poses.some(([, y]) => y < -40)) liftedFrames++;
    }
    if (!reducedMotion) assert.ok(grippedFrames > 10);
    if (!reducedMotion && (!custom || buildTrace(custom).at(-1).swaps)) assert.ok(liftedFrames > 10);
    timeline.kill();
  }
  const expected = buildTrace(custom || [5, 2, 8, 1, 7, 3, 6, 4]).at(-1);
  assert.deepEqual(bars.map(bar => ({ value: Number(bar.dataset.value), x: numbers(bar.attrs.transform)[0] })).sort((a, b) => a.x - b.x).map(bar => bar.value), expected.values);
  assert.equal(get("#comparisons").textContent, expected.comparisons);
  assert.equal(get("#swaps").textContent, expected.swaps);
  assert.ok(bars.every(bar => bar.classList.contains("finished")));
  assert.equal(get("#comparator").opacity, 0);
  return { grippedFrames, liftedFrames };
}
console.log("Mechanical timeline:", exerciseTimeline());
exerciseTimeline(true);
exerciseTimeline(false, true);
for (const custom of [[99, 1], [1, 99], [5, 5, 5], [2, 1, 2, 1], [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88]]) exerciseTimeline(false, false, custom);

function advanceLine(harness) {
  harness.get("#step").dispatch("click");
  for (let t = harness.timeline.time() + .02; t <= harness.timeline.duration() + .02; t += .02) {
    harness.timeline.time(Math.min(t, harness.timeline.duration()), false);
    if (harness.get("#execution-state").textContent !== "动作单步") break;
  }
}

function checkControls() {
  const harness = createHarness();
  const { get } = harness;
  get("#play-toggle").dispatch("click");
  assert.equal(get("#execution-state").textContent, "已暂停");
  assert.ok(harness.timeline.paused());
  get("#play-toggle").dispatch("click");
  assert.equal(get("#execution-state").textContent, "运行中");
  get("#play-toggle").dispatch("click");
  for (let i = 0; i < 5; i++) advanceLine(harness);
  assert.equal(highlightedLine(harness), 7);
  // Pause mid-lift, then finish only the tmp assignment with one step.
  get("#play-toggle").dispatch("click");
  harness.timeline.time(harness.timeline.time() + .12, false);
  get("#play-toggle").dispatch("click");
  const frozen = get("#bars").children.map(bar => bar.attrs.transform);
  assert.ok(harness.timeline.paused());
  assert.deepEqual(get("#bars").children.map(bar => bar.attrs.transform), frozen);
  get("#step").dispatch("click");
  assert.ok(get("#step").disabled);
  for (let t = harness.timeline.time() + .02; t < harness.timeline.duration(); t += .02) {
    harness.timeline.time(t, false);
    if (get("#execution-state").textContent === "已暂停") break;
  }
  assert.equal(get("#comparisons").textContent, 1);
  assert.equal(get("#swaps").textContent, 0);
  assert.equal(get("#var-tmp").textContent, 5);
  assert.equal(highlightedLine(harness), 8);
  assert.equal(get("#execution-state").textContent, "已暂停");
  assert.equal(harness.editor.decorations.length, 1);
  get("#restart").dispatch("click");
  assert.equal(get("#comparisons").textContent, 0);
  assert.equal(get("#swaps").textContent, 0);
  assert.equal(get("#comparator").opacity, 1);
  const saved = harness.timeline;
  const savedBars = get("#bars").children;
  get("#array-input").value = "0, 2";
  get("#array-form").dispatch("submit");
  assert.equal(harness.timeline, saved);
  assert.equal(get("#bars").children, savedBars);
  assert.equal(get("#array-input").attrs["aria-invalid"], "true");
  get("#array-input").value = "3, 1, 3, 2";
  get("#array-form").dispatch("submit");
  assert.equal(get("#bars").children.length, 4);
  assert.equal(get("#var-n").textContent, 4);
  assert.equal(get("#array-error").textContent, "");
  assert.equal(get("#execution-state").textContent, "已暂停");
  const expected = buildExecutionTrace([3, 1, 3, 2]);
  for (const snapshot of expected.slice(1)) {
    advanceLine(harness);
    assert.equal(get("#comparisons").textContent, snapshot.comparisons);
    assert.equal(get("#swaps").textContent, snapshot.swaps);
    assert.equal(get("#status").textContent, `a = [${snapshot.values.join(", ")}]`);
    // Legacy rendering fixtures can have more lines than the editable C++.
    // Monaco intentionally clamps highlights to the current model's last line.
    assert.equal(highlightedLine(harness), Math.min(snapshot.line, harness.editor.model.getLineCount()));
    assert.ok(harness.timeline.paused());
  }
  advanceLine(harness);
  assert.equal(get("#execution-state").textContent, "已完成");
  assert.ok(get("#step").disabled);
  assert.ok(get("#play-toggle").disabled);
  assert.equal(get("#status").textContent, "a = [1, 2, 3, 3]");
  harness.timeline.kill();
}
checkControls();

function checkAlgorithmTabs() {
  const harness = createHarness();
  const expected = {
    selection: ["选择排序", "selection sort"],
    insertion: ["插入排序", "insertion sort"],
    merge: ["归并排序", "merge sort"],
    quick: ["快速排序", "quick sort"],
    heap: ["堆排序", "heap sort"],
    shell: ["希尔排序", "shell sort"],
    comb: ["梳排序", "comb sort"],
    counting: ["计数排序", "counting sort"]
  };
  for (const [algorithm, [title, subtitle]] of Object.entries(expected)) {
    const previousTimeline = harness.timeline;
    harness.tabs.find(tab => tab.dataset.algorithm === algorithm).dispatch("click");
    assert.equal(harness.get("h1").textContent, title);
    assert.equal(harness.get(".subtitle").textContent, subtitle);
    assert.equal(harness.get("#status").textContent, "a = [5, 2, 8, 1, 7, 3, 6, 4]");
    assert.equal(harness.get("#execution-state").textContent, "运行中");
    assert.notEqual(harness.timeline, previousTimeline);
    assert.ok(harness.tabs.find(tab => tab.dataset.algorithm === algorithm).attrs["aria-selected"] === "true");
    assert.equal(harness.explorerFiles.find(file => file.dataset.algorithm === algorithm).attrs["aria-current"], "true");
    assert.equal(harness.explorerFiles.filter(file => file.attrs["aria-current"] === "true").length, 1);
    assert.equal(harness.editor.model.getValue().includes(`${algorithm}_sort`), true);
    const activeTab = harness.tabs.find(tab => tab.dataset.algorithm === algorithm);
    const strip = harness.get(".code-tabs");
    assert.ok(activeTab.offsetLeft >= strip.scrollLeft && activeTab.offsetLeft + activeTab.offsetWidth <= strip.scrollLeft + strip.clientWidth);
    const baseline = algorithm === "counting" ? 330 : 460;
    assert.equal(harness.get(".chart").attrs.viewBox, `0 0 484 ${baseline + 30}`);
    assert.equal(Number(harness.get("#array-baseline").attrs.y1), baseline);
    harness.timeline.kill();
  }
  harness.tabs[0].dispatch("click");
  assert.equal(harness.get(".chart").attrs.viewBox, "0 0 484 360");
  assert.equal(Number(harness.get("#array-baseline").attrs.y1), 330);
  harness.timeline.kill();
}
checkAlgorithmTabs();

function checkExplorerNavigation(narrowLayout = false) {
  const harness = createHarness(false, false, narrowLayout);
  const { get, explorerFiles, tabs } = harness;
  const toggle = get("#explorer-toggle");
  function checkExpanded(expanded) {
    assert.equal(toggle.attrs["aria-expanded"], String(expanded));
    assert.equal(toggle.attrs["aria-label"], expanded ? "隐藏资源管理器" : "显示资源管理器");
    assert.equal(toggle.attrs.title, toggle.attrs["aria-label"]);
    assert.equal(get(".workspace").classList.contains("explorer-collapsed"), !expanded);
    assert.equal(get("#explorer-sidebar").inert, !expanded);
    assert.equal(get("#explorer-sidebar").attrs["aria-hidden"], String(!expanded));
  }
  checkExpanded(!narrowLayout);
  get("#array-input").value = "4, 2, 3, 1";
  get("#array-form").dispatch("submit");
  for (const file of explorerFiles) {
    if (toggle.attrs["aria-expanded"] !== "true") toggle.dispatch("click");
    checkExpanded(true);
    const previous = harness.timeline;
    file.dispatch("click");
    assert.equal(file.attrs["aria-current"], "true");
    assert.equal(explorerFiles.filter(item => item.attrs["aria-current"] === "true").length, 1);
    assert.equal(tabs.find(tab => tab.dataset.algorithm === file.dataset.algorithm).attrs["aria-selected"], "true");
    assert.equal(harness.editor.model.getValue().includes(`${file.dataset.algorithm}_sort`), true);
    assert.equal(get("#explorer-current").textContent, get("h1").textContent);
    assert.equal(get("#status").textContent, "a = [4, 2, 3, 1]");
    checkExpanded(!narrowLayout);
    if (file.dataset.algorithm !== "bubble") {
      assert.notEqual(harness.timeline, previous);
      assert.equal(get("#execution-state").textContent, "运行中");
      get("#play-toggle").dispatch("click");
      const selectedTimeline = harness.timeline;
      file.dispatch("click");
      assert.equal(harness.timeline, selectedTimeline, "Selecting the current file must preserve a paused animation");
      assert.equal(get("#execution-state").textContent, "已暂停");
    }
  }
  tabs[0].dispatch("click");
  assert.equal(explorerFiles[0].attrs["aria-current"], "true");
  assert.equal(get("#explorer-current").textContent, "冒泡排序");
  // Opening/closing the explorer must not recreate or rewind the animation.
  const timeline = harness.timeline;
  timeline.seek(timeline.duration() * .35, false);
  const runningTime = timeline.time();
  toggle.dispatch("click");
  toggle.dispatch("click");
  assert.equal(harness.timeline, timeline);
  assert.equal(timeline.time(), runningTime);
  assert.equal(get("#execution-state").textContent, "运行中");
  get("#play-toggle").dispatch("click");
  const time = timeline.time();
  const chart = get("#bars").children.map(bar => bar.attrs.transform);
  const codeLine = highlightedLine(harness);
  const state = get("#execution-state").textContent;
  for (let click = 0; click < 4; click++) {
    const expanded = toggle.attrs["aria-expanded"] !== "true";
    toggle.dispatch("click");
    checkExpanded(expanded);
    assert.equal(harness.timeline, timeline);
    assert.equal(timeline.time(), time);
    assert.deepEqual(get("#bars").children.map(bar => bar.attrs.transform), chart);
    assert.equal(highlightedLine(harness), codeLine);
    assert.equal(get("#execution-state").textContent, state);
  }
  if (toggle.attrs["aria-expanded"] !== "true") toggle.dispatch("click");
  harness.document.activeElement = explorerFiles[0];
  get("#explorer-sidebar").dispatch("keydown", { key: "Escape" });
  checkExpanded(false);
  assert.equal(toggle.focused, true, "Closing the sidebar returns keyboard focus to the visible toggle");
  harness.timeline.kill();
}
checkExplorerNavigation();
checkExplorerNavigation(true);

function checkEditorDraftIsolation() {
  const harness = createHarness();
  const { get, editor, tabs } = harness;
  const model = editor.model;
  const original = model.getValue();
  assert.equal(original.includes("<span"), false, "Templates must be plain C++, not handcrafted highlighting HTML");
  assert.equal(model.getLineCount(), require('./algorithms.cjs').bubble.split('\n').length);
  const draft = "// draft\n" + original.replace("a[j] > a[j+1]", "a[j] < a[j+1]");
  const timeline = harness.timeline;
  model.setValue(draft);
  assert.equal(harness.timeline, timeline, "Editing source does not create a new timeline before compilation");
  assert.ok(get("#play-toggle").disabled, "Editing invalidates playback immediately");
  assert.equal(editor.decorations.length, 0);
  assert.match(get("#editor-notice").textContent, /代码已修改/);
  tabs[1].dispatch("click");
  const selection = editor.model;
  assert.ok(highlightedLine(harness));
  selection.setValue("// selection draft\n" + selection.getValue());
  tabs[0].dispatch("click");
  assert.equal(editor.model, model);
  assert.equal(model.getValue(), draft);
  get("#restart").dispatch("click");
  assert.equal(model.getValue(), draft);
  get("#array-input").value = "4, 2, 3, 1";
  get("#array-form").dispatch("submit");
  assert.equal(model.getValue(), draft);
  harness.timeline.seek(harness.timeline.duration(), false);
  assert.equal(get("#status").textContent, "a = [1, 2, 3, 4]", "The renderer installs the collector's returned fixture");
  assert.ok(highlightedLine(harness), "Successfully traced drafts have execution highlighting");
  model.setValue(original);
  assert.equal(editor.decorations.length, 0, "Undo invalidates the current trace too");
  get("#restart").dispatch("click");
  assert.ok(highlightedLine(harness));
  tabs[1].dispatch("click");
  assert.equal(editor.model, selection);
  assert.match(selection.getValue(), /^\/\/ selection draft/);
  harness.timeline.kill();

  const fallback = createHarness(false, false, false, true);
  assert.match(fallback.get("#code").textContent, /void bubble_sort/);
  assert.match(fallback.get("#editor-notice").textContent, /Monaco 未加载/);
  assert.equal(fallback.timeline, undefined, "No template animation fallback if the compiler cannot read a model");
  assert.ok(fallback.get("#play-toggle").disabled);
}
checkEditorDraftIsolation();

function checkStaleCollection() {
  const harness = createHarness(false, false, false, false, null, true);
  const {get, compiler, editor} = harness;
  const first = compiler.requests[0];
  assert.equal(harness.timeline, undefined);
  assert.ok(get('#step').disabled);
  editor.model.setValue('// new draft\n' + editor.model.getValue());
  compiler.deliver(first);
  assert.equal(harness.timeline, undefined, 'Discard results for a source edited during compilation');
  get('#restart').dispatch('click');
  const second = compiler.requests.at(-1);
  compiler.deliver(first);
  assert.equal(harness.timeline, undefined, 'An old result must not replace an active newer collection');
  compiler.deliver(second);
  assert.ok(harness.timeline);
  assert.ok(editor.decorations.length);
  const count = compiler.requests.length;
  get('#restart').dispatch('click');
  assert.equal(compiler.requests.length, count, 'Unchanged restart reuses the completed C++ trace');
  get('#array-input').value = '4, 3, 2, 1';
  get('#array-input').dispatch('input');
  assert.ok(get('#play-toggle').disabled);
  assert.equal(editor.decorations.length, 0);
  compiler.deliver(second);
  assert.ok(get('#play-toggle').disabled, 'Input changes also reject old callbacks');
  get('#array-form').dispatch('submit');
  compiler.deliver(compiler.requests.at(-1), {status:'stopped',message:'已停止。'});
  assert.equal(get('#execution-state').textContent,'无法播放');
  assert.ok(get('#step').disabled);
}
checkStaleCollection();

const expectedCodeLines = { bubble: 14, selection: 14, insertion: 10, merge: 13, quick: 13, heap: 17, shell: 8, comb: 15, counting: 9 };
for (const algorithm of Object.keys(expectedCodeLines)) {
  for (const event of buildExecutionTrace([5, 2, 8, 1, 7, 3, 6, 4], algorithm)) {
    assert.ok(event.line >= 1 && event.line <= expectedCodeLines[algorithm], `${algorithm} highlights missing line ${event.line}`);
  }
}

function exerciseExtendedTimeline(algorithm, input = [5, 2, 8, 1, 7, 3, 6, 4]) {
  const harness = createHarness();
  harness.get("#array-input").value = input.join(", ");
  harness.get("#array-form").dispatch("submit");
  const tab = harness.tabs.find(item => item.dataset.algorithm === algorithm);
  tab.dispatch("click");
  const timeline = harness.timeline;
  const bars = harness.get("#bars").children;
  const slotX = harness.get("#indices").children.map(index => Number(index.attrs.x));
  const pitch = slotX[1] - slotX[0];
  let previousBoxes = null, travelingFrames = 0;
  for (let time = 0; time <= timeline.duration() + .02; time += .02) {
    timeline.time(Math.min(time, timeline.duration()), false);
    const boxes = bars.map(bar => {
      const [x, offsetY] = numbers(bar.attrs.transform);
      const rect = bar.children[0].attrs;
      return { x, y: Number(rect.y) + offsetY, width: Number(rect.width), height: Number(rect.height) };
    });
    if (previousBoxes) boxes.forEach((box, id) => {
      assert.ok(Math.abs(box.x - previousBoxes[id].x) < pitch * .8,
        `${algorithm}: bar ${id} teleports horizontally at ${time.toFixed(2)}s`);
      assert.ok(Math.abs(box.y - previousBoxes[id].y) < 50,
        `${algorithm}: bar ${id} teleports vertically at ${time.toFixed(2)}s`);
    });
    if (boxes.some(box => slotX.every(x => Math.abs(box.x + box.width / 2 - x) > .1))) travelingFrames++;
    previousBoxes = boxes;
    if (algorithm !== "counting") for (let side = 0; side < 2; side++) {
      const [x, y] = numbers(harness.get(`#claw-${side}`).attrs.transform);
      assert.ok(y >= 37, `${algorithm}: gripper must stay below the rail`);
      const gap = -numbers(harness.get(`#claw-${side} .jaw-left`).attrs.d)[2];
      if (Math.abs(gap - (boxes[0].width / 2 + 4)) < .001) {
        assert.ok(boxes.some(box => Math.abs(box.x + box.width / 2 - x) < .001 && Math.abs(box.y - 14 - y) < .001),
          `${algorithm}: closed gripper must follow its carried bar`);
      }
    }
    for (let left = 0; left < boxes.length; left++) for (let right = left + 1; right < boxes.length; right++) {
      const overlapX = Math.min(boxes[left].x + boxes[left].width, boxes[right].x + boxes[right].width) - Math.max(boxes[left].x, boxes[right].x);
      const overlapY = Math.min(boxes[left].y + boxes[left].height, boxes[right].y + boxes[right].height) - Math.max(boxes[left].y, boxes[right].y);
      assert.ok(overlapX < .001 || overlapY < .001, `${algorithm} bars overlap at ${time.toFixed(2)}s`);
    }
  }
  const expected = buildExecutionTrace(input, algorithm).at(-1);
  if (expected.swaps) assert.ok(travelingFrames > 0, `${algorithm}: exchanges must include visible horizontal travel`);
  assert.deepEqual(bars.map(bar => ({ value: Number(bar.dataset.value), x: numbers(bar.attrs.transform)[0] })).sort((a, b) => a.x - b.x).map(bar => bar.value), expected.values);
  assert.equal(harness.get("#comparisons").textContent, algorithm === "counting" ? expected.scans : expected.comparisons);
  assert.equal(harness.get("#swaps").textContent, algorithm === "counting" ? expected.writes : expected.swaps);
  assert.deepEqual(bars.map(bar => Number(bar.children[1].textContent)).sort((a, b) => a - b), expected.values);
  timeline.kill();
}
for (const algorithm of extendedAlgorithms) {
  for (const input of [[5, 2, 8, 1, 7, 3, 6, 4], [3, 1, 3, 2], [5, 4, 3, 2, 1]]) exerciseExtendedTimeline(algorithm, input);
}
// Non-adjacent swaps must clear tall bars between the exchanged values, and a
// swap near the value ceiling must still leave both grippers below the rail.
for (const input of [[2, 99, 99, 99, 1], [99, 98, 98, 99, 98], [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88]]) exerciseExtendedTimeline("selection", input);
for (const algorithm of ["heap", "shell", "comb"]) {
  exerciseExtendedTimeline(algorithm, [2, 99, 99, 99, 1]);
  exerciseExtendedTimeline(algorithm, [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88]);
}

function checkSelectionExchangePause() {
  const input = [2, 99, 99, 99, 1];
  const harness = createHarness();
  harness.get("#array-input").value = input.join(", ");
  harness.get("#array-form").dispatch("submit");
  harness.tabs.find(tab => tab.dataset.algorithm === "selection").dispatch("click");
  const { timeline, get } = harness;
  const travel = timeline.getChildren().find(tween => tween.vars.x !== undefined && tween.targets()[0].height !== undefined);
  const halfway = travel.startTime() + travel.duration() / 2;
  for (let time = 0; time < halfway; time += .02) timeline.time(time, false);
  timeline.time(halfway, false);
  get("#play-toggle").dispatch("click");
  assert.equal(get("#execution-state").textContent, "已暂停");
  const slots = get("#indices").children.map(index => Number(index.attrs.x));
  const carried = get("#bars").children[0];
  const width = Number(carried.children[0].attrs.width);
  const midX = numbers(carried.attrs.transform)[0] + width / 2;
  assert.ok(midX > slots[0] && midX < slots.at(-1), "Pause must preserve an intermediate exchange position");
  advanceLine(harness);
  const afterSwap = buildExecutionTrace(input, "selection").find(event => event.action === "release");
  get("#bars").children.forEach((bar, id) => {
    const [x, y] = numbers(bar.attrs.transform);
    assert.ok(Math.abs(x + width / 2 - slots[afterSwap.order.indexOf(id)]) < .001);
    assert.equal(y, 0);
  });
  assert.equal(get("#execution-state").textContent, "已暂停");
  timeline.kill();
}
checkSelectionExchangePause();

function checkAdjacentExchange(algorithm) {
  const input = [2, 1];
  const harness = createHarness();
  harness.get("#array-input").value = input.join(", ");
  harness.get("#array-form").dispatch("submit");
  harness.tabs.find(tab => tab.dataset.algorithm === algorithm).dispatch("click");
  harness.get("#play-toggle").dispatch("click");
  const trace = buildExecutionTrace(input, algorithm);
  const liftIndex = trace.findIndex(event => event.action === "lift");
  for (let index = 0; index <= liftIndex; index++) advanceLine(harness);
  const [left, right] = trace[liftIndex].pair.map(id => harness.get("#bars").children[id]);
  assert.ok(numbers(left.attrs.transform)[1] < 0, `${algorithm}: adjacent exchange lifts the left bar`);
  assert.equal(numbers(right.attrs.transform)[1], 0, `${algorithm}: adjacent exchange must not lift the right bar`);
  harness.get("#step").dispatch("click");
  for (let time = harness.timeline.time() + .02; time <= harness.timeline.duration(); time += .02) {
    harness.timeline.time(time, false);
    assert.equal(numbers(right.attrs.transform)[1], 0, `${algorithm}: right bar stays on the baseline throughout adjacent exchange`);
    if (harness.get("#execution-state").textContent !== "动作单步") break;
  }
  assert.equal(harness.get("#execution-state").textContent, "已暂停");
  const ordered = harness.get("#bars").children.map(bar => ({ x: numbers(bar.attrs.transform)[0], value: Number(bar.dataset.value) })).sort((a, b) => a.x - b.x);
  assert.deepEqual(ordered.map(bar => bar.value), [1, 2]);
  assert.equal(numbers(left.attrs.transform)[1], 0);
  harness.timeline.kill();
}
for (const algorithm of ["selection", "insertion", "merge", "quick", "heap", "shell", "comb"]) checkAdjacentExchange(algorithm);

for (const algorithm of ["heap", "shell", "comb"]) {
  const input = [4, 1, 3, 2];
  const harness = createHarness();
  harness.get("#array-input").value = input.join(", ");
  harness.get("#array-form").dispatch("submit");
  harness.tabs.find(tab => tab.dataset.algorithm === algorithm).dispatch("click");
  harness.get("#play-toggle").dispatch("click");
  assert.equal(harness.get("#gap-variable").style.display, algorithm === "heap" ? "none" : "");
  assert.equal(harness.get("#heap-variable").style.display, algorithm === "heap" ? "" : "none");
  assert.equal(harness.get("#algorithm-note").hidden, false);
  const events = buildExecutionTrace(input, algorithm);
  for (let index = 0; index < events.length - 1; index++) {
    const event = events[index];
    advanceLine(harness);
    assert.equal(highlightedLine(harness), Math.min(events[index + 1].line, harness.editor.model.getLineCount()), `${algorithm}: step previews the next source site`);
    assert.equal(harness.get("#comparisons").textContent, event.comparisons);
    assert.equal(harness.get("#swaps").textContent, event.swaps);
    assert.equal(harness.get("#var-gap").textContent, event.gap ?? "-");
    assert.equal(harness.get("#var-heap-size").textContent, event.heapSize ?? "-");
    assert.equal(harness.get("#algorithm-note").textContent, event.note);
    assert.equal(harness.get("#status").textContent, `a = [${event.values.join(", ")}]`);
  }
  advanceLine(harness);
  assert.equal(harness.get("#execution-state").textContent, "已完成");
  harness.get("#restart").dispatch("click");
  assert.equal(harness.get("#execution-state").textContent, "已暂停");
  assert.equal(harness.get("#swaps").textContent, 0);
  assert.equal(harness.get("#comparisons").textContent, 0);
  harness.timeline.kill();
}

for (const algorithm of extendedAlgorithms) {
  const harness = createHarness(true);
  harness.tabs.find(item => item.dataset.algorithm === algorithm).dispatch("click");
  const final = buildExecutionTrace([5, 2, 8, 1, 7, 3, 6, 4], algorithm).at(-1);
  const expected = final.values;
  assert.deepEqual(harness.get("#bars").children.map(bar => ({ value: Number(bar.dataset.value), x: numbers(bar.attrs.transform)[0] })).sort((a, b) => a.x - b.x).map(bar => bar.value), expected);
  assert.equal(harness.get("#comparator").opacity, 0);
  if (algorithm === "counting") {
    assert.equal(harness.get("#comparisons").textContent, final.scans);
    assert.equal(harness.get("#swaps").textContent, final.writes);
    const max = Math.max(5, 2, 8, 1, 7, 3, 6, 4);
    assert.deepEqual(harness.get("#bars").children.map(bar => Number(bar.children[0].attrs.height)).sort((a, b) => a - b), final.values.map(value => 28 + (value - 1) * 90 / (max - 1)).sort((a, b) => a - b));
  }
}
gsap.ticker.sleep();
console.log("Passed: parsing, sorting, pause/resume, single-step, restart, custom arrays, input errors, stable IDs, current-slot grip targets, centered slider, jump-free attachment, uncrossed approach, continuous exchanges, separated carrying lanes, mid-exchange pause/step, collision-free grippers, and completion states.");
module.exports = { createHarness };
