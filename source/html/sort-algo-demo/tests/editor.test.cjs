const assert = require("node:assert/strict");
const { createSortCodeEditor } = require("../frontend/js/code-editor.js");

// Test our integration contract, not Monaco's own rendering implementation.
function createMonacoStub() {
  const models = new Map();
  const instance = {
    model: null, decorations: [], viewState: null, textFocus: false, revealed: [],
    setModel(model) { this.model = model; },
    saveViewState() { return this.viewState; },
    restoreViewState(state) { this.viewState = state; },
    hasTextFocus() { return this.textFocus; },
    revealLineInCenterIfOutsideViewport(line, scroll) { this.revealed.push([line, scroll]); },
    createDecorationsCollection() {
      return {
        clear: () => { this.decorations = []; },
        set: decorations => { this.decorations = decorations; }
      };
    },
    dispose() { this.disposed = true; }
  };
  return {
    models, instance,
    Uri: { parse: value => value },
    Range: class Range {
      constructor(startLineNumber, startColumn, endLineNumber, endColumn) {
        Object.assign(this, { startLineNumber, startColumn, endLineNumber, endColumn });
      }
    },
    editor: {
      ScrollType: { Immediate: 1 },
      defineTheme(name, theme) { instance.theme = { name, theme }; },
      create(container, options) { instance.options = options; return instance; },
      createModel(value, language, uri) {
        const listeners = new Set();
        const model = {
          value, language, uri,
          getValue() { return this.value; },
          setValue(next) { this.value = next; listeners.forEach(callback => callback()); },
          getLineCount() { return this.value.split("\n").length; },
          onDidChangeContent(callback) { listeners.add(callback); return { dispose: () => listeners.delete(callback) }; },
          dispose() { this.disposed = true; listeners.clear(); }
        };
        models.set(uri, model);
        return model;
      }
    }
  };
}
module.exports = { createMonacoStub };

if (require.main === module) {
  const notice = { textContent: "", classList: { toggle() {} } };
  const tabs = ["bubble", "selection"].map(algorithm => ({
    dataset: { algorithm }, attrs: {},
    setAttribute(key, value) { this.attrs[key] = value; }
  }));
  const drafts = new Map();
  const storage = {
    getItem: key => drafts.get(key) ?? null,
    setItem: (key, value) => drafts.set(key, value),
    removeItem: key => drafts.delete(key)
  };
  const source = "void bubble_sort() {\n  // original\n}";
  const monaco = createMonacoStub();
  const changes = [];
  const bridge = createSortCodeEditor(monaco, {}, { notice, tabs, storage, onChange: algorithm => changes.push(algorithm) });
  bridge.showFile("bubble", source);
  const model = monaco.instance.model;
  assert.equal(model.language, "cpp");
  assert.equal(model.uri, "inmemory://sort-algo-demo/bubble_sort.cpp");
  assert.deepEqual(bridge.getActiveFile(), { algorithm: 'bubble', filename: 'bubble_sort.cpp', source });
  assert.equal(monaco.instance.options.automaticLayout, true);
  assert.equal(monaco.instance.options.model, null);
  bridge.highlight(2);
  assert.equal(monaco.instance.decorations.length, 0, "No execution position before successful tracing");
  bridge.setExecutionSource(source);
  bridge.highlight(2);
  assert.equal(monaco.instance.decorations[0].range.startLineNumber, 2);
  assert.equal(monaco.instance.decorations[0].options.isWholeLine, true);

  monaco.instance.viewState = { selection: "2:4", scrollTop: 28 };
  monaco.instance.textFocus = true;
  const reveals = monaco.instance.revealed.length;
  bridge.highlight(3);
  assert.equal(monaco.instance.revealed.length, reveals, "Animation must not scroll an editor being typed in");
  const draft = "// added a line\n" + source.replace("original", "edited");
  model.setValue(draft);
  assert.equal(bridge.getActiveFile().source, draft, 'Compiler reads the edited model, not the original template');
  assert.equal(tabs[0].attrs["data-modified"], "true");
  assert.match(notice.textContent, /轨迹已失效/);
  assert.deepEqual(changes, ["bubble"]);
  assert.equal(monaco.instance.decorations.length, 0);
  bridge.highlight(2);
  assert.equal(monaco.instance.decorations.length, 0, "Modified source must never show a built-in execution location");
  assert.equal(drafts.get("sort-algo-demo:draft:v1:bubble"), draft);

  bridge.showFile("selection", "void selection_sort() {} ");
  assert.equal(monaco.instance.decorations.length, 0);
  bridge.setExecutionSource("void selection_sort() {} ");
  assert.equal(monaco.instance.decorations[0].range.startLineNumber, 1);
  bridge.showFile("bubble", source);
  assert.equal(monaco.instance.model, model, "Switching must reuse the original model and its undo stack");
  assert.equal(model.getValue(), draft);
  assert.deepEqual(monaco.instance.viewState, { selection: "2:4", scrollTop: 28 });
  bridge.showFile("bubble", source);
  assert.equal(model.getValue(), draft, "Restarting an animation cannot reset code edits");
  bridge.setExecutionSource(draft);
  bridge.highlight(2);
  assert.equal(monaco.instance.decorations[0].range.startLineNumber, 2, "A successfully traced draft has real execution locations");
  bridge.setExecutionSource(null);

  const restored = createMonacoStub();
  const second = createSortCodeEditor(restored, {}, { notice, tabs, storage });
  second.showFile("bubble", source);
  assert.equal(restored.instance.model.getValue(), draft);
  assert.equal(restored.instance.decorations.length, 0);
  model.setValue(source); // Equivalent to undoing all edits in the real editor.
  bridge.highlight(2);
  assert.equal(tabs[0].attrs["data-modified"], "false");
  assert.equal(monaco.instance.decorations.length, 0, "Undo also invalidates the old execution");
  bridge.setExecutionSource(source);
  bridge.highlight(2);
  assert.equal(monaco.instance.decorations[0].range.startLineNumber, 2);
  assert.equal(drafts.has("sort-algo-demo:draft:v1:bubble"), false);
  model.setValue("");
  assert.equal(drafts.get("sort-algo-demo:draft:v1:bubble"), "");
  bridge.dispose();
  second.dispose();
  assert.ok([...monaco.models.values()].every(file => file.disposed));

  const blockedStorage = {
    getItem() { throw new Error("blocked"); },
    setItem() { throw new Error("full"); },
    removeItem() { throw new Error("blocked"); }
  };
  const offline = createMonacoStub();
  const third = createSortCodeEditor(offline, {}, { notice, tabs, storage: blockedStorage });
  third.showFile("bubble", source);
  offline.instance.model.setValue("changed");
  third.showFile("selection", "selection");
  third.showFile("bubble", source);
  assert.equal(offline.instance.model.getValue(), "changed");
  third.dispose();
  console.log("Passed: Monaco model reuse, view states, execution decorations, edit isolation, draft persistence, undo-to-original behavior, and unavailable storage.");
}
