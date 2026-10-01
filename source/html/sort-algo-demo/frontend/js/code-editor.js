function createSortCodeEditor(monaco, container, { notice, tabs, storage = null, sources = null, onChange = () => {} }) {
  monaco.editor.defineTheme("sort-blueprint", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "569CD6" },
      { token: "type", foreground: "4EC9B0" },
      { token: "number", foreground: "B5CEA8" },
      { token: "string", foreground: "CE9178" },
      { token: "comment", foreground: "6A9955" }
    ],
    colors: {
      "editor.background": "#1E1E1E",
      "editor.foreground": "#D4D4D4",
      "editorLineNumber.foreground": "#858585",
      "editorLineNumber.activeForeground": "#FFD184",
      "editorCursor.foreground": "#FFD184",
      "editor.selectionBackground": "#264F78",
      "editorGutter.background": "#1E1E1E",
      "editorWidget.background": "#252526",
      "editorIndentGuide.background1": "#404040",
      "editorIndentGuide.activeBackground1": "#707070"
    }
  });
  const editor = monaco.editor.create(container, {
    model: null,
    theme: "sort-blueprint",
    ariaLabel: "可编辑的 C++ 算法代码",
    automaticLayout: true,
    fixedOverflowWidgets: true,
    fontFamily: "Menlo, Consolas, monospace",
    fontSize: 13,
    lineHeight: 24,
    tabSize: 2,
    insertSpaces: true,
    minimap: { enabled: false },
    glyphMargin: true,
    lineNumbersMinChars: 2,
    scrollBeyondLastLine: false,
    renderLineHighlight: "none",
    stickyScroll: { enabled: false },
    padding: { top: 12, bottom: 12 },
    scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
    // Syntax suggestions only; compilation is attached separately, not clangd.
    wordBasedSuggestions: "currentDocument"
  });
  const files = new Map();
  const execution = editor.createDecorationsCollection();
  let active = null;
  let executionLine = 1;
  let executionSource = null;

  function isModified(file) { return file.model.getValue() !== file.source; }
  function refresh(reveal = false) {
    const file = files.get(active);
    if (!file) return;
    const modified = isModified(file);
    const current = executionSource !== null && executionSource === file.model.getValue();
    notice.textContent = current
      ? "动画来自当前 C++ 执行轨迹 · 单步按真实动作事件推进。"
      : modified ? "代码已修改：旧轨迹已失效，请编译并演示。" : "编译并演示执行当前 C++ · 自动记录比较、交换与写入。";
    notice.classList.toggle("modified", modified);
    if (!current || !executionLine) {
      execution.clear();
      return;
    }
    const line = Math.min(executionLine, file.model.getLineCount());
    execution.set([{
      range: new monaco.Range(line, 1, line, 1),
      options: {
        isWholeLine: true,
        className: "execution-line",
        glyphMarginClassName: "execution-arrow"
      }
    }]);
    // Do not move the caret or steal focus while the animation is playing.
    if (reveal && !editor.hasTextFocus()) editor.revealLineInCenterIfOutsideViewport(line, monaco.editor.ScrollType.Immediate);
  }
  function markModified(algorithm, file) {
    const tab = tabs.find(item => item.dataset.algorithm === algorithm);
    const modified = isModified(file);
    tab.setAttribute("data-modified", String(modified));
    tab.setAttribute("aria-label", `${algorithm}_sort.cpp${modified ? "（已修改的草稿）" : ""}`);
  }
  const unsubscribe = sources?.subscribe(record => {
    const file = files.get(record.id);
    if (!file) return;
    file.source = record.diskSource;
    markModified(record.id, file);
    if (active === record.id) refresh();
  });
  return {
    getActiveFile() {
      const file = files.get(active);
      return file ? { algorithm: active, filename: `${active}_sort.cpp`, source: file.model.getValue() } : null;
    },
    showFile(algorithm, source) {
      if (!files.has(algorithm)) {
        let draft = null;
        if (sources) draft = sources.get(algorithm).source;
        else try { draft = storage?.getItem(`sort-algo-demo:draft:v1:${algorithm}`); } catch { /* Storage may be disabled. */ }
        const model = monaco.editor.createModel(draft ?? source, "cpp", monaco.Uri.parse(`inmemory://sort-algo-demo/${algorithm}_sort.cpp`));
        const file = { model, source, viewState: null };
        files.set(algorithm, file);
        model.onDidChangeContent(() => {
          if (file.loading) return;
          markModified(algorithm, file);
          if (sources) sources.edit(algorithm, model.getValue());
          else try {
            const key = `sort-algo-demo:draft:v1:${algorithm}`;
            if (isModified(file)) storage?.setItem(key, model.getValue());
            else storage?.removeItem(key);
          } catch { /* Keep editing in memory if local storage is unavailable. */ }
          if (active === algorithm) { executionSource = null; refresh(); }
          onChange(algorithm);
        });
      }
      if (active !== algorithm) {
        if (active) files.get(active).viewState = editor.saveViewState();
        // Remove decorations before changing models so they cannot leak between files.
        execution.clear();
        active = algorithm;
        const file = files.get(active);
        editor.setModel(file.model);
        if (file.viewState) editor.restoreViewState(file.viewState);
        executionLine = 1;
        executionSource = null;
      }
      markModified(algorithm, files.get(algorithm));
      refresh();
    },
    highlight(line) {
      const changed = line !== executionLine;
      executionLine = line;
      refresh(changed);
    },
    setExecutionSource(source) {
      executionSource = source;
      refresh();
    },
    reloadFile(algorithm, source) {
      const file = files.get(algorithm);
      if (!file) return;
      file.loading = true;
      try { file.source = source; file.model.setValue(source); }
      finally { file.loading = false; }
      markModified(algorithm, file);
      if (active === algorithm) { executionSource = null; refresh(); }
      onChange(algorithm);
    },
    dispose() {
      unsubscribe?.();
      execution.clear();
      editor.dispose();
      files.forEach(file => file.model.dispose());
    }
  };
}

if (typeof module !== "undefined") module.exports = { createSortCodeEditor };
if (typeof window !== "undefined") window.createSortCodeEditor = createSortCodeEditor;
