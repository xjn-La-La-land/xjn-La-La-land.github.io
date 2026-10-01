// Source analysis for the pinned Clang 8 WASM toolchain. No algorithm simulation.
(function (exports) {
  const unwrap = n => n && ['ImplicitCastExpr', 'ParenExpr'].includes(n.kind) && n.children.length === 1 ? unwrap(n.children[0]) : n;
  function parseTraceAst(dump, source, name, filename) {
    const sections = dump.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').split(/^Dumping /m).filter(s => s.startsWith(name + ':\n'));
    if (sections.length !== 1) throw new Error(`无法确定 ${name} 的函数定义；暂不支持多个重载、宏生成入口或其他命名空间入口。`);
    const lines = source.split('\n'), starts = []; let offset = 0, currentLine = 1;
    for (const line of lines) { starts.push(offset); offset += line.length + 1; }
    function position(token) {
      const explicit = token.match(/^(.*):(\d+):(\d+)$/), columnOnly = token.match(/^col:(\d+)$/);
      let column;
      if (explicit) {
        if (explicit[1] !== 'line' && explicit[1] !== filename) throw new Error('AST 包含不支持的宏展开或其他文件位置：' + token);
        currentLine = Number(explicit[2]); column = Number(explicit[3]);
      } else if (columnOnly) column = Number(columnOnly[1]);
      else throw new Error('无法映射 AST 位置：' + token);
      if (currentLine < 1 || currentLine > lines.length) throw new Error('AST 行号超出原始源码。');
      const bytes = new TextEncoder().encode(lines[currentLine - 1]);
      if (column < 1 || column > bytes.length + 1) throw new Error('AST 列号超出原始源码。');
      return { line: currentLine, column, offset: starts[currentLine - 1] + new TextDecoder('utf-8', { fatal: true }).decode(bytes.slice(0, column - 1)).length };
    }
    const root = { kind: 'Root', children: [] }, stack = [root];
    for (const line of sections[0].slice(name.length + 2).split('\n')) {
      const m = line.match(/^([ |`-]*)([A-Za-z][A-Za-z0-9_]*)\s+(0x[0-9a-f]+)\s+<([^>]+)>(.*)$/);
      if (!m) continue;
      const [, prefix, kind, id, range, detail] = m;
      if (range.includes('invalid sloc')) continue; // Implicit zero-initialization has no source edit.
      const points = range.split(', '), begin = position(points[0]), endPoint = points.length === 1 ? begin : position(points[1]);
      const extra = detail.match(/^\s+((?:line:\d+:|col:)\d+)/); if (extra) position(extra[1]);
      const token = source.slice(endPoint.offset).match(/^(?:[A-Za-z_][A-Za-z_0-9]*|\d+|\+\+|--|<=|>=|==|!=|&&|\|\||.)/s);
      if (!token) throw new Error('AST 结束位置没有源码 token。');
      const depth = prefix.length / 2; stack.length = depth + 1;
      if (!Number.isInteger(depth) || !stack[depth]) throw new Error('工具链 AST 格式不受支持：' + line.slice(0, 180));
      const n = { kind, id, detail, begin, end: endPoint.offset + token[0].length, children: [], parent: stack[depth] };
      n.op = detail.match(/'([^']+)'\s*$/)?.[1];
      n.ref = detail.match(/\b(?:Var|ParmVar|Function) (0x[0-9a-f]+) '([^']+)'/);
      n.type = detail.match(/'([^']+)'/)?.[1];
      n.name = detail.match(/^\s*(?:(?:line:\d+:|col:)\d+\s+)?(?:(?:used|referenced|implicit|constexpr)\s+)*([A-Za-z_]\w*)\s+'/)?.[1];
      stack[depth].children.push(n); stack[depth + 1] = n;
    }
    if (root.children.length !== 1 || root.children[0].kind !== 'FunctionDecl') throw new Error('缺少可分析的函数 AST。');
    return root.children[0];
  }

  function instrumentTraceSource({ source, ast: dump, name, filename }) {
    if (/^\s*#\s*(?:define|line|if|pragma|undef)/m.test(source)) throw new Error('动画模式暂不支持宏或自定义行号指令；普通编译运行仍可使用。');
    const ast = parseTraceAst(dump, source, name, filename), nodes = [], seen = new Set();
    (function collect(n) { if (!seen.has(n.id)) { seen.add(n.id); nodes.push(n); } for (const c of n.children) collect(c); })(ast);
    const mainParam = ast.children.find(n => n.kind === 'ParmVarDecl' && n.type === 'int *');
    if (!mainParam) throw new Error('动画入口需要一个 int a[] / int* 数组参数。');
    const text = n => source.slice(n.begin.offset, n.end);
    const fail = (n, message) => { throw new Error(`${filename}:${n.begin.line}:${n.begin.column}: ${message}`); };
    const mainRef = n => unwrap(n)?.ref?.[1] === mainParam.id;
    const containsMain = n => mainRef(n) || !!n?.children.some(containsMain);
    const array = n => unwrap(n)?.kind === 'ArraySubscriptExpr' ? unwrap(n) : null;
    const primary = n => !!array(n) && mainRef(array(n).children[0]) && array(n).type === 'int';
    const auxDecls = new Map(nodes.filter(n => n.kind === 'VarDecl' && /^int \[\d+\]$/.test(n.type || '')).map(n => [n.id, n]));
    const auxiliary = n => !!array(n) && auxDecls.has(unwrap(array(n).children[0])?.ref?.[1]);
    const effect = n => !!n && (['CallExpr', 'CXXOperatorCallExpr'].includes(n.kind) || n.kind === 'UnaryOperator' && ['++', '--'].includes(n.op) || n.kind === 'BinaryOperator' && n.op === '=' || n.children.some(effect));
    // A replaced parent expression must not hide an extra array action or a
    // helper call inside its index. Plain i++ swap/write indices remain valid.
    for (const n of nodes.filter(primary)) {
      const index = array(n).children[1];
      const unsafeIndex = x => ['ArraySubscriptExpr', 'CallExpr', 'CXXOperatorCallExpr'].includes(x.kind) || x.children.some(unsafeIndex);
      if (unsafeIndex(index)) fail(n, '数组下标中的嵌套数组或函数调用暂不支持完整追踪。');
    }
    const keys = new Map(); for (const n of nodes.filter(n => ['VarDecl', 'ParmVarDecl'].includes(n.kind))) keys.set(n.id, keys.size + 1);
    const scopes = new Set(['CompoundStmt', 'ForStmt', 'IfStmt', 'WhileStmt', 'FunctionDecl', 'CXXMethodDecl']);
    const declarations = nodes.filter(n => ['VarDecl', 'ParmVarDecl'].includes(n.kind) && n.name && /^(?:const )?(?:int|bool|size_t|unsigned long|unsigned int|long)$/.test(n.type || ''));
    function variables(at) {
      const visible = new Map();
      for (const d of declarations) {
        if (d.kind === 'VarDecl' && (!/\b(?:cinit|listinit)\b/.test(d.detail) || d.end > at.begin.offset)) continue;
        let scope = d.parent; while (scope && !scopes.has(scope.kind)) scope = scope.parent;
        if (!scope?.begin || scope.begin.offset > at.begin.offset || scope.end < at.end) continue;
        const previous = visible.get(d.name);
        if (!previous || scope.begin.offset > previous.scope.begin.offset || d.begin.offset > previous.d.begin.offset) visible.set(d.name, { d, scope });
      }
      return '{' + [...visible].slice(0, 32).map(([key]) => `{"${key}", static_cast<long long>(${key})}`).join(',') + '}';
    }
    const edits = [], sites = [], handled = new Set();
    function mark(n) { handled.add(n.id); for (const c of n.children) mark(c); }
    function add(n, expression, kind) {
      // Snapshot caller locals before evaluating side-effecting operands.
      const replacement = `([&]() -> decltype(auto) { auto sort_trace_vars = std::initializer_list<sort_trace::Variable>${variables(n)}; return ${expression}; }())`;
      edits.push({ start: n.begin.offset, end: n.end, replacement: replacement + '\n'.repeat((text(n).match(/\n/g) || []).length) });
      sites.push({ kind, line: n.begin.line, column: n.begin.column, original: text(n) }); mark(n);
    }
    const location = n => `${n.begin.line}, ${n.begin.column}, sort_trace_vars`;
    function observed(n) {
      n = unwrap(n);
      if (primary(n)) return `sort_trace::element(${text(n)})`;
      if (n.kind === 'DeclRefExpr' && n.type === 'int') return `sort_trace::scalar(${text(n)}, ${keys.get(n.ref?.[1]) || 0})`;
      if (n.kind === 'IntegerLiteral' && n.type === 'int') return `sort_trace::scalar(${text(n)}, 0)`;
      fail(n, '该比较/写入表达式需要可追踪的 int 数组元素、整型变量或常量。');
    }
    // Function/lambda scopes make saved-variable provenance recursion-safe.
    const guarded = new Set();
    for (const n of nodes) if (['FunctionDecl', 'CXXMethodDecl'].includes(n.kind)) {
      const body = n.children.find(c => c.kind === 'CompoundStmt');
      if (body && !guarded.has(body.begin.offset)) { guarded.add(body.begin.offset); edits.push({ start: body.begin.offset + 1, end: body.begin.offset + 1, replacement: ' sort_trace::Frame sort_trace_frame; ' }); }
    }
    // Preserve the three individual C++ statements of a proven manual exchange.
    for (const block of nodes.filter(n => n.kind === 'CompoundStmt')) {
      const kids = block.children;
      for (let i = 0; i + 2 < kids.length; i++) {
        const [decl, first, second] = kids.slice(i, i + 3), v = decl.kind === 'DeclStmt' && decl.children.length === 1 ? decl.children[0] : null;
        const initial = unwrap(v?.children[0]), a = unwrap(first.children[0]), b = unwrap(first.children[1]), c = unwrap(second.children[0]), d = unwrap(second.children[1]);
        if (v?.kind !== 'VarDecl' || !primary(initial) || first.kind !== 'BinaryOperator' || first.op !== '=' || second.kind !== 'BinaryOperator' || second.op !== '=' || !primary(a) || !primary(b) || !primary(c) || d?.ref?.[1] !== v.id) continue;
        if (text(initial).replace(/\s/g, '') !== text(a).replace(/\s/g, '') || text(b).replace(/\s/g, '') !== text(c).replace(/\s/g, '')) continue;
        if ([initial, a, b, c].some(effect)) fail(first, '三语句交换的下标不能带副作用，请使用 std::swap。');
        add(initial, `sort_trace::save(${text(initial)}, ${keys.get(v.id)}, ${location(initial)})`, 'save');
        add(first, `sort_trace::shift(${text(a)}, ${text(b)}, ${keys.get(v.id)}, ${location(first)})`, 'shift');
        add(second, `sort_trace::drop(${text(c)}, ${text(d)}, ${keys.get(v.id)}, ${location(second)})`, 'drop'); i += 2;
      }
    }
    for (const n of nodes) {
      if (handled.has(n.id)) continue;
      const left = unwrap(n.children[0]), right = unwrap(n.children[1]);
      if (n.kind === 'BinaryOperator' && ['>', '<', '>=', '<=', '==', '!='].includes(n.op) && (primary(left) || primary(right))) {
        if (effect(left) || effect(right)) fail(n, '比较操作数中暂不支持副作用。');
        add(n, `sort_trace::compare(${observed(left)}, ${observed(right)}, "${n.op}", ${location(n)})`, 'compare'); continue;
      }
      if (n.kind === 'CallExpr') {
        const callee = unwrap(n.children[0]), args = n.children.slice(1).map(unwrap), spelling = callee ? text(callee).replace(/\s/g, '') : '';
        if (spelling === 'std::swap' && args.length === 2 && args.every(primary)) {
          add(n, `sort_trace::exchange(${args.map(text).join(', ')}, ${location(n)})`, 'swap'); continue;
        }
        if (spelling === 'std::rotate' && args.length === 3 && args.every(a => a.type === 'int *' && containsMain(a))) {
          if (args.some(effect)) fail(n, 'rotate 的边界不支持副作用。');
          add(n, `sort_trace::rotate(${args.map(text).join(', ')}, ${location(n)})`, 'rotate'); continue;
        }
      }
      if (n.kind === 'UnaryOperator' && ['++', '--'].includes(n.op) && auxiliary(left)) {
        const a = array(left), base = unwrap(a.children[0]), index = unwrap(a.children[1]);
        const value = primary(index) ? `sort_trace::scan(${text(index)}, ${location(n)})` : text(index);
        add(n, `sort_trace::bucket(${text(base)}, ${value}, ${n.op === '++' ? 1 : -1}, ${/postfix/.test(n.detail)}, ${location(n)})`, 'bucket'); continue;
      }
      if (n.kind === 'BinaryOperator' && n.op === '=' && primary(left)) {
        if (effect(right)) fail(n, '数组写入右侧暂不支持副作用或函数调用。');
        // C++17 assignment evaluates RHS before LHS. Keep that order explicitly.
        const expr = `([&]() -> int& { auto sort_trace_value = ${observed(right)}; auto &sort_trace_target = (${text(left)}); return sort_trace::write(sort_trace_target, sort_trace_value.value, ${location(n)}); }())`;
        add(n, expr, 'write'); continue;
      }
      if (n.kind === 'VarDecl' && primary(unwrap(n.children[0])) && !handled.has(unwrap(n.children[0]).id)) {
        const init = unwrap(n.children[0]);
        add(init, `sort_trace::read(${text(init)}, ${keys.get(n.id)}, ${location(init)})`, 'read'); continue;
      }
      if (n.kind === 'ArraySubscriptExpr' && primary(n)) {
        let parent = n.parent; while (parent && ['ImplicitCastExpr', 'ParenExpr'].includes(parent.kind)) parent = parent.parent;
        if (parent?.kind === 'CompoundAssignOperator' || parent?.kind === 'BinaryOperator' && /=$/.test(parent.op || '') || parent?.kind === 'UnaryOperator' && ['++', '--', '&'].includes(parent.op)) fail(n, '该数组左值操作未被完整追踪。');
        add(n, `sort_trace::read(${text(n)}, 0, ${location(n)})`, 'read');
      }
    }
    // Refuse pointer escapes, aliases, unhandled array writes and unknown helpers.
    for (const n of nodes) {
      if (handled.has(n.id)) continue;
      if (n.kind === 'DeclRefExpr' && n.ref?.[1] === mainParam.id) {
        let p = n.parent; while (p && ['ImplicitCastExpr', 'ParenExpr', 'BinaryOperator'].includes(p.kind)) p = p.parent;
        if (p?.kind === 'LambdaExpr' || p?.kind === 'CallExpr' && unwrap(p.children[0])?.ref?.[1] === ast.id) continue;
        fail(n, '数组指针逃逸、别名或未知库操作无法保证完整 trace。');
      }
      if (n.kind === 'ArraySubscriptExpr') fail(n, '该数组操作尚未支持追踪。');
    }
    edits.sort((a, b) => b.start - a.start || b.end - a.end);
    let result = source, boundary = source.length;
    for (const e of edits) { if (e.end > boundary) throw new Error('源码记录点重叠，拒绝生成不完整 trace。'); result = result.slice(0, e.start) + e.replacement + result.slice(e.end); boundary = e.start; }
    return { source: result, sites: sites.sort((a, b) => a.line - b.line || a.column - b.column) };
  }
  Object.assign(exports, { parseTraceAst, instrumentTraceSource });
  if (typeof window !== 'undefined') window.SortTraceSource = exports;
  else if (typeof self !== 'undefined') self.SortTraceSource = exports;
  if (typeof module === 'object' && module && module.exports) module.exports = exports;
})({});
