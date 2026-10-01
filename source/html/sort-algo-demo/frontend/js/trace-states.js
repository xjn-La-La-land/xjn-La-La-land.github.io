// Conservative semantic contracts for the nine reviewed C++ implementations.
// Recognition uses AST declaration bindings, never variable spelling or final values.
(function(exports) {
  const CONTRACTS = {
  "bubble": "void bubble_sort ( int v0 [ ] , size_t v1 ) { for ( size_t v2 = 0 ; v2 + 1 < v1 ; ++ v2 ) { bool v3 = false ; for ( size_t v4 = 0 ; v4 + 1 < v1 - v2 ; ++ v4 ) { if ( v0 [ v4 ] > v0 [ v4 + 1 ] ) { std :: swap ( v0 [ v4 ] , v0 [ v4 + 1 ] ) ; v3 = true ; } } if ( ! v3 ) break ; } }",
  "selection": "void selection_sort ( int v0 [ ] , size_t v1 ) { for ( size_t v2 = 0 ; v2 + 1 < v1 ; ++ v2 ) { size_t v3 = v2 ; for ( size_t v4 = v2 + 1 ; v4 < v1 ; ++ v4 ) { if ( v0 [ v4 ] < v0 [ v3 ] ) { v3 = v4 ; } } if ( v3 == v2 ) continue ; std :: swap ( v0 [ v2 ] , v0 [ v3 ] ) ; } }",
  "insertion": "void insertion_sort ( int v0 [ ] , size_t v1 ) { for ( size_t v2 = 1 ; v2 < v1 ; ++ v2 ) { size_t v3 = v2 ; while ( v3 > 0 && v0 [ v3 - 1 ] > v0 [ v3 ] ) { std :: swap ( v0 [ v3 - 1 ] , v0 [ v3 ] ) ; -- v3 ; } } }",
  "merge": "void merge_sort ( int v0 [ ] , size_t v1 ) { for ( size_t v2 = 1 ; v2 < v1 ; v2 * = 2 ) { for ( size_t v3 = 0 ; v3 < v1 ; v3 + = 2 * v2 ) { size_t v4 = std :: min ( v3 + v2 , v1 ) ; size_t v5 = std :: min ( v3 + 2 * v2 , v1 ) ; size_t v6 = v3 , v7 = v4 ; while ( v6 < v4 && v7 < v5 ) { if ( v0 [ v6 ] <= v0 [ v7 ] ) ++ v6 ; else { std :: rotate ( v0 + v6 , v0 + v7 , v0 + v7 + 1 ) ; ++ v6 ; ++ v7 ; ++ v4 ; } } } } }",
  "quick": "void quick_sort ( int v0 [ ] , int v1 , int v2 ) { if ( v1 >= v2 ) return ; int v3 = v0 [ v2 ] ; int v4 = v1 ; for ( int v5 = v1 ; v5 < v2 ; ++ v5 ) { if ( v0 [ v5 ] < v3 ) { std :: swap ( v0 [ v4 ++ ] , v0 [ v5 ] ) ; } } std :: swap ( v0 [ v4 ] , v0 [ v2 ] ) ; quick_sort ( v0 , v1 , v4 - 1 ) ; quick_sort ( v0 , v4 + 1 , v2 ) ; }",
  "heap": "void heap_sort ( int v0 [ ] , int v1 ) { auto v2 = [ v0 ] ( int v3 , int v4 ) { while ( 2 * v3 + 1 < v4 ) { int v5 = 2 * v3 + 1 ; if ( v5 + 1 < v4 && v0 [ v5 ] < v0 [ v5 + 1 ] ) ++ v5 ; if ( v0 [ v3 ] >= v0 [ v5 ] ) break ; std :: swap ( v0 [ v3 ] , v0 [ v5 ] ) ; v3 = v5 ; } } ; for ( int v6 = v1 / 2 - 1 ; v6 >= 0 ; -- v6 ) v2 ( v6 , v1 ) ; for ( int v7 = v1 - 1 ; v7 > 0 ; -- v7 ) { std :: swap ( v0 [ 0 ] , v0 [ v7 ] ) ; v2 ( 0 , v7 ) ; } }",
  "shell": "void shell_sort ( int v0 [ ] , int v1 ) { for ( int v2 = v1 / 2 ; v2 > 0 ; v2 / = 2 ) { for ( int v3 = v2 ; v3 < v1 ; ++ v3 ) { for ( int v4 = v3 ; v4 >= v2 && v0 [ v4 - v2 ] > v0 [ v4 ] ; v4 - = v2 ) std :: swap ( v0 [ v4 - v2 ] , v0 [ v4 ] ) ; } } }",
  "comb": "void comb_sort ( int v0 [ ] , int v1 ) { int v2 = v1 ; bool v3 = true ; while ( v2 > 1 || v3 ) { v2 = std :: max ( 1 , v2 * 10 / 13 ) ; v3 = false ; for ( int v4 = 0 ; v4 + v2 < v1 ; ++ v4 ) { int v5 = v4 + v2 ; if ( v0 [ v4 ] > v0 [ v5 ] ) { std :: swap ( v0 [ v4 ] , v0 [ v5 ] ) ; v3 = true ; } } } }",
  "counting": "void counting_sort ( int v0 [ ] , size_t v1 ) { int v2 [ 100 ] = { } ; for ( size_t v3 = 0 ; v3 < v1 ; ++ v3 ) ++ v2 [ v0 [ v3 ] ] ; size_t v4 = 0 ; for ( int v5 = 1 ; v5 < 100 ; ++ v5 ) while ( v2 [ v5 ] -- > 0 ) v0 [ v4 ++ ] = v5 ; }"
};
  function context(ast, source) {
    const nodes=[], seen=new Set();
    (function walk(n){if(seen.has(n.id))return;seen.add(n.id);nodes.push(n);n.children.forEach(walk);})(ast);
    const declarations=nodes.filter(n=>['VarDecl','ParmVarDecl'].includes(n.kind));
    const keys=new Map(declarations.map((n,i)=>[n.id,`v${i}`]));
    const tokens=[...source.matchAll(/\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_]\w*|\d+|::|\+\+|--|<=|>=|==|!=|&&|\|\||[^\s]/g)].filter(m=>!m[0].startsWith('//')&&!m[0].startsWith('/*'));
    const bindings=new Map();
    for(const d of declarations){const token=tokens.find(t=>t.index>=d.begin.offset&&t.index<d.end&&t[0]===d.name);if(token)bindings.set(token.index,keys.get(d.id));}
    for(const n of nodes)if(n.kind==='DeclRefExpr'&&keys.has(n.ref?.[1]))bindings.set(n.begin.offset,keys.get(n.ref[1]));
    const signature=tokens.map(t=>bindings.get(t.index)||t[0]).join(' ');
    return {nodes,declarations,signature};
  }
  function plan(ast,source,name) {
    const algorithm=name.replace(/_sort$/,''),c=context(ast,source);
    if(c.signature!==CONTRACTS[algorithm])return {recognized:false,edits:[]};
    const nodes=c.nodes,vars=c.declarations.map(d=>d.name),edits=[];
    const roles={bubble:'a n i swapped j',selection:'a n i minIndex j',insertion:'a n i j',merge:'a n width left mid right i j',quick:'a lo hi pivot store j',heap:'a n sift i heapSize j i heapSize',shell:'a n gap i j',comb:'a n gap swapped i j',counting:'a n count i out value'}[algorithm].split(' ');
    const aliases=Object.fromEntries(vars.map((name,i)=>[name,roles[i]]));
    const body=ast.children.find(n=>n.kind==='CompoundStmt');
    const loops=nodes.filter(n=>n.kind==='ForStmt'), whiles=nodes.filter(n=>n.kind==='WhileStmt');
    const blocks=loops.map(n=>n.children.at(-1));
    const calls=nodes.filter(n=>n.kind==='CallExpr'&&source.slice(n.begin.offset,n.end).startsWith('std::swap'));
    const stmtEnd=n=>n.end+(source.slice(n.end).match(/^\s*;/)?.[0].length||0);
    const insert=(at,n,phase,values)=>{
      const locals={...values};
      for(const expression of Object.values(values))if(vars.includes(expression)&&!Object.hasOwn(locals,aliases[expression]))locals[aliases[expression]]=expression;
      const fields=Object.entries(locals).map(([k,v])=>`{"${k}",static_cast<long long>(${v})}`).join(',');
      const lines=source.slice(0,at).split('\n'),line=lines.length,column=new TextEncoder().encode(lines.at(-1)).length+1;
      edits.push({start:at,end:at,replacement:` sort_trace::phase("${algorithm}.${phase}",${line},${column},{${fields}}); `});
    };
    const before=(n,p,v)=>insert(n.begin.offset,n,p,v),after=(n,p,v)=>insert(stmtEnd(n),n,p,v);
    const enter=(b,p,v)=>insert(b.begin.offset+1,b,p,v),leave=(b,p,v)=>insert(b.end-1,b,p,v);
    const wrap=(n,p,v)=>{
      edits.push({start:n.begin.offset,end:n.begin.offset,replacement:'{'});
      after(n,p,v);
      // Insert closing brace after the phase at the same offset. Stable edit order
      // makes the end text "phase; }", without changing the original statement.
      edits.push({start:stmtEnd(n),end:stmtEnd(n),replacement:'}'});
    };
    const size=vars[1];
    enter(body,'start',algorithm==='quick'?{lo:vars[1],hi:vars[2]}:{n:size});
    if(algorithm==='bubble') {
      const v={pass:vars[2],end:`${size}-${vars[2]}`,swapped:vars[3]};
      before(loops[1],'begin',v);after(loops[1],'end',v);
    } else if(algorithm==='selection') {
      const v={index:vars[2],candidate:vars[3],n:size};
      before(loops[1],'begin',v);after(loops[1],'scan-end',v);
      const decision=blocks[1].children.at(-1);after(decision,'candidate',{...v,cursor:vars[4]});
      // Commit even when the selected minimum is already at the left boundary.
      before(blocks[0].children.find(n=>n.kind==='IfStmt'),'end',v);
      after(calls.at(-1),'placed',v);
    } else if(algorithm==='insertion') {
      const v={index:vars[2],cursor:vars[3]};
      before(whiles[0],'begin',v);leave(blocks[0],'end',v);
    } else if(algorithm==='merge') {
      const v={width:vars[2],left:vars[3],mid:vars[4],right:vars[5],cursor:vars[6],other:vars[7]};
      before(whiles[0],'begin',v);after(whiles[0].children.at(-1).children.at(-1),'progress',v);leave(blocks[1],'end',v);
    } else if(algorithm==='quick') {
      const v={lo:vars[1],hi:vars[2],pivot:vars[3],store:vars[4]};
      before(loops[0],'begin',v);enter(blocks[0],'scan',{...v,cursor:vars[5]});
      after(calls.at(-1),'placed',v);leave(body,'end',v);
      const branch=body.children.find(n=>n.kind==='IfStmt'),ret=branch.children.at(-1);
      // Base cases have no pivot declaration, and must still mark singleton slots.
      edits.push({start:ret.begin.offset,end:ret.begin.offset,replacement:'{'});
      before(ret,'base',{lo:vars[1],hi:vars[2]});
      edits.push({start:stmtEnd(ret),end:stmtEnd(ret),replacement:'}'});
    } else if(algorithm==='heap') {
      const lambdaBody=nodes.find(n=>n.kind==='CXXMethodDecl')?.children.find(n=>n.kind==='CompoundStmt');
      enter(lambdaBody,'sift',{root:vars[3],end:vars[4]});
      before(whiles[0].children.at(-1).children.find(n=>n.kind==='IfStmt'),'sift',{root:vars[3],end:vars[4],child:vars[5]});
      after(calls.at(-1),'extract',{end:vars[7]});
    } else if(algorithm==='shell') {
      enter(blocks[0],'gap',{gap:vars[2],n:size});enter(blocks[1],'group',{gap:vars[2],index:vars[3],n:size});
    } else if(algorithm==='comb') {
      before(loops[0],'gap',{gap:vars[2],n:size});
    } else if(algorithm==='counting') {
      const assignment=nodes.find(n=>n.kind==='BinaryOperator'&&n.op==='=');
      wrap(assignment,'write',{out:vars[4],value:vars[5]});
    }
    // Merge insertions at equal offsets before applying source edits.
    const merged=new Map();for(const e of edits)merged.set(e.start,(merged.get(e.start)||'')+e.replacement);
    return {recognized:true,aliases,edits:[...merged].map(([start,replacement])=>({start,end:start,replacement}))};
  }
  function validatePhase(e,n) {
    const schema={bubble:{start:'n',begin:'pass end swapped',end:'pass end swapped'},selection:{start:'n',begin:'index candidate n','scan-end':'index candidate n',candidate:'index candidate cursor n',end:'index candidate n',placed:'index candidate n'},insertion:{start:'n',begin:'index cursor',end:'index cursor'},merge:{start:'n',begin:'width left mid right cursor other',progress:'width left mid right cursor other',end:'width left mid right cursor other'},quick:{start:'lo hi',begin:'lo hi pivot store',scan:'lo hi pivot store cursor',placed:'lo hi pivot store',base:'lo hi',end:'lo hi pivot store'},heap:{start:'n',sift:'root end',extract:'end'},shell:{start:'n',gap:'gap n',group:'gap index n'},comb:{start:'n',gap:'gap n'},counting:{start:'n',write:'out value'}};
    if(typeof e.phase!=='string')return false;
    const [algorithm,phase,...extra]=e.phase.split('.'),fields=schema[algorithm]?.[phase];
    if(!fields||extra.length||!fields.split(' ').every(key=>Number.isSafeInteger(e.variables[key])))return false;
    const v=e.variables;
    if(v.n!=null&&v.n!==n)return false;
    for(const key of ['pass','index','candidate','cursor','other','root','child','end','left','mid','right','store','lo','out','gap','width'])if(v[key]!=null&&(v[key]<0||v[key]>n))return false;
    if(v.hi!=null&&(v.hi < -1||v.hi>=n))return false;
    if(v.swapped!=null&&![0,1].includes(v.swapped))return false;
    if(v.gap!=null&&v.gap<1)return false;
    return true;
  }
  function createTracker(initial) {
    let state=null;const n=initial.length;
    const fresh=()=>({algorithm:null,fixed:[],ranges:[],candidate:null,pivot:null,gap:null,group:null,buckets:{},bucket:null,out:null,label:''});
    const range=(start,end,kind)=>({start:Math.max(0,start),end:Math.min(n,end),kind});
    const fix=(start,end)=>{for(let p=Math.max(0,start);p<Math.min(n,end);p++)if(!state.fixed.includes(p))state.fixed.push(p);};
    const sorted=(a,b)=>a.every((v,i)=>i===0||a[i-1]<=v)&&(!b||a.slice().sort((x,y)=>x-y).join(',')===b.slice().sort((x,y)=>x-y).join(','));
    function update(e) {
      if(e.kind==='phase') {
        const [algorithm,phase]=e.phase.split('.'),v=e.variables;
          if(phase==='start'){if(!state)state=fresh();state.algorithm=algorithm;state.ranges=[];state.candidate=null;state.pivot=null;state.label=algorithm==='counting'?'统计各值出现次数':algorithm==='heap'?'构建最大堆':'开始识别算法阶段';return JSON.parse(JSON.stringify(state));}
        if(!state||state.algorithm!==algorithm)return null;
        if(algorithm==='bubble') {
          if(phase==='begin'){state.ranges=[range(0,v.end,'active')];state.label=`第 ${v.pass+1} 轮 · 扫描 [0, ${v.end})`;}
          if(phase==='end'){fix(v.swapped?v.end-1:0,n);state.ranges=[range(0,v.swapped?v.end-1:0,'active')];state.label=v.swapped?`第 ${v.pass+1} 轮结束 · 右侧 ${state.fixed.length} 个元素已归位`:'本轮无交换 · 全部元素已归位';}
        } else if(algorithm==='selection') {
          state.candidate=e.ids[v.candidate];state.ranges=[range(v.index,n,'active')];
          if(phase==='placed'||phase==='end'&&v.candidate===v.index){fix(0,v.index+1);state.candidate=null;}
          state.label=`选择位置 ${v.index} · 最小值候选 a[${v.candidate}]`;
        } else if(algorithm==='insertion') {
          state.insertionEnd=v.index+1;
          if(phase==='begin')state.candidate=e.ids[v.index];else state.candidate=null;state.ranges=[range(0,v.index+(phase==='end'?1:0),'ordered')];
          state.label=phase==='end'?`前 ${v.index+1} 个元素局部有序`:`将 a[${v.index}] 插入左侧有序段`;
        } else if(algorithm==='merge') {
          state.mergeRange=[v.left,v.right];
          state.ranges=phase==='end'?[range(v.left,v.right,'ordered')]:[range(v.left,v.mid,'left'),range(v.mid,v.right,'right')];
          state.label=phase==='end'?`合并完成 [${v.left}, ${v.right})`:`合并 [${v.left}, ${v.mid}) 与 [${v.mid}, ${v.right})`;
        } else if(algorithm==='quick') {
          if(phase==='begin')state.pivot=e.ids[v.hi];
          if(phase==='placed')fix(v.store,v.store+1);
          if(phase==='base'&&v.lo===v.hi)fix(v.lo,v.lo+1);
          state.ranges=v.lo<=v.hi?[range(v.lo,v.hi+1,'active')]:[];
          if(phase==='scan')state.ranges=[range(v.lo,v.store,'left'),range(v.store,v.cursor,'right'),range(v.cursor,v.hi+1,'active')];
          if(phase==='base'||phase==='end')state.pivot=null;
          state.label=phase==='placed'?`pivot 已归位于 a[${v.store}]`:`当前分区 [${v.lo}, ${v.hi+1})${v.pivot==null?'':` · pivot = ${v.pivot}`}`;
        } else if(algorithm==='heap') {
          if(phase==='extract')fix(v.end,n);
          state.ranges=[range(0,v.end??n,'active')];state.candidate=v.root==null?null:e.ids[v.root];
          state.label=phase==='extract'?`堆顶已移到 a[${v.end}] · 堆大小 ${v.end}`:phase==='sift'?`调整子树 a[${v.root}] · 堆大小 ${v.end}`:'构建最大堆';
        } else if(algorithm==='shell'||algorithm==='comb') {
          if(v.gap!=null){state.gap=v.gap;state.group=v.index==null?null:v.index%v.gap;state.ranges=[range(0,n,'active')];state.label=`gap = ${v.gap}${state.group==null?'':` · 当前组下标 % gap = ${state.group}`}`;}
        } else if(algorithm==='counting') {
          if(phase==='start')state.label='统计各值出现次数';
          if(phase==='write'){state.out=v.out;fix(0,v.out);state.candidate=v.out<n?e.ids[v.out]:null;state.ranges=[range(v.out,n,'active')];state.label=`已写入 ${v.out} / ${n} 个元素`;}
        }
      }
      // rotate is presented as several movements. Its temporary visual states
      // need not preserve the two original ordered runs; restore them only at
      // the subsequent real C++ progress boundary.
      if(state?.algorithm==='merge'&&e.kind==='rotate') {
        state.ranges=[range(...state.mergeRange,'active')];state.label=`正在合并 [${state.mergeRange[0]}, ${state.mergeRange[1]})`;
      }
      if(state?.algorithm==='counting'&&e.kind==='bucket') {
        if(e.x<0){state.out??=0;state.candidate=state.out<n?e.ids[state.out]:null;state.ranges=[range(state.out,n,'active')];}
        state.buckets[e.bucket]=e.count;state.bucket=e.bucket;state.label=`count[${e.bucket}] = ${e.count}${state.out==null?' · 统计中':` · 写入位置 ${state.out}`}`;
      }
      if(e.kind==='finish'&&state) {
        const valid=sorted(e.values,initial);state.fixed=valid?Array.from({length:n},(_,i)=>i):[];state.ranges=[];state.candidate=null;state.pivot=null;state.gap=null;
        state.label=valid?'排序完成 · 全部元素已归位':'回放完成 · 结果未通过升序排序检查';
      }
      return state?JSON.parse(JSON.stringify(state)):null;
    }
    return {update};
  }
  function projectState(state,ids) {
    if(!state)return null;
    const copy=JSON.parse(JSON.stringify(state));
    // In exchange-style insertion, the original ordered elements stay ordered
    // around the moving key; the key itself is excluded from both blue ranges.
    if(copy.algorithm==='insertion'&&copy.candidate!=null) {
      const at=ids.indexOf(copy.candidate);
      copy.ranges=[{start:0,end:at,kind:'ordered'},{start:at+1,end:copy.insertionEnd,kind:'ordered'}].filter(r=>r.end>r.start);
    }
    return copy;
  }
  Object.assign(exports,{context,plan,validatePhase,createTracker,projectState});
  if(typeof window!=='undefined')window.SortTraceStates=exports;
  else if(typeof self!=='undefined')self.SortTraceStates=exports;
  if(typeof module==='object'&&module?.exports)module.exports=exports;
})({});
