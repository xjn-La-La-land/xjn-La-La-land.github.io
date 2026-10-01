(function (exports) {
  const stateAPI = typeof SortTraceStates !== 'undefined' ? SortTraceStates : typeof require === 'function' ? require('./trace-states.js') : null;
  const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  function parseTrace(text, initial, sourceLines) {
    const records = text.trim().split('\n').map(line => JSON.parse(line));
    const header = records.shift();
    if (header?.kind !== 'header' || header.version !== 1 || !equal(header.initial, initial)) throw new Error('轨迹输入与当前源码运行不匹配。');
    if (!records.length || records.length > 10000 || records.at(-1).kind !== 'finish') throw new Error('轨迹未完整结束或超限。');
    let values = initial.slice(), ids = initial.map((_, i) => i), saved = null;
    const n = initial.length, operators = { '>': (a,b)=>a>b, '<': (a,b)=>a<b, '>=': (a,b)=>a>=b, '<=': (a,b)=>a<=b, '==': (a,b)=>a===b, '!=': (a,b)=>a!==b };
    const check = (ok, message) => { if (!ok) throw new Error('无效执行轨迹：' + message); };
    const position = p => Number.isInteger(p) && p >= 0 && p < n;
    for (let i = 0; i < records.length; i++) {
      const e = records[i];
      check(e.seq === i + 1 && Number.isInteger(e.line) && e.line >= 1 && e.line <= sourceLines && Number.isInteger(e.column) && e.column >= 1, '事件位置');
      for (const key of ['before','values','beforeIds','ids']) check(Array.isArray(e[key]) && e[key].length === n && e[key].every(Number.isInteger), key);
      check(equal(e.before, values) && equal(e.beforeIds, ids), '前态不连续');
      check(e.variables && typeof e.variables === 'object' && !Array.isArray(e.variables) && Object.keys(e.variables).length <= 32 && Object.entries(e.variables).every(([key,v]) => /^[A-Za-z_]\w*$/.test(key) && Number.isSafeInteger(v)), '变量快照');
      let expected = values.slice(), nextIds = ids.slice();
      if (['read','scan','save'].includes(e.kind)) {
        check(position(e.left) && e.x === values[e.left], '读取值');
        if (e.kind === 'save') { check(!saved,'交换保存嵌套'); saved = { left:e.left, value:e.x, id:ids[e.left], right:null }; }
      } else if (e.kind === 'compare') {
        check(Object.hasOwn(operators,e.operator) && Number(operators[e.operator](e.x,e.y)) === e.result, '比较结果');
        check(e.left === -1 || position(e.left) && values[e.left] === e.x, '左操作数');
        check(e.right === -1 || position(e.right) && values[e.right] === e.y, '右操作数');
      } else if (e.kind === 'swap') {
        check(position(e.left) && position(e.right), '交换下标');
        [expected[e.left],expected[e.right]]=[expected[e.right],expected[e.left]];
        [nextIds[e.left],nextIds[e.right]]=[nextIds[e.right],nextIds[e.left]];
      } else if (e.kind === 'shift') {
        check(saved && saved.left === e.left && position(e.right) && saved.right === null,'交换移动'); saved.right=e.right;
        expected[e.left]=values[e.right]; nextIds[e.left]=ids[e.right];
      } else if (e.kind === 'drop') {
        check(saved && e.left===saved.left && e.right===saved.right,'交换放下');expected[e.right]=saved.value;nextIds[e.right]=saved.id;saved=null;
      } else if (e.kind === 'write') {
        check(position(e.left) && Number.isInteger(e.x),'写入');expected[e.left]=e.x;
      } else if (e.kind === 'rotate') {
        check([e.left,e.middle,e.right].every(Number.isInteger) && e.left>=0 && e.left<=e.middle && e.middle<=e.right && e.right<=n,'rotate 边界');
        const rotate = a => [...a.slice(0,e.left),...a.slice(e.middle,e.right),...a.slice(e.left,e.middle),...a.slice(e.right)];expected=rotate(values);nextIds=rotate(ids);
      } else if (e.kind === 'phase') {
        check(!saved && stateAPI?.validatePhase(e,n), '阶段标记');
      } else if (e.kind === 'finish') check(i===records.length-1 && !saved,'未完成交换');
      else check(e.kind==='bucket' && Number.isInteger(e.bucket) && e.bucket>=0 && Number.isInteger(e.count) && [1,-1].includes(e.x) && e.count===e.y+e.x,'未知动作/计数桶');
      check(equal(e.values,expected) && equal(e.ids,nextIds),'操作后态');
      if (!saved || saved.right===null) check(equal(e.ids.slice().sort((a,b)=>a-b), initial.map((_,j)=>j)),'元素身份');
      values=e.values; ids=e.ids;
    }
    return records;
  }

  function toAnimationTrace(records, initial, sourceLines) {
    let comparisons=0, swaps=0, scans=0, writes=0, manual=null, lastScan=null;
    const tracker=stateAPI?.createTracker(initial);let state=null;
    const output=[];
    function base(e, before=false) {
      const v=e.variables||{};
      return {line:e.kind==='finish'?sourceLines:e.line,column:e.column,action:'none',pair:[],result:null,
        order:(before?e.beforeIds:e.ids).slice(),values:(before?e.before:e.values).slice(),variables:v,
        i:v.i??null,j:v.j??null,minIndex:v.minIndex??null,tmp:v.tmp??null,swapped:v.swapped==null?null:!!v.swapped,
        gap:v.gap??null,heapSize:v.heapSize??null,comparisons,swaps,scans,writes,sortedFrom:null,state,note:''};
    }
    function push(e,action,pair=[],before=false,extra={}){const event={...base(e,before),action,pair:pair.slice(),...extra};event.state=stateAPI?.projectState(state,event.order)||null;output.push(event);return event;}
    function exchange(e,left,right,previousValues=e.before,previousIds=e.beforeIds,nextValues=e.values,nextIds=e.ids,logicalSwap=false) {
      const pair=[Math.min(left,right),Math.max(left,right)].map(p=>previousIds[p]);
      if(left===right){push(e,'none');return;}
      const before={order:previousIds.slice(),values:previousValues.slice()};
      push(e,'grip',pair,true,before);push(e,'lift',pair,true,before);
      if(logicalSwap)swaps++;
      push(e,'exchange',pair,false,{order:nextIds.slice(),values:nextValues.slice()});push(e,'release',pair,false,{order:nextIds.slice(),values:nextValues.slice()});
    }
    push({kind:'entry',line:1,column:1,ids:initial.map((_,id)=>id),values:initial,variables:{}},'none');
    for(let index=0;index<records.length;index++) {
      const e=records[index];
      state=tracker?.update(e)||null;
      if(e.kind==='phase'){push(e,'state');continue;}
      if(e.kind==='compare') {
        const pair=[e.left,e.right].filter(p=>p>=0).map(p=>e.ids[p]).filter((id,i,a)=>a.indexOf(id)===i);
        if(pair.length===2)push(e,'grip',pair,true);
        comparisons++;
        push(e,'compare-keep',pair,false,{result:!!e.result,operator:e.operator,relation:`${e.x} ${e.operator} ${e.y} → ${e.result?'true':'false'}`});
        if(pair.length===2)push(e,'release',pair);continue;
      }
      if(e.kind==='save') {
        const next=records[index+1];
        if(next?.kind!=='shift')throw new Error('保存事件没有完整交换后继。');
        manual={pair:[Math.min(e.left,next.right),Math.max(e.left,next.right)].map(p=>e.ids[p]),savedId:e.ids[e.left],shiftId:e.ids[next.right],left:e.left,right:next.right};
        if(manual.left===manual.right)push(e,'none',[],false,{tmp:e.x});
        else {push(e,'grip',manual.pair,true);push(e,'lift',manual.pair,false,{tmp:e.x});}continue;
      }
      if(e.kind==='shift'){push(e,manual.left===manual.right?'none':'shift',manual.pair,false,{target:e.left,movingId:manual.shiftId,tmp:records[index-1].x});continue;}
      if(e.kind==='drop') {
        if(e.left!==e.right)swaps++;
        if(e.left===e.right)push(e,'none',[],false,{tmp:e.x});
        else {push(e,'drop',manual.pair,false,{target:e.right,movingId:manual.savedId,tmp:e.x});push(e,'release',manual.pair);}manual=null;continue;
      }
      if(e.kind==='swap'){exchange(e,e.left,e.right,e.before,e.beforeIds,e.values,e.ids,true);continue;}
      if(e.kind==='rotate') {
        // Presentation decomposition, not extra C++ swaps or another sorting algorithm.
        let ids=e.beforeIds.slice(),values=e.before.slice();
        for(let target=e.left;target<e.right;target++) {
          let at=ids.indexOf(e.ids[target]);
          while(at>target) {
            const nextIds=ids.slice(),nextValues=values.slice();
            [nextIds[at-1],nextIds[at]]=[nextIds[at],nextIds[at-1]];[nextValues[at-1],nextValues[at]]=[nextValues[at],nextValues[at-1]];
            exchange(e,at-1,at,values,ids,nextValues,nextIds);ids=nextIds;values=nextValues;at--;
          }
        }
        continue;
      }
      if(e.kind==='write'){writes++;push(e,'write',[],false,{target:e.left,result:e.x});continue;}
      if(e.kind==='scan'){scans++;lastScan=e.left;push(e,'none',[e.ids[e.left]],false,{note:`读取 a[${e.left}] = ${e.x}`});continue;}
      if(e.kind==='bucket') {
        push(e,e.x>0?'count':'none',lastScan===null?[]:[e.ids[lastScan]],false,{result:e.bucket,bucketCount:e.count,compact:e.x===-1 && e.y===0,note:`count[${e.bucket}] = ${e.count}`});continue;
      }
      const verifiedSorted=e.kind==='finish' && e.values.every((v,i)=>i===0||e.values[i-1]<=v) && equal(e.values.slice().sort((a,b)=>a-b),initial.slice().sort((a,b)=>a-b));
      push(e,e.kind==='finish'?'finish':'none',[],false,{verifiedSorted,note:e.kind==='read'?`读取 a[${e.left}] = ${e.x}`:''});
    }
    if(!output.length)throw new Error('没有可播放事件。');
    // Keep the existing grip only when the actual next operation lifts the
    // same two elements from the same state. Do not infer swaps from `result`:
    // edited C++ can swap on false, or compare without swapping on true.
    const continuous=[];
    const samePair=(a,b)=>a.length===2 && b.length===2 && a.every(id=>b.includes(id));
    for(let i=0;i<output.length;i++) {
      const compare=output[i],release=output[i+1],grip=output[i+2],lift=output[i+3];
      continuous.push(compare);
      if(compare.action==='compare-keep' && release?.action==='release' && grip?.action==='grip' && lift?.action==='lift'
        && samePair(compare.pair,grip.pair) && samePair(grip.pair,lift.pair)
        && equal(compare.order,grip.order) && equal(compare.values,grip.values))i+=2;
    }
    // Plan each release independently. Pure reads between two pair operations
    // can preserve a grip, but writes/counting/completion end the pair flow.
    for(let i=0;i<continuous.length;i++) {
      const release=continuous[i];
      if(release.action!=='release')continue;
      release.retain=[];
      let j=i+1;
      while(['none','state'].includes(continuous[j]?.action) && equal(release.order,continuous[j].order) && equal(release.values,continuous[j].values))j++;
      const grip=continuous[j];
      if(grip?.action!=='grip' || !equal(release.order,grip.order) || !equal(release.values,grip.values))continue;
      release.retain=release.pair.filter(id=>grip.pair.includes(id));
      grip.retain=release.retain.slice();
    }
    return continuous;
  }
  Object.assign(exports,{parseTrace,toAnimationTrace});
  // A browser host can also expose `module`; CommonJS must not suppress the
  // explicit page/worker API. Export both, as compiler.js already does.
  if(typeof window!=='undefined')window.SortTraceEvents=exports;
  else if(typeof self!=='undefined')self.SortTraceEvents=exports;
  if(typeof module==='object' && module && module.exports)module.exports=exports;
})({});
