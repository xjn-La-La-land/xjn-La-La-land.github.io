const assert = require('node:assert/strict');
const {parseTrace, toAnimationTrace} = require('../frontend/js/trace-events.js');
const fs = require('node:fs'), vm = require('node:vm');

for(const file of ['trace-events.js','trace-source.js','trace-states.js']) {
  const globalName={'trace-events.js':'SortTraceEvents','trace-source.js':'SortTraceSource','trace-states.js':'SortTraceStates'}[file];
  const method={'trace-events.js':'toAnimationTrace','trace-source.js':'instrumentTraceSource','trace-states.js':'createTracker'}[file];
  for(const mode of ['page','page-with-module','worker','worker-with-module','commonjs']) {
    const context={};
    if(mode.startsWith('page'))context.window=context;
    if(mode.startsWith('worker'))context.self=context;
    if(mode.includes('module') || mode==='commonjs')context.module={exports:{}};
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(__dirname+'/../frontend/js/'+file,'utf8'),context,{filename:file});
    if(mode!=='commonjs')assert.equal(typeof context[globalName]?.[method],'function',file+' '+mode+' exports its global API');
    if(context.module)assert.equal(typeof context.module.exports[method],'function',file+' '+mode+' preserves CommonJS');
  }
}

const initial = [3, 1, 2], identities = [0, 1, 2];
function event(kind, seq, before = initial, values = before, beforeIds = identities, ids = beforeIds, extra = {}) {
  return {kind, seq, line:1, column:1, left:-1, right:-1, middle:-1, x:0, y:0, operator:'', result:-1, bucket:-1, count:0,
    before, values, beforeIds, ids, variables:{n:3}, ...extra};
}
function parse(events) {
  return parseTrace([{kind:'header', version:1, initial}, ...events].map(e=>JSON.stringify(e)).join('\n'), initial, 1);
}
const exchanged = [1, 3, 2], exchangedIds = [1, 0, 2];
const compare = event('compare', 1, initial, initial, identities, identities, {left:0,right:1,x:3,y:1,operator:'>',result:1});
const swap = event('swap', 2, initial, exchanged, identities, exchangedIds, {left:0,right:1,x:3,y:1});
const finish = event('finish', 3, exchanged, exchanged, exchangedIds, exchangedIds);
const valid = parse([compare, swap, finish]);
const animation = toAnimationTrace(valid, initial, 1);
assert.equal(animation[0].comparisons, 0);
assert.equal(animation[0].swaps, 0);
assert.equal(animation.find(e=>e.action==='grip').comparisons, 0, 'Decorative grip is not a comparison');
assert.equal(animation.find(e=>e.action==='compare-keep').relation, '3 > 1 → true');
assert.equal(animation.find(e=>e.action==='lift').swaps, 0, 'Decorative lift is not an executed swap');
assert.equal(animation.find(e=>e.action==='exchange').swaps, 1);
assert.deepEqual(animation.at(-1).values, exchanged);
assert.deepEqual(animation.map(e=>e.action), ['none','grip','compare-keep','lift','exchange','release','finish'], 'Comparison flows directly into the recorded same-pair swap');
const keepOnFalse=toAnimationTrace(parse([{...compare,operator:'<',result:0},swap,finish]),initial,1);
assert.equal(keepOnFalse[keepOnFalse.findIndex(e=>e.action==='compare-keep')+1].action,'lift', 'Actual swap, not a truthy comparison, controls grip reuse');
const noSwap=toAnimationTrace(parse([compare,event('finish',2)]),initial,1);
assert.equal(noSwap[noSwap.findIndex(e=>e.action==='compare-keep')+1].action,'release', 'True comparisons without a following swap still retract');
const otherValues=[2,1,3],otherIds=[2,1,0];
const otherPair=toAnimationTrace(parse([compare,event('swap',2,initial,otherValues,identities,otherIds,{left:0,right:2,x:3,y:2}),event('finish',3,otherValues,otherValues,otherIds,otherIds)]),initial,1);
assert.equal(otherPair[otherPair.findIndex(e=>e.action==='compare-keep')+1].action,'release', 'A different exchange pair must release and reposition');
assert.deepEqual(otherPair.find(e=>e.action==='release').retain,[0],'A different pair reuses its common element instead of releasing both');
const nextCompare=event('compare',2,initial,initial,identities,identities,{left:2,right:1,x:2,y:1,operator:'<',result:0,variables:{minIndex:1,j:2,n:3}});
const selectionComparisons=toAnimationTrace(parse([compare,nextCompare,event('finish',3)]),initial,1);
assert.deepEqual(selectionComparisons.find(e=>e.action==='release').retain,[1],'Unchanged selection minimum stays gripped');
assert.deepEqual(selectionComparisons.filter(e=>e.action==='grip')[1].retain,[1],'Only the scanning gripper needs to approach the next element');
const repeatedPair=toAnimationTrace(parse([compare,{...compare,seq:2},event('finish',3)]),initial,1);
assert.deepEqual(repeatedPair.find(e=>e.action==='release').retain,[0,1],'Repeated comparisons retain both operands');
const afterSwap=toAnimationTrace(parse([compare,swap,event('compare',3,exchanged,exchanged,exchangedIds,exchangedIds,{left:0,right:2,x:1,y:2,operator:'<',result:1}),{...finish,seq:4}]),initial,1);
assert.deepEqual(afterSwap.find(e=>e.action==='release').retain,[1],'Reuse follows element identity after exchanging positions');
assert.deepEqual(afterSwap.filter(e=>e.action==='grip')[1].retain,[1]);
const interrupted=toAnimationTrace(parse([compare,event('write',2,initial,initial,identities,identities,{left:0,x:3}),{...nextCompare,seq:3},event('finish',4)]),initial,1);
assert.deepEqual(interrupted.find(e=>e.action==='release').retain,[],'An intervening write ends the old operation grip');
for (const events of [[compare, swap], [{...compare,result:0}, swap, finish], [compare,{...swap,values:[1,2,3]},finish], [compare,swap,{...finish,before:initial}], [compare,swap,{...finish,seq:4}], [compare,swap,{...finish,line:2}]]) {
  assert.throws(()=>parse(events));
}
const partial = [3, 3, 2], partialIds = [0, 0, 2];
const reversed = parse([
  event('save',1,initial,initial,identities,identities,{left:1,x:1}),
  event('shift',2,initial,partial,identities,partialIds,{left:1,right:0,x:1,y:3}),
  event('drop',3,partial,exchanged,partialIds,exchangedIds,{left:1,right:0,x:1}), finish
].map((e,i)=>({...e,seq:i+1})));
const reverseAnimation = toAnimationTrace(reversed,initial,1);
assert.deepEqual(reverseAnimation.find(e=>e.action==='lift').pair,[0,1], 'Adjacent exchange always lifts spatial-left');
assert.equal(reverseAnimation.find(e=>e.action==='shift').movingId,0);
assert.equal(reverseAnimation.find(e=>e.action==='drop').movingId,1);
const manualAfterCompare=toAnimationTrace(parse([compare,...reversed.map(e=>({...e,seq:e.seq+1}))]),initial,1);
assert.equal(manualAfterCompare.filter(e=>e.action==='grip').length,1,'Manual exchanges reuse the grip even when the operand order is reversed');
assert.equal(manualAfterCompare[manualAfterCompare.findIndex(e=>e.action==='compare-keep')+1].action,'lift');
assert.throws(()=>parse(reversed.slice(0,2).concat(event('finish',3,partial,partial,partialIds,partialIds))));
const rotated=[1,2,3], rotatedIds=[1,2,0];
const rotation=parse([event('rotate',1,initial,rotated,identities,rotatedIds,{left:0,middle:1,right:3}),event('finish',2,rotated,rotated,rotatedIds,rotatedIds)]);
const rotatedAnimation=toAnimationTrace(rotation,initial,1);
assert.deepEqual(rotatedAnimation.at(-1).order,rotatedIds);
assert.equal(rotatedAnimation.at(-1).swaps,0, 'rotate presentation is not fabricated C++ swap counts');
assert.throws(()=>parse([event('bucket',1,initial,initial,identities,identities,{bucket:3,count:5,x:1,y:0}),event('finish',2)]));
const phase=event('phase',1,initial,initial,identities,identities,{phase:'bubble.start',variables:{n:3}});
assert.equal(toAnimationTrace(parse([phase,event('finish',2)]),initial,1).find(e=>e.action==='state').state.algorithm,'bubble');
for(const extra of [{phase:'bubble.sift'},{variables:{}},{variables:{n:4}}])assert.throws(()=>parse([{...phase,...extra},event('finish',2)]));
console.log('Passed: trace continuity, semantic validation, complete finish, source positions, real counters, rotate identities, reversed manual exchange and bucket integrity.');
