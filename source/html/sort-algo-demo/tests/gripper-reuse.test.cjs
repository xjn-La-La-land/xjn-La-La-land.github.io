const assert = require('node:assert/strict');
const {parseTrace,toAnimationTrace} = require('../frontend/js/trace-events.js');
const numbers = value => value.match(/-?\d+(?:\.\d+)?/g).map(Number);
const original = [5,2,8,1,7,3,6,4];

function fixture(values = original) {
  let state=values.slice(), ids=values.map((_,id)=>id), records=[];
  function emit(kind,left=-1,right=-1,variables={}) {
    const before=state.slice(), beforeIds=ids.slice(), x=state[left]??0, y=state[right]??0;
    if(kind==='swap') { [state[left],state[right]]=[state[right],state[left]];[ids[left],ids[right]]=[ids[right],ids[left]]; }
    records.push({kind,seq:records.length+1,line:4,column:1,left,right,middle:-1,x,y,
      operator:kind==='compare'?'<':'',result:kind==='compare'?Number(x<y):-1,bucket:-1,count:0,
      before,values:state.slice(),beforeIds,ids:ids.slice(),variables});
  }
  emit('compare',2,1,{j:2,minIndex:1});
  emit('compare',3,1,{j:3,minIndex:1}); // unchanged minimum; only scanner moves
  emit('compare',4,3,{j:4,minIndex:3}); // new minimum changes operand role
  emit('compare',4,3,{j:4,minIndex:3}); // repeated pair: neither claw moves
  emit('swap',4,3);                    // same-pair comparison -> swap
  emit('compare',5,3,{j:5,minIndex:3}); // identity moved by previous swap
  emit('compare',7,0,{j:7,minIndex:0}); // disjoint pair: release both
  emit('finish');
  return parseTrace([{kind:'header',version:1,initial:values},...records].map(e=>JSON.stringify(e)).join('\n'),values,14);
}

function checkReusePhysics(records, values = original, algorithm = 'selection') {
  const {createHarness} = require('./test.cjs');
  // The initial harness uses eight elements. A minimal complete trace supplies
  // its preview before the requested input/algorithm is installed below.
  const collect=request => request.values.length===values.length && request.values.every((v,i)=>v===values[i])
    ? records : fixture(request.values);
  const harness=createHarness(false,false,false,false,collect);
  if(values.length!==original.length || !values.every((v,i)=>v===original[i])) {
    harness.get('#array-input').value=values.join(', ');harness.get('#array-form').dispatch('submit');
  }
  if(algorithm!=='bubble')harness.tabs.find(tab=>tab.dataset.algorithm===algorithm).dispatch('click');
  const timeline=harness.timeline;
  assert.ok(timeline,algorithm+' installs the semantic trace');
  const animation=toAnimationTrace(records,values,harness.editor.model.getLineCount());
  const targets=timeline.getChildren().flatMap(t=>t.targets());
  const claws=[0,1].map(side=>targets.find(t=>t.element===harness.get('#claw-'+side)));
  if(!claws.every(Boolean)){timeline.kill();return {partial:0,both:0,crossings:0};}
  const width=Number(harness.get('#bars').children[0].children[0].attrs.width), closed=width/2+4;
  const slotX=harness.get('#indices').children.map(node=>Number(node.attrs.x));
  const maxFrameStep=Math.max(32,(slotX.at(-1)-slotX[0])*.15);
  const transitions=[];
  animation.forEach((event,index)=>{
    if(event.action!=='release' || !event.retain?.length)return;
    let next=index+1;while(['none','state'].includes(animation[next]?.action))next++;
    assert.equal(animation[next].action,'grip');
    transitions.push({event,start:timeline.labels['action-'+index]+.001,end:timeline.labels['action-'+(next+1)]-.001,
      next:animation[next],held:null});
  });
  let partial=0,both=0,crossings=0,previous=null;
  for(let time=0;time<=timeline.duration()+.01;time+=.01) {
    timeline.time(Math.min(time,timeline.duration()),false);
    const poses=claws.map(claw=>numbers(claw.element.attrs.transform));
    poses.forEach((pose,i)=>{
      assert.ok(pose[1]>=37,algorithm+' stays below rail');
      if(previous)assert.ok(Math.hypot(pose[0]-previous[i][0],pose[1]-previous[i][1])<maxFrameStep,algorithm+' claws do not teleport');
    });
    previous=poses;
    // The heads/jaws may pass horizontally only in separate vertical lanes.
    const overlapX=Math.min(poses[0][0]+claws[0].gap,poses[1][0]+claws[1].gap)-Math.max(poses[0][0]-claws[0].gap,poses[1][0]-claws[1].gap);
    const overlapY=Math.min(poses[0][1]+25,poses[1][1]+25)-Math.max(poses[0][1]-4,poses[1][1]-4);
    assert.ok(overlapX<.01 || overlapY<.01,`${algorithm}: gripper heads intersect at ${time.toFixed(2)}s`);
    for(const transition of transitions) {
      if(time<transition.start || time>transition.end)continue;
      if(!transition.held) {
        transition.held=transition.event.retain.map(id=>{
          const claw=claws.find(c=>c.attached===id);assert.ok(claw,'Retained element is attached before release');
          return {id,claw,pose:numbers(claw.element.attrs.transform)};
        });
        if(transition.held.length===1) {
          partial++;
          const kept=transition.held[0].id;
          const oldOther=transition.event.pair.find(id=>id!==kept),newOther=transition.next.pair.find(id=>id!==kept);
          const position=id=>transition.event.order.indexOf(id);
          if((position(oldOther)-position(kept))*(position(newOther)-position(kept))<0)crossings++;
        } else both++;
      }
      transition.held.forEach(({id,claw,pose})=>{
        assert.equal(claw.attached,id,`${algorithm}: reused claw never detaches`);
        assert.ok(Math.abs(claw.gap-closed)<.001,`${algorithm}: reused claw never opens`);
        const current=numbers(claw.element.attrs.transform);
        assert.ok(Math.hypot(current[0]-pose[0],current[1]-pose[1])<.001,`${algorithm}: reused claw never retracts or changes position`);
      });
    }
  }
  timeline.time(timeline.duration(),false);
  assert.ok(claws.every(c=>c.attached===null && c.gap>closed),'Finish releases all claws');
  assert.deepEqual(harness.get('#bars').children.map(bar=>({value:Number(bar.dataset.value),x:numbers(bar.attrs.transform)[0]})).sort((a,b)=>a.x-b.x).map(bar=>bar.value),records.at(-1).values);
  // Replay rebuilds the per-claw assignment; cancellation opens any retained
  // claw and leaves no half-attached state behind.
  harness.get('#restart').dispatch('click');
  const replay=harness.timeline,first=transitions.find(t=>t.event.retain.length===1);
  if(first) {
    for(let time=0;time<first.start+.1;time+=.01)replay.time(time,false);
    const replayTargets=replay.getChildren().flatMap(t=>t.targets());
    const kept=replayTargets.find(c=>c.element && c.attached===first.event.retain[0]);
    assert.ok(kept,'Replay reconstructs the retained physical claw');
    const frozen=numbers(kept.element.attrs.transform);
    for(let action=0;action<2;action++) {
      harness.get('#step').dispatch('click');
      for(let time=replay.time()+.01;time<replay.duration();time+=.01) {
        replay.time(time,false);
        assert.equal(kept.attached,first.event.retain[0],'Single-step preserves the held operand');
        const position=numbers(kept.element.attrs.transform);
        assert.ok(Math.hypot(position[0]-frozen[0],position[1]-frozen[1])<.001,'Paused/stepped retained claw stays stationary');
        if(harness.get('#execution-state').textContent!=='动作单步')break;
      }
      assert.equal(harness.get('#execution-state').textContent,'已暂停');
      assert.ok(replay.paused());
    }
    harness.editor.model.setValue('// changed\n'+harness.editor.model.getValue());
    const replayClaws=replay.getChildren().flatMap(t=>t.targets()).filter(t=>claws.some(c=>t.element===c.element));
    assert.ok(replayClaws.every(c=>c.attached===null && c.gap>closed),'Editing resets retained grips');
    assert.ok(harness.get('#step').disabled);
  }
  replay.kill();timeline.kill();
  return {partial,both,crossings};
}

module.exports={checkReusePhysics,fixture};
if(require.main===module) {
  const {gsap}=require('../frontend/assets/gsap.min.js');
  const normal=checkReusePhysics(fixture());
  assert.ok(normal.partial>=3 && normal.both>=1 && normal.crossings>=1);
  const repeated=Array(12).fill(7);
  const duplicates=checkReusePhysics(fixture(repeated),repeated);
  assert.ok(duplicates.partial>=3 && duplicates.crossings>=1,'Equal keys reuse by identity, with narrow-layout clearance');
  gsap.ticker.sleep();
  console.log('Passed: per-element grip retention, unchanged/changed minimum, reversed operand roles, post-swap identities, duplicate keys, crossing avoidance, replay and edit cleanup.',normal,duplicates);
}
