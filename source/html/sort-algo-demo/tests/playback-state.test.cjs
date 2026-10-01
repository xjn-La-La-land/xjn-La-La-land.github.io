const assert = require('node:assert/strict');
const {parseTrace, toAnimationTrace} = require('../frontend/js/trace-events.js');
const {createHarness} = require('./test.cjs');
const {gsap} = require('../frontend/assets/gsap.min.js');
const initial = [5, 2, 8, 1, 7, 3, 6, 4];

function recordsFor(values, fill) {
  let array = values.slice(), ids = values.map((_, i) => i);
  const records = [];
  const emit = (kind, fields = {}) => {
    const before = array.slice(), beforeIds = ids.slice();
    if (kind === 'swap') {
      [array[fields.left], array[fields.right]] = [array[fields.right], array[fields.left]];
      [ids[fields.left], ids[fields.right]] = [ids[fields.right], ids[fields.left]];
    }
    if (kind === 'write') array[fields.left] = fields.x;
    records.push({kind, seq: records.length + 1, line: 4, column: 1,
      left: -1, right: -1, middle: -1, x: 0, y: 0, operator: '', result: -1,
      bucket: -1, count: 0, variables: {}, ...fields,
      before, beforeIds, values: array.slice(), ids: ids.slice()});
  };
  fill(emit);
  emit('finish');
  return parseTrace([{kind: 'header', version: 1, initial: values}, ...records].map(JSON.stringify).join('\n'), values, 14);
}
function advance(timeline, target) {
  for (let t = timeline.time() + .005; t < target; t += .005) timeline.time(t, false);
  timeline.time(target, false);
}
const status = (h, values) => assert.equal(h.get('#status').textContent, `a = [${values.join(', ')}]`);

for (const [left, right] of [[0, 1], [0, 7]]) {
  const records = recordsFor(initial, emit => emit('swap', {left, right}));
  const actions = toAnimationTrace(records, initial, 14);
  const h = createHarness(false, false, false, false, () => records), tl = h.timeline;
  const exchange = actions.findIndex(e => e.action === 'exchange');
  advance(tl, tl.labels['action-' + exchange] - .001);
  status(h, initial); assert.equal(h.get('#swaps').textContent, 0, 'Lift boundary cannot commit an exchange');
  advance(tl, tl.labels['action-' + exchange] + .15);
  status(h, initial); assert.equal(h.get('#swaps').textContent, 0, 'Mid-exchange still shows the preceding state');
  advance(tl, tl.labels['action-' + (exchange + 1)] - .001);
  status(h, records.at(-1).values); assert.equal(h.get('#swaps').textContent, 1);
  const physical = h.get('#bars').children.map(b => ({value: Number(b.dataset.value), x: Number(b.attrs.transform.match(/-?\d+(?:\.\d+)?/)[0])})).sort((a, b) => a.x - b.x).map(b => b.value);
  assert.deepEqual(physical, records.at(-1).values, 'Paused exchange state agrees with physical slots');
  tl.kill();
}

const counting = values => recordsFor(values, emit => {
  const counts = new Map();
  emit('phase', {phase: 'counting.start', variables: {n: values.length}});
  values.forEach((value, i) => {
    emit('scan', {left: i, x: value});
    const before = counts.get(value) || 0; counts.set(value, before + 1);
    emit('bucket', {bucket: value, x: 1, y: before, count: before + 1});
  });
  let out = 0;
  for (let value = 1; value < 100; value++) {
    let count = counts.get(value) || 0;
    while (count > 0) {
      emit('bucket', {bucket: value, x: -1, y: count, count: --count});
      emit('write', {left: out++, x: value});
      emit('phase', {phase: 'counting.write', variables: {out, value}});
    }
    emit('bucket', {bucket: value, x: -1, y: 0, count: -1});
  }
});
const h = createHarness(false, false, false, false, r => counting(r.values));
h.tabs.find(t => t.dataset.algorithm === 'counting').dispatch('click');
const tl = h.timeline, records = counting(initial), actions = toAnimationTrace(records, initial, 14);
const firstWrite = actions.findIndex(e => e.action === 'write');
advance(tl, tl.labels['action-' + firstWrite] + .12);
status(h, initial); assert.equal(h.get('#swaps').textContent, 0, 'Write count waits for the bar update');
assert.equal(h.get('#bars').children[0].children[1].textContent, 5);
advance(tl, tl.labels['action-' + (firstWrite + 1)] - .001);
status(h, actions[firstWrite].values); assert.equal(h.get('#swaps').textContent, 1);
assert.equal(h.get('#bars').children[0].children[1].textContent, 1);

const lastWrite = actions.findLastIndex(e => e.action === 'write');
assert.equal(actions.filter(e => e.compact).length, 99, 'Every failed bucket check is retained');
const tail = tl.labels['action-' + (actions.length - 1)] - (tl.labels['action-' + lastWrite] + .30);
assert.ok(tail < 1.5, 'Empty bucket tail takes under 1.5 seconds');
const firstEmpty = actions.findIndex(e => e.compact && e.result > 8);
advance(tl, tl.labels['action-' + firstEmpty] - .001);
for (let i = firstEmpty; i < firstEmpty + 3; i++) {
  h.get('#step').dispatch('click');
  advance(tl, tl.time() + .02);
  assert.equal(h.get('#execution-state').textContent, '已暂停');
  assert.match(h.get('#state-description').textContent, new RegExp(`count\\[${actions[i].result}\\] = -1`), 'A compact event is still individually inspectable');
}
advance(tl, tl.duration());
status(h, initial.slice().sort((a, b) => a - b));
assert.equal(h.get('#execution-state').textContent, '已完成');
tl.kill(); gsap.ticker.sleep();
console.log(`Passed: exchange/write commits agree with paused visuals; compact bucket events remain stepable; counting tail ${tail.toFixed(2)}s.`);
