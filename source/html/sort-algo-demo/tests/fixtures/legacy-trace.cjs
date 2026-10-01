// Legacy JavaScript sorting simulations are rendering fixtures only.
"use strict";

function buildTrace(values, algorithm = "bubble") {
  if (algorithm === "selection") return buildSelectionTrace(values);
  const items = values.map((value, id) => ({ value, id }));
  const events = [];
  let comparisons = 0, swaps = 0, i = 0, j = 0, swapped = false, sortedFrom = values.length;
  const record = (kind) => events.push({ kind, i, j, swapped, sortedFrom, comparisons, swaps, order: items.map(item => item.id), values: items.map(item => item.value) });
  record("initial");
  for (i = 0; i + 1 < items.length; i++) {
    swapped = false;
    j = 0;
    record("pass");
    for (j = 0; j + 1 < items.length - i; j++) {
      comparisons++;
      record("compare");
      if (items[j].value > items[j + 1].value) {
        [items[j], items[j + 1]] = [items[j + 1], items[j]];
        swaps++;
        swapped = true;
        record("swap");
      }
    }
    sortedFrom = swapped ? items.length - i - 1 : 0;
    record("settle");
    if (!swapped) break;
  }
  sortedFrom = 0;
  record("finish");
  return events;
}

function buildSelectionTrace(values) {
  const items = values.map((value, id) => ({ value, id }));
  const events = [];
  let comparisons = 0, swaps = 0, i = 0, j = 0, minIndex = 0, sortedFrom = values.length;
  const record = kind => events.push({ kind, i, j, minIndex, sortedFrom, comparisons, swaps, order: items.map(item => item.id), values: items.map(item => item.value) });
  record("initial");
  for (i = 0; i < items.length - 1; i++) {
    minIndex = i;
    record("pass");
    for (j = i + 1; j < items.length; j++) {
      comparisons++;
      record("compare");
      if (items[j].value < items[minIndex].value) {
        minIndex = j;
        record("minimum");
      }
    }
    if (minIndex !== i) {
      [items[i], items[minIndex]] = [items[minIndex], items[i]];
      swaps++;
      record("swap");
    }
    sortedFrom = i + 1;
    record("settle");
  }
  sortedFrom = 0;
  record("finish");
  return events;
}



function buildExecutionTrace(values, algorithm = "bubble") {
  if (algorithm === "selection") return buildSelectionExecutionTrace(values);
  if (["heap", "shell", "comb"].includes(algorithm)) return buildCompareExchangeTrace(values, algorithm);
  if (algorithm !== "bubble") return buildAlgorithmExecutionTrace(values, algorithm);
  const a = [...values];
  const ids = values.map((_, id) => id);
  const events = [];
  let i = 0, j = null, swapped = null, tmp = null;
  let comparisons = 0, swaps = 0, sortedFrom = values.length;
  const record = (line, action = "none", pair = [], result = null) => events.push({
    line, action, pair: [...pair], result, order: ids.slice(), i, j, minIndex: null, swapped, tmp,
    comparisons, swaps, sortedFrom, values: [...a]
  });
  // Snapshots are BEFORE execution, like a debugger's next-statement cursor.
  for (;;) {
    record(2);
    if (i + 1 >= a.length) break;
    record(3);
    swapped = false;
    j = 0;
    for (;;) {
      const inside = j + 1 < a.length - i;
      const pair = inside ? [ids[j], ids[j + 1]] : [];
      record(4, inside ? "grip" : "none", pair);
      if (!inside) break;
      const exchange = a[j] > a[j + 1];
      record(5, exchange ? "compare-swap" : "compare-keep", pair, exchange);
      comparisons++;
      if (exchange) {
        record(6, "lift", pair);
        tmp = a[j];
        record(7, "shift", pair);
        a[j] = a[j + 1];
        record(8, "drop", pair);
        a[j + 1] = tmp;
        [ids[j], ids[j + 1]] = [ids[j + 1], ids[j]];
        swaps++;
        record(9, "release", pair);
        swapped = true;
        tmp = null;
      }
      j++;
    }
    sortedFrom = values.length - i - 1;
    record(12);
    if (!swapped) break;
    i++;
  }
  sortedFrom = 0;
  record(14, "finish");
  return events;
}

function buildCompareExchangeTrace(values, algorithm) {
  const a = [...values], ids = values.map((_, id) => id), events = [];
  let i = 0, j = null, gap = null, heapSize = null, swapped = null, tmp = null;
  let comparisons = 0, swaps = 0, sortedFrom = values.length, minIndex = null, note = "";
  const record = (line, action = "none", pair = [], result = null, operator = null) => events.push({
    line, action, pair: [...pair], result, operator, order: [...ids], values: [...a],
    i, j, gap, heapSize, swapped, tmp, comparisons, swaps, sortedFrom, minIndex, note
  });
  const compare = (left, right, line, result, trueOperator, falseOperator) => {
    const pair = [ids[left], ids[right]];
    record(line, "grip", pair);
    comparisons++;
    record(line, "compare-select", pair, result, result ? trueOperator : falseOperator);
    return result;
  };
  const exchange = (left, right, line, alreadyGripped = false) => {
    // Animation pairs are always in spatial order, including Shell's backwards
    // scan; the first carried bar must be the one currently on the left.
    if (left > right) [left, right] = [right, left];
    const pair = [ids[left], ids[right]];
    if (!alreadyGripped) record(line, "grip", pair);
    tmp = a[left]; record(line, "lift", pair);
    [a[left], a[right]] = [a[right], a[left]];
    [ids[left], ids[right]] = [ids[right], ids[left]];
    swaps++;
    record(line, "exchange", pair);
    tmp = null; record(line, "release", pair);
  };

  if (algorithm === "heap") {
    const sift = (root, size, stage) => {
      i = root; j = null; heapSize = size; minIndex = null;
      for (;;) {
        note = `${stage}：检查父节点 a[${i}] 的子节点`;
        record(3);
        if (2 * i + 1 >= heapSize) break;
        record(4); j = 2 * i + 1;
        if (j + 1 < heapSize) {
          note = `${stage}：比较子节点 a[${j}] 与 a[${j + 1}]，选择较大者`;
          if (compare(j, j + 1, 5, a[j] < a[j + 1], "<", "≥")) j++;
        } else record(5);
        minIndex = j;
        note = `${stage}：比较父节点 a[${i}] 与较大子节点 a[${j}]`;
        if (!compare(i, j, 6, a[i] < a[j], "<", "≥")) break;
        note = `${stage}：交换父节点 a[${i}] 与子节点 a[${j}]`;
        exchange(i, j, 7, true);
        record(8); i = j; j = null; minIndex = null;
      }
    };
    heapSize = a.length;
    for (let root = Math.floor(a.length / 2) - 1; root >= 0; root--) {
      i = root; j = null; minIndex = null; note = `建堆：从父节点 a[${root}] 开始向下调整`;
      record(11); record(12); sift(root, a.length, "建堆");
    }
    i = -1; note = "建堆完成，开始逐个取出最大值"; record(11);
    for (let size = a.length - 1; size > 0; size--) {
      heapSize = size; i = 0; j = size; minIndex = null;
      note = `取出最大值：交换堆顶 a[0] 与末尾 a[${size}]`;
      record(13); exchange(0, size, 14);
      sortedFrom = size;
      note = `调整堆：a[${size}..${a.length - 1}] 已排序，堆区间为 a[0..${size - 1}]`;
      record(15); sift(0, size, "调整堆");
    }
    heapSize = 1; i = 0; j = null; minIndex = null; note = "堆中只剩一个元素，所有元素已就位";
    record(13);
  } else if (algorithm === "shell") {
    for (gap = Math.floor(a.length / 2); gap > 0; gap = Math.floor(gap / 2)) {
      note = `gap = ${gap}：对相隔 ${gap} 个位置的元素进行插入排序`;
      record(2);
      for (i = gap; i < a.length; i++) {
        record(3);
        for (j = i; j >= gap; j -= gap) {
          if (!compare(j - gap, j, 4, a[j - gap] > a[j], ">", "≤")) break;
          exchange(j - gap, j, 5, true);
        }
        record(4);
      }
      record(3);
    }
    note = "所有间隔轮次已完成"; record(2);
  } else if (algorithm === "comb") {
    record(2); gap = a.length;
    record(3); swapped = true;
    for (;;) {
      record(4);
      if (gap === 1 && !swapped) break;
      record(5); gap = Math.max(1, Math.floor(gap * 10 / 13));
      note = gap > 1 ? `gap = ${gap}：逐对比较并交换，下一轮继续缩小间隔` : "gap = 1：检查相邻元素，直到整轮没有交换";
      record(6); swapped = false;
      for (i = 0; i + gap < a.length; i++) {
        record(7); record(8); j = i + gap;
        if (compare(i, j, 9, a[i] > a[j], ">", "≤")) {
          exchange(i, j, 10, true);
          record(11); swapped = true;
        }
      }
      record(7);
    }
  }
  sortedFrom = 0; note = "排序完成";
  record({ heap: 17, shell: 8, comb: 15 }[algorithm], "finish");
  return events;
}

function buildAlgorithmExecutionTrace(values, algorithm) {
  const a = [...values], ids = values.map((_, id) => id), events = [];
  let comparisons = 0, swaps = 0, i = 0, j = null, minIndex = null, tmp = null, sortedFrom = algorithm === "insertion" ? 1 : 0;
  let scans = 0, writes = 0;
  const record = (line, action = "none", pair = [], result = null) => {
    events.push({ line, action, pair: [...pair], result, order: ids.slice(), i, j, minIndex, tmp, comparisons, swaps, scans, writes, sortedFrom, values: [...a] });
    return events.at(-1);
  };
  const swap = (left, right, line = 9, singleLine = false) => {
    if (left === right) return;
    const pair = [ids[left], ids[right]];
    record(line, "grip", pair); tmp = a[left]; record(singleLine ? line : line + 1, "lift", pair);
    [a[left], a[right]] = [a[right], a[left]]; [ids[left], ids[right]] = [ids[right], ids[left]]; swaps++;
    tmp = null; record(singleLine ? line : line + 2, "exchange", pair);
    record(singleLine ? line : line + 3, "release", pair);
  };
  if (algorithm === "insertion") {
    record(2);
    for (i = 1; i < a.length; i++) {
      j = i; record(3);
      while (j > 0) {
        const pair = [ids[j - 1], ids[j]];
        record(4, "grip", pair);
        const moves = a[j - 1] > a[j];
        comparisons++;
        record(4, "compare-select", pair, moves);
        if (!moves) { record(4, "release", pair); break; }
        tmp = a[j - 1]; record(5, "lift", pair);
        [a[j - 1], a[j]] = [a[j], a[j - 1]]; [ids[j - 1], ids[j]] = [ids[j], ids[j - 1]]; swaps++;
        record(5, "exchange", pair); tmp = null; record(6, "release", pair);
        j--; record(6);
      }
      sortedFrom = i + 1; record(8);
    }
  } else if (algorithm === "merge") {
    record(2);
    for (let width = 1; width < a.length; width *= 2) {
      for (let left = 0; left < a.length; left += 2 * width) {
        const mid = Math.min(left + width, a.length), right = Math.min(left + 2 * width, a.length);
        let end = mid;
        for (i = left, j = mid; i < end && j < right;) {
          const outOfOrder = a[j] < a[i];
          const pair = [ids[i], ids[j]];
          record(8, "grip", pair); comparisons++; record(8, "compare-select", pair, outOfOrder);
          if (!outOfOrder) { i++; continue; }
          for (let k = j; k > i; k--) {
            const adjacent = [ids[k - 1], ids[k]];
            record(9, "grip", adjacent); tmp = a[k - 1]; record(9, "lift", adjacent);
            [a[k - 1], a[k]] = [a[k], a[k - 1]]; [ids[k - 1], ids[k]] = [ids[k], ids[k - 1]]; swaps++;
            tmp = null; record(9, "exchange", adjacent); record(9, "release", adjacent);
          }
          i++; j++; end++;
        }
      }
      record(12);
    }
  } else if (algorithm === "quick") {
    const sort = (lo, hi) => {
      if (lo >= hi) return;
      const pivot = hi; let store = lo;
      minIndex = pivot;
      record(2); record(3); record(4);
      for (j = lo; j < hi; j++) {
        const less = a[j] < a[pivot];
        const pair = [ids[j], ids[pivot]];
        record(6, "grip", pair); comparisons++; record(7, "compare-select", pair, less);
        if (less) { swap(store, j, 8, true); store++; }
      }
      swap(store, pivot, 10, true); sort(lo, store - 1); sort(store + 1, hi);
    };
    sort(0, a.length - 1);
  } else if (algorithm === "counting") {
    record(2); const counts = Array(100).fill(0);
    for (i = 0; i < a.length; i++) {
      j = a[i]; scans++; counts[j]++;
      const event = record(4, "count", [ids[i]], j);
      event.bucketCount = counts[j];
    }
    let out = 0;
    for (let value = 1; value < counts.length; value++) while (counts[value]-- > 0) {
      i = out; j = value; a[out] = value; writes++;
      const event = record(8, "write", [ids[out]], value);
      event.bucketCount = counts[value];
      out++; sortedFrom = out;
    }
  }
  sortedFrom = 0;
  record(({ insertion: 10, merge: 13, quick: 13, counting: 9 })[algorithm] || 14, "finish");
  return events;
}

function buildSelectionExecutionTrace(values) {
  const a = [...values], ids = values.map((_, id) => id), events = [];
  let i = 0, j = null, minIndex = null, tmp = null, comparisons = 0, swaps = 0, sortedFrom = 0;
  const record = (line, action = "none", pair = [], result = null) => events.push({ line, action, pair: [...pair], result, order: ids.slice(), i, j, minIndex, tmp, comparisons, swaps, sortedFrom, values: [...a] });
  record(2);
  for (i = 0; i + 1 < a.length; i++) {
    record(3);
    minIndex = i;
    record(4);
    for (j = i + 1; j < a.length; j++) {
      const pair = [ids[minIndex], ids[j]];
      record(5, "grip", pair);
      const isNewMinimum = a[j] < a[minIndex];
      record(6, "compare-select", pair, isNewMinimum);
      comparisons++;
      if (isNewMinimum) {
        record(7, "mark-min", pair);
        minIndex = j;
      }
      record(8);
    }
    if (minIndex !== i) {
      const pair = [ids[i], ids[minIndex]];
      record(9, "grip", pair);
      record(10, "lift", pair);
      tmp = a[i];
      [a[i], a[minIndex]] = [a[minIndex], a[i]];
      [ids[i], ids[minIndex]] = [ids[minIndex], ids[i]];
      swaps++;
      record(11, "exchange", pair);
      tmp = null;
      record(12, "release", pair);
    }
    sortedFrom = i + 1;
    record(13);
  }
  sortedFrom = 0;
  record(14, "finish");
  return events;
}

if (typeof module !== "undefined") module.exports = { buildTrace, buildExecutionTrace };
