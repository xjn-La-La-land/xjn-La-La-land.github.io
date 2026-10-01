#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>
using std::size_t;

void bubble_sort(int a[], size_t n) {
  for (size_t i = 0; i + 1 < n; ++i) {
    bool swapped = false;
    for (size_t j = 0; j + 1 < n-i; ++j) {
      if (a[j] > a[j+1]) {
        int tmp = a[j];
        a[j] = a[j+1];
        a[j+1] = tmp;
        swapped = true;
      }
    }
    if (!swapped) break;
  }
}

void selection_sort(int a[], size_t n) {
  for (size_t i = 0; i + 1 < n; ++i) {
    size_t minIndex = i;
    for (size_t j = i + 1; j < n; ++j) {
      if (a[j] < a[minIndex]) {
        minIndex = j;
      }
    }
    if (minIndex == i) continue;
    int tmp = a[i];
    a[i] = a[minIndex];
    a[minIndex] = tmp;
  }
}

void insertion_sort(int a[], size_t n) {
  for (size_t i = 1; i < n; ++i) {
    size_t j = i;
    while (j > 0 && a[j-1] > a[j]) {
      std::swap(a[j-1], a[j]);
      --j;
    }
    // a[0..i] is sorted
  }
}

void merge_sort(int a[], size_t n) {
  for (size_t width = 1; width < n; width *= 2) {
    for (size_t left = 0; left < n; left += 2 * width) {
      size_t mid = std::min(left + width, n);
      size_t right = std::min(left + 2 * width, n);
      size_t i = left, j = mid;
      while (i < mid && j < right) {
        if (a[i] <= a[j]) ++i;
        else { std::rotate(a + i, a + j, a + j + 1); ++i; ++j; ++mid; }
      }
    }
  }
}

void quick_sort(int a[], int lo, int hi) {
  if (lo >= hi) return;
  int pivot = a[hi];
  int store = lo;
  for (int j = lo; j < hi; ++j) {
    if (a[j] < pivot) {
      std::swap(a[store++], a[j]);
    }
  }
  std::swap(a[store], a[hi]);
  quick_sort(a, lo, store - 1);
  quick_sort(a, store + 1, hi);
}

void heap_sort(int a[], int n) {
  auto sift = [&](int i, int heapSize) {
    while (2*i + 1 < heapSize) {
      int j = 2*i + 1;
      if (j+1 < heapSize && a[j] < a[j+1]) ++j;
      if (a[i] >= a[j]) break;
      std::swap(a[i], a[j]);
      i = j;
    }
  };
  for (int i = n/2-1; i >= 0; --i)
    sift(i, n);
  for (int heapSize = n-1; heapSize > 0; --heapSize) {
    std::swap(a[0], a[heapSize]);
    sift(0, heapSize);
  }
}

void shell_sort(int a[], int n) {
  for (int gap = n/2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      for (int j = i; j >= gap && a[j-gap] > a[j]; j -= gap)
        std::swap(a[j-gap], a[j]);
    }
  }
}

void comb_sort(int a[], int n) {
  int gap = n;
  bool swapped = true;
  while (gap > 1 || swapped) {
    gap = std::max(1, gap * 10 / 13);
    swapped = false;
    for (int i = 0; i + gap < n; ++i) {
      int j = i + gap;
      if (a[i] > a[j]) {
        std::swap(a[i], a[j]);
        swapped = true;
      }
    }
  }
}

void counting_sort(int a[], size_t n) {
  int count[100] = {};
  for (size_t i = 0; i < n; ++i)
    ++count[a[i]];
  size_t out = 0;
  for (int value = 1; value < 100; ++value)
    while (count[value]-- > 0)
      a[out++] = value;
}

int main() {
  // Counting sort's current template supports values 1..99 only.
  const std::vector<std::vector<int>> cases = {
    {5, 2, 8, 1, 7, 3, 6, 4}, {9, 1, 9, 2, 2, 99, 1},
    {1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}, {42}, {}
  };
  const std::vector<std::vector<int>> answers = {
    {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 2, 2, 9, 9, 99},
    {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {42}, {}
  };
  int passed = 0;

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    bubble_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: bubble\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: bubble (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    selection_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: selection\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: selection (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    insertion_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: insertion\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: insertion (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    merge_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: merge\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: merge (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    quick_sort(a.data(), 0, static_cast<int>(a.size()) - 1);
    if (a != expected) { std::cerr << "FAIL: quick\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: quick (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    heap_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: heap\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: heap (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    shell_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: shell\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: shell (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    comb_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: comb\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: comb (6 cases)\n";

  for (size_t c = 0; c < cases.size(); ++c) {
    auto a = cases[c];
    const auto& expected = answers[c];
    counting_sort(a.data(), a.size());
    if (a != expected) { std::cerr << "FAIL: counting\n"; return 1; }
    ++passed;
  }
  std::cout << "PASS: counting (6 cases)\n";
  std::cout << "TOTAL: " << passed << "/54 PASS\n";
  return 0;
}
