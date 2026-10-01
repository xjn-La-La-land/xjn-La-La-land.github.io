void selection_sort(int a[], size_t n) {
  for (size_t i = 0; i + 1 < n; ++i) {
    size_t minIndex = i;
    for (size_t j = i + 1; j < n; ++j) {
      if (a[j] < a[minIndex]) {
        minIndex = j;
      }
    }
    if (minIndex == i) continue;
    std::swap(a[i], a[minIndex]);
  }
}
