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
