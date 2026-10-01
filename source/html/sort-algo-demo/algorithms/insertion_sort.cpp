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
