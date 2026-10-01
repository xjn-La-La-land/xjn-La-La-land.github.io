void bubble_sort(int a[], size_t n) {
  for (size_t i = 0; i + 1 < n; ++i) {
    bool swapped = false;
    for (size_t j = 0; j + 1 < n-i; ++j) {
      if (a[j] > a[j+1]) {
        std::swap(a[j], a[j+1]);
        swapped = true;
      }
    }
    if (!swapped) break;
  }
}
