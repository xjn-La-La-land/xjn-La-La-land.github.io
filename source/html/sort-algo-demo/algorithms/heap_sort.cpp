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
