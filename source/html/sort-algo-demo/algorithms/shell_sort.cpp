void shell_sort(int a[], int n) {
  for (int gap = n/2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      for (int j = i; j >= gap && a[j-gap] > a[j]; j -= gap)
        std::swap(a[j-gap], a[j]);
    }
  }
}
