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
