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
