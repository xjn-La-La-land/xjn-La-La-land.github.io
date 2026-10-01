void counting_sort(int a[], size_t n) {
  int count[100] = {};
  for (size_t i = 0; i < n; ++i)
    ++count[a[i]];
  size_t out = 0;
  for (int value = 1; value < 100; ++value)
    while (count[value]-- > 0)
      a[out++] = value;
}
