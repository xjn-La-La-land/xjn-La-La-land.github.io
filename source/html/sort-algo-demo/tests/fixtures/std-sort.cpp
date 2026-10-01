// Experimental regression: the old wasm-clang demo fails to link this.
#include <algorithm>
#include <cstdio>
int main() {
  int a[] = {3, 1, 2};
  std::sort(a, a + 3);
  for (int v : a) std::printf("%d ", v);
  return 0;
}
