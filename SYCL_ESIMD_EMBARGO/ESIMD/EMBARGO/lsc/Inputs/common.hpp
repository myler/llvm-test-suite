#include <stdlib.h>

template <int case_num> class KernelID;

template <typename T> T get_rand() {
  T v = rand();
  if constexpr (sizeof(T) > 4)
    v = (v << 32) | rand();
  return v;
}
