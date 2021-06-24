#include <stdlib.h>

template <typename T> T get_rand() {
  T v = rand();
  if constexpr (sizeof(T) > 4)
    v = (v << 32) | rand();
  return v;
}
