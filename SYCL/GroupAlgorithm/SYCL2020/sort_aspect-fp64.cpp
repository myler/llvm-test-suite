// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "sort.hpp"

namespace oneapi_exp = sycl::ext::oneapi::experimental;

int main(int argc, char *argv[]) {
  sycl::queue q(sycl::default_selector{}, async_handler_);

  if (!q.get_device().has(sycl::aspect::fp64) {
    std::cout << "Skipping test\n";
    return 0;
  }

  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  std::vector<int> sizes{1, 12, 32};

  for (int i = 0; i < sizes.size(); ++i) {
    test_sort_by_type<std::int8_t>(q, sizes[i]);
    test_sort_by_type<std::uint16_t>(q, sizes[i]);
    test_sort_by_type<std::int32_t>(q, sizes[i]);
    test_sort_by_type<std::uint32_t>(q, sizes[i]);
    test_sort_by_type<float>(q, sizes[i]);
    test_sort_by_type<sycl::half>(q, sizes[i]);
    test_sort_by_type<std::size_t>(q, sizes[i]);

    test_custom_type(q, sizes[i]);
  }
  std::cout << "Test passed." << std::endl;
}
