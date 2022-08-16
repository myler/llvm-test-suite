// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// XFAIL: hip
// Expected failure because hip does not have atomic64 check implementation

#include "assignment.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;
  if (!q.get_device().has(aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }
  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;
  assignment_test<double>(q, N);

  std::cout << "Test passed." << std::endl;
}
