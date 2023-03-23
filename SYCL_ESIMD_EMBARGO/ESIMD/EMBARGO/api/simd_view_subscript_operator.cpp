//==----- simd_view_subscript_operator.cpp  - DPC++ ESIMD on-device test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: TEMPORARY DISABLED
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that it's possible to write through the simd_view subscript
// operator. E.g.:
//   simd<int, 4> v = 1;
//   auto v1 = v.select<2, 1>(0);
//   v1[0] = 0; // v1[0] returns writable simd_view
//
// TODO: Reenable the test once HW supporting hf8 and bf8 is available

#define SKIP_MAIN
#include "../../../../SYCL/ESIMD/api/simd_view_subscript_operator.cpp"

using hf8 = sycl::ext::intel::experimental::esimd::hf8;
using bf8 = sycl::ext::intel::experimental::esimd::bf8;
namespace esimd_test {
TID(sycl::ext::intel::experimental::esimd::hf8)
TID(sycl::ext::intel::experimental::esimd::bf8)
} // namespace esimd_test

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;

  passed &= test<hf8>(q);
  passed &= test<bf8>(q);

  std::cout << (passed ? "Test Passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
