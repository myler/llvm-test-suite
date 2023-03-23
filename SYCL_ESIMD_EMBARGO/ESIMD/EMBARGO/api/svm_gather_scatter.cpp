//==---- svm_gather_scatter.cpp  - DPC++ ESIMD svm gather/scatter test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: TEMPORARY_DISABLED
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl-device-code-split=per_kernel -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Regression test for SVM gather/scatter API.
//
// TODO: Reenable the test once HW supporting hf8 and bf8 is available

#define SKIP_MAIN
#include "../../../../SYCL/ESIMD/api/svm_gather_scatter.cpp"

using hf8 = sycl::ext::intel::experimental::esimd::hf8;
using bf8 = sycl::ext::intel::experimental::esimd::bf8;
namespace esimd_test {
TID(sycl::ext::intel::experimental::esimd::hf8)
TID(sycl::ext::intel::experimental::esimd::bf8)
} // namespace esimd_test

int main(void) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>()
            << "\n";

  bool Pass = true;

  Pass &= test<hf8, 1>(Q);
  Pass &= test<bf8, 1>(Q);
  Pass &= test<hf8, 2>(Q);
  Pass &= test<hf8, 4>(Q);
  Pass &= test<hf8, 8>(Q);
  Pass &= test<hf8, 16>(Q);
  Pass &= test<hf8, 32>(Q);
  Pass &= test<bf8, 2>(Q);
  Pass &= test<bf8, 4>(Q);
  Pass &= test<bf8, 8>(Q);
  Pass &= test<bf8, 16>(Q);
  Pass &= test<bf8, 32>(Q);

  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
