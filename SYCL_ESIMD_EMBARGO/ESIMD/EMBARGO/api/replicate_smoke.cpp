//==------- replicate_smoke.cpp  - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: TEMPORARY_DISABLED
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks main functionality of the esimd::replicate_vs_w_hs function.

// Temporarily disable while the failure is being investigated.
// UNSUPPORTED: windows
//
// TODO: Reenable the test once HW supporting hf8 and bf8 is available

#define SKIP_MAIN
#include "../../../../SYCL/ESIMD/api/replicate_smoke.cpp"

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

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
