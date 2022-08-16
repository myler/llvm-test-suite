//==- simd_view_select_2d_fp_aspect-fp64.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, aspect-fp64
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'single_task()' method
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Smoke test for 2D region select API which can be used to represent 2D tiles.
// Tests FP types.

#include "simd_view_select_2d.hpp"

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  if (!q.get_device().has(sycl::aspect::fp64) {
    std::cout << "Skipping test\n";
    return 0;
  }

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;

  passed &= test<double>(q);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
