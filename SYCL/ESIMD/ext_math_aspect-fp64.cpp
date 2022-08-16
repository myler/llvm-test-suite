//==----- ext_math_aspect-fp64.cpp  - DPC++ ESIMD extended math test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, aspect-fp64
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'half' type
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks extended math operations. Combinations of
// - argument type - half, float
// - math function - sin, cos, ..., div_ieee, pow
// - SYCL vs ESIMD APIs

#include "ext_math.hpp"

using namespace cl::sycl;
using namespace sycl::ext::intel;

// --- The entry point

int main(void) {
  queue Q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  if (!Q.get_device().has(sycl::aspect::fp64) {
    std::cout << "Skipping test\n";
    return 0;
  }
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << "\n";
  bool Pass = true;

  // Not support IEEE-conformant sqrt operations for single precision data.
  Pass &= testESIMDSqrtIEEE<float, 16>(Q);
  Pass &= testESIMDDivIEEE<float, 8>(Q);

  Pass &= testESIMDSqrtIEEE<double, 32>(Q);
  Pass &= testESIMDDivIEEE<double, 32>(Q);
  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
