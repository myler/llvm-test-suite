//==---------------- ext_math.cpp  - DPC++ ESIMD extended math test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
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

using namespace sycl;
using namespace sycl::ext::intel;

// --- The entry point

int main(void) {
  queue Q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << "\n";
  bool Pass = true;
  Pass &= testESIMD<half, 8>(Q);
  Pass &= testESIMD<float, 16>(Q);
  Pass &= testESIMD<float, 32>(Q);
  Pass &= testSYCL<float, 8>(Q);
  Pass &= testSYCL<float, 32>(Q);
  Pass &= testESIMDPow<float, 8>(Q);
  Pass &= testESIMDPow<half, 32>(Q);
  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
