//==--------------- bin_un_cmp_ops_heavy_aspect-fp64.cpp  - DPC++ ESIMD
// on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-fp64, gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'half' type
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests various binary operations applied to simd objects.

// TODO
// Arithmetic operations behaviour depends on Gen's control regiter's rounding
// mode, which is RTNE by default:
//    cr0.5:4 is 00b = Round to Nearest or Even (RTNE)
// For half this leads to divergence between Gen and host (emulated) results
// larger than certain threshold. Might need to tune the cr0 once this feature
// is available in ESIMD.
//

#include "bin_and_cmp_ops_heavy.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  if (!q.get_device().has(aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  bool passed = true;
  using BinOp = esimd_test::BinaryOp;

  auto arith_ops = esimd_test::ArithBinaryOps;
  passed &= test<short, double, 7, BinOp, VSf, IDf>(arith_ops, q);
  passed &= test<double, half, 7, BinOp, VSf, IDf>(arith_ops, q);

  using CmpOp = esimd_test::CmpOp;
  auto cmp_ops = esimd_test::CmpOps;
  passed &= test<short, double, 7, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<double, half, 7, CmpOp, VSf, IDf>(cmp_ops, q);

  std::cout << (passed ? "Test PASSED\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
