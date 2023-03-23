//==-------------- bin_and_cmp_ops_heavy.cpp  - DPC++ ESIMD on-device test -==//
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

// Tests various binary operations applied to simd objects.

// TODO
// Arithmetic operations behaviour depends on Gen's control register's rounding
// mode, which is RTNE by default:
//    cr0.5:4 is 00b = Round to Nearest or Even (RTNE)
// For half and tfloat32 this leads to divergence between Gen and host
// (emulated) results larger than certain threshold. Might need to tune the cr0
// once this feature is available in ESIMD.
//
//
// TODO: Reenable the test once HW supporting hf8 and bf8 is available

#define SKIP_MAIN
#include "../../../../SYCL/ESIMD/api/bin_and_cmp_ops_heavy.cpp"

using hf8 = sycl::ext::intel::experimental::esimd::hf8;
using bf8 = sycl::ext::intel::experimental::esimd::bf8;
namespace esimd_test {
TID(sycl::ext::intel::experimental::esimd::hf8)
TID(sycl::ext::intel::experimental::esimd::bf8)
} // namespace esimd_test

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;
  using BinOp = esimd_test::BinaryOp;

  bool SupportsDouble = dev.has(aspect::fp64);
  bool SupportsHalf = dev.has(aspect::fp16);

  auto arith_ops = esimd_test::ArithBinaryOpsNoDiv;

  passed &= test<hf8, float, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);
  passed &= test<tfloat32, hf8, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);
  if (SupportsHalf)
    passed &= test<hf8, half, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);
  passed &= test<bf8, hf8, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);
  passed &= test<hf8, bf8, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);
  passed &= test<bf8, float, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);
  passed &= test<tfloat32, bf8, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);
  if (SupportsHalf)
    passed &= test<bf8, half, 32, BinOp, VEfa, IDf>(arith_ops, q, 0.001f);

  // Test division separately, as error probability is higher.
  auto div_op = esimd_test::BinaryOpSeq<BinOp::div>{};

  passed &= test<hf8, float, 32, BinOp, VEf, IDf>(div_op, q, 0.001f);
  if (SupportsHalf)
    passed &= test<hf8, half, 32, BinOp, VEf, IDf>(div_op, q, 0.001f);

  passed &= test<hf8, bf8, 32, BinOp, VEf, IDf>(div_op, q, 0.01f);
  passed &= test<bf8, hf8, 32, BinOp, VEf, IDf>(div_op, q, 0.01f);
  passed &= test<bf8, float, 32, BinOp, VEf, IDf>(div_op, q, 0.001f);
  passed &= test<bf8, bf8, 32, BinOp, VEf, IDf>(div_op, q, 0.1f);

  if (SupportsHalf)
    passed &= test<bf8, half, 32, BinOp, VEf, IDf>(div_op, q, 0.001f);

  using CmpOp = esimd_test::CmpOp;
  auto cmp_ops = esimd_test::CmpOps;

  passed &= test<hf8, float, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<hf8, hf8, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<char, hf8, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  if (SupportsHalf)
    passed &= test<hf8, half, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<hf8, unsigned char, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<hf8, short, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<bf8, hf8, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<hf8, bf8, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<bf8, float, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<bf8, bf8, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<char, bf8, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  if (SupportsHalf)
    passed &= test<bf8, half, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<bf8, unsigned char, 32, CmpOp, VSf, IDf>(cmp_ops, q);
  passed &= test<bf8, short, 32, CmpOp, VSf, IDf>(cmp_ops, q);

  std::cout << (passed ? "Test PASSED\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
