//==------- saturation_smoke.cpp  - DPC++ ESIMD on-device test -------------==//
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
//
// The test checks main functionality of esimd::saturate function.

#include "saturation_smoke.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// clang-format on

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<half, int, FpToInt>(q);
  passed &= test<half, unsigned char, FpToInt>(q);
  passed &= test<float, int, FpToInt>(q);

  passed &= test<unsigned char, char, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned short, short, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned int, int, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned int, char, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned short, unsigned char, UIntToSameOrNarrowAnyInt>(q);

  passed &= test<char, unsigned int, IntToWiderUInt>(q);
  passed &= test<char, unsigned short, IntToWiderUInt>(q);
  passed &= test<short, unsigned int, IntToWiderUInt>(q);

  passed &= test<short, char, SIntToNarrowAnyInt>(q);
  passed &= test<int, unsigned char, SIntToNarrowAnyInt>(q);

  passed &= test<float, float, FpToFp>(q);
  passed &= test<half, half, FpToFp>(q);

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
