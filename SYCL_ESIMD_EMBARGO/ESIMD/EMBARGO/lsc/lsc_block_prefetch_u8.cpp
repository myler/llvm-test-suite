/*========================== begin_copyright_notice ============================
INTEL CONFIDENTIAL
Copyright (C) 2018-2021 Intel Corporation
This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.
This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.
============================= end_copyright_notice ===========================*/
// TODO enable this test on PVC fullsim when LSC patch is merged
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx -fsycl %s -o %t.out
// RUNx: %ESIMD_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_block_load.hpp"

constexpr uint32_t seed = 322;
using T = uint8_t;

constexpr CacheHint L1H = CacheHint::Uncached;
constexpr CacheHint L3H = CacheHint::Uncached;

int main(void) {
  srand(seed);
  bool passed = true;

  // These parameters require unpadding. It is not implemented yet
  // passed &= test<0, T, 2, 2, 2, 2>(16, 4, 16, 1, 1);

  // non transposed, non transformed
  passed &= test<1, T, 1, 1, 16, 32, 2, false, false, L1H, L3H, true>(
      40, 64, 64, 4, 21);
  passed &=
      test<2, T, 2, 2, 8, 8, 2, false, false, L1H, L3H, true>(16, 16, 64, 8, 5);
  passed &= test<3, T, 1, 1, 8, 32, 2, false, false, L1H, L3H, true>(16, 80, 64,
                                                                     4, 1);

  // transformed
  passed &= test<4, T, 1, 1, 16, 4, 4, false, true, L1H, L3H, true>(100, 10,
                                                                    128, 16, 5);
  passed &= test<5, T, 1, 1, 12, 20, 1, false, true, L1H, L3H, true>(16, 40, 64,
                                                                     0, 0);
  passed &=
      test<6, T, 1, 1, 16, 4, 2, false, true, L1H, L3H, true>(32, 4, 64, 4, 1);
  passed &=
      test<7, T, 2, 2, 4, 16, 2, false, true, L1H, L3H, true>(4, 20, 64, 0, 3);
  passed &= test<8, T, 1, 1, 16, 32, 1, false, true, L1H, L3H, true>(24, 80, 64,
                                                                     4, 14);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
