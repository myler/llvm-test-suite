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

#include "Inputs/lsc_surf_load.hpp"

constexpr uint32_t seed = 198;
constexpr lsc_data_size DS = lsc_data_size::u64;

constexpr CacheHint L1H = CacheHint::Uncached;
constexpr CacheHint L3H = CacheHint::Uncached;

int main(void) {
  srand(seed);
  bool passed = true;

  // non transpose
  passed &= test<0, uint64_t, 1, 4, 32, 1, false, DS, L1H, L3H, true>(rand());
  passed &= test<1, uint64_t, 1, 4, 32, 2, false, DS, L1H, L3H, true>(rand());
  passed &= test<2, uint64_t, 1, 4, 16, 2, false, DS, L1H, L3H, true>(rand());
  passed &= test<3, uint64_t, 1, 4, 4, 1, false, DS, L1H, L3H, true>(rand());
  passed &= test<4, uint64_t, 1, 1, 1, 1, false, DS, L1H, L3H, true>(1);
  passed &= test<5, uint64_t, 2, 1, 1, 1, false, DS, L1H, L3H, true>(1);

  // transpose
  passed &= test<8, uint64_t, 1, 4, 1, 32, true, DS, L1H, L3H, true>();
  passed &= test<9, uint64_t, 2, 2, 1, 16, true, DS, L1H, L3H, true>();
  passed &= test<10, uint64_t, 4, 4, 1, 4, true, DS, L1H, L3H, true>();

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
