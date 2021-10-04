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

#include "Inputs/lsc_slm_store.hpp"

constexpr uint32_t seed = 276;

int main(void) {
  srand(seed);
  bool passed = true;

  // non transpose
  passed &= test<0, uint64_t, 1, 4, 32, 1, false>(rand());
  passed &= test<1, uint64_t, 1, 4, 32, 2, false>(rand());
  passed &= test<2, uint64_t, 1, 4, 16, 2, false>(rand());
  passed &= test<3, uint64_t, 1, 4, 4, 1, false>(rand());
  passed &= test<4, uint64_t, 1, 1, 1, 1, false>(1);
  passed &= test<5, uint64_t, 2, 1, 1, 1, false>(1);
  // passed &= test<6, uint64_t, 1, 4, 8, 2>(rand()); // merge fail
  // passed &= test<7, uint64_t, 1, 4, 8, 3>(rand()); // exec fail

  // transpose
  passed &= test<8, uint64_t, 1, 4, 1, 32, true>();
  passed &= test<9, uint64_t, 2, 2, 1, 16, true>();
  passed &= test<10, uint64_t, 4, 4, 1, 4, true>();

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
