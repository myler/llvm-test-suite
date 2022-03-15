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
using T = uint32_t;

int main(void) {
  srand(seed);
  bool passed = true;

  // These parameters require unpadding. It is not implemented yet
  // passed &= test<0, T, 2, 2, 2, 2>(16, 4, 16, 1, 1);

  // non transposed, non transformed
  passed &= test<1, T, 1, 1, 16, 4>(16, 16, 32, 2, 1);
  passed &= test<2, T, 2, 2, 8, 4>(16, 16, 16, 1, 5);
  passed &= test<3, T, 1, 1, 8, 2, 2>(16, 4, 16, 3, 1);

  // transposed
  passed &= test<4, T, 2, 2, 1, 16, 1, true>(16, 20, 16, 1, 2);
  passed &= test<5, T, 1, 1, 2, 8, 1, true>(12, 10, 16, 10, 1);
  passed &= test<6, T, 1, 1, 4, 8, 1, true>(16, 10, 24, 11, 1);
  passed &= test<7, T, 1, 1, 3, 8, 1, true>(16, 10, 20, 11, 1);
  passed &= test<8, T, 1, 1, 8, 2, 1, true>(16, 6, 32, 10, 3);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
