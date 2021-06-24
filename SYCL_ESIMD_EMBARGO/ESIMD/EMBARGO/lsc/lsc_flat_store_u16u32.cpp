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

#include "Inputs/lsc_flat_store.hpp"

constexpr uint32_t seed = 286;
constexpr lsc_data_size DS = lsc_data_size::u16u32;

int main(void) {
  srand(seed);
  bool passed = true;

  passed &= test<uint32_t, 1, 4, 32, 1, false, DS>(0, rand());

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
