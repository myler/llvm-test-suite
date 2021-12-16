//==--------- nb_exec_in_order.cpp.cpp - DPC++ ESIMD on-device test -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO enable this test on PVC fullsim when named barriers patch is merged
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx -fsycl %s -o %t.out
// RUNx: %ESIMD_RUN_PLACEHOLDER %t.out
//
// Test checks support of named barrier in ESIMD kernel.
// Threads are executed in ascending order of their local ID and each thread
// stores data to addresses that partially overlap with addresses used by
// previous thread.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

#include "Inputs/exec_in_order.hpp"

int main() {
  bool passed = true;

  passed &= test<1, 4, 8>();
  passed &= test<2, 4, 32>();
  passed &= test<3, 8, 16>();
  passed &= test<4, 8, 32>();
  passed &= test<5, 32, 64>();

  return passed ? 0 : 1;
}
