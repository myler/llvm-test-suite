//==--- nb_exec_in_order_branched.cpp.cpp - DPC++ ESIMD on-device test ----===//
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
// previous thread. Same as "nb_exec_in_order.cpp", but each thread in separate
// branch

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

#include "Inputs/exec_in_order_branched.hpp"

int main() {
  bool passed = true;

  passed &= test<1, 8>();
  passed &= test<2, 16>();
  passed &= test<3, 32>();

  return passed ? 0 : 1;
}
