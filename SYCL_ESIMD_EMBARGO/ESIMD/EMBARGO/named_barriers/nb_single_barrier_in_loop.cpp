//==--- nb_single_barrier_in_loop.cpp.cpp - DPC++ ESIMD on-device test ----===//
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
// Test checks support of named barrier in a loop in ESIMD kernel.
// SLM and surface size is 32 bytes, 16 bytes per iteration.
// Each iteration has 1 barrier and 1 producer. Producer stores data to SLM,
// then all threads read SLM and store data to surface.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

#include "Inputs/single_barrier_in_loop.hpp"

int main() {
  bool passed = true;

  passed &= test<1, 1, 8, 64>();

  return passed ? 0 : 1;
}
