//==---------- nb_single_wg.cpp.cpp - DPC++ ESIMD on-device test ----------===//
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
// Basic case with 1 work-group and 16 threads: 4 producer and 12 consumer.
// SLM and surface size is 64 bytes.
// Producers store data to SLM, then all threads read SLM and store data to
// surface.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

#include "Inputs/single_wg.hpp"

int main() {
  bool passed = true;

  passed &= test<1, 4, 12>();
  passed &= test<2, 2, 14>();
  passed &= test<3, 8, 24>();
  passed &= test<4, 2, 2>();

  return passed ? 0 : 1;
}
