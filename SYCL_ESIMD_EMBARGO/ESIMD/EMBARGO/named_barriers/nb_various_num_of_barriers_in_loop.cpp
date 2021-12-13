//=- nb_various_num_of_barriers_in_loop.cpp.cpp - DPC++ ESIMD on-device test =//
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
// First iteration has 1 barrier and 1 producer, second - 2 barriers and 2
// producers. Producer stores data to SLM, then all threads read SLM and store
// data to surface.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

#include "Inputs/common.hpp"
#include "Inputs/various_num_of_barriers_in_loop.hpp"

int main() { return test<1, 8, 64>(); }
