//==-------- nb_multiple_wg.cpp.cpp - DPC++ ESIMD on-device test ----------===//
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
// Basic case with 2 work-groups.
// SLM and surface size is 16 bytes, 8 bytes per group.
// Each work-group contain 2 threads: 1 producer and 1 consumer.
// Producers store to SLM; consumers read SLM and store data to surface.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

#include "Inputs/common.hpp"
#include "Inputs/multiple_wg.hpp"

int main() { return test<2, 2, 16>(); }
