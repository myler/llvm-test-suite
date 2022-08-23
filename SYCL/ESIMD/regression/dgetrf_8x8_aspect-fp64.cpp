//==------- dgetrf_8x8_aspect-fp64.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, aspect-fp64
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -DENABLE_FP64 %s -I%S/.. -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out 1
//
// Reduced version of dgetrf.cpp - M = 8, N = 8, single batch.
//
#include "dgetrf_8x8.cpp"
