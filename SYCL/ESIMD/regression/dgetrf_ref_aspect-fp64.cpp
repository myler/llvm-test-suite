//==------- dgetrf_ref_aspect-fp64.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Enable FP64 part of <dgetrf_ref.cpp>. To be removed once DPC++
// supports optional device features and the code could be enabled
// unconditionally without causing failures in speculative compilation
// of the kernels.
//
// REQUIRES: gpu, aspect-fp64
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -DUSE_REF -DENABLE_FP64 %s -I%S/.. -o %t.ref.out
// RUN: %GPU_RUN_PLACEHOLDER %t.ref.out 3 2 1
//

#include "Inputs/dgetrf.hpp"
