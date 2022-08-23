//==------ unary_ops_heavy_aspect-fp64.cpp  - DPC++ ESIMD on-device test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, aspect-fp64
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'half' type
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl -DENABLE_FP64 %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests various unary operations applied to simd objects.

// TODO
// Arithmetic operations behaviour depends on Gen's control regiter's rounding
// mode, which is RTNE by default:
//    cr0.5:4 is 00b = Round to Nearest or Even (RTNE)
// For half this leads to divergence between Gen and host (emulated) results
// larger than certain threshold. Might need to tune the cr0 once this feature
// is available in ESIMD.
//

#include "unary_ops_heavy.cpp"
