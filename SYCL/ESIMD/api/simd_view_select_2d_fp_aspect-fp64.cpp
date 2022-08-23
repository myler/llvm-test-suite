//==- simd_view_select_2d_fp_aspect-fp64.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, aspect-fp64
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'single_task()' method
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl -DENABLE_FP64 %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Smoke test for 2D region select API which can be used to represent 2D tiles.
// Tests FP types.

#include "simd_view_select_2d.cpp"
