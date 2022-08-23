//==---- saturation_smoke_aspect-fp64.cpp  - DPC++ ESIMD on-device test
//-----==//
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
//
// The test checks main functionality of esimd::saturate function.

#include "saturation_smoke.cpp"
