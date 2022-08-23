// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API -DENABLE_FP64 %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// Specialization constants are not supported on FPGA h/w and emulator.
// UNSUPPORTED: cuda || hip
//
//==----------- specialization_constants_override.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Checks that set_spec_constant can be used twice on the same program

#include "specialization_constants_override.cpp"
