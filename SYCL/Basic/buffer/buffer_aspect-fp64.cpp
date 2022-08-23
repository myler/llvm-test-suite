// REQUIRES: aspect-fp64
// RUN: %clangxx %cxx_std_optionc++17 -DENABLE_FP64 %s -o %t1.out %sycl_options
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: %HOST_RUN_PLACEHOLDER %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out

//==------------------- buffer.cpp - SYCL buffer basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buffer.cpp"
