// See https://github.com/intel/llvm/issues/2922 for more info
// UNSUPPORTED: cuda || hip
// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DENABLE_FP64 %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==------------- info.cpp - SYCL sub_group parameters test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "info.cpp"
