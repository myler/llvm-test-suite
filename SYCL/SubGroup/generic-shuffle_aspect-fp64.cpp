// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//==-- generic_shuffle_aspect-fp64.cpp - SYCL sub_group generic shuffle test *-
//C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "generic-shuffle.hpp"

using namespace sycl;

int main() {
  queue Queue;
  if (Queue.get_device().is_host() or
      !Queue.get_device().has(sycl::aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  auto ComplexDoubleGenerator = [state = std::complex<double>(0, 1)]() mutable {
    return state += std::complex<double>(2, 2);
  };
  check_struct<class KernelName_CjlHUmnuxWtyejZFD, std::complex<double>>(
      Queue, ComplexDoubleGenerator);

  std::cout << "Test passed." << std::endl;
  return 0;
}
