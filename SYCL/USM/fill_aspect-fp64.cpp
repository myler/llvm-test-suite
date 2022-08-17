//==---- fill_aspect-fp64.cpp - USM fill test for double type
//---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

#include "fill.hpp";

using namespace sycl;

int main() {
  queue q;

  if (!q.get_device().has(aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (dev.get_info<info::device::usm_host_allocations>()) {
    runHostTests<double>(dev, ctxt, q, 4.24242);
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    runHostTests<double>(dev, ctxt, q, 4.24242);
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    runHostTests<double>(dev, ctxt, q, 4.24242);
  }

  return 0;
}
