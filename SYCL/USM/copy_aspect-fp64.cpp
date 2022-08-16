//==---- copy_aspect-fp64.cp - USM copy test ------------------------------------------==//
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

#include "copy.hpp";

using namespace sycl;
using namespace sycl::usm;

int main() {
  queue q;

  if (!q.get_device().has(aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  auto dev = q.get_device();

  if (dev.has(aspect::usm_host_allocations)) {
    runTests<double>(q, 4.24242, alloc::host, alloc::host);
  }

  if (dev.has(aspect::usm_shared_allocations)) {
    runTests<double>(q, 4.24242, alloc::shared, alloc::shared);
  }

  if (dev.has(aspect::usm_device_allocations)) {
    runTests<double>(q, 4.24242, alloc::device, alloc::device);
  }

  if (dev.has(aspect::usm_host_allocations) &&
      dev.has(aspect::usm_shared_allocations)) {
    runTests<double>(q, 4.24242, alloc::host, alloc::shared);
  }

  if (dev.has(aspect::usm_host_allocations) &&
      dev.has(aspect::usm_device_allocations)) {
    runTests<double>(q, 4.24242, alloc::host, alloc::device);
  }

  if (dev.has(aspect::usm_shared_allocations) &&
      dev.has(aspect::usm_device_allocations)) {
    runTests<double>(q, 4.24242, alloc::shared, alloc::device);
  }

  return 0;
}
