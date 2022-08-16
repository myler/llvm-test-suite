//==---- copy.cpp - USM copy test ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

#include "copy.hpp";

using namespace sycl;
using namespace sycl::usm;

struct test_struct {
  short a;
  int b;
  long c;
  long long d;
  half e;
  float f;
};

bool operator==(const test_struct &lhs, const test_struct &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d &&
         lhs.e == rhs.e && lhs.f == rhs.f;
}

int main() {
  queue q;
  auto dev = q.get_device();

  test_struct test_obj{4, 42, 424, 4242, 4.2f, 4.242f};

  if (dev.has(aspect::usm_host_allocations)) {
    runTests<short>(q, 4, alloc::host, alloc::host);
    runTests<int>(q, 42, alloc::host, alloc::host);
    runTests<long>(q, 424, alloc::host, alloc::host);
    runTests<long long>(q, 4242, alloc::host, alloc::host);
    runTests<half>(q, half(4.2f), alloc::host, alloc::host);
    runTests<float>(q, 4.242f, alloc::host, alloc::host);
    runTests<test_struct>(q, test_obj, alloc::host, alloc::host);
  }

  if (dev.has(aspect::usm_shared_allocations)) {
    runTests<short>(q, 4, alloc::shared, alloc::shared);
    runTests<int>(q, 42, alloc::shared, alloc::shared);
    runTests<long>(q, 424, alloc::shared, alloc::shared);
    runTests<long long>(q, 4242, alloc::shared, alloc::shared);
    runTests<half>(q, half(4.2f), alloc::shared, alloc::shared);
    runTests<float>(q, 4.242f, alloc::shared, alloc::shared);
    runTests<test_struct>(q, test_obj, alloc::shared, alloc::shared);
  }

  if (dev.has(aspect::usm_device_allocations)) {
    runTests<short>(q, 4, alloc::device, alloc::device);
    runTests<int>(q, 42, alloc::device, alloc::device);
    runTests<long>(q, 424, alloc::device, alloc::device);
    runTests<long long>(q, 4242, alloc::device, alloc::device);
    runTests<half>(q, half(4.2f), alloc::device, alloc::device);
    runTests<float>(q, 4.242f, alloc::device, alloc::device);
    runTests<test_struct>(q, test_obj, alloc::device, alloc::device);
  }

  if (dev.has(aspect::usm_host_allocations) &&
      dev.has(aspect::usm_shared_allocations)) {
    runTests<short>(q, 4, alloc::host, alloc::shared);
    runTests<int>(q, 42, alloc::host, alloc::shared);
    runTests<long>(q, 424, alloc::host, alloc::shared);
    runTests<long long>(q, 4242, alloc::host, alloc::shared);
    runTests<half>(q, half(4.2f), alloc::host, alloc::shared);
    runTests<float>(q, 4.242f, alloc::host, alloc::shared);
    runTests<test_struct>(q, test_obj, alloc::host, alloc::shared);
  }

  if (dev.has(aspect::usm_host_allocations) &&
      dev.has(aspect::usm_device_allocations)) {
    runTests<short>(q, 4, alloc::host, alloc::device);
    runTests<int>(q, 42, alloc::host, alloc::device);
    runTests<long>(q, 424, alloc::host, alloc::device);
    runTests<long long>(q, 4242, alloc::host, alloc::device);
    runTests<half>(q, half(4.2f), alloc::host, alloc::device);
    runTests<float>(q, 4.242f, alloc::host, alloc::device);
    runTests<test_struct>(q, test_obj, alloc::host, alloc::device);
  }

  if (dev.has(aspect::usm_shared_allocations) &&
      dev.has(aspect::usm_device_allocations)) {
    runTests<short>(q, 4, alloc::shared, alloc::device);
    runTests<int>(q, 42, alloc::shared, alloc::device);
    runTests<long>(q, 424, alloc::shared, alloc::device);
    runTests<long long>(q, 4242, alloc::shared, alloc::device);
    runTests<half>(q, half(4.2f), alloc::shared, alloc::device);
    runTests<float>(q, 4.242f, alloc::shared, alloc::device);
    runTests<test_struct>(q, test_obj, alloc::shared, alloc::device);
  }

  return 0;
}
