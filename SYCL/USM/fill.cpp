//==---- fill.cpp - USM fill test ------------------------------------------==//
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

#include "fill.hpp"

using namespace sycl;

struct test_struct {
  short a;
  int b;
  long c;
  long long d;
  sycl::half e;
  float f;
};

bool operator==(const test_struct &lhs, const test_struct &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d &&
         lhs.e == rhs.e && lhs.f == rhs.f;
}


int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  test_struct test_obj{4, 42, 424, 4242, 4.2f, 4.242};

  if (dev.get_info<info::device::usm_host_allocations>()) {
    runHostTests<short>(dev, ctxt, q, 4);
    runHostTests<int>(dev, ctxt, q, 42);
    runHostTests<long>(dev, ctxt, q, 424);
    runHostTests<long long>(dev, ctxt, q, 4242);
    runHostTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runHostTests<float>(dev, ctxt, q, 4.242f);
    runHostTests<test_struct>(dev, ctxt, q, test_obj);
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    runSharedTests<short>(dev, ctxt, q, 4);
    runSharedTests<int>(dev, ctxt, q, 42);
    runSharedTests<long>(dev, ctxt, q, 424);
    runSharedTests<long long>(dev, ctxt, q, 4242);
    runSharedTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runSharedTests<float>(dev, ctxt, q, 4.242f);
    runSharedTests<test_struct>(dev, ctxt, q, test_obj);
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    runDeviceTests<short>(dev, ctxt, q, 4);
    runDeviceTests<int>(dev, ctxt, q, 42);
    runDeviceTests<long>(dev, ctxt, q, 420);
    runDeviceTests<long long>(dev, ctxt, q, 4242);
    runDeviceTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runDeviceTests<float>(dev, ctxt, q, 4.242f);
    runDeviceTests<test_struct>(dev, ctxt, q, test_obj);
  }

  return 0;
}
