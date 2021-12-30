//==------------------ user-sot.cpp - DPC++ task_sequence ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aoc, accelerator
// TODO: As the -fintelfpga flag will implicitly enable -g, this test will hang
// on finishing queue. Need to replace the -fsycl-targets=spir64_fpga flag with
// the -fintelfpga flag when the
// https://jira.devtools.intel.com/browse/CMPLRLLVM-32751 JR will be resolved.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include "sycl/ext/intel/experimental/task_sequence.hpp"

using namespace cl::sycl;

constexpr int NSIZE = 128;

int dot_product(int *a, int *b, int N) {
  int total = 0;
  for (int i = 0; i < N; ++i) {
    total += a[i] * b[i];
  }
  return total;
}

int lib_sot(int* data1, int* data2, int N) {
  ext::intel::experimental::task_sequence<dot_product, 10, 10> first_half;
  ext::intel::experimental::task_sequence<dot_product, 10, 10> second_half;

  first_half.async(data1, data2, N/2);
  second_half.async(&(data1[N/2]), &(data2[N/2]), N/2);
  return first_half.get() + second_half.get();
}

int user_sot(int* data1, int* data2, int N) {
  ext::intel::experimental::task_sequence<lib_sot, 10, 10> unique_task1;
  ext::intel::experimental::task_sequence<lib_sot, 10, 10> unique_task2;

  unique_task1.async(data1, data2, N/2);
  unique_task2.async(&(data1[N/2]), &(data2[N/2]), N/2);

  return unique_task1.get() + unique_task2.get();
}

int main () {
  sycl::queue myQueue;

  int result = 0;
  sycl::buffer<int, 1> res_buf(&result, sycl::range<1>(1));

  myQueue.submit([&](sycl::handler& cgh) {
    auto res_acc = res_buf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task([=](kernel_handler kh) {
      int d1[NSIZE], d2[NSIZE];
      for (int i = 0; i < NSIZE; ++i)
        d1[i] = d2[i] = i;
      ext::intel::experimental::task_sequence<user_sot, 10, 10> sot_object;
      sot_object.async(d1, d2, NSIZE);
      res_acc[0] = sot_object.get();
    });
  });
  myQueue.wait();

  // Check result:
  int sum = 0;
  for (int i = 0; i < NSIZE; ++i)
    sum += i * i;
  if (result == sum)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
}

