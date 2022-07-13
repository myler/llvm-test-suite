//==------ sum-of-products-more-parallelism.cpp - DPC++ task_sequence ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (aoc || accelerator)
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include "sycl/ext/intel/experimental/task_sequence.hpp"

using namespace cl::sycl;

constexpr int NSIZE = 128;

int mult(int a, int b) {
  return a * b;
}

int sum_of_products_more_parallelism(int *data1, int *data2) {
  ext::intel::experimental::task_sequence<mult, NSIZE, NSIZE> task1, task2;
  // Launch the tasks
  for (int i = 0; i < NSIZE; i += 2) {
    task1.async(data1[i], data2[i]);
    task2.async(data1[i + 1], data2[i + 1]);
  }

  // Collect the results
  int total = 0;
  for (int i = 0; i < (NSIZE >> 1); ++i) {
    total += task1.get() + task2.get();
  }
  return total;
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
      ext::intel::experimental::task_sequence<sum_of_products_more_parallelism,
                                              NSIZE, NSIZE> sot_object;
      sot_object.async(d1, d2);
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
