//==-------------- consumer-producer.cpp - DPC++ task_sequence -------------==//
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

using inter_pipe = pipe<class my_sot, int>;

void produce_data(int *a, int *b, int N) {
  for (int i = 0; i < N; ++i)
    inter_pipe::write(a[i] * b[i]);
}
int consume_data(int N) {
  int total = 0;
  for (int i = 0; i < N; ++i)
    total += (inter_pipe::read() + 42);
  return total;
}
int my_sot(int *data1, int *data2, int N) {
  ext::intel::experimental::task_sequence<produce_data, 1, 1> producer;
  ext::intel::experimental::task_sequence<consume_data, 1, 1> consumer;

  producer.async(data1, data2, N);
  consumer.async(N);
  return consumer.get();
}

int main () {
  sycl::queue myQueue;

  int result = 0;
  sycl::buffer<int, 1> res_buf(&result, sycl::range<1>(1));

  myQueue.submit([&](handler& cgh) {
    auto res_acc = res_buf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task([=](kernel_handler kh) {
      int d1[NSIZE], d2[NSIZE];
      for (int i = 0; i < NSIZE; ++i)
        d1[i] = d2[i] = i;
      ext::intel::experimental::task_sequence<my_sot, 1, 1> sot_object;
      sot_object.async(d1, d2, NSIZE);
      res_acc[0] = sot_object.get();
    });
  });
  myQueue.wait();

  // Check result:
  int sum = 0;
  for (int i = 0; i < NSIZE; ++i)
    sum += i * i + 42;
  if (result == sum)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
}

