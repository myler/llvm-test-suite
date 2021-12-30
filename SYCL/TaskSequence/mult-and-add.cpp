//==---------------- mult-and-add.cpp - DPC++ task_sequence ----------------==//
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

using namespace sycl;

int mult(int a, int b) { return a * b; }

int mult_and_add(int a1, int b1, int a2, int b2) {
  ext::intel::experimental::task_sequence<mult, 1, 1> task1, task2;

  task1.async(a1, b1);
  task2.async(a2, b2);
  return task1.get() + task2.get();
}

int main() {
  sycl::queue myQueue;

  int result = 0;
  sycl::buffer<int, 1> res_buf(&result, sycl::range<1>(1));

  myQueue.submit([&](handler &cgh) {
    auto res_acc = res_buf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task([=](kernel_handler kh) {
      res_acc[0] = mult_and_add(1, 2, 3, 4);
    });
  });
  myQueue.wait();

  // Check result:
  if (result == 1 * 2 + 3 * 4)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
}
