// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_SubgroupBlockReadINTEL, __spirv_SubgroupBlockWriteINTEL on
// AMD
// XFAIL: hip_amd
//
//==----- load_store_aspect-fp64.cpp - SYCL sub_group load/store test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "load_store.hpp"

using namespace sycl;

int main() {
  queue Queue;
  if (Queue.get_device().is_host() or !Queue.get_device().has(sycl::aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }
  std::string PlatformName =
      Queue.get_device().get_platform().get_info<info::platform::name>();
  auto Vec = Queue.get_device().get_info<info::device::extensions>();
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups_long") !=
          std::end(Vec) ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef double aligned_double __attribute__((aligned(16)));
    check<aligned_double>(Queue);
    check<aligned_double, 1>(Queue);
    check<aligned_double, 2>(Queue);
    check<aligned_double, 3>(Queue);
    check<aligned_double, 4>(Queue);
    check<aligned_double, 8>(Queue);
    check<aligned_double, 16>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
