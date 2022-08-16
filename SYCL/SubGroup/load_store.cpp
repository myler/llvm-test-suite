// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_SubgroupBlockReadINTEL, __spirv_SubgroupBlockWriteINTEL on
// AMD
// XFAIL: hip_amd
//
//==----------- load_store.cpp - SYCL sub_group load/store test ------------==//
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
  if (Queue.get_device().is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }
  std::string PlatformName =
      Queue.get_device().get_platform().get_info<info::platform::name>();
  auto Vec = Queue.get_device().get_info<info::device::extensions>();
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups") !=
          std::end(Vec) ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef bool aligned_char __attribute__((aligned(16)));
    check<aligned_char>(Queue);
    typedef int aligned_int __attribute__((aligned(16)));
    check<aligned_int>(Queue);
    check<aligned_int, 1>(Queue);
    check<aligned_int, 2>(Queue);
    check<aligned_int, 3>(Queue);
    check<aligned_int, 4>(Queue);
    check<aligned_int, 8>(Queue);
    check<aligned_int, 16>(Queue);
    typedef unsigned int aligned_uint __attribute__((aligned(16)));
    check<aligned_uint>(Queue);
    check<aligned_uint, 1>(Queue);
    check<aligned_uint, 2>(Queue);
    check<aligned_uint, 3>(Queue);
    check<aligned_uint, 4>(Queue);
    check<aligned_uint, 8>(Queue);
    check<aligned_uint, 16>(Queue);
    typedef float aligned_float __attribute__((aligned(16)));
    check<aligned_float>(Queue);
    check<aligned_float, 1>(Queue);
    check<aligned_float, 2>(Queue);
    check<aligned_float, 3>(Queue);
    check<aligned_float, 4>(Queue);
    check<aligned_float, 8>(Queue);
    check<aligned_float, 16>(Queue);
  }
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups_short") !=
          std::end(Vec) ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef short aligned_short __attribute__((aligned(16)));
    check<aligned_short>(Queue);
    check<aligned_short, 1>(Queue);
    check<aligned_short, 2>(Queue);
    check<aligned_short, 3>(Queue);
    check<aligned_short, 4>(Queue);
    check<aligned_short, 8>(Queue);
    check<aligned_short, 16>(Queue);
    if (Queue.get_device().has(sycl::aspect::fp16) ||
        PlatformName.find("CUDA") != std::string::npos) {
      typedef half aligned_half __attribute__((aligned(16)));
      check<aligned_half>(Queue);
      check<aligned_half, 1>(Queue);
      check<aligned_half, 2>(Queue);
      check<aligned_half, 3>(Queue);
      check<aligned_half, 4>(Queue);
      check<aligned_half, 8>(Queue);
      check<aligned_half, 16>(Queue);
    }
  }
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups_long") !=
          std::end(Vec) ||
      PlatformName.find("CUDA") != std::string::npos) {
    typedef long aligned_long __attribute__((aligned(16)));
    check<aligned_long>(Queue);
    check<aligned_long, 1>(Queue);
    check<aligned_long, 2>(Queue);
    check<aligned_long, 3>(Queue);
    check<aligned_long, 4>(Queue);
    check<aligned_long, 8>(Queue);
    check<aligned_long, 16>(Queue);
    typedef unsigned long aligned_ulong __attribute__((aligned(16)));
    check<aligned_ulong>(Queue);
    check<aligned_ulong, 1>(Queue);
    check<aligned_ulong, 2>(Queue);
    check<aligned_ulong, 3>(Queue);
    check<aligned_ulong, 4>(Queue);
    check<aligned_ulong, 8>(Queue);
    check<aligned_ulong, 16>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
