/*========================== begin_copyright_notice ============================
INTEL CONFIDENTIAL
Copyright (C) 2018-2021 Intel Corporation
This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.
This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.
============================= end_copyright_notice ===========================*/
// TODO enable this test on PVC fullsim when named barriers patch is merged
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx -fsycl %s -o %t.out
// RUNx: %ESIMD_RUN_PLACEHOLDER %t.out
//
// Test checks support of named barrier in ESIMD kernel:
//   2 workgroups
//   2 threads: 1 producer per workgroup, 1 consumer per workgroup
//   1 barrier

constexpr unsigned bnum = 2;
constexpr unsigned barrier = 1;

constexpr unsigned Groups = 2;
constexpr unsigned Threads = 2;

// 1 producer and 1 consumer per workgroup
constexpr unsigned producers = 1;
constexpr unsigned consumers = 1;
constexpr unsigned NUM = Threads * Groups;

constexpr unsigned VL = 4;
constexpr unsigned Size = VL * NUM; // 16; 8 ints per workgroup

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  esimd_nbarrier_init<bnum>();

  unsigned int localID = ndi.get_local_id(0);
  unsigned int groupID = ndi.get_group(0);
  unsigned int globalID = ndi.get_global_id(0);
  unsigned int groupSize = ndi.get_local_range(0);
  unsigned int group_off = VL * groupID * groupSize * sizeof(int);
  unsigned int global_off = VL * globalID * sizeof(int);

  slm_init(VL * NUM * sizeof(int));
  slm_block_store(global_off, simd<int, VL>(0));
  esimd_barrier();

  bool is_producer = localID == 1;
  bool is_consumer = !is_producer;
  unsigned int flag = is_producer ? 0x1 : 0x2;
  int v = 0xdead0000 | (groupID << 8) | localID;

  if (is_producer) {
    slm_block_store(group_off, simd<int, Size / 2>(v));
  }

  esimd_nbarrier_signal(barrier, flag, producers, consumers);

  if (is_consumer) {
    esimd_nbarrier_wait(barrier);
    auto ret = slm_block_load<int, Size / 2>(group_off);
    lsc_surf_store<int, Size / 2>(ret, acc, group_off);
  }
}

bool check(std::vector<int> out) {
  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = (i < Size / 2) ? 0xdead0001 : 0xdead0101;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}

#include "Inputs/common.hpp"
