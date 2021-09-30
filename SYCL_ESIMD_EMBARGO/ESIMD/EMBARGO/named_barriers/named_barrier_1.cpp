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
//   1 workgroup
//   16 threads: 4 producers, 12 consumers
//   1 barrier

constexpr unsigned Groups = 1;
constexpr unsigned Threads = 16;

constexpr unsigned Size = 64;
constexpr unsigned VL = Size / Threads;

constexpr unsigned bnum = 2;
constexpr unsigned barrier = 1;

constexpr unsigned producers = 4;
constexpr unsigned consumers = Threads - producers;
constexpr unsigned batch = 1 + (consumers / producers);

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  esimd_nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  bool is_producer = (idx % producers) == 3;
  bool is_consumer = !is_producer;
  unsigned int flag = is_producer ? 0x1 : 0x2;

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  esimd_barrier();

  if (is_producer) {
    unsigned int x = idx - 3;
    unsigned int p_off = x * VL * sizeof(int);
    simd<int, batch * VL> init(0xdead0000 + x * VL, 1);
    slm_block_store(p_off, init);
  }

  esimd_nbarrier_signal(barrier, flag, producers, consumers);

  if (is_consumer)
    esimd_nbarrier_wait(barrier);

  auto val = slm_block_load<int, VL>(off);
  lsc_surf_store<int, VL>(val, acc, off);
}

bool check(std::vector<int> out) {
  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 0xdead0000 + i;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}

#include "Inputs/common.hpp"
