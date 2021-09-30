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
//   8 threads: 2 different producers - 1 on each iteration
//   2 barriers
//   2 sequential overlaping stores in a loop by producers to SLM

constexpr unsigned Groups = 1;
constexpr unsigned Threads = 8;

constexpr unsigned Size = 32;
constexpr unsigned VL = Size / Threads;

constexpr unsigned bnum = 3;

constexpr unsigned producers = 1;
constexpr unsigned consumers = Threads;

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

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  esimd_barrier();

  for (int b = 1; b < bnum; b++) {
    bool is_producer = idx == b;
    bool is_consumer = !is_producer;
    unsigned int flag = is_producer ? 0x0 : 0x2; // producer is also a consumer

    unsigned int b_off = (b - 1) * sizeof(int) * Size / 4;

    if (is_producer) {
      int v = 0xdead0000 + (b - 1) * Size / 2;
      simd<int, Size / 2> init(v, 1);
      slm_block_store(b_off, init);
    }

    esimd_nbarrier_signal(b, flag, producers, consumers);
    esimd_nbarrier_wait(b);

    auto val = slm_block_load<int, VL>(off + b_off);
    lsc_surf_store<int, VL>(val, acc, off + b_off);
  }
}

bool check(std::vector<int> out) {
  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 0;
    if (i < 3 * Size / 4)
      etalon = 0xdead0000 + (i < Size / 4 ? i : i + Size / 4);
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}

#include "Inputs/common.hpp"
