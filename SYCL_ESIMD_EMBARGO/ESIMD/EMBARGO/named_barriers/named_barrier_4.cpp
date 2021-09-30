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
//   8 threads: 2 producers for 1st barrier, 1 producer for 2nd
//   2 barriers
//   several sequential overlaping stores in a loop by different producers to
//   SLM

constexpr unsigned Groups = 1;
constexpr unsigned Threads = 8;

constexpr unsigned Size = 32;
constexpr unsigned VL = Size / Threads;

constexpr unsigned bnum = 3;

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

  unsigned int indexes[2][2] = {{1, 2}, {3, 3}}; // local ids of producers
  unsigned int prods[2] = {2, 1};                // number of producers

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  esimd_barrier();

  for (int b = bnum - 1; b > 0; b--) {
    int j = bnum - b - 1;

    bool is_producer = idx == indexes[j][0] || idx == indexes[j][1];
    bool is_consumer = !is_producer;
    unsigned int flag = is_producer ? 0x1 : 0x2;

    unsigned int producers = prods[j];
    unsigned int consumers = Threads - producers;

    if (is_producer) {
      unsigned int p_off = j * sizeof(int) * Size / 4;
      p_off += (producers == 2 ? (idx - 1) : 0) * sizeof(int) * Size / 2;
      int v = 0xdead0000 + idx;
      simd<int, Size / 2> init(v);
      slm_block_store(p_off, init);
    }

    esimd_nbarrier_signal(b, flag, producers, consumers);

    if (is_consumer)
      esimd_nbarrier_wait(b);

    auto val = slm_block_load<int, VL>(off);
    lsc_surf_store<int, VL>(val, acc, off);
  }
}

bool check(std::vector<int> out) {
  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 0xdead0003;
    if (i < Size / 4)
      etalon = 0xdead0001;
    else if (i >= 3 * Size / 4)
      etalon = 0xdead0002;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}

#include "Inputs/common.hpp"
