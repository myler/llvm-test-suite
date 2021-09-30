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
//   4 threads: expected to be executed in strict order (0, 1, 2, 3)
//   3 barriers
//   sequential overlaping stores to output buffer

constexpr unsigned Groups = 1;
constexpr unsigned Threads = 4;

constexpr unsigned Size = 32;
constexpr unsigned VL = Size / (2 * Threads);

constexpr unsigned bnum = Threads - 1;

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

  // we use producer-consumer mode
  int flag = 0;
  // each barrier is going to be signalled and waited on by two threads
  int producers = 2;
  int consumers = 2;

  if (idx > 0) {
    int barrier_id = idx - 1;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  }

  simd<int, VL * 2> val(idx);
  lsc_surf_store<int, VL * 2>(val, acc, off);

  if (idx < bnum) {
    int barrier_id = idx;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  }
}

bool check(std::vector<int> out) {
  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = i / 4;
    if (etalon * 4 == Size / 2)
      etalon -= 1;
    if (etalon * 4 > Size / 2)
      etalon = 0;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}

#include "Inputs/common.hpp"
