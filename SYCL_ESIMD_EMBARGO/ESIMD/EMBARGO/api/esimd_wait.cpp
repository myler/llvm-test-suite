//==------------------- esimd_wait.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Smoke test for the esimd wait API.

#include "../../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>

#include <iostream>

using namespace sycl;

int main() {
  constexpr unsigned Size = 16;
  constexpr unsigned VL = 16;
  constexpr unsigned GroupSize = 1;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  auto ctxt = q.get_context();
  auto *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  auto *B =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  auto *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  for (auto i = 0; i != Size; i++) {
    A[i] = 1.0f;
    B[i] = 3.0f;
  }

  // iteration space
  nd_range<1> Range(range<1>(Size / VL), range<1>(GroupSize));

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(Range, [=](nd_item<1> i) SYCL_ESIMD_KERNEL {
      using namespace __ESIMD_NS;
      using namespace __ESIMD_ENS;

      simd<uint32_t, VL> address(0, 1);
      address = address * sizeof(float);
      simd_mask<VL> pred(0);
      pred[0] = 1;
      simd<float, VL> data =
          lsc_gather<float, 1, lsc_data_size::default_size, cache_hint::cached,
                     cache_hint::cached, VL>(A, address);
      wait(data.bit_cast_view<uint16_t>()[0]);
      simd<float, VL> tmp =
          lsc_gather<float, 1, lsc_data_size::default_size, cache_hint::cached,
                     cache_hint::cached, VL>(B, address);
      wait(tmp.bit_cast_view<uint16_t>()[0]);
      data.merge(tmp, pred);
      lsc_block_store<float, VL, lsc_data_size::default_size,
                      cache_hint::write_back, cache_hint::write_back>(C, data);
    });
  });
  e.wait();

  bool passed = true;
  for (auto i = 0; i != Size; i++) {
    std::cout << " C[" << i << "]:" << C[i] << std::endl;
    if ((i == 0 && C[i] != B[i]) || (i > 0 && C[i] != A[i])) {
      passed = false;
      break;
    }
  }

  free(A, q);
  free(B, q);
  free(C, q);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
