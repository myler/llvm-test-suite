//==---------------- vadd_usm.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl %s -DESIMD_GEN12_7 -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <iostream>
#include <stdlib.h>

class Test;

#define DTYPE float

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

ESIMD_INLINE void atomic_add_float(nd_item<1> ndi, DTYPE *sA) {
  simd<uint32_t, 16> offsets = {0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  simd<float, 16> mat = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  lsc_flat_atomic<float, sycl::ext::intel::experimental::esimd::atomic_op::fadd, 1,
                  lsc_data_size::default_size, CacheHint::Uncached,
                  CacheHint::WriteBack, 16>((float *)sA,
                                            offsets * sizeof(float), mat, 1);
}

int main(void) {
  constexpr unsigned Size = 256;
  constexpr unsigned VL = 4;
  constexpr unsigned NElts = 1;
  constexpr unsigned GroupSize = 4;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  DTYPE *A = malloc_shared<DTYPE>(Size / 16, q);
  DTYPE *B = malloc_shared<DTYPE>(Size / 16, q);

  for (unsigned i = 0; i < Size / 16; ++i) {
    A[i] = 0;
    B[i] = 1 * 64 * 0.5;
  }

  range<1> GlobalRange{Size / VL / NElts};
  range<1> LocalRange{GroupSize};
  nd_range<1> Range(GlobalRange, LocalRange);

  std::vector<kernel_id> kernelId1 = {get_kernel_id<Test>()};
  setenv("SYCL_PROGRAM_COMPILE_OPTIONS", "-vc-codegen -doubleGRF", 1);
  auto inputBundle1 = get_kernel_bundle<bundle_state::input>(ctxt, kernelId1);
  auto exeBundle1 = build(inputBundle1);
  try {
    q.submit([&](handler &cgh) {
      cgh.use_kernel_bundle(exeBundle1);
      cgh.parallel_for<Test>(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
        atomic_add_float(ndi, A);
      });
    }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(B, q);
    return 1;
  }

  for (unsigned i = 0; i < Size / 16; ++i) {
    std::cout << A[i] << " ";
  }
  int err_cnt = 0;

  for (unsigned i = 0; i < Size / 16; ++i) {
    if (A[i] != B[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << B[i] << " != " << A[i]
                  << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  free(A, q);
  free(B, q);

  return err_cnt > 0 ? 1 : 0;
}
