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

// This test checks 2d flat lsc intrinsics

// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl %s -DESIMD_GEN12_7 -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sycl/ext/intel/experimental/esimd.hpp>

int main() {
  using namespace cl::sycl;
  using namespace sycl::ext::intel::experimental::esimd;
  unsigned data_height = 4;
  unsigned data_width = 9;
  unsigned data_pitch = 16;
  unsigned x = 0;
  unsigned y = 0;
  unsigned size = data_height * data_pitch;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<info::device::name>()
            << std::endl;

  auto *input = malloc_shared<int>(size, q);
  std::iota(input, input + size, 0);

  constexpr unsigned Width = 4;
  constexpr unsigned Height = 4;
  constexpr unsigned NumBlocks = 1;
  auto *block_store = malloc_shared<int>(size, q);

  auto *ref = new int[size];
  // Fill dst and ref data which is untouched with random values
  for (int i = 0; i < size; i++)
    block_store[i] = ref[i] = rand() % 128;

  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      ref[y * data_pitch + i * data_pitch + x + j] =
          input[y * data_pitch + i * data_pitch + x + j];
    }
  }
  try {
    q.submit([&](handler &h) {
      h.parallel_for<class SimplestKernel>(
          range<1>{1}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            lsc_flat_prefetch2d<int, Width, Height, NumBlocks, false, false,
                                CacheHint::Uncached, CacheHint::Uncached>(
                input, (data_width * sizeof(int)) - 1, data_height - 1,
                (data_pitch * sizeof(int)) - 1, x, y);
            auto data =
                lsc_flat_load2d<int, Width, Height, NumBlocks, false, false,
                                CacheHint::Uncached, CacheHint::Uncached>(
                    input, (data_width * sizeof(int)) - 1, data_height - 1,
                    (data_pitch * sizeof(int)) - 1, x, y);
            lsc_flat_store2d<int, Width, Height, false, false,
                             CacheHint::Uncached, CacheHint::Uncached>(
                block_store, (data_width * sizeof(int)) - 1, data_height - 1,
                (data_pitch * sizeof(int)) - 1, x, y, data);
          });
    });
    q.wait();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    free(input, q);
    free(block_store, q);
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i < size; ++i)
    error += std::abs(ref[i] - block_store[i]);
  free(input, q);
  free(block_store, q);
  std::cout << (error != 0 ? "FAILED" : "PASSED") << std::endl;
  return error;
}