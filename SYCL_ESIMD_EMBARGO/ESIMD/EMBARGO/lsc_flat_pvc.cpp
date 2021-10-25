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

// This test checks 1d flat lsc intrinsics
// TODO enable this test on PVC fullsim when LSC patch is merged
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx -fsycl %s -DESIMD_GEN12_7 -o %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sycl/ext/intel/experimental/esimd.hpp>

int main() {
  using namespace cl::sycl;
  using namespace sycl::ext::intel::experimental::esimd;
  auto size = size_t{128};
  auto constexpr SIMDSize = unsigned{4};

  auto q =
      queue{esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler()};
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<info::device::name>()
            << std::endl;

  auto *vec_0 = malloc_shared<int>(size, q);
  auto *vec_1 = malloc_shared<int>(size, q);
  auto *vec_2 = malloc_shared<int>(size, q);
  auto *vec_3 = malloc_shared<int>(size, q);
  auto *vec_4 = malloc_shared<int>(size, q);
  std::iota(vec_0, vec_0 + size, 0);
  std::iota(vec_1, vec_1 + size, 0);
  std::iota(vec_2, vec_2 + size, 0);
  std::iota(vec_3, vec_3 + size, 0);
  std::iota(vec_4, vec_4 + size, 0);

  try {
    q.submit([&](handler &h) {
      h.parallel_for<class SimplestKernel>(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id[0] * SIMDSize;
            auto offsets = simd<uint32_t, SIMDSize>(id * SIMDSize * sizeof(int),
                                                    sizeof(int));
            auto pred = simd_mask<SIMDSize>(1);
            auto add = simd<uint16_t, SIMDSize>(5);
            auto compare = simd<uint32_t, SIMDSize>(id * SIMDSize, 1);
            auto swap = compare * 2;

            lsc_flat_prefetch<int, SIMDSize, lsc_data_size::default_size,
                              CacheHint::Uncached, CacheHint::Uncached>(vec_0 +
                                                                        offset);
            auto data_0 = lsc_flat_load<int, SIMDSize>(vec_0 + offset);
            lsc_flat_store<int, SIMDSize>(vec_0 + offset, data_0 * 2);

            lsc_flat_prefetch<int, 1, lsc_data_size::default_size,
                              CacheHint::Uncached, CacheHint::Uncached>(
                vec_1, offsets);
            auto data_1 = lsc_flat_load<int>(vec_1, offsets);
            lsc_flat_store<int>(vec_1, data_1 * 2, offsets);

            lsc_flat_atomic<int, atomic_op::inc>(vec_2, offsets, pred);
            lsc_flat_atomic<int, atomic_op::add>(vec_3, offsets, add, pred);
            lsc_flat_atomic<int, atomic_op::cmpxchg>(vec_4, offsets, compare,
                                                     swap, pred);
          });
    });
    q.wait();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    sycl::free(vec_0, q);
    sycl::free(vec_1, q);
    sycl::free(vec_2, q);
    sycl::free(vec_3, q);
    sycl::free(vec_4, q);
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i != size; ++i) {
    error += std::abs(vec_0[i] - 2 * i);
    error += std::abs(vec_1[i] - 2 * i);
    error += std::abs(vec_2[i] - (i + 1));
    error += std::abs(vec_3[i] - (i + 5));
    error += std::abs(vec_4[i] - (i * 2));
  }
  sycl::free(vec_0, q);
  sycl::free(vec_1, q);
  sycl::free(vec_2, q);
  sycl::free(vec_3, q);
  sycl::free(vec_4, q);
  std::cout << (error != 0 ? "FAILED" : "PASSED") << std::endl;
  return error;
}
