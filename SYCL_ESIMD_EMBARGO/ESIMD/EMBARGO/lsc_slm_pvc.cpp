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

// This test checks 1d slm lsc intrinsics
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

  auto vec_0 = std::vector<int>(size);
  auto vec_1 = std::vector<int>(size);
  auto vec_2 = std::vector<int>(size);
  auto vec_3 = std::vector<int>(size);
  auto vec_4 = std::vector<int>(size);
  auto buf_0 = buffer{vec_0};
  auto buf_1 = buffer{vec_1};
  auto buf_2 = buffer{vec_2};
  auto buf_3 = buffer{vec_3};
  auto buf_4 = buffer{vec_4};

  try {
    q.submit([&](handler &h) {
      auto access_0 = buf_0.template get_access<access::mode::read_write>(h);
      auto access_1 = buf_1.template get_access<access::mode::read_write>(h);
      auto access_2 = buf_2.template get_access<access::mode::read_write>(h);
      auto access_3 = buf_3.template get_access<access::mode::read_write>(h);
      auto access_4 = buf_4.template get_access<access::mode::read_write>(h);
      h.parallel_for<class SimplestKernel>(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id * SIMDSize * sizeof(int);
            auto offsets =
                simd<uint32_t, SIMDSize>(id * SIMDSize, 1) * sizeof(int);
            auto data = simd<int, SIMDSize>(id * SIMDSize, 1);
            auto pred = simd<uint16_t, SIMDSize>(1);
            auto add = simd<uint16_t, SIMDSize>(5);
            auto compare = simd<uint32_t, SIMDSize>(id * SIMDSize, 1);
            auto swap = compare * 2;

            slm_init(4096);
            lsc_slm_store<int, SIMDSize>(data * 2, offset);
            auto data_0 = lsc_slm_load<int, SIMDSize>(offset);
            lsc_surf_store<int, SIMDSize>(data_0, access_0, offset);

            lsc_slm_store<int>(data * 2, offsets);
            auto data_1 = lsc_slm_load<int>(offsets);
            lsc_surf_store<int, SIMDSize>(data_1, access_1, offset);

            lsc_slm_store<int, SIMDSize>(data, offset);
            lsc_slm_atomic<int, atomic_op::inc>(offsets, pred);
            auto data_2 = lsc_slm_load<int, SIMDSize>(offset);
            lsc_surf_store<int, SIMDSize>(data_2, access_2, offset);

            lsc_slm_store<int, SIMDSize>(data, offset);
            lsc_slm_atomic<int, atomic_op::add>(offsets, add, pred);
            auto data_3 = lsc_slm_load<int, SIMDSize>(offset);
            lsc_surf_store<int, SIMDSize>(data_3, access_3, offset);

            lsc_slm_store<int, SIMDSize>(data, offset);
            lsc_slm_atomic<int, atomic_op::cmpxchg>(offsets, compare, swap,
                                                    pred);
            auto data_4 = lsc_slm_load<int, SIMDSize>(offset);
            lsc_surf_store<int, SIMDSize>(data_4, access_4, offset);
          });
    });
    q.wait();
    buf_0.template get_access<access::mode::read_write>();
    buf_1.template get_access<access::mode::read_write>();
    buf_2.template get_access<access::mode::read_write>();
    buf_3.template get_access<access::mode::read_write>();
    buf_4.template get_access<access::mode::read_write>();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i != size; ++i) {
    error += std::abs(vec_0[i] - (i * 2));
    error += std::abs(vec_1[i] - (i * 2));
    error += std::abs(vec_2[i] - (i + 1));
    error += std::abs(vec_3[i] - (i + 5));
    error += std::abs(vec_4[i] - (i * 2));
  }
  std::cout << (error != 0 ? "FAILED" : "PASSED") << std::endl;
  return error;
}
