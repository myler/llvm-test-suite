//==---------------- fp_conversions_ats.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl %s -o %t.out
// RUN %GPU_RUN_PLACEHOLDER %t.out

// This test checks conversions between floating-point types:
// - bfloat16 <-> float

#include "../../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <int N> class Test;
using byte = unsigned char;

struct TestBF16Impl {
  float *FP32;
  bfloat16 *BF16;
  bfloat16 *FP32_to_BF16;
  float *BF16_to_FP32;
  int N;
  queue q;
  size_t Size;

  TestBF16Impl(queue q, int N) : q(q), N(N) {
    FP32 = sycl::malloc_shared<float>(Size, q);
    BF16 = sycl::malloc_shared<bfloat16>(Size, q);
    FP32_to_BF16 = sycl::malloc_shared<bfloat16>(Size, q);
    BF16_to_FP32 = sycl::malloc_shared<float>(Size, q);
    Size = 1024 * 128;
  }

  ~TestBF16Impl() {
    if (FP32) sycl::free(FP32, q);
    if (BF16) sycl::free(BF16, q);
    if (FP32_to_BF16) sycl::free(FP32_to_BF16, q);
    if (BF16_to_FP32) sycl::free(BF16_to_FP32, q);
  }

  virtual void testImpl(byte *BF16_ptr, byte *FP32_to_BF16_ptr, float *BF16_to_FP32, float *FP32) = 0;

  bool test() {
    std::cout << "Testing VL=" << N << "\n";

    // bloat16 difference from float32 is only the number of bits in the
    // significand, hence float32 -> bloat16 conversion is just truncation of 16
    // least significant bits (with rounding to nearest or even),
    // bfloat16->float32 is adding 16 zeroes

    constexpr size_t Nvals = 4;
    uint32_t fp32_values[Nvals] = {
      //               V                  // bf16 truncates here
      //x                                 // sign
      // xxxxxxxx                         // mantissa
      //         xxxxxxxxxxxxxxxxxxxxxxx  // significand
      0b01000000010010010000000000000000, // 3.150625
      0b00000000110110011000000000000000, // 9.98711e-39
      0b00000000110110010100000000000000, // 9.975631e-39
      0b11111111011111111111111111111111  // -3.4028234e+38
    };

    // Golden values for float -> bfloat16 conversion. Assumes control regiter's
    // rounding mode is RTNE:
    //    cr0.5:4 is 00b = Round to Nearest or Even (RTNE)
    //
    // Also, a source for bfloat16->float conversion and golden value
    // calculation.
    uint16_t bf16_values[Nvals] = {
      //x                 // sign
      // xxxxxxxx         // mantissa
      //         xxxxxxx  // significand
      0b0100000001001001, // 3.150625, exact match to float
      0b0000000011011010, // 1.0010069e-38, rounded to even (up)
      0b0000000011011001, // 9.964151e-39, rounded to even (down)
      0b1111111110000000  // -3.3961775e+38, rounded to even (up)
    };


    float *FP32 = sycl::malloc_shared<float>(Size, q);
    bfloat16 *BF16 = sycl::malloc_shared<bfloat16>(Size, q);
    bfloat16 *FP32_to_BF16 = sycl::malloc_shared<bfloat16>(Size, q);
    float *BF16_to_FP32 = sycl::malloc_shared<float>(Size, q);

    for (unsigned i = 0; i < Size; ++i) {
      FP32[i] = *reinterpret_cast<float*>(&fp32_values[i%Nvals]);
      BF16[i] = *reinterpret_cast<bfloat16*>(&bf16_values[i%Nvals]);
      FP32_to_BF16[i] = 0;
      BF16_to_FP32[i] = 0;
    }
    try {
      // TODO temporarily bfloat16 maps to different types on host and device, so
      // need to cast pointers to common type before they are captured by the
      // lambda to ensure the same kernel name mangling (otherwise the kernel
      // won't be resolved)...
      byte *BF16_ptr = reinterpret_cast<byte*>(BF16);
      byte *FP32_to_BF16_ptr = reinterpret_cast<byte*>(FP32_to_BF16);
      testImpl(BF16_ptr, FP32_to_BF16_ptr, BF16_to_FP32, FP32);
    }
    catch (cl::sycl::exception const &e) {
      std::cout << "SYCL exception caught: " << e.what() << '\n';
      sycl::free(FP32, q);
      sycl::free(BF16, q);
      sycl::free(FP32_to_BF16, q);
      sycl::free(BF16_to_FP32, q);
      return false;
    }
    int err_cnt = 0;
    std::cout << "  Verifying bfloat16->float results...\n";
    int cur_err_cnt = 0;

    for (unsigned i = 0; i < Size; ++i) {
      // take BF16[i] and add 16 zeroes to the right
      uint32_t gold = 0;
      *(((uint16_t*)&gold) + 1) = *((uint16_t*)&BF16[i]);
      uint32_t res = *((uint32_t*)&BF16_to_FP32[i]);

      if (gold != res) {
        if (++cur_err_cnt < 10) {
          std::cout << "    failed at index " << i << ": " << res << " != " << gold
            << " (gold)\n";
        }
      }
    }
    if (cur_err_cnt > 0) {
      std::cout << "    pass rate: "
        << ((float)(Size - cur_err_cnt) / (float)Size) * 100.0f << "% ("
        << (Size - cur_err_cnt) << "/" << Size << ")\n";
    }
    else {
      std::cout << "    OK\n";
    }
    err_cnt += cur_err_cnt;

    std::cout << "  Verifying float->bfloat16 results...\n";
    cur_err_cnt = 0;

    for (unsigned i = 0; i < Size; ++i) {
      uint16_t gold = bf16_values[i%Nvals];
      uint16_t res = *((uint16_t*)&FP32_to_BF16[i]);

      if (gold != res) {
        if (++cur_err_cnt < 20) {
          std::cout << "    failed at index " << i << ": " << res << " != " << gold
            << " (gold)\n";
        }
      }
    }
    if (cur_err_cnt > 0) {
      std::cout << "    pass rate: "
        << ((float)(Size - cur_err_cnt) / (float)Size) * 100.0f << "% ("
        << (Size - cur_err_cnt) << "/" << Size << ")\n";
    }
    else {
      std::cout << "    OK\n";
    }
    err_cnt += cur_err_cnt;
    sycl::free(FP32, q);
    sycl::free(BF16, q);
    sycl::free(FP32_to_BF16, q);
    sycl::free(BF16_to_FP32, q);
    return err_cnt == 0;
  }
};

template <int VL>
struct TestBF16 : TestBF16Impl {
  using TestBF16Impl::TestBF16Impl;

  TestBF16(queue q) : TestBF16Impl(q, VL) {}

  void testImpl(byte *BF16_ptr, byte *FP32_to_BF16_ptr, float *BF16_to_FP32, float *FP32) {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<Test<VL>>(
        range<1>{Size/VL}, [=](id<1> i) SYCL_ESIMD_KERNEL {
          // TODO ... now cast back from byte* to specific types
          bfloat16 *BF16 = reinterpret_cast<bfloat16*>(BF16_ptr);
          bfloat16 *FP32_to_BF16 = reinterpret_cast<bfloat16*>(FP32_to_BF16_ptr);

          auto offset = i * VL;
          simd<bfloat16, VL> vbf16;
          simd<float, VL> vfp32;
          
          if constexpr (VL == 1) {
            vbf16[0] = BF16[offset];
            vfp32[0] = FP32[offset];
          }
          else {
            vbf16.copy_from(BF16 + offset);
            vfp32.copy_from(FP32 + offset);
          }
          simd<bfloat16, VL> vbf16conv = convert_to_bf16(vfp32);
          simd<float, VL> vfp32conv = convert_from_bf16(vbf16);
          
          if constexpr (VL == 1) {
            FP32_to_BF16[offset] = vbf16conv[0];
            BF16_to_FP32[offset] = vfp32conv[0];
          }
          else {
            vbf16conv.copy_to(FP32_to_BF16 + offset);
            vfp32conv.copy_to(BF16_to_FP32 + offset);
          }
        });
    });
    e.wait();
  }
};

struct ScalarTestBF16 : TestBF16Impl {
  using TestBF16Impl::TestBF16Impl;

  ScalarTestBF16(queue q) : TestBF16Impl(q, 0) {}

  void testImpl(byte *BF16_ptr, byte *FP32_to_BF16_ptr, float *BF16_to_FP32, float *FP32) {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<ScalarTestBF16>(
        range<1>{Size}, [=](id<1> i) SYCL_ESIMD_KERNEL {
          // TODO ... now cast back from byte* to specific types
          bfloat16 *BF16 = reinterpret_cast<bfloat16*>(BF16_ptr);
          bfloat16 *FP32_to_BF16 = reinterpret_cast<bfloat16*>(FP32_to_BF16_ptr);
          FP32_to_BF16[i] = convert_to_bf16(FP32[i]);
          BF16_to_FP32[i] = convert_from_bf16(BF16[i]);
        });
    });
    e.wait();
  }
};

int main() {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  // TODO: enable two testcases below when XDEPS-2248 is fixed:
  //{ passed &= ScalarTestBF16{q}.test(); }
  //{ passed &= TestBF16<1>{q}.test(); }
  { passed &= TestBF16<8>{q}.test(); }
  { passed &= TestBF16<16>{q}.test(); }
  { passed &= TestBF16<32>{q}.test(); }
  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
