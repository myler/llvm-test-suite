// UNSUPPORTED: cuda || hip_nvidia
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

template <typename T> constexpr T IMM_ARGUMENT = T(0.5);

template <typename T1, typename T2>
struct KernelFunctor : WithInputBuffers<T2, 1>, WithOutputBuffer<T2> {
  KernelFunctor(const std::vector<T2> &input)
      : WithInputBuffers<T2, 1>(input), WithOutputBuffer<T2>(input.size()) {}

  void operator()(sycl::handler &cgh) {
    auto A =
        this->getInputBuffer(0).template get_access<sycl::access::mode::read>(
            cgh);
    auto B =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);

    cgh.parallel_for<KernelFunctor<T1, T2>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm("mul (M1, 8) %0(0, 0)<1> %1(0, 0)<1;1,0> %2"
              : "=rw"(B[wiID])
              : "rw"(A[wiID]), "i"(IMM_ARGUMENT<T1>));
#else
          B[wiID] = A[wiID] * IMM_ARGUMENT<T1>;
#endif
        });
  }
};

template <typename T1, typename T2> bool check() {
  constexpr T1 IMM_ARGUMENT = T1(0.5);

  std::vector<T2> input(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++)
    input[i] = (T1)1 / std::pow(2, i);

  KernelFunctor<T1, T2> f(input);
  if (!launchInlineASMTest(f))
    return true;

  auto &B = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; ++i) {
    if (B[i] != input[i] * IMM_ARGUMENT) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << B[i] << " != " << input[i] * IMM_ARGUMENT << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  bool Passed = true;

  Passed &= check<float, sycl::cl_float>();
#ifdef ENABLE_FP64
  Passed &= check<double, sycl::cl_double>();
#endif

  return Passed ? 0 : 1;
}
