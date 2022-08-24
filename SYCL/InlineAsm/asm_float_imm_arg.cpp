// UNSUPPORTED: cuda || hip_nvidia
// REQUIRES: gpu,linux,aspect-fp64
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#ifdef ENABLE_FP64
using fptype = double;
using dataType = sycl::cl_double;
#else
using fptype = float;
using dataType = sycl::cl_float;
#endif

constexpr fptype IMM_ARGUMENT = 0.5;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 1>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input)
      : WithInputBuffers<T, 1>(input), WithOutputBuffer<T>(input.size()) {}

  void operator()(sycl::handler &cgh) {
    auto A =
        this->getInputBuffer(0).template get_access<sycl::access::mode::read>(
            cgh);
    auto B =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm("mul (M1, 8) %0(0, 0)<1> %1(0, 0)<1;1,0> %2"
              : "=rw"(B[wiID])
              : "rw"(A[wiID]), "i"(IMM_ARGUMENT));
#else
          B[wiID] = A[wiID] * IMM_ARGUMENT;
#endif
        });
  }
};

int main() {
  std::vector<dataType> input(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++)
    input[i] = (fptype)1 / std::pow(2, i);

  KernelFunctor<> f(input);
  if (!launchInlineASMTest(f))
    return 0;

  auto &B = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; ++i) {
    if (B[i] != input[i] * IMM_ARGUMENT) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << B[i] << " != " << input[i] * IMM_ARGUMENT << "\n";
      return 1;
    }
  }
  return 0;
}
