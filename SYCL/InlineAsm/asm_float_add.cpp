// UNSUPPORTED: cuda || hip_nvidia
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

template <typename T>
struct KernelFunctor : WithInputBuffers<T, 2>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input1, const std::vector<T> &input2)
      : WithInputBuffers<T, 2>(input1, input2), WithOutputBuffer<T>(
                                                    input1.size()) {}

  void operator()(sycl::handler &cgh) {
    auto A =
        this->getInputBuffer(0).template get_access<sycl::access::mode::read>(
            cgh);
    auto B =
        this->getInputBuffer(1).template get_access<sycl::access::mode::read>(
            cgh);
    auto C =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm("add (M1, 8) %0(0, 0)<1> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>"
              : "=rw"(C[wiID])
              : "rw"(A[wiID]), "rw"(B[wiID]));
#else
          C[wiID] = A[wiID] + B[wiID];
#endif
        });
  }
};

template <typename T1, typename T2>
bool check() {
  std::vector<T2> inputA(DEFAULT_PROBLEM_SIZE),
      inputB(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    inputA[i] = (T1)1 / std::pow(2, i);
    inputB[i] = (T1)2 / std::pow(2, i);
  }

  KernelFunctor<T2> f(inputA, inputB);
  if (!launchInlineASMTest(f))
    return true;

  auto &C = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    if (C[i] != inputA[i] + inputB[i]) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << C[i] << " != " << inputA[i] + inputB[i] << "\n";
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

