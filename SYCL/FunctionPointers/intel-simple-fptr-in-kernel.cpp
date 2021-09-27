// UNSUPPORTED: windows
// UNSUPPORTED: cuda
// CUDA does not support the function pointer as kernel argument extension.

// RUN: %clangxx -fsycl -fsycl-enable-function-pointers %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

// CHECK: Test PASSED.

#include <CL/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

[[intel::device_indirectly_callable]] int add(int A, int B) { return A + B; }

[[intel::device_indirectly_callable]] int sub(int A, int B) { return A - B; }

int main() {
  const int Size = 100;
  std::vector<long> A(Size, 1);
  std::vector<long> B(Size, 2);

  cl::sycl::queue Q;

  cl::sycl::buffer<long> BufA(A.data(), cl::sycl::range<1>(Size));
  cl::sycl::buffer<long> BufB(B.data(), cl::sycl::range<1>(Size));

  Q.submit([&](cl::sycl::handler &CGH) {
    auto AccA =
        BufA.template get_access<cl::sycl::access::mode::read_write>(CGH);
    auto AccB = BufB.template get_access<cl::sycl::access::mode::read>(CGH);
    CGH.parallel_for<class K>(cl::sycl::range<1>(Size),
                              [=](cl::sycl::id<1> Index) {
                                int (*Fptr)(int, int) = nullptr;
                                if (Index % 2 == 0)
                                  Fptr = add;
                                else
                                  Fptr = sub;
                                AccA[Index] = Fptr(AccA[Index], AccB[Index]);
                              });
  });

  auto HostAcc = BufA.get_access<cl::sycl::access::mode::read>();
  auto *Data = HostAcc.get_pointer();

  bool Passed = true;
  for (int i = 0; i < Size; ++i) {
    Passed &= (i % 2 == 0) ? (Data[i] == 3) : (Data[i] == -1);
  }
  if (Passed) {
    std::cout << "Test PASSED." << std::endl;
  } else {
    std::cout << "Test FAILED." << std::endl;
    for (int I = 0; I < Size; ++I) {
      std::cout << HostAcc[I] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
