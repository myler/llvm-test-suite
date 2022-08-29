// This test checks kernel execution with union type as kernel parameters.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <cstdio>
#include <sycl/sycl.hpp>

template <typename T> union TestUnion {
public:
  int myint;
  char mychar;
  T mytype;

  TestUnion() { mytype = 0.0; };
};

template <typename T> bool check() {
  TestUnion<T> x;
  x.mytype = 5.0;
  T mytype = 0.0;

  sycl::queue queue;
  {
    sycl::buffer<T, 1> buf(&mytype, 1);
    queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.template get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class test>([=]() { acc[0] = x.mytype; });
    });
  }

  if (mytype != 5.0) {
    printf("FAILED\nmytype = %d\n", mytype);
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  bool Passed = true;

  Passed &= check<float>();
#ifdef ENABLE_FP64
  Passed &= check<double>();
#endif

  return Passed ? 0 : 1;
}

