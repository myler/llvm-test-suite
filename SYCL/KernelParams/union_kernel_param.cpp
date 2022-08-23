// This test checks kernel execution with union type as kernel parameters.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <cstdio>
#include <sycl/sycl.hpp>

#ifdef ENABLE_FP64
typedef double fptype;
#else
typedef float fptype;
#endif

union TestUnion {
public:
  int myint;
  char mychar;
  fptype mytype;

  TestUnion() { mytype = 0.0; };
};

int main(int argc, char **argv) {
  TestUnion x;
  x.mytype = 5.0;
  fptype mytype = 0.0;

  sycl::queue queue;
  {
    sycl::buffer<fptype, 1> buf(&mytype, 1);
    queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class test>([=]() { acc[0] = x.mytype; });
    });
  }

  if (mytype != 5.0) {
    printf("FAILED\nmytype = %d\n", mytype);
    return 1;
  }
  return 0;
}
