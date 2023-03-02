// REQUIRES: cpu || gpu
// UNSUPPORTED: cuda,hip
//
// RUN: %clangxx -fsycl -Xclang -fsycl-allow-virtual-functions -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Purpose of this test is to check that we can pass an object of class with
// virtual functions between kernels and that those virtual functions will
// continue to work correctly.

#include <sycl/sycl.hpp>

#define BASE 1
#define DERIVED1 2
#define DERIVED2 3

class Base {
public:
  virtual int display() { return BASE; }
};

class Derived1 : public Base {
public:
  int display() { return DERIVED1; }
};

class Derived2 : public Base {
public:
  int display() { return DERIVED2; }
};

int main() {
  sycl::queue Q;

  auto *Storage =
      sycl::malloc_device<char>(sizeof(Derived1) + sizeof(Derived2), Q);
  auto *Ptrs = sycl::malloc_device<Base *>(2, Q);
  auto *Result = sycl::malloc_shared<int>(2, Q);

  Q.single_task([=] {
     Ptrs[0] = new (&Storage[0]) Derived1;
     Ptrs[1] = new (&Storage[sizeof(Derived1)]) Derived2;
   }).wait();

  Q.single_task([=] {
     for (int i = 0; i < 2; ++i) {
       Result[i] = Ptrs[i]->display();
     }
   }).wait();

  assert(Result[0] == DERIVED1);
  assert(Result[1] == DERIVED2);

  return 0;
}
