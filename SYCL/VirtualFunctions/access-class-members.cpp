// REQUIRES: cpu || gpu
// UNSUPPORTED: cuda,hip
//
// RUN: %clangxx -fsycl -Xclang -fsycl-allow-virtual-functions -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Purpose of this test is to check that we can access class member fields from
// virtual functions just fine.

#include <sycl/sycl.hpp>

class Base {
protected:
  int base_data = 42;

public:
  virtual int display() { return base_data; }
};

class Derived1 : public Base {
  int data;

public:
  Derived1(int _data) : data(_data) {}
  int display() { return base_data + data; }
};

class Derived2 : public Base {
  int data1;
  int data2;

public:
  Derived2(int _data1, int _data2) : data1(_data1), data2(_data2) {}
  int display() { return base_data + data1 + 2 * data2; }
};

int main() {
  sycl::queue Q;

  auto *Storage =
      sycl::malloc_device<char>(sizeof(Derived1) + sizeof(Derived2), Q);
  auto *Ptrs = sycl::malloc_device<Base *>(2, Q);
  auto *Result = sycl::malloc_shared<int>(2, Q);

  Q.single_task([=] {
     Ptrs[0] = new (&Storage[0]) Derived1(1);
     Ptrs[1] = new (&Storage[sizeof(Derived1)]) Derived2(1, 2);
   }).wait();

  Q.single_task([=] {
     for (int i = 0; i < 2; ++i) {
       Result[i] = Ptrs[i]->display();
     }
   }).wait();

  assert(Result[0] == 42 + 1);
  assert(Result[1] == 42 + 1 + 2 * 2);

  return 0;
}
