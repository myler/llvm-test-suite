// REQUIRES: aspect-fp64
// RUN: %clangxx %cxx_std_optionc++17 %s -o %t1.out %sycl_options
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: %HOST_RUN_PLACEHOLDER %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out

//==------------------- buffer.cpp - SYCL buffer basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>
#include <memory>

using namespace sycl;

template <class T> constexpr T write_back_result = T(3);
template <> constexpr double write_back_result<double> = double(7.5);
template <class T> class fill_buffer_for_write_back {};

template <class T, int D> void check_set_write_back() {
  size_t size = 32;
  sycl::range<D> r(size);
  std::shared_ptr<T> shrd(new T[size], [](T *data) { delete[] data; });
  std::vector<T> vector;
  vector.reserve(size);
  sycl::queue Queue;
  std::mutex m;
  {
    sycl::buffer<T, D> buf_shrd(
        shrd, r, sycl::property_list{sycl::property::buffer::use_mutex(m)});
    m.lock();
    std::fill(shrd.get(), (shrd.get() + size), T());
    m.unlock();
    buf_shrd.set_final_data(vector.begin());
    buf_shrd.set_write_back(true);
    Queue.submit([&](sycl::handler &cgh) {
      auto Accessor =
          buf_shrd.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<class fill_buffer_for_write_back<T>>(r, [=](sycl::id<1> WIid) {
        Accessor[WIid] = write_back_result<T>;
      });
    });
  } // Data is copied back
  for (size_t i = 0; i < size; i++) {
    if (vector[i] != write_back_result<T>) {
      assert(false && "Data was not copied back");
    }
  }
}

int main() {
  // Check that data is copied back after forcing write-back using
  // set_write_back
  queue q;
  if (!q.get_device().has(aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  check_set_write_back<double, 1>();
  return 0;
}