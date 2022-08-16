// This test is intended to check basic operations with SYCL 2020 specialization
// constants using sycl::kernel_bundle and sycl::kernel_handler APIs:
// - test that specialization constants can be accessed in kernel and they
//   have their default values if `set_specialization_constants` wasn't called
// - test that specialization constant values can be set and retrieved through
//   kernel_bundle APIs on host
// - test that specialization constant values can be set through kernel_bundle
//   API and correctly retrieved within a kernel
//
// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// FIXME: SYCL 2020 specialization constants are not supported on host device
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// FIXME: ACC devices use emulation path, which is not yet supported
// UNSUPPORTED: hip

#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

#include "common.hpp"

constexpr sycl::specialization_id<double> double_id(3.14);

class TestDefaultValuesKernel;
class EmptyKernel;
class TestSetAndGetOnDevice;

bool test_default_values(sycl::queue q);
bool test_set_and_get_on_host(sycl::queue q);
bool test_set_and_get_on_device(sycl::queue q);

int main() {
  auto exception_handler = [&](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "An async SYCL exception was caught: " << e.what()
                  << std::endl;
        std::exit(1);
      }
    }
  };

  sycl::queue q(exception_handler);

  if (!q.get_device().has(sycl::aspect::fp64) {
    std::cout << "Skipping test\n";
    return 0;
  }

  if (!test_default_values(q)) {
    std::cout << "Test for default values of specialization constants failed!"
              << std::endl;
    return 1;
  }

  if (!test_set_and_get_on_host(q)) {
    std::cout << "Test for set and get API on host failed!" << std::endl;
    return 1;
  }

  if (!test_set_and_get_on_device(q)) {
    std::cout << "Test for set and get API on device failed!" << std::endl;
    return 1;
  }

  return 0;
};

bool test_default_values(sycl::queue q) {
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(q.get_context())) {
    std::cout << "Cannot obtain kernel_bundle in input state, skipping default "
                 "values test"
              << std::endl;
    // TODO: check that online_compielr aspec is not available
    return true;
  }

  sycl::buffer<double> double_buffer(1);

  auto input_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(q.get_context());
  auto exec_bundle = sycl::build(input_bundle);

  q.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(exec_bundle);
    auto double_acc = double_buffer.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<TestDefaultValuesKernel>([=](sycl::kernel_handler kh) {
      double_acc[0] = kh.get_specialization_constant<double_id>();
    });
  });

  auto double_acc = double_buffer.get_access<sycl::access::mode::read>();
  if (!check_value(3.14, double_acc[0], "double specialization constant"))
    return false;

  return true;
}

bool test_set_and_get_on_host(sycl::queue q) {
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(q.get_context())) {
    std::cout << "Cannot obtain kernel_bundle in input state, skipping default "
                 "values test"
              << std::endl;
    // TODO: check that online_compielr aspec is not available
    return true;
  }

  unsigned errors = 0;

  try {
    auto input_bundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(q.get_context());

    if (!input_bundle.contains_specialization_constants()) {
      std::cout
          << "Obtained kernel_bundle is expected to contain specialization "
             "constants, but it doesn't!"
          << std::endl;
      return false;
    }

    if (!check_value(3.14,
                     input_bundle.get_specialization_constant<double_id>(),
                     "double specializaiton constant before setting any value"))
      ++errors;

    // Update values
    double new_double_value = 3.0;

    input_bundle.set_specialization_constant<double_id>(new_double_value);

    // And re-check them again
    if (!check_value(new_double_value,
                     input_bundle.get_specialization_constant<double_id>(),
                     "double specializaiton constant after setting a value"))
      ++errors;

    // Let's try to build the bundle
    auto exec_bundle = sycl::build(input_bundle);

    // And ensure that updated spec constant values are still there
    if (!check_value(new_double_value,
                     exec_bundle.get_specialization_constant<double_id>(),
                     "double specializaiton constant after build"))
      ++errors;
  } catch (sycl::exception &e) {
  }

  return 0 == errors;
}

bool test_set_and_get_on_device(sycl::queue q) {
  sycl::buffer<double> double_buffer(1);

  double new_double_value = 3.0;

  auto input_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(q.get_context());
  input_bundle.set_specialization_constant<double_id>(new_double_value);
  auto exec_bundle = sycl::build(input_bundle);

  q.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(exec_bundle);
    auto double_acc = double_buffer.get_access<sycl::access::mode::write>(cgh);

    cgh.single_task<TestSetAndGetOnDevice>([=](sycl::kernel_handler kh) {
      double_acc[0] = kh.get_specialization_constant<double_id>();
    });
  });

  auto double_acc = double_buffer.get_access<sycl::access::mode::read>();
  if (!check_value(new_double_value, double_acc[0],
                   "double specialization constant"))
    return false;

  return true;
}
