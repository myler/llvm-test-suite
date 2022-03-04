//===-- operator_assignment.hpp - Functions for tests on simd assignment
//      operators. --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd assignment operators.
///
//===----------------------------------------------------------------------===//

#pragma once
<<<<<<< HEAD
<<<<<<< HEAD
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS
=======
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))

// The test proxy is used to verify the move assignment was called actually.
#define __ESIMD_ENABLE_TEST_PROXY

#include "../shared_element.hpp"
#include "common.hpp"

namespace esimd_test::api::functional::operators {

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
<<<<<<< HEAD
<<<<<<< HEAD
template <typename DataT, typename SizeT, typename TestCaseT> class run_test {
  static constexpr int NumElems = SizeT::value;
  using TestDescriptionT = TestDescription<NumElems, TestCaseT>;
<<<<<<< HEAD
=======
template <typename DataT, typename DimT, typename TestCaseT> class run_test {
  static constexpr int NumElems = DimT::value;
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
template <typename DataT, typename SizeT, typename TestCaseT> class run_test {
  static constexpr int NumElems = SizeT::value;
>>>>>>> e737b795e ([SYCL][ESIMD] Add tests for simd broadcast constructor (#690))
=======
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    log::trace<TestDescriptionT>(data_type);

    if (should_skip_test_with<DataT>(queue.get_device())) {
      return true;
    }

<<<<<<< HEAD
    bool passed = true;
<<<<<<< HEAD
    log::trace<TestDescriptionT>(data_type);

    if (should_skip_test_with<DataT>(queue.get_device())) {
      return true;
    }

=======
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed = run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type);
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<DataT> &ref_data,
                        const std::string &data_type) {
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);

<<<<<<< HEAD
<<<<<<< HEAD
    // Initialize operator correctness flag with pre-defined invalid value
    constexpr bool is_move_expected = TestCaseT::is_move_expected();
    shared_element<bool> was_moved(queue, !is_move_expected);
=======
    shared_element<bool> is_correct_operator(queue, false);
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
    // Initialize operator correctness flag with pre-defined invalid value
    constexpr bool is_move_expected = TestCaseT::is_move_expected();
    shared_element<bool> was_moved(queue, !is_move_expected);
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();
<<<<<<< HEAD
<<<<<<< HEAD
      const auto was_moved_ptr = was_moved.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *was_moved_ptr = TestCaseT::template run<DataT, NumElems>(ref, out);
=======
      const auto is_correct_operator_storage = is_correct_operator.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *is_correct_operator_storage =
                TestCaseT::template run<DataT, NumElems>(ref, out);
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
      const auto was_moved_ptr = was_moved.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *was_moved_ptr = TestCaseT::template run<DataT, NumElems>(ref, out);
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
<<<<<<< HEAD
<<<<<<< HEAD
      const auto &expected = ref_data[i];
      const auto &retrieved = result[i];

      if (!are_bitwise_equal(expected, retrieved)) {
        passed = false;
        log::fail(TestDescriptionT(data_type), "Unexpected value at index ", i,
                  ", retrieved: ", retrieved, ", expected: ", expected);
      }
    }

    if (was_moved.value() != is_move_expected) {
      passed = false;

      if constexpr (is_move_expected) {
        log::fail(TestDescriptionT(data_type),
                  "A copy operator instead of a move operator was used");
      } else {
        log::fail(TestDescriptionT(data_type),
                  "Unexpected simd vector move operator called");
      }
=======
      if (!are_bitwise_equal(ref_data[i], result[i])) {
        passed = false;
=======
      const auto &expected = ref_data[i];
      const auto &retrieved = result[i];
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))

      if (!are_bitwise_equal(expected, retrieved)) {
        passed = false;
        log::fail(TestDescriptionT(data_type), "Unexpected value at index ", i,
                  ", retrieved: ", retrieved, ", expected: ", expected);
      }
    }

    if (was_moved.value() != is_move_expected) {
      passed = false;
<<<<<<< HEAD
      log::note("Test failed due to " + TestCaseT::get_description() +
                " hasn't called for simd<" + data_type + ", " +
                std::to_string(NumElems) + ">.");
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======

      if constexpr (is_move_expected) {
        log::fail(TestDescriptionT(data_type),
                  "A copy operator instead of a move operator was used");
      } else {
        log::fail(TestDescriptionT(data_type),
                  "Unexpected simd vector move operator called");
      }
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
    }

    return passed;
  }
};

} // namespace esimd_test::api::functional::operators
