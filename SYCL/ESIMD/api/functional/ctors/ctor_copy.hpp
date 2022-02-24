//===-- ctor_copy.hpp - Functions for tests on simd copy constructor definition.
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_copy.hpp
<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_copy.hpp
=======
>>>>>>> c1366f1d7 ([SYCL][ESIMD] Split tests on simd constructors into core and fp_extra (#748)):SYCL/ESIMD/api/functional/ctors/ctor_copy.cpp
///
/// \file
/// This file provides functions for tests on simd copy constructor.
///
//===----------------------------------------------------------------------===//

#pragma once
<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_copy.hpp
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS
=======
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Test for esimd copy constructor.
>>>>>>> 4dd90b8a6 ([SYCL][ESIMD] Enable simd copy constructor tests (#722)):SYCL/ESIMD/api/functional/ctors/ctor_copy.cpp

#include "common.hpp"

namespace esimd = sycl::ext::intel::esimd;
=======

#include "common.hpp"

namespace esimd = sycl::ext::intel::experimental::esimd;
>>>>>>> c1366f1d7 ([SYCL][ESIMD] Split tests on simd constructors into core and fp_extra (#748)):SYCL/ESIMD/api/functional/ctors/ctor_copy.cpp

namespace esimd_test::api::functional::ctors {

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> simd_by_init =
        esimd::simd<DataT, NumElems>(source_simd);
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> simd_by_var_decl(source_simd);
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = esimd::simd<DataT, NumElems>(source_simd);
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    call_simd_by_const_ref<DataT, NumElems>(
        esimd::simd<DataT, NumElems>(source_simd), out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *out) {
    simd_by_const_ref.copy_to(out);
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
<<<<<<< HEAD
<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_copy.hpp
template <typename DataT, typename SizeT, typename TestCaseT> class run_test {
  static constexpr int NumElems = SizeT::value;
  using TestDescriptionT = ctors::TestDescription<NumElems, TestCaseT>;
=======
template <typename DataT, typename DimT, typename TestCaseT> class run_test {
  static constexpr int NumElems = DimT::value;
>>>>>>> 6870ea3ee ([SYCL][ESIMD] Provide the for_all_combinations utility (#721)):SYCL/ESIMD/api/functional/ctors/ctor_copy.cpp
=======
template <typename DataT, typename SizeT, typename TestCaseT> class run_test {
  static constexpr int NumElems = SizeT::value;
>>>>>>> e37c07509 ([SYCL][ESIMD] Replace "dim", "dimensions" with "size", "sizes", etc. (#803))

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    if (should_skip_test_with<DataT>(queue.get_device())) {
      return true;
    }

    bool passed = true;
    log::trace<TestDescriptionT>(data_type);

    if (should_skip_test_with<DataT>(queue.get_device())) {
      return true;
    }

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

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(ref, out);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      const auto &expected = ref_data[i];
      const auto &retrieved = result[i];

      if (!are_bitwise_equal(expected, retrieved)) {
        passed = false;
        log::fail(TestDescriptionT(data_type), "Unexpected value at index ", i,
                  ", retrieved: ", retrieved, ", expected: ", expected);
      }
    }

    return passed;
  }
};

<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_copy.hpp
<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_copy.hpp
} // namespace esimd_test::api::functional::ctors
=======
int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::all>();
  const auto dims = get_all_dimensions();
  const auto contexts = unnamed_type_pack<initializer, var_decl, rval_in_expr,
                                          const_ref>::generate();

  passed &= for_all_combinations<run_test>(types, dims, contexts, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
>>>>>>> 6870ea3ee ([SYCL][ESIMD] Provide the for_all_combinations utility (#721)):SYCL/ESIMD/api/functional/ctors/ctor_copy.cpp
=======
} // namespace esimd_test::api::functional::ctors
>>>>>>> c1366f1d7 ([SYCL][ESIMD] Split tests on simd constructors into core and fp_extra (#748)):SYCL/ESIMD/api/functional/ctors/ctor_copy.cpp
