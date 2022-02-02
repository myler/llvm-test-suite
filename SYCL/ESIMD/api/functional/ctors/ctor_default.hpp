//===-- ctor_copy.hpp - Functions for tests on simd copy constructor definition.
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd copy constructor.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"

namespace esimd = sycl::ext::intel::experimental::esimd;

namespace esimd_test::api::functional::ctors {

// Descriptor class for the case of calling constructor in initializer context
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    const auto simd_by_init = esimd::simd<DataT, NumElems>();
    simd_by_init.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    esimd::simd<DataT, NumElems> simd_by_var_decl;
    simd_by_var_decl.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    esimd::simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = esimd::simd<DataT, NumElems>();
    simd_by_rval.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context
struct const_ref {
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *output_data) {
    simd_by_const_ref.copy_to(output_data);
  }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    call_simd_by_const_ref<DataT, NumElems>(esimd::simd<DataT, NumElems>(),
                                            output_data);
  }
};

// Struct that calls simd in provided context and then verifies obtained result.
template <typename DataT, typename DimT, typename TestCaseT> struct run_test {
  static constexpr int NumElems = DimT::value;

  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    DataT default_val{};

    shared_vector<DataT> result(NumElems, shared_allocator<DataT>(queue));

    queue.submit([&](sycl::handler &cgh) {
      DataT *const out = result.data();
      cgh.single_task<ctors::Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(out);
          });
    });

    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i] != default_val) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, TestCaseT>(
                i, result[i], default_val, data_type);
        log::fail(description);
      }
    }

    return passed;
  }
};

<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_default.hpp
<<<<<<< HEAD:SYCL/ESIMD/api/functional/ctors/ctor_default.hpp
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
>>>>>>> 6870ea3ee ([SYCL][ESIMD] Provide the for_all_combinations utility (#721)):SYCL/ESIMD/api/functional/ctors/ctor_default.cpp
=======
} // namespace esimd_test::api::functional::ctors
>>>>>>> c1366f1d7 ([SYCL][ESIMD] Split tests on simd constructors into core and fp_extra (#748)):SYCL/ESIMD/api/functional/ctors/ctor_default.cpp
