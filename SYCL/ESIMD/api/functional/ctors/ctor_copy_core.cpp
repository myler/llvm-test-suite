//==------- ctor_copy_core.cpp  - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Test for simd copy constructor.
<<<<<<< HEAD
// This test uses different data types, sizes and different simd constructor
// invocation contexts.
=======
// This test uses different data types, dimensionality and different simd
// constructor invocation contexts.
>>>>>>> c1366f1d7 ([SYCL][ESIMD] Split tests on simd constructors into core and fp_extra (#748))
// The test do the following actions:
//  - construct simd object with golden values calls copy constructor from
//    constructed simd object
//  - bitwise comparing expected and retrieved values

#include "ctor_copy.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::core>();
<<<<<<< HEAD
  const auto sizes = get_all_sizes();
=======
  const auto dims = get_all_dimensions();
>>>>>>> c1366f1d7 ([SYCL][ESIMD] Split tests on simd constructors into core and fp_extra (#748))
  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();

<<<<<<< HEAD
  passed &=
      for_all_combinations<ctors::run_test>(types, sizes, contexts, queue);
=======
  passed &= for_all_combinations<ctors::run_test>(types, dims, contexts, queue);
>>>>>>> c1366f1d7 ([SYCL][ESIMD] Split tests on simd constructors into core and fp_extra (#748))

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
