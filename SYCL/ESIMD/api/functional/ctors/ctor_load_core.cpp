//==------- ctor_load_core.cpp  - DPC++ ESIMD on-device test ---------------==//
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
// Test for simd load constructor.
// The test uses reference data and different alignment flags. Invokes simd
// constructors in different contexts with provided reference data and alignment
// flag.
// It is expected for destination simd instance to store a bitwise same data as
// the reference one.

#include "ctor_load.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> d43bc4e32 ([SYCL][ESIMD] Decrease type coverage for core tests (#832))
  const auto types = get_tested_types<tested_types::core>();
<<<<<<< HEAD
=======
  const auto types = get_tested_types<tested_types::all>();
>>>>>>> 1548e68f8 ([SYCL][ESIMD] Add test on simd load ctor (#769))
=======
  const auto types = get_tested_types<tested_types::core>();
>>>>>>> 3caa01663 ([SYCL][ESIMD] Replace using tested_types::all with tested_types::core (#798))
  const auto dims = get_all_dimensions();
=======
  const auto sizes = get_all_sizes();
>>>>>>> e37c07509 ([SYCL][ESIMD] Replace "dim", "dimensions" with "size", "sizes", etc. (#803))

  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();
  const auto alignments =
      unnamed_type_pack<ctors::alignment::element, ctors::alignment::vector,
                        ctors::alignment::overal>::generate();

  passed &= for_all_combinations<ctors::run_test>(types, sizes, contexts,
                                                  alignments, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
