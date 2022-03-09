//==------- operator_assignment_move_and_copy_core.cpp  - DPC++ ESIMD on-device
//          test -----------------------------------------------------------==//
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
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *
// TODO Remove XFAIL once the simd vector provides move assignment operator
//
// Test for simd move and copy assignment operators.
// The test creates source simd instance with reference data and invokes move or
// copy assignment operator from source to destination simd instance. It is
// expected for destination simd instance to store a bitwise same data as the
// reference one.

#include "operator_assignment.hpp"

<<<<<<< HEAD
<<<<<<< HEAD
using namespace sycl::ext::intel::esimd;
=======
using namespace sycl::ext::intel::experimental::esimd;
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
using namespace sycl::ext::intel::esimd;
>>>>>>> b2897f953 ([SYCL][ESIMD] Move some ESIMD APIs outside of experimental namespace (#892))
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling move assignment operator.
struct move_assignment {
  static std::string get_description() { return "move assignment operator"; }

<<<<<<< HEAD
<<<<<<< HEAD
  static constexpr bool is_move_expected() { return true; }

=======
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
  static constexpr bool is_move_expected() { return true; }

>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
  template <typename DataT, int NumElems>
  static bool run(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_obj;
    simd_obj = std::move(source_simd);
    simd_obj.copy_to(out);
<<<<<<< HEAD
<<<<<<< HEAD
    return simd_obj.get_test_proxy().was_move_destination();
=======
    return simd_obj.get_test_proxy().was_move_destination() == true;
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
    return simd_obj.get_test_proxy().was_move_destination();
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
  }
};

// Descriptor class for the case of calling copy assignment operator.
struct copy_assignment {
  static std::string get_description() { return "copy assignment operator"; }

<<<<<<< HEAD
<<<<<<< HEAD
  static constexpr bool is_move_expected() { return false; }

=======
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
  static constexpr bool is_move_expected() { return false; }

>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
  template <typename DataT, int NumElems>
  static bool run(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_obj;
    simd_obj = source_simd;
    simd_obj.copy_to(out);
<<<<<<< HEAD
<<<<<<< HEAD
    return simd_obj.get_test_proxy().was_move_destination();
=======
    return simd_obj.get_test_proxy().was_move_destination() == false;
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
    return simd_obj.get_test_proxy().was_move_destination();
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

<<<<<<< HEAD
<<<<<<< HEAD
  const auto types = get_tested_types<tested_types::all>();
<<<<<<< HEAD
  const auto all_sizes = get_all_sizes();
=======
=======
  const auto types = get_tested_types<tested_types::core>();
>>>>>>> d26b27f31 (Hotfix types for operator_assignment_move_and_copy_core (#833))
  const auto dims = get_all_dimensions();
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
  const auto types = get_tested_types<tested_types::all>();
  const auto all_sizes = get_all_sizes();
>>>>>>> e737b795e ([SYCL][ESIMD] Add tests for simd broadcast constructor (#690))

  const auto context =
      unnamed_type_pack<move_assignment, copy_assignment>::generate();

<<<<<<< HEAD
<<<<<<< HEAD
  passed &= for_all_combinations<operators::run_test>(types, all_sizes, context,
                                                      queue);
=======
  passed &=
      for_all_combinations<operators::run_test>(types, dims, context, queue);
>>>>>>> 1017d075e ([SYCL][ESIMD] Add tests on simd copy and move assignment operators (#762))
=======
  passed &= for_all_combinations<operators::run_test>(types, all_sizes, context,
                                                      queue);
>>>>>>> e737b795e ([SYCL][ESIMD] Add tests for simd broadcast constructor (#690))

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
