//==------- ctor_fill_core.cpp  - DPC++ ESIMD on-device test ---------------==//
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
// Test for simd fill constructor for core types.
// This test uses different data types, dimensionality, base and step values and
// different simd constructor invocation contexts. The test do the following
// actions:
//  - construct simd with pre-defined base and step value
//  - bitwise comparing expected and retrieved values

#include "ctor_fill.hpp"

using namespace esimd_test::api::functional;
using init_val = ctors::init_val;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;
  // Checks are run for specific combinations of types, vector length,
  // invocation contexts, base and step values accumulating the result in
  // boolean flag.

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 6870ea3ee ([SYCL][ESIMD] Provide the for_all_combinations utility (#721))
  {
    // Validate basic functionality works for every invocation context
    const auto types = named_type_pack<char, int>::generate("char", "int");
    const auto dims = get_dimensions<1, 8>();
    const auto contexts =
        unnamed_type_pack<ctors::initializer, ctors::var_decl,
                          ctors::rval_in_expr, ctors::const_ref>::generate();
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::min_half>();
      const auto step_values = ctors::get_init_values_pack<init_val::zero>();
      passed &= for_all_combinations<ctors::run_test>(
          types, dims, contexts, base_values, step_values, queue);
    }
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::min_half, init_val::zero>();
      const auto step_values =
          ctors::get_init_values_pack<init_val::positive>();
      passed &= for_all_combinations<ctors::run_test>(
          types, dims, contexts, base_values, step_values, queue);
    }
  }
  {
    // Validate basic functionality works for every type
<<<<<<< HEAD
    const auto types = get_tested_types<tested_types::core>();
=======
    const auto types = get_tested_types<tested_types::all>();
>>>>>>> 6870ea3ee ([SYCL][ESIMD] Provide the for_all_combinations utility (#721))
    const auto dims = get_all_dimensions();
    const auto contexts = unnamed_type_pack<ctors::var_decl>::generate();
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::min, init_val::max_half>();
      const auto step_values = ctors::get_init_values_pack<init_val::zero>();
      passed &= for_all_combinations<ctors::run_test>(
          types, dims, contexts, base_values, step_values, queue);
    }
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::zero, init_val::max_half>();
      const auto step_values =
          ctors::get_init_values_pack<init_val::positive, init_val::negative>();
      passed &= for_all_combinations<ctors::run_test>(
          types, dims, contexts, base_values, step_values, queue);
    }
  }
  {
    // Verify specific cases for different type groups
    const auto dims = get_dimensions<8>();
    const auto contexts = unnamed_type_pack<ctors::var_decl>::generate();
    {
      const auto types = get_tested_types<tested_types::uint>();
      {
        const auto base_values =
            ctors::get_init_values_pack<init_val::min, init_val::max>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::positive,
                                        init_val::negative>();
        passed &= for_all_combinations<ctors::run_test>(
            types, dims, contexts, base_values, step_values, queue);
      }
    }
    {
      const auto types = get_tested_types<tested_types::sint>();
      {
        const auto base_values = ctors::get_init_values_pack<init_val::min>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::positive>();
        passed &= for_all_combinations<ctors::run_test>(
            types, dims, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values = ctors::get_init_values_pack<init_val::max>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::negative>();
        passed &= for_all_combinations<ctors::run_test>(
            types, dims, contexts, base_values, step_values, queue);
      }
    }
    {
      const auto types = get_tested_types<tested_types::fp>();
      {
        const auto base_values =
            ctors::get_init_values_pack<init_val::neg_inf>();
        const auto step_values = ctors::get_init_values_pack<init_val::max>();
        passed &= for_all_combinations<ctors::run_test>(
            types, dims, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values = ctors::get_init_values_pack<init_val::max>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::neg_inf>();
        passed &= for_all_combinations<ctors::run_test>(
            types, dims, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values = ctors::get_init_values_pack<init_val::nan>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::negative>();
        passed &= for_all_combinations<ctors::run_test>(
            types, dims, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values =
            ctors::get_init_values_pack<init_val::negative>();
        const auto step_values = ctors::get_init_values_pack<init_val::nan>();
        passed &= for_all_combinations<ctors::run_test>(
            types, dims, contexts, base_values, step_values, queue);
      }
    }
  }
<<<<<<< HEAD
=======
  const auto two_dims = values_pack<1, 8>();
  const auto char_int_types = named_type_pack<char, int>({"char", "int"});

  // Run for specific combinations of types, vector length, base and step values
  // and invocation contexts.
  // The first init_val value it's a base value and the second init_val value
  // it's a step value.
  passed &=
      ctors::run_verification<ctors::initializer, ctors::init_val::min_half,
                              ctors::init_val::zero>(queue, two_dims,
                                                     char_int_types);

  passed &= ctors::run_verification<ctors::initializer, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);
  passed &=
      ctors::run_verification<ctors::initializer, ctors::init_val::min_half,
                              ctors::init_val::positive>(queue, two_dims,
                                                         char_int_types);

  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min_half,
                                    ctors::init_val::zero>(queue, two_dims,
                                                           char_int_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min_half,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);

  passed &=
      ctors::run_verification<ctors::rval_in_express, ctors::init_val::min_half,
                              ctors::init_val::zero>(queue, two_dims,
                                                     char_int_types);
  passed &=
      ctors::run_verification<ctors::rval_in_express, ctors::init_val::zero,
                              ctors::init_val::positive>(queue, two_dims,
                                                         char_int_types);
  passed &=
      ctors::run_verification<ctors::rval_in_express, ctors::init_val::min_half,
                              ctors::init_val::positive>(queue, two_dims,
                                                         char_int_types);

  passed &= ctors::run_verification<ctors::const_ref, ctors::init_val::min_half,
                                    ctors::init_val::zero>(queue, two_dims,
                                                           char_int_types);
  passed &= ctors::run_verification<ctors::const_ref, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);
  passed &= ctors::run_verification<ctors::const_ref, ctors::init_val::min_half,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);

  const auto all_dims = get_all_dimensions();
  const auto all_types = get_tested_types<tested_types::all>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::zero>(queue, all_dims,
                                                           all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max_half,
                                    ctors::init_val::zero>(queue, all_dims,
                                                           all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, all_dims,
                                                               all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max_half,
                                    ctors::init_val::positive>(queue, all_dims,
                                                               all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::zero,
                                    ctors::init_val::negative>(queue, all_dims,
                                                               all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max_half,
                                    ctors::init_val::negative>(queue, all_dims,
                                                               all_types);

  const auto single_dim = values_pack<8>();
  const auto uint_types = get_tested_types<tested_types::uint>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::positive>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::negative>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::positive>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::negative>(
      queue, single_dim, uint_types);

  const auto sint_types = get_tested_types<tested_types::sint>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::positive>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::negative>(
      queue, single_dim, uint_types);

  const auto fp_types = get_tested_types<tested_types::fp>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::neg_inf,
                                    ctors::init_val::max>(queue, single_dim,
                                                          fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::neg_inf>(queue, single_dim,
                                                              fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::nan,
                                    ctors::init_val::negative>(
      queue, single_dim, fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::negative,
                                    ctors::init_val::nan>(queue, single_dim,
                                                          fp_types);
>>>>>>> 7ff842b8f ([SYCL][ESIMD] Enable verification for 32 simd vector length (#758))
=======
>>>>>>> 6870ea3ee ([SYCL][ESIMD] Provide the for_all_combinations utility (#721))

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
