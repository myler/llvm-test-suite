//==-------- joint_matrix_bfloat16.cpp  - DPC++ joint_matrix----------- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8

<<<<<<< HEAD
<<<<<<<< HEAD:SYCL/Matrix/Legacy/XMX8/joint_matrix_bfloat16.cpp
// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=1
========
// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
>>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391)):SYCL/Matrix/XMX8/joint_matrix_bfloat16_use.cpp
=======
// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=1
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
<<<<<<< HEAD
using bfloat16 = sycl::ext::oneapi::bfloat16;
=======
using bfloat16 = sycl::ext::oneapi::experimental::bfloat16;
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))

#define SG_SZ 8

#include "../joint_matrix_bfloat16_impl.hpp"
