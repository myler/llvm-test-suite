<<<<<<< HEAD
<<<<<<<< HEAD:SYCL/Matrix/Legacy/joint_matrix_bfloat16_32x64.cpp
//==----- joint_matrix_bfloat16_32x64.cpp  - DPC++ joint_matrix-------------==//
========
//==-------- joint_matrix_bf16_vnni.cpp  - DPC++ joint_matrix---------------==//
>>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391)):SYCL/Matrix/Legacy/joint_matrix_int8_vnni.cpp
=======
//==----- joint_matrix_bfloat16_32x64.cpp  - DPC++ joint_matrix-------------==//
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=1
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// XFAIL: *

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
<<<<<<< HEAD
<<<<<<< HEAD

#define SG_SZ 16

<<<<<<<< HEAD:SYCL/Matrix/Legacy/joint_matrix_bfloat16_32x64.cpp
#include "joint_matrix_bfloat16_32x64_impl.hpp"
========
#include "joint_matrix_int8_vnni_impl.hpp"
>>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391)):SYCL/Matrix/Legacy/joint_matrix_int8_vnni.cpp
=======
using bfloat16 = sycl::ext::oneapi::experimental::bfloat16;
=======
using bfloat16 = sycl::ext::oneapi::bfloat16;
>>>>>>> 87f7445c7 ([SYCL][Matrix] Fix bfloat16 namespace in the legacy tests (#1478))

#define SG_SZ 16

#include "joint_matrix_bfloat16_32x64_impl.hpp"
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
