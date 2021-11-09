// REQUIRES: windows
// RUN: %clangxx -DSYCL_ENABLE_FALLBACK_ASSERT -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// FIXME Windows versionprints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_one_kernel.hpp:10: {{<unknown func>|(null)}}: global id: [{{[0-3]}},0,0], local id: [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] != 0 && "from assert statement"` failed.
// CHECK-NOT:  The test ended.

#include "assert_in_one_kernel.hpp"
