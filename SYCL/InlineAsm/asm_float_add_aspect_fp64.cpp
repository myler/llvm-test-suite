// Enable FP64 part of <asm_float_add.cpp>. To be removed once DPC++
// supports optional device features and the code could be enabled
// unconditionally without causing failures in speculative compilation
// of the kernels.
//
// UNSUPPORTED: cuda || hip_nvidia
// REQUIRES: gpu,linux,aspect-fp64
// RUN: %clangxx -fsycl -DENABLE_FP64 %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "asm_float_add.cpp"
