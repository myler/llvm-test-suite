// Enable FP64 part of <handler-api.cpp>. To be removed once DPC++
// supports optional device features and the code could be enabled
// unconditionally without causing failures in speculative compilation
// of the kernels.
//
// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DENABLE_FP64 %s -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// FIXME: SYCL 2020 specialization constants are not supported on host device
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// FIXME: ACC devices use emulation path, which is not yet supported
// UNSUPPORTED: hip

#include "handler-api.cpp"
