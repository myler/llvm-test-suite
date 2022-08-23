// This test is intended to check basic operations with SYCL 2020 specialization
// constants using sycl::kernel_bundle and sycl::kernel_handler APIs:
// - test that specialization constants can be accessed in kernel and they
//   have their default values if `set_specialization_constants` wasn't called
// - test that specialization constant values can be set and retrieved through
//   kernel_bundle APIs on host
// - test that specialization constant values can be set through kernel_bundle
//   API and correctly retrieved within a kernel
//
// REQUIRES: aspect-fp64
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DENABLE_FP64 %s -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// FIXME: SYCL 2020 specialization constants are not supported on host device
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// FIXME: ACC devices use emulation path, which is not yet supported
// UNSUPPORTED: hip

#include "kernel-bundle-api.cpp"
