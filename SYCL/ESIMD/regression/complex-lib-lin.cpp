// This test is intended to check a certain approach of compiling libraries and
// application, when both regular SYCL and ESIMD are used.
//
// We used to have a bug, when under some circumstances compiler created empty
// device images, but at the same time it stated that they contain some kernels.
// More details can be found in intel/llvm#4927.
//
// REQUIRES: linux,gpu
<<<<<<< HEAD
// UNSUPPORTED: cuda || hip
// TODO/DEBUG Segmentation fault occurs with esimd_emulator backend
// XFAIL: esimd_emulator
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
// The test hangs on 22.05.22297 GPU RT on Linux
// UNSUPPORTED: linux && (opencl || level_zero) && gpu
=======
>>>>>>> d98407d06 ([SYCL][ESIMD][EMU] Marking ESIMD kernels for esimd_emulator backend (#751))
=======
// The test hangs on 22.05.22297 GPU RT on Linux
// UNSUPPORTED: linux && (opencl || level_zero) && gpu
>>>>>>> 5e8f630e2 ([SYCL] Align tests with 22.05.22297 GPU RT (#871))
=======
>>>>>>> b8c62d2d5 ([ESIMD] Fix the complex-lib-lin checking compilation from static library (#923))
=======
// UNSUPPORTED: cuda || hip || esimd_emulator
// TODO: running non-ESIMD kernels on esimd_emulator backend.
>>>>>>> b17112500 ([SYCL][ESIMD][EMU] Running non-ESIMD kernels on esimd_emulator backend is not supported. (#953))
//
// RUN: %clangxx -fsycl -fPIC -O3 %S/Inputs/complex-lib-sycl.cpp -c -o %t-lib-sycl.o
// RUN: %clangxx -fsycl -fPIC -O3 %S/Inputs/complex-lib-esimd.cpp -c -o %t-lib-esimd.o
// RUN: %clangxx -fsycl -fPIC -O3 %S/Inputs/complex-lib-test.cpp -c -o %t-test.o
//
// RUN: ar crsv %t-lib-sycl.a %t-lib-sycl.o
// RUN: ar crsv %t-lib-esimd.a %t-lib-esimd.o
//
// One shared library is built using static libraries
//
// RUN: %clangxx -fsycl -shared %t-lib-sycl.a %t-lib-esimd.a \
// RUN:  -fsycl-device-code-split=per_kernel -Wl,--whole-archive \
// RUN:  %t-lib-sycl.a %t-lib-esimd.a -Wl,--no-whole-archive -Wl,-soname,%S -o %t-lib-a.so
//
// And another one is constructed directly from object files
//
// RUN: %clangxx -fsycl -shared %t-lib-sycl.o %t-lib-esimd.o \
// RUN:  -fsycl-device-code-split=per_kernel -Wl,-soname,%S -o %t-lib-o.so
//
// RUN: %clangxx -fsycl %t-test.o %t-lib-a.so -o %t-a.run
// RUN: %clangxx -fsycl %t-test.o %t-lib-o.so -o %t-o.run
//
// FIXME: is there better way to handle libraries loading than LD_PRELOAD?
// There is no LIT substitution, which would point to a directory, where
// temporary files are located. There is %T, but it is marked as "deprecated,
// do not use"
// RUN: %GPU_RUN_PLACEHOLDER LD_PRELOAD=%t-lib-a.so %t-a.run
// RUN: %GPU_RUN_PLACEHOLDER LD_PRELOAD=%t-lib-o.so %t-o.run
