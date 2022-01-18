// REQUIRES: windows,timelimit,(opencl||level_zero)
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %timelimit %t.out > %t.txt 2>&1 || true
// RUN: cat %t.txt | FileCheck %s
// CHECK: timelimit:{{.*}}sycl!{{.*}}sycl::detail::enqueueImpKernel{{.*}}
// CHECK: timelimit:{{.*}}sycl!{{.*}}sycl::handler::finalize{{.*}}
// CHECK: timelimit:{{.*}}sycl!{{.*}}sycl::detail::queue_impl::finalizeHandler{{.*}}
// CHECK: timelimit:{{.*}}sycl!{{.*}}sycl::detail::queue_impl::submit_impl{{.*}}

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue Q;
  // nd_range_error is generated due to 0-based local range.
  Q.submit([&](handler &CGH) {
     CGH.parallel_for_work_group<class f>(range<2>(5, 33), range<2>(1, 0),
         [=](group<2>) {});
  });
  return 0;
}
