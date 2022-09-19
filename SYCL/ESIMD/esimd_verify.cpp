// REQUIRES: gpu
// RUN: not %clangxx -fsycl -fsycl-device-only %s 2>&1 | FileCheck %s

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// CHECK: error: function 'sycl::{{.+}}multi_ptr<{{.+}}> sycl::{{.+}}accessor<{{.+}}>::get_pointer<{{.+}}>() const' is not supported in ESIMD context

SYCL_EXTERNAL auto
test(accessor<int, 1, access::mode::read_write, access::target::device> &acc)
    SYCL_ESIMD_FUNCTION {
  return acc.get_pointer();
}
