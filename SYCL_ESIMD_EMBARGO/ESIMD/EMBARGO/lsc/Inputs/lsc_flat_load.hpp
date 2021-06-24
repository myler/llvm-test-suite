#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

#include "common.hpp"

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T, uint32_t Groups, uint32_t Threads, uint16_t VL,
          uint16_t VS, bool transpose = false,
          lsc_data_size DS = lsc_data_size::default_size,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
bool test(uint16_t case_num, uint32_t pmask = 0) {
  static_assert((VL == 1) || !transpose, "Transpose must have exec size 1");
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(!transpose, "Conversion types may not use vector");
    static_assert(VS == 1, "Only D32 and D64 support vector load");
  }

  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");
  static_assert(sizeof(T) >= 4,
                "D8 and D16 are valid only in 2D block load/store");

  uint16_t Size = Threads * VL * VS;

  T vmask = (T)-1;
  if constexpr (DS == lsc_data_size::u8u32)
    vmask = (T)0xff;
  if constexpr (DS == lsc_data_size::u16u32)
    vmask = (T)0xffff;
  if constexpr (DS == lsc_data_size::u16u32h)
    vmask = (T)0xffff0000;

  T old_val = get_rand<T>();
  T zero_val = (T)0;

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto dev = q.get_device();
  std::cout << "Running case #" << case_num << " on "
            << dev.get_info<info::device::name>() << "\n";
  auto ctx = q.get_context();

  // workgroups
  cl::sycl::range<1> GlobalRange{Groups};
  // threads in each group
  cl::sycl::range<1> LocalRange{Threads};
  cl::sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    out[i] = old_val;

  T *in = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    in[i] = get_rand<T>();

  std::vector<uint16_t> p(VL, 0);
  if constexpr (!transpose)
    for (int i = 0; i < VL; i++)
      p[i] = (pmask >> i) & 1;

  try {
    buffer<uint16_t, 1> bufp(p.data(), p.size());

    auto e = q.submit([&](handler &cgh) {
      auto accp = bufp.template get_access<access::mode::read>(cgh);
      cgh.parallel_for<class Test>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint16_t idx = ndi.get_local_id(0);
            uint16_t o = idx * VL * VS;
            uint32_t f = o * sizeof(T);

            if constexpr (transpose) {
              auto vals = lsc_flat_load<T, VS, DS, L1H, L3H>(in + o);
              lsc_flat_store<T, VS, lsc_data_size::default_size,
                             CacheHint::None, CacheHint::None>(out + o, vals);
            } else {
              simd<uint32_t, VL> offset(f, VS * sizeof(T));
              simd<uint16_t, VL> pred = lsc_surf_load<uint16_t, VL>(accp, 0);
              simd<T, VS * VL> vals(0);
              simd<uint16_t, VS * VL> mask(0);
              for (int i = 0; i < VS; i++)
                mask.template select<VL, 1>(i * VL) = pred;
              vals.merge(
                  lsc_flat_load<T, VS, DS, L1H, L3H, VL>(in, offset, pred),
                  mask);
              lsc_flat_store<T, VS, lsc_data_size::default_size,
                             CacheHint::None, CacheHint::None, VL>(out, vals,
                                                                   offset);
            }
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(out, ctx);
    sycl::free(in, ctx);
    return e.get_cl_code();
  }

  bool passed = true;

  if (!transpose) {
    uint16_t add_per_thread = VL * VS;
    uint16_t add_per_vs = VL;
    for (int t = 0; t < Threads; t++) {
      for (int i = 0; i < VL; i++) {
        for (int s = 0; s < VS; s++) {
          int idx = s + i * VS + t * VL * VS;
          T e = ((pmask >> i) & 1) == 0 ? 0 : in[idx] & vmask;
          if (out[idx] != e) {
            passed = false;
            std::cout << "out[" << idx << "] = 0x" << std::hex
                      << (uint64_t)out[idx] << " vs etalon = 0x" << (uint64_t)e
                      << std::dec << std::endl;
          }
        }
      }
    }
  } else {
    for (int i = 0; i < Size; i++) {
      if (out[i] != in[i]) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << (uint64_t)out[i]
                  << " vs etalon = 0x" << (uint64_t)in[i] << std::dec
                  << std::endl;
      }
    }
  }

  if (!passed)
    std::cout << "Case #" << case_num << " FAILED" << std::endl;

  sycl::free(out, ctx);
  sycl::free(in, ctx);

  return passed;
}
