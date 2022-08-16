#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <class T, int VL, class Ops> class TestID;

// Helpers for printing
template <class T> auto cast(T val) { return val; }
template <> auto cast<char>(char val) { return (int)val; }
template <> auto cast<unsigned char>(unsigned char val) {
  return (unsigned int)val;
}
#ifdef __SYCL_DEVICE_ONLY__
template <> auto cast<_Float16>(_Float16 val) { return (float)val; }
#endif

// Main test function.
// T - operand type,
// VL - vector length,
// Ops - a compile-time sequence of operations to test.
//
template <class T, int VL, class Ops, template <class, int> class SimdT = simd>
bool test(Ops ops, queue &q) {
  using OpClass = esimd_test::UnaryOp;
  // Log test case info
  std::cout << "Testing T=" << typeid(T).name() << ", VL=" << VL << " ...\n";
  std::cout << "Operations:";
  esimd_test::iterate_ops(ops, [=](OpClass op) {
    std::cout << " '" << esimd_test::Op2Str(op) << "'";
  });
  std::cout << "\n";

  // initialize test data
  constexpr int Size = 1024 * 7;
  T *A = sycl::malloc_shared<T>(Size, q);
  constexpr int NumOps = (int)Ops::size;
  int CSize = NumOps * Size;
  T *C = sycl::malloc_shared<T>(CSize, q);

  for (int i = 0; i < Size; ++i) {
    if constexpr (std::is_unsigned_v<T>) {
      A[i] = i;
    } else {
      A[i] = i - Size / 2;
    }
    C[i] = 0;
  }

  // submit the kernel
  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<TestID<T, VL, Ops>>(
          Size / VL, [=](id<1> i) SYCL_ESIMD_KERNEL {
            unsigned off = i * VL;
            SimdT<T, VL> va(A + off);
            // applies each of the input operations to the va,
            // then invokes the lambda below, passing the result of the
            // operation, its ID and sequential number within the input sequence
            esimd_test::apply_unary_ops(
                ops, va, [=](SimdT<T, VL> res, OpClass op, unsigned op_num) {
                  unsigned res_off = off * NumOps + op_num * VL;
                  res.copy_to(C + res_off);
                });
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(C, q);
    return false;
  }

  int err_cnt = 0;

  // now verify the results using provided verification function type
  for (unsigned i = 0; i < Size / VL; ++i) {
    unsigned off = i * VL;

    for (int j = 0; j < VL; ++j) {
      T a = A[off + j];

      esimd_test::apply_unary_ops(
          ops, a, [&](T Gold, OpClass op, unsigned op_num) {
            unsigned res_off = off * NumOps + op_num * VL;
            T Res = C[res_off + j];
            using Tint = esimd_test::int_type_t<sizeof(T)>;
            Tint ResBits = *(Tint *)&Res;
            Tint GoldBits = *(Tint *)&Gold;
            // allow 1 bit discrepancy for half on modifying op
            int delta = ((int)op >= (int)OpClass::minus_minus_pref) &&
                                ((int)op <= (int)OpClass::plus_plus_inf) &&
                                std::is_same_v<T, half>
                            ? 1
                            : 0;

            if ((Gold != Res) && (abs(ResBits - GoldBits) > delta)) {
              if (++err_cnt < 10) {
                std::cout << "  failed at index " << (res_off + j) << ", op "
                          << esimd_test::Op2Str(op) << ": " << cast(Res)
                          << "(0x" << std::hex << ResBits << ")"
                          << " != " << cast(Gold) << "(0x" << std::hex
                          << GoldBits << ") [" << esimd_test::Op2Str(op) << " "
                          << std::dec << cast(a) << "]\n";
              }
            }
          });
    }
  }
  if (err_cnt > 0) {
    auto Size1 = NumOps * Size;
    std::cout << "  pass rate: "
              << ((float)(Size1 - err_cnt) / (float)Size1) * 100.0f << "% ("
              << (Size1 - err_cnt) << "/" << Size1 << ")\n";
  }

  free(A, q);
  free(C, q);
  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt == 0;
}
