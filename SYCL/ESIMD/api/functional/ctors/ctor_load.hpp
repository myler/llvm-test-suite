//===-- ctor_load.hpp - Functions for tests on simd load constructor.
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd load constructor
///
//===----------------------------------------------------------------------===//

#pragma once
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS
=======
>>>>>>> 11cb2778d ([SYCL][ESIMD] Rename tests on USM simd load constructors  (#950))

#include "common.hpp"

#include <string>

namespace esimd = sycl::ext::intel::esimd;
<<<<<<< HEAD
=======
=======
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))

#include "common.hpp"

namespace esimd = sycl::ext::intel::experimental::esimd;
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
>>>>>>> b2897f953 ([SYCL][ESIMD] Move some ESIMD APIs outside of experimental namespace (#892))

namespace esimd_test::api::functional::ctors {

// Dummy kernel for submitting some code into device side.
template <typename DataT, int NumElems, typename T, typename Alignment>
struct Kernel_for_load_ctor;

// Alignment tags
namespace alignment {

struct element {
<<<<<<< HEAD
<<<<<<< HEAD
  static std::string to_string() { return "element_aligned"; }
<<<<<<< HEAD
=======
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
  static std::string to_string() { return "element_aligned"; }
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
  template <typename DataT, int> static size_t get_size() {
=======
  template <typename DataT, int> static constexpr size_t get_size() {
>>>>>>> 034142eb8 ([SYCL][ESIMD] Add tests on simd load from accessors (#921))
    return alignof(DataT);
  }
  static constexpr auto get_value() { return esimd::element_aligned; }
};

struct vector {
<<<<<<< HEAD
<<<<<<< HEAD
  static std::string to_string() { return "vector_aligned"; }
=======
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
  static std::string to_string() { return "vector_aligned"; }
<<<<<<< HEAD
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
  template <typename DataT, int NumElems> static size_t get_size() {
=======
  template <typename DataT, int NumElems> static constexpr size_t get_size() {
>>>>>>> 034142eb8 ([SYCL][ESIMD] Add tests on simd load from accessors (#921))
    // Referring to the simd class specialization on the host side is by design.
    return alignof(esimd::simd<DataT, NumElems>);
  }
  static constexpr auto get_value() { return esimd::vector_aligned; }
};

<<<<<<< HEAD
struct overal {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  static std::string to_string() { return "overaligned"; }
=======
>>>>>>> 78c3d9b33 ([SYCL][ESIMD] Replace std::max_align_t with 16 for overaligned (#846))
=======
  static std::string to_string() { return "overaligned"; }
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
=======
template <unsigned int size = 16 /*oword alignment*/> struct overal {
>>>>>>> 034142eb8 ([SYCL][ESIMD] Add tests on simd load from accessors (#921))
  // Use 16 instead of std::max_align_t because of the fact that long double is
  // not a native type in Intel GPUs. So 16 is not driven by any type, but
  // rather the "oword alignment" requirement for all block loads. In that
  // sense, std::max_align_t would give wrong idea.

<<<<<<< HEAD
  static constexpr auto get_value() { return esimd::overaligned<oword_align>; }
<<<<<<< HEAD
=======
  template <typename, int> static size_t get_size() {
    return alignof(std::max_align_t);
  }
  static constexpr auto get_value() {
    return esimd::overaligned<alignof(std::max_align_t)>;
  }
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
>>>>>>> 78c3d9b33 ([SYCL][ESIMD] Replace std::max_align_t with 16 for overaligned (#846))
=======
  static std::string to_string() {
    return "overaligned<" + std::to_string(size) + ">";
  }

  template <typename DataT, int> static constexpr size_t get_size() {
    static_assert(size % alignof(DataT) == 0,
                  "Unsupported data type alignment");
    return size;
  }

  static constexpr auto get_value() { return esimd::overaligned<size>; }
>>>>>>> 034142eb8 ([SYCL][ESIMD] Add tests on simd load from accessors (#921))
};

} // namespace alignment

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
// Detailed test case description to use for logs
template <int NumElems, typename TestCaseT>
class LoadCtorTestDescription : public ITestDescription {
public:
  LoadCtorTestDescription(const std::string &data_type,
                          const std::string &alignment_name)
      : m_description(data_type), m_alignment_name(alignment_name) {}

  std::string to_string() const override {
    return m_description.to_string() + ", with alignment: " + m_alignment_name;
  }

private:
  const ctors::TestDescription<NumElems, TestCaseT> m_description;
  const std::string m_alignment_name;
};

<<<<<<< HEAD
<<<<<<< HEAD
// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, typename SizeT, typename TestCaseT,
          typename AlignmentT>
class run_test {
  static constexpr int NumElems = SizeT::value;
  using TestDescriptionT = LoadCtorTestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type,
                  const std::string &alignment_name) {
    bool passed = true;
    log::trace<TestDescriptionT>(data_type, alignment_name);

=======
=======
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, typename SizeT, typename TestCaseT,
          typename AlignmentT>
class run_test {
  static constexpr int NumElems = SizeT::value;
  using TestDescriptionT = LoadCtorTestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type,
                  const std::string &alignment_name) {
    bool passed = true;
<<<<<<< HEAD
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
    log::trace<TestDescriptionT>(data_type, alignment_name);

>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
<<<<<<< HEAD
<<<<<<< HEAD
        passed =
            run_verification(queue, {ref_data[i]}, data_type, alignment_name);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type, alignment_name);
=======
        passed = run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type);
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
        passed =
            run_verification(queue, {ref_data[i]}, data_type, alignment_name);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type, alignment_name);
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<DataT> &ref_data,
<<<<<<< HEAD
<<<<<<< HEAD
                        const std::string &data_type,
                        const std::string &alignment_name) {
=======
                        const std::string &data_type) {
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
                        const std::string &data_type,
                        const std::string &alignment_name) {
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    const size_t alignment_value =
        AlignmentT::template get_size<DataT, NumElems>();
    const size_t container_extra_size = alignment_value / sizeof(DataT) + 1;
    const size_t offset = 1;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(NumElems + container_extra_size +
                                             offset,
                                         shared_allocator<DataT>(queue));

    const size_t object_size = NumElems * sizeof(DataT);
    size_t buffer_size = object_size + container_extra_size * sizeof(DataT);

    // When we allocate USM there is a high probability that this memory will
    // have stronger alignment that required. We increment our pointer by fixed
    // offset value to avoid bigger alignment of USM shared.
    // The std::align can provide expected alignment on the small values of an
    // alignment.
    void *ref = shared_ref_data.data() + offset;
    if (std::align(alignment_value, object_size, ref, buffer_size) == nullptr) {
      return false;
    }
    DataT *const ref_aligned = static_cast<DataT *>(ref);

    for (size_t i = 0; i < NumElems; ++i) {
      ref_aligned[i] = ref_data[i];
    }

    queue.submit([&](sycl::handler &cgh) {
      DataT *const out = result.data();

      cgh.single_task<
          Kernel_for_load_ctor<DataT, NumElems, TestCaseT, AlignmentT>>(
          [=]() SYCL_ESIMD_KERNEL {
            const auto alignment = AlignmentT::get_value();
            TestCaseT::template call_simd_ctor<DataT, NumElems>(ref_aligned,
                                                                out, alignment);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
      const auto &expected = ref_data[i];
      const auto &retrieved = result[i];

      if (!are_bitwise_equal(expected, retrieved)) {
<<<<<<< HEAD
        passed = false;

        log::fail(TestDescriptionT(data_type, alignment_name),
                  "Unexpected value at index ", i, ", retrieved: ", retrieved,
                  ", expected: ", expected);
=======
      if (!are_bitwise_equal(ref_data[i], result[i])) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, TestCaseT>(
                i, result[i], ref_data[i], data_type);
        log::fail(description);
>>>>>>> 7ffc560aa ([SYCL][ESIMD] Add test on simd load constructor for fp_extra types (#797))
=======
        passed = false;

        log::fail(TestDescriptionT(data_type, alignment_name),
                  "Unexpected value at index ", i, ", retrieved: ", retrieved,
                  ", expected: ", expected);
>>>>>>> 05418ade9 ([SYCL][ESIMD] Make logs architecture more flexible (#838))
      }
    }

    return passed;
  }
};

=======
>>>>>>> 11cb2778d ([SYCL][ESIMD] Rename tests on USM simd load constructors  (#950))
} // namespace esimd_test::api::functional::ctors
