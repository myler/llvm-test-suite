#pragma once

#ifndef TEST_GENERIC_IN_LOCAL
#define TEST_GENERIC_IN_LOCAL 0
#endif

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void add_fetch_local_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(123456));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
       auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       accessor<T, 1, access::mode::read_write, access::target::local> loc(1,
                                                                           cgh);

       cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
         int gid = it.get_global_id(0);
         if (gid == 0)
           loc[0] = 0;
         it.barrier(access::fence_space::local_space);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (loc[0]);
         out[gid] = atm.fetch_add(Difference(1), order);
         it.barrier(access::fence_space::local_space);
         if (gid == 0)
           sum[0] = loc[0];
       });
     }).wait_and_throw();
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // Fetch returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == 0 && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void add_fetch_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
       auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       cgh.parallel_for(range<1>(N), [=](item<1> it) {
         int gid = it.get_id(0);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (sum[0]);
         out[gid] = atm.fetch_add(Difference(1), order);
       });
     }).wait_and_throw();
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // Fetch returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == 0 && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void add_plus_equal_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (sum[0]);
        out[gid] = atm += Difference(1);
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // += returns updated value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void add_pre_inc_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (sum[0]);
        out[gid] = ++atm;
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // Pre-increment returns updated value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void add_post_inc_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (sum[0]);
        out[gid] = atm++;
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == T(N));

  // Post-increment returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void add_test(queue q, size_t N) {
  constexpr bool do_local_tests =
      space == access::address_space::local_space ||
      (space == access::address_space::generic_space && TEST_GENERIC_IN_LOCAL);
  constexpr bool do_global_tests =
      space == access::address_space::global_space ||
      (space == access::address_space::generic_space && !TEST_GENERIC_IN_LOCAL);
  constexpr bool do_ext_tests = space != access::address_space::generic_space;
  if constexpr (do_local_tests) {
<<<<<<< HEAD
<<<<<<< HEAD
#ifdef RUN_DEPRECATED
=======
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#ifdef RUN_DEPRECATED
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
    if constexpr (do_ext_tests) {
      add_fetch_local_test<::sycl::ext::oneapi::atomic_ref, space, T,
                           Difference, order, scope>(q, N);
    }
<<<<<<< HEAD
<<<<<<< HEAD
#else
    add_fetch_local_test<::sycl::atomic_ref, space, T, Difference, order,
                         scope>(q, N);
#endif
  }
  if constexpr (do_global_tests) {
#ifdef RUN_DEPRECATED
=======
=======
#else
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
    add_fetch_local_test<::sycl::atomic_ref, space, T, Difference, order,
                         scope>(q, N);
#endif
  }
  if constexpr (do_global_tests) {
<<<<<<< HEAD
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#ifdef RUN_DEPRECATED
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
    if constexpr (do_ext_tests) {
      add_fetch_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                     order, scope>(q, N);
      add_plus_equal_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                          order, scope>(q, N);
      if constexpr (!std::is_floating_point_v<T>) {
        add_pre_inc_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                         order, scope>(q, N);
        add_post_inc_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                          order, scope>(q, N);
      }
    }
<<<<<<< HEAD
<<<<<<< HEAD
#else
=======
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#else
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
    add_fetch_test<::sycl::atomic_ref, space, T, Difference, order, scope>(q,
                                                                           N);
    add_plus_equal_test<::sycl::atomic_ref, space, T, Difference, order, scope>(
        q, N);
    if constexpr (!std::is_floating_point_v<T>) {
      add_pre_inc_test<::sycl::atomic_ref, space, T, Difference, order, scope>(
          q, N);
      add_post_inc_test<::sycl::atomic_ref, space, T, Difference, order, scope>(
          q, N);
    }
<<<<<<< HEAD
<<<<<<< HEAD
#endif
=======
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#endif
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
  }
}

template <access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed>
void add_test_scopes(queue q, size_t N) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
<<<<<<< HEAD
<<<<<<< HEAD
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) !=
      scopes.end()) {
    add_test<space, T, Difference, order, memory_scope::system>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    add_test<space, T, Difference, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    add_test<space, T, Difference, order, memory_scope::sub_group>(q, N);
  }
  add_test<space, T, Difference, order, memory_scope::device>(q, N);
=======
#if defined(SYSTEM)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
=======
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) !=
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
      scopes.end()) {
    add_test<space, T, Difference, order, memory_scope::system>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    add_test<space, T, Difference, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    add_test<space, T, Difference, order, memory_scope::sub_group>(q, N);
  }
  add_test<space, T, Difference, order, memory_scope::device>(q, N);
<<<<<<< HEAD
#endif
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
}

template <access::address_space space, typename T, typename Difference = T>
void add_test_orders_scopes(queue q, size_t N) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
<<<<<<< HEAD
<<<<<<< HEAD
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) !=
      orders.end()) {
    add_test_scopes<space, T, Difference, memory_order::acq_rel>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    add_test_scopes<space, T, Difference, memory_order::acquire>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    add_test_scopes<space, T, Difference, memory_order::release>(q, N);
  }
  add_test_scopes<space, T, Difference, memory_order::relaxed>(q, N);
=======
#if defined(ACQ_REL)
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) ==
=======
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) !=
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
      orders.end()) {
    add_test_scopes<space, T, Difference, memory_order::acq_rel>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    add_test_scopes<space, T, Difference, memory_order::acquire>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    add_test_scopes<space, T, Difference, memory_order::release>(q, N);
  }
  add_test_scopes<space, T, Difference, memory_order::relaxed>(q, N);
<<<<<<< HEAD
#endif
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
}

template <access::address_space space> void add_test_all() {
  queue q;

  constexpr int N = 32;
<<<<<<< HEAD
<<<<<<< HEAD
#ifdef FULL_ATOMIC64_COVERAGE
  add_test_orders_scopes<space, double>(q, N);
=======
#ifdef ATOMIC64
  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return;
  }

  add_test_orders_scopes<space, double>(q, N);
#ifndef FP_TESTS_ONLY
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#ifdef FULL_ATOMIC64_COVERAGE
  add_test_orders_scopes<space, double>(q, N);
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
  if constexpr (sizeof(long) == 8) {
    add_test_orders_scopes<space, long>(q, N);
    add_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(long long) == 8) {
    add_test_orders_scopes<space, long long>(q, N);
    add_test_orders_scopes<space, unsigned long long>(q, N);
  }
  if constexpr (sizeof(char *) == 8) {
    add_test_orders_scopes<space, char *, ptrdiff_t>(q, N);
  }
#endif
<<<<<<< HEAD
<<<<<<< HEAD
  add_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
=======
#else
  add_test_orders_scopes<space, float>(q, N);
#ifndef FP_TESTS_ONLY
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
  add_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
  add_test_orders_scopes<space, int>(q, N);
  add_test_orders_scopes<space, unsigned int>(q, N);
  if constexpr (sizeof(long) == 4) {
    add_test_orders_scopes<space, long>(q, N);
    add_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(char *) == 4) {
    add_test_orders_scopes<space, char *, ptrdiff_t>(q, N);
  }
#endif
<<<<<<< HEAD
<<<<<<< HEAD
=======
#endif

>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
  std::cout << "Test passed." << std::endl;
}
