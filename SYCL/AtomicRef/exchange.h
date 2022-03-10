#pragma once

#ifndef TEST_GENERIC_IN_LOCAL
#define TEST_GENERIC_IN_LOCAL 0
#endif

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void exchange_local_test(queue q, size_t N) {
  const T initial = T(N);
  T cum = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(123456));
  {
    buffer<T> cum_buf(&cum, 1);
    buffer<T> output_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
       auto cum = cum_buf.template get_access<access::mode::read_write>(cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       accessor<T, 1, access::mode::read_write, access::target::local> loc(1,
                                                                           cgh);

       cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
         int gid = it.get_global_id(0);
         if (gid == 0)
           loc[0] = initial;
         it.barrier(access::fence_space::local_space);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (loc[0]);
         out[gid] = atm.exchange(T(gid), order);
         it.barrier(access::fence_space::local_space);
         if (gid == 0)
           cum[0] = loc[0];
       });
     }).wait_and_throw();
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be unique; each work-item replaces the value it
  // reads with its own ID
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void exchange_global_test(queue q, size_t N) {
  const T initial = T(N);
  T exchange = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> exchange_buf(&exchange, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto exc =
          exchange_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (exc[0]);
        out[gid] = atm.exchange(T(gid), order);
      });
    });
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be unique; each work-item replaces the value it
  // reads with its own ID
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void exchange_test(queue q, size_t N) {
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
      exchange_local_test<::sycl::ext::oneapi::atomic_ref, space, T, order,
                          scope>(q, N);
    }
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
#else
    exchange_local_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
#endif
  }
  if constexpr (do_global_tests) {
#ifdef RUN_DEPRECATED
<<<<<<< HEAD
=======
    exchange_local_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
  }
  if constexpr (do_global_tests) {
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
    if constexpr (do_ext_tests) {
      exchange_global_test<::sycl::ext::oneapi::atomic_ref, space, T, order,
                           scope>(q, N);
    }
<<<<<<< HEAD
<<<<<<< HEAD
#else
    exchange_global_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
#endif
=======
    exchange_global_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#else
    exchange_global_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
#endif
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed>
void exchange_test_scopes(queue q, size_t N) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
<<<<<<< HEAD
<<<<<<< HEAD
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) !=
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::system>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::sub_group>(q, N);
  }
  exchange_test<space, T, order, memory_scope::device>(q, N);
=======
#if defined(SYSTEM)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
=======
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) !=
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::system>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::sub_group>(q, N);
  }
  exchange_test<space, T, order, memory_scope::device>(q, N);
<<<<<<< HEAD
#endif
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
}

template <access::address_space space, typename T>
void exchange_test_orders_scopes(queue q, size_t N) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
<<<<<<< HEAD
<<<<<<< HEAD
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::acq_rel>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::acquire>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::release>(q, N);
  }
  exchange_test_scopes<space, T, memory_order::relaxed>(q, N);
=======
#if defined(ACQ_REL)
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) ==
=======
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) !=
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::acq_rel>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::acquire>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::release>(q, N);
  }
  exchange_test_scopes<space, T, memory_order::relaxed>(q, N);
<<<<<<< HEAD
#endif
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
}

template <access::address_space space> void exchange_test_all() {
  queue q;

  constexpr int N = 32;
<<<<<<< HEAD
<<<<<<< HEAD
#ifdef FULL_ATOMIC64_COVERAGE
=======
#ifdef ATOMIC64
  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return;
  }
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#ifdef FULL_ATOMIC64_COVERAGE
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
  exchange_test_orders_scopes<space, double>(q, N);
  if constexpr (sizeof(long) == 8) {
    exchange_test_orders_scopes<space, long>(q, N);
    exchange_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(long long) == 8) {
    exchange_test_orders_scopes<space, long long>(q, N);
    exchange_test_orders_scopes<space, unsigned long long>(q, N);
  }
<<<<<<< HEAD
<<<<<<< HEAD
#endif
  exchange_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
  exchange_test_orders_scopes<space, int>(q, N);
  exchange_test_orders_scopes<space, unsigned int>(q, N);
=======
  if constexpr (sizeof(char *) == 8) {
    exchange_test_orders_scopes<space, char *>(q, N);
  }
#else
  exchange_test_orders_scopes<space, int>(q, N);
  exchange_test_orders_scopes<space, unsigned int>(q, N);
  exchange_test_orders_scopes<space, float>(q, N);

>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
#endif
  exchange_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
  exchange_test_orders_scopes<space, int>(q, N);
  exchange_test_orders_scopes<space, unsigned int>(q, N);
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
  if constexpr (sizeof(long) == 4) {
    exchange_test_orders_scopes<space, long>(q, N);
    exchange_test_orders_scopes<space, unsigned long>(q, N);
  }
<<<<<<< HEAD
<<<<<<< HEAD
=======
  if constexpr (sizeof(char *) == 4) {
    exchange_test_orders_scopes<space, char *>(q, N);
  }
>>>>>>> 88ee9d1a0 ([SYCL] Add tests for atomics with various memory orders and scopes (#534))
=======
>>>>>>> a5f90c0cd ([SYCL] Speed up atomic_ref tests (#879))
#endif

  std::cout << "Test passed." << std::endl;
}
