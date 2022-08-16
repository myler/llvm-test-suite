#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::usm;

template <typename T> class transfer;

static constexpr int N = 100; // should be even

template <typename T> T *regular(queue q, alloc kind) {
  return malloc<T>(N, q, kind);
}

template <typename T> T *aligned(queue q, alloc kind) {
  return aligned_alloc<T>(alignof(long long), N, q, kind);
}

template <typename T> void test(queue q, T val, T *src, T *dst, bool dev_dst) {
  q.fill(src, val, N).wait();

  // Use queue::copy for the first half and handler::copy for the second
  q.copy(src, dst, N / 2).wait();
  q.submit([&](handler &h) { h.copy(src + N / 2, dst + N / 2, N / 2); }).wait();

  T *out = dst;

  std::array<T, N> arr;
  if (dev_dst) { // if copied to device, transfer data back to host
    buffer buf{arr};
    q.submit([&](handler &h) {
      accessor acc{buf, h};
      h.parallel_for<transfer<T>>(N, [=](id<1> i) { acc[i] = dst[i]; });
    });
    out = arr.data();
  }

  for (int i = 0; i < N; ++i) {
    assert(out[i] == val);
  }

  free(src, q);
  free(dst, q);
}

template <typename T> void runTests(queue q, T val, alloc kind1, alloc kind2) {
  bool dev_dst1 = (kind1 == alloc::device);
  bool dev_dst2 = (kind2 == alloc::device);
  test(q, val, regular<T>(q, kind1), regular<T>(q, kind2), dev_dst2);
  test(q, val, regular<T>(q, kind2), regular<T>(q, kind1), dev_dst1);
  test(q, val, aligned<T>(q, kind1), aligned<T>(q, kind2), dev_dst2);
  test(q, val, aligned<T>(q, kind2), aligned<T>(q, kind1), dev_dst1);
  test(q, val, regular<T>(q, kind1), aligned<T>(q, kind2), dev_dst2);
  test(q, val, regular<T>(q, kind2), aligned<T>(q, kind1), dev_dst1);
  test(q, val, aligned<T>(q, kind1), regular<T>(q, kind2), dev_dst2);
  test(q, val, aligned<T>(q, kind2), regular<T>(q, kind1), dev_dst1);
}