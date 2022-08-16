#include <sycl/sycl.hpp>

using namespace cl::sycl;

template <typename T> class usm_device_transfer;
template <typename T> class usm_aligned_device_transfer;

static constexpr int N = 100;

template <typename T>
void runHostTests(device dev, context ctxt, queue q, T val) {
  T *array;

  array = (T *)malloc_host(N * sizeof(T), q);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);

  array = (T *)aligned_alloc_host(alignof(long long), N * sizeof(T), ctxt);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);
}

template <typename T>
void runSharedTests(device dev, context ctxt, queue q, T val) {
  T *array;

  array = (T *)malloc_shared(N * sizeof(T), q);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);

  array =
      (T *)aligned_alloc_shared(alignof(long long), N * sizeof(T), dev, ctxt);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);
}

template <typename T>
void runDeviceTests(device dev, context ctxt, queue q, T val) {
  T *array;
  std::vector<T> out;
  out.resize(N);

  array = (T *)malloc_device(N * sizeof(T), q);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();

  {
    buffer<T, 1> buf{&out[0], range<1>{N}};
    q.submit([&](handler &h) {
       auto acc = buf.template get_access<access::mode::write>(h);
       h.parallel_for<usm_device_transfer<T>>(
           range<1>(N), [=](id<1> item) { acc[item] = array[item]; });
     }).wait();
  }

  for (int i = 0; i < N; ++i) {
    assert(out[i] == val);
  }
  free(array, ctxt);

  out.clear();
  out.resize(N);

  array =
      (T *)aligned_alloc_device(alignof(long long), N * sizeof(T), dev, ctxt);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();

  {
    buffer<T, 1> buf{&out[0], range<1>{N}};
    q.submit([&](handler &h) {
       auto acc = buf.template get_access<access::mode::write>(h);
       h.parallel_for<usm_aligned_device_transfer<T>>(
           range<1>(N), [=](id<1> item) { acc[item] = array[item]; });
     }).wait();
  }

  for (int i = 0; i < N; ++i) {
    assert(out[i] == val);
  }
  free(array, ctxt);
}
