// REQUIRES: ats, linux
//
// RUN: %clangxx -fsycl %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out -lpthread
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <thread>
#include <unistd.h>

namespace sycl = cl::sycl;

#define MAX 5
#define N 1024*1024*4
#define LOOP 1000
void test_multi(float* host_buff, float* dev_buff, float* host_buff1, sycl::queue* q) {
  //q->memcpy(dev_buff, host_buff, N * sizeof(float));
  q->submit([&](sycl::handler &h) {
          h.parallel_for<class KernelA_ReLU>(cl::sycl::range<1>(cl::sycl::range<1>{N}), 
            [=](cl::sycl::item<1> item) {
                auto id = item.get_linear_id();
                if (dev_buff[id] <= 0.0) dev_buff[id] = 0.0;
            });
        });
  //auto ev = q->memcpy(host_buff, dev_buff, N * sizeof(float));
  //ev.wait();
}

void submit_kernels(float* host_buff, float* dev_buff, float* host_buff1, sycl::queue* q) {
  for (int i = 0; i < LOOP * 10; ++i) {
    int rt = rand() % 1000 + 100;
    usleep(rt);
    test_multi(host_buff, dev_buff, host_buff1, q);
    printf("submit %d\n", i);
    fflush(0);
  }
}

void h2d(float* host_buff, float* dev_buff, float* host_buff1, sycl::queue* q) {
  for (int i = 0; i < LOOP * 10; ++i) {
    int rt = rand() % 1000 + 100;
    usleep(rt);
    q->memcpy(dev_buff, host_buff, N * sizeof(float));
    printf("h2d %d\n", i);
    fflush(0);
  }
}

void d2h(float* host_buff, float* dev_buff, float* host_buff1, sycl::queue* q) {
  for (int i = 0; i < LOOP * 10; ++i) {
    int rt = rand() % 1000 + 100;
    usleep(rt);
    q->memcpy(host_buff, dev_buff, N * sizeof(float));
    printf("d2h %d\n", i);
    fflush(0);
  }
}

void just_wait(sycl::queue* q) {
  int tid = gettid();
  for (int i = 0; i < LOOP; ++i) {
    int rt = rand() % 1000 + 100;
    usleep(rt);
    q->wait();
    q->wait();
    printf("wait %d, tid is %d\n", i, tid);
    fflush(0);
  }
}

int main() {

  sycl::device dev = sycl::device(sycl::gpu_selector());
  std::vector<sycl::device> subdev = {};
  subdev = dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
  sycl::queue q = sycl::queue(subdev[0], {cl::sycl::property::queue::in_order()});
  //sycl::queue q = sycl::queue(subdev[0]);

  float* dev_buff = sycl::malloc_device<float>(N, q);
  float* host_buff = sycl::malloc_host<float>(N, q);
  float* host_buff1 = sycl::malloc_host<float>(N, q);

  std::thread pool[MAX];
  pool[0] = std::thread(submit_kernels, host_buff, dev_buff, host_buff1, &q);
  pool[1] = std::thread(just_wait, &q);
  pool[2] = std::thread(h2d, host_buff, dev_buff, host_buff1, &q);
  pool[3] = std::thread(d2h, host_buff, dev_buff, host_buff1, &q);
  pool[4] = std::thread(just_wait, &q);
  for (auto i = 0; i < MAX; ++i) {
    pool[i].join();
  } 

  q.wait();
  return 0;
}
