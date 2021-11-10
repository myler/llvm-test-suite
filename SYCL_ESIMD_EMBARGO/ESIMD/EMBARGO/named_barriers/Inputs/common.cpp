/*========================== begin_copyright_notice ============================
INTEL CONFIDENTIAL
Copyright (C) 2018-2021 Intel Corporation
This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.
This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.
============================= end_copyright_notice ===========================*/

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void case_1(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 2;
  constexpr unsigned barrier = 1;

  constexpr unsigned producers = 4;
  constexpr unsigned consumers = Threads - producers;

  constexpr unsigned NUM = 1 + (consumers / producers);
  constexpr unsigned VL = Size / Threads;

  esimd_nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  bool is_producer = (idx % producers) == 3;
  bool is_consumer = !is_producer;
  unsigned int flag = is_producer ? 0x1 : 0x2;

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  esimd_barrier();

  if (is_producer) {
    unsigned int x = idx - 3;
    unsigned int p_off = x * VL * sizeof(int);
    simd<int, NUM * VL> init(0xdead0000 + x * VL, 1);
    slm_block_store(p_off, init);
  }

  esimd_nbarrier_signal(barrier, flag, producers, consumers);

  if (is_consumer)
    esimd_nbarrier_wait(barrier);

  auto val = slm_block_load<int, VL>(off);
  lsc_surf_store<int, VL>(val, acc, off);
}

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void case_2(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 2;
  constexpr unsigned barrier = 1;

  constexpr unsigned producers = 1;
  constexpr unsigned consumers = 1;

  constexpr unsigned NUM = Threads * Groups;
  constexpr unsigned VL = Size / NUM;

  esimd_nbarrier_init<bnum>();

  unsigned int localID = ndi.get_local_id(0);
  unsigned int groupID = ndi.get_group(0);
  unsigned int globalID = ndi.get_global_id(0);
  unsigned int groupSize = ndi.get_local_range(0);
  unsigned int group_off = VL * groupID * groupSize * sizeof(int);
  unsigned int global_off = VL * globalID * sizeof(int);

  slm_init(VL * NUM * sizeof(int));
  slm_block_store(global_off, simd<int, VL>(0));
  esimd_barrier();

  bool is_producer = localID == 1;
  bool is_consumer = !is_producer;
  unsigned int flag = is_producer ? 0x1 : 0x2;
  int v = 0xdead0000 | (groupID << 8) | localID;

  if (is_producer) {
    slm_block_store(group_off, simd<int, Size / 2>(v));
  }

  esimd_nbarrier_signal(barrier, flag, producers, consumers);

  if (is_consumer) {
    esimd_nbarrier_wait(barrier);
    auto ret = slm_block_load<int, Size / 2>(group_off);
    lsc_surf_store<int, Size / 2>(ret, acc, group_off);
  }
}

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void case_3(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 3;

  constexpr unsigned producers = 1;
  constexpr unsigned consumers = Threads;

  constexpr unsigned VL = Size / Threads;

  esimd_nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  esimd_barrier();

  for (int b = 1; b < bnum; b++) {
    bool is_producer = idx == b;
    bool is_consumer = !is_producer;
    unsigned int flag = is_producer ? 0x0 : 0x2; // producer is also a consumer

    unsigned int b_off = (b - 1) * sizeof(int) * Size / 4;

    if (is_producer) {
      int v = 0xdead0000 + (b - 1) * Size / 2;
      simd<int, Size / 2> init(v, 1);
      slm_block_store(b_off, init);
    }

    esimd_nbarrier_signal(b, flag, producers, consumers);
    esimd_nbarrier_wait(b);

    auto val = slm_block_load<int, VL>(off + b_off);
    lsc_surf_store<int, VL>(val, acc, off + b_off);
  }
}

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void case_4(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 3;

  constexpr unsigned VL = Size / Threads;

  esimd_nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  unsigned int indexes[2][2] = {{1, 2}, {3, 3}}; // local ids of producers
  unsigned int prods[2] = {2, 1};                // number of producers

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  esimd_barrier();

  for (int b = bnum - 1; b > 0; b--) {
    int j = bnum - b - 1;

    bool is_producer = idx == indexes[j][0] || idx == indexes[j][1];
    bool is_consumer = !is_producer;
    unsigned int flag = is_producer ? 0x1 : 0x2;

    unsigned int producers = prods[j];
    unsigned int consumers = Threads - producers;

    if (is_producer) {
      unsigned int p_off = j * sizeof(int) * Size / 4;
      p_off += (producers == 2 ? (idx - 1) : 0) * sizeof(int) * Size / 2;
      int v = 0xdead0000 + idx;
      simd<int, Size / 2> init(v);
      slm_block_store(p_off, init);
    }

    esimd_nbarrier_signal(b, flag, producers, consumers);

    if (is_consumer)
      esimd_nbarrier_wait(b);

    auto val = slm_block_load<int, VL>(off);
    lsc_surf_store<int, VL>(val, acc, off);
  }
}

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void case_5(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = Threads - 1;

  constexpr unsigned VL = Size / (2 * Threads);

  esimd_nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  int flag = 0;
  int producers = 2;
  int consumers = 2;

  if (idx > 0) {
    int barrier_id = idx - 1;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  }

  simd<int, VL * 2> val(idx);
  lsc_surf_store<int, VL * 2>(val, acc, off);

  if (idx < bnum) {
    int barrier_id = idx;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  }
}

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void case_6(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = Threads - 1;

  constexpr unsigned VL = Size / (2 * Threads);

  esimd_nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  int flag = 0;
  int producers = 2;
  int consumers = 2;

  simd<int, VL * 2> val(idx);

  if (idx == 0) { // first thread within a work-group
    const int barrier_id = idx;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  } else if (idx == 1) {
    const int barrier_id = idx - 1;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);

    const int barrier_id2 = idx;
    esimd_nbarrier_signal(barrier_id2, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id2);
  } else if (idx == 2) {
    const int barrier_id = idx - 1;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);

    const int barrier_id2 = idx;
    esimd_nbarrier_signal(barrier_id2, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id2);
  } else { // idx == bnum (last thread within a work-group)
    const int barrier_id = idx - 1;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  }

  lsc_surf_store<int, VL * 2>(val, acc, off);
}

template <int CASE_NUM> class KernelID;

template <int CASE_NUM, unsigned Groups, unsigned Threads, unsigned Size>
int test() {
  std::cout << "Case #" << CASE_NUM << "\n";

  std::vector<int> out(Size, 0);
  std::vector<int> etalon(Size, 0);

  switch (CASE_NUM) {
  case 1:
    for (int i = 0; i < Size; i++)
      etalon[i] = 0xdead0000 + i;
    break;
  case 2:
    for (int i = 0; i < Size; i++)
      etalon[i] = (i < Size / 2) ? 0xdead0001 : 0xdead0101;
    break;
  case 3:
    for (int i = 0; i < Size; i++) {
      if (i < 3 * Size / 4)
        etalon[i] = 0xdead0000 + (i < Size / 4 ? i : i + Size / 4);
    }
    break;
  case 4:
    for (int i = 0; i < Size; i++) {
      if (i < Size / 4)
        etalon[i] = 0xdead0001;
      else if (i < 3 * Size / 4)
        etalon[i] = 0xdead0003;
      else
        etalon[i] = 0xdead0002;
    }
    break;
  case 5:
  case 6:
    for (int i = 0; i < Size; i++) {
      etalon[i] = i / 4;
      if (etalon[i] * 4 == Size / 2)
        etalon[i] -= 1;
      if (etalon[i] * 4 > Size / 2)
        etalon[i] = 0;
    }
    break;
  }

  try {
    buffer<int, 1> buf(out.data(), out.size());

    // workgroups
    cl::sycl::range<1> GlobalRange{Groups};
    // threads in each group
    cl::sycl::range<1> LocalRange{Threads};
    cl::sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

    auto GPUSelector = gpu_selector{};
    auto q = queue{GPUSelector};
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<KernelID<CASE_NUM>>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            if constexpr (CASE_NUM == 1)
              case_1<Groups, Threads, Size>(acc, ndi);
            if constexpr (CASE_NUM == 2)
              case_2<Groups, Threads, Size>(acc, ndi);
            if constexpr (CASE_NUM == 3)
              case_3<Groups, Threads, Size>(acc, ndi);
            if constexpr (CASE_NUM == 4)
              case_4<Groups, Threads, Size>(acc, ndi);
            if constexpr (CASE_NUM == 5)
              case_5<Groups, Threads, Size>(acc, ndi);
            if constexpr (CASE_NUM == 6)
              case_6<Groups, Threads, Size>(acc, ndi);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return -1;
  }

  bool passed = true;
  for (int i = 0; i < Size; i++) {
    if (out[i] != etalon[i]) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs "
                << etalon[i] << std::dec << std::endl;
    }
  }

  std::cout << (passed ? " Passed\n" : " FAILED\n");
  return passed ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Skipped! Specify case number" << std::endl;
    return 1;
  }

  int case_num = atoi(argv[1]);
  switch (case_num) {
  case 1:
    return test<1, 1, 16, 64>();
  case 2:
    return test<2, 2, 2, 16>();
  case 3:
    return test<3, 1, 8, 32>();
  case 4:
    return test<4, 1, 8, 32>();
  case 5:
    return test<5, 1, 4, 32>();
  case 6:
    return test<6, 1, 4, 32>();
  }

  std::cerr << "Invalid case number: " << case_num << "\n";
  return 1;
}
