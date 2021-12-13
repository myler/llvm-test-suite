using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  static_assert(Threads == 4, "Threads number expect to be 4");

  static_assert(Size % (2 * Threads) == 0,
                "Surface size must be a multiple of the number of threads");
  static_assert(Size >= (2 * Threads),
                "Surface size must be greater than the number of threads");

  // (Threads - 1) named barriers required, but id 0 reserved for unnamed
  constexpr unsigned bnum = Threads;
  // number of ints stored by each thread
  constexpr unsigned VL = Size / (2 * Threads);

  nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  int flag = 0; // producer-consumer mode
  int producers = 2;
  int consumers = 2;

  simd<int, VL * 2> val(idx);

  // Threads are executed in ascending order of their local ID and each thread
  // stores data to addresses that partially overlap with addresses used by
  // previous thread.
  if (idx == 0) {
    // T0 signals barrier 1 and locks, waiting for first signal from T1
    const int barrier_id = idx + 1;
    nbarrier_signal(barrier_id, flag, producers, consumers);
    nbarrier_wait(barrier_id);
  } else if (idx == 1) {
    // T1 signals barrier 1 and locks, waiting for signal from T0
    const int barrier_id = idx;
    nbarrier_signal(barrier_id, flag, producers, consumers);
    nbarrier_wait(barrier_id);

    // T1 signals barrier 2 and locks, waiting for first signal from T2
    const int barrier_id2 = idx + 1;
    nbarrier_signal(barrier_id2, flag, producers, consumers);
    nbarrier_wait(barrier_id2);
  } else if (idx == 2) {
    // T2 signals barrier 2 and locks, waiting for second signal from T1
    const int barrier_id = idx;
    nbarrier_signal(barrier_id, flag, producers, consumers);
    nbarrier_wait(barrier_id);

    // T2 signals barrier 3 and locks, waiting for signal from T3
    const int barrier_id2 = idx + 1;
    nbarrier_signal(barrier_id2, flag, producers, consumers);
    nbarrier_wait(barrier_id2);
  } else {
    // T3 signals barrier 3 and locks, waiting for second signal from T2
    const int barrier_id = idx;
    nbarrier_signal(barrier_id, flag, producers, consumers);
    nbarrier_wait(barrier_id);
  }

  lsc_surf_store<int, VL * 2>(val, acc, off);
}

template <int case_num> class KernelID;

template <int case_num, unsigned Size> bool test() {
  constexpr unsigned Groups = 1;
  constexpr unsigned Threads = 4;

  std::vector<int> out(Size, 0);

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
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            work<Threads, Size>(acc, ndi);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return -1;
  }

  bool passed = true;
  constexpr unsigned VL = Size / (2 * Threads);
  for (int i = 0; i < Size; i++) {
    int etalon = i / VL;   // namely number of thread that stored that data
    if (etalon == Threads) // last stored chunk
      etalon -= 1;
    if (etalon > Threads)  // excessive part of surface
      etalon = 0;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }

  std::cout << "#" << case_num << (passed ? " Passed\n" : " FAILED\n");
  return passed;
}
