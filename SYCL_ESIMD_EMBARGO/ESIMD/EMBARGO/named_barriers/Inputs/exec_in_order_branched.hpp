using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum =
      Threads; // 3 named barriers, id 0 reserved for unnamed
  constexpr unsigned VL = Size / (2 * Threads); // 4

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

bool check(std::vector<int> out) {
  bool passed = true;
  int size = out.size();
  for (int i = 0; i < size; i++) {
    int etalon = i / 4;
    if (etalon * 4 == size / 2)
      etalon -= 1;
    if (etalon * 4 > size / 2)
      etalon = 0;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}
