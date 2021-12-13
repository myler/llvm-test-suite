using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  // 3 named barriers, id 0 reserved for unnamed
  constexpr unsigned bnum = Threads;
  constexpr unsigned VL = Size / (2 * Threads); // 4

  esimd_nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  int flag = 0; // producer-consumer mode
  int producers = 2;
  int consumers = 2;

  // Threads are executed in ascending order of their local ID and each thread
  // stores data to addresses that partially overlap with addresses used by
  // previous thread.

  if (idx > 0) {
    // thread 0 skips this branch and goes straight to lsc_surf_store
    // thread 1 signals barrier 0
    // thread 2 signals barrier 1
    // thread 3 signals barrier 2
    int barrier_id = idx;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  }

  simd<int, VL * 2> val(idx);
  lsc_surf_store<int, VL * 2>(val, acc, off);

  if (idx < bnum) {
    // thread 0 arrives here first and signals barrier 0, unlocking thread 1
    // thread 1 arrives here next and signals barrier 1, unlocking thread 2
    // thread 2 arrives here next and signals barrier 2, unlocking thread 3
    // thread 3 skips this branch
    int barrier_id = idx + 1;
    esimd_nbarrier_signal(barrier_id, flag, producers, consumers);
    esimd_nbarrier_wait(barrier_id);
  }
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
