using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
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
