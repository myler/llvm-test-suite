using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
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

bool check(std::vector<int> out) {
  bool passed = true;
  int size = out.size();
  for (int i = 0; i < size; i++) {
    int etalon = 0;
    if (i < 3 * size / 4)
      etalon = 0xdead0000 + (i < size / 4 ? i : i + size / 4);
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}
