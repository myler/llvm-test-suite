using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 3; // 2 named barriers, id 0 reserved for unnamed

  constexpr unsigned producers = 1;
  constexpr unsigned consumers = Threads; // 8

  // SLM size is half of output surface size so
  // content of SLM can be copied to out buffer on each iteration
  constexpr unsigned SlmSize = Size / 2;
  // number of ints read/written by single thread
  constexpr unsigned VL = SlmSize / Threads;

  nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  slm_init(SlmSize * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  barrier();

  for (int b = 1; b < bnum; b++) {
    int i = b - 1;

    // local ID 1 is producer on first iteration, local ID 2 on second
    bool is_producer = idx == b;
    bool is_consumer = !is_producer;
    unsigned int flag = is_producer ? 0x0 : 0x2; // producer is also a consumer

    if (is_producer) {
      // second iteration store partialy overlaps data stored on first iteration
      unsigned int b_off = i * sizeof(int) * SlmSize / 4;

      int v = 0xdead0000 + idx;
      simd<int, SlmSize> init(v, 1);
      slm_block_store(b_off, init); // producer stores to SLM
    }

    nbarrier_signal(b, flag, producers, consumers);
    nbarrier_wait(b); // consumers waiting for signal

    auto val = slm_block_load<int, VL>(off); // reading SLM
    // and storing it to output surface
    lsc_surf_store<int, VL>(val, acc, off + i * SlmSize);
  }
}

bool check(std::vector<int> out) {
  bool passed = true;
  int size = out.size();
  for (int i = 0; i < size; i++) {
    int etalon = 0xdead0002;
    if (i < size / 4)
      etalon = 0xdead0001;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}
