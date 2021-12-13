using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 3; // 2 named barriers, id 0 reserved for unnamed

  // SLM size is half of output surface size so
  // content of SLM can be copied to out buffer on each iteration
  constexpr unsigned SlmSize = Size / 2;     // 32
  constexpr unsigned VL = SlmSize / Threads; // 4

  nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  // 2 producers on first iteration, 1 producer on second
  unsigned int indexes[2][2] = {{1, 2}, {3, 3}}; // local ids of producers
  unsigned int prods[2] = {2, 1};                // number of producers

  slm_init(SlmSize * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  barrier();

  for (int b = bnum - 1; b > 0; b--) {
    int j = bnum - b - 1; // iteration index

    bool is_producer = idx == indexes[j][0] || idx == indexes[j][1];
    bool is_consumer = !is_producer;
    // only-consumer or only-producer modes
    unsigned int flag = is_producer ? 0x1 : 0x2;

    unsigned int producers = prods[j];
    unsigned int consumers = Threads - producers;

    if (is_producer) {
      unsigned int p_off = j * sizeof(int) * SlmSize / 4;
      // second iteration store partialy overlaps first iteration stores
      p_off += (producers == 2 ? (idx - 1) : 0) * sizeof(int) * SlmSize / 2;
      int v = 0xdead0000 + idx;
      simd<int, SlmSize / 2> init(v);
      slm_block_store(p_off, init); // producer stores to SLM
    }

    nbarrier_signal(b, flag, producers, consumers);

    if (is_consumer)
      nbarrier_wait(b); // consumers waiting for signal

    auto val = slm_block_load<int, VL>(off); // reading SLM
    // and storing it to output surface
    lsc_surf_store<int, VL>(val, acc, off + j * SlmSize);
  }
}

bool check(std::vector<int> out) {
  bool passed = true;
  int size = out.size();
  for (int i = 0; i < size; i++) {
    int etalon = 0xdead0003;
    if (i < size / 4)
      etalon = 0xdead0001;
    else if (i >= 3 * size / 4)
      etalon = 0xdead0002;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}
