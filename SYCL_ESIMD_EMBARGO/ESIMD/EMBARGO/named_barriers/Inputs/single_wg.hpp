using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 2; // 1 named barrier, id 0 reserved for unnamed
  constexpr unsigned bid = 1;

  constexpr unsigned producers = 4;
  constexpr unsigned consumers = Threads - producers; // 12

  // number of ints stored to SLM by producer
  constexpr unsigned NUM = 1 + (consumers / producers); // 4
  // number of ints loaded/stored by each thread
  constexpr unsigned VL = Size / Threads; // 16

  nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  // threads with linear ids 3, 7, 11 and 15 are producers
  bool is_producer = (idx % producers) == 3;
  bool is_consumer = !is_producer;
  // only-consumer or only-producer modes
  unsigned int flag = is_producer ? 0x1 : 0x2;

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  barrier();

  if (is_producer) {
    unsigned int x = idx - 3;
    unsigned int p_off = x * VL * sizeof(int);
    // each producer stores x4 data
    simd<int, NUM * VL> init(0xdead0000 + x * VL, 1);
    slm_block_store(p_off, init); // producers store data to SLM
  }

  // signaling after data stored
  nbarrier_signal(bid, flag, producers, consumers);

  if (is_consumer)
    nbarrier_wait(bid); // consumers waiting for signal

  auto val = slm_block_load<int, VL>(off); // reading SLM
  lsc_surf_store<int, VL>(val, acc, off);  // and storing it to output surface
}

bool check(std::vector<int> out) {
  bool passed = true;
  for (int i = 0; i < out.size(); i++) {
    int etalon = 0xdead0000 + i;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}
