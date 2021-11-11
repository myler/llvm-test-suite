using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
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
