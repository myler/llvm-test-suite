using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  constexpr unsigned bnum = 2; // 1 named barrier, id 0 reserved for unnamed
  constexpr unsigned bid = 1;

  constexpr unsigned producers = 1;
  constexpr unsigned consumers = 1;

  constexpr unsigned NUM = Threads * Groups; // 4
  constexpr unsigned VL = Size / NUM;        // 4

  esimd_nbarrier_init<bnum>();

  unsigned int localID = ndi.get_local_id(0);
  unsigned int groupID = ndi.get_group(0);
  unsigned int globalID = ndi.get_global_id(0);
  unsigned int groupSize = ndi.get_local_range(0);
  unsigned int group_off = VL * groupID * groupSize * sizeof(int);
  unsigned int global_off = VL * globalID * sizeof(int);

  slm_init(VL * NUM * sizeof(int));
  slm_block_store(global_off, simd<int, VL>(0));
  barrier();

  // thread with local id 1 is producer in each work-group
  bool is_producer = localID == 1;
  bool is_consumer = !is_producer;
  // only-producer or only-comsumer modes
  unsigned int flag = is_producer ? 0x1 : 0x2;

  if (is_producer) {
    int v = 0xdead0000 | (groupID << 8) | localID;
    // producer stores data to SLM
    slm_block_store(group_off, simd<int, Size / 2>(v));
  }

  // signaling after data stored
  esimd_nbarrier_signal(bid, flag, producers, consumers);

  if (is_consumer) {
    esimd_nbarrier_wait(bid); // consumers waiting here for signal from producer
    auto ret = slm_block_load<int, Size / 2>(group_off); // reading SLM
    lsc_surf_store<int, Size / 2>(ret, acc, group_off);  // storing it to output
  }
}

bool check(std::vector<int> out) {
  bool passed = true;
  int size = out.size();
  for (int i = 0; i < size; i++) {
    int etalon = (i < size / 2) ? 0xdead0001 : 0xdead0101;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }
  return passed;
}
