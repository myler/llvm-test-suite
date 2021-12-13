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
    lsc_surf_store<int, VL>(val, acc, off + j * SlmSize * sizeof(int));

    barrier();
  }
}

template <int case_num> class KernelID;

template <unsigned case_num, unsigned Groups, unsigned Threads, unsigned Size>
bool test() {
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
            work<Groups, Threads, Size>(acc, ndi);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return -1;
  }

  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 0xdead0002;
    if (i < Size / 4)
      etalon = 0xdead0001;
    if (i >= Size / 2) {
      if (i < (7 * Size / 8)) {
        if (i < (5 * Size / 8))
          etalon = 0xdead0001;
        else
          etalon = 0xdead0003;
      }
    }
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }

  std::cout << "#" << case_num << (passed ? " Passed\n" : " FAILED\n");
  return passed;
}
