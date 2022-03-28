using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;
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
      simd<int, SlmSize / 2> init(v);
      slm_block_store(b_off, init); // producer stores to SLM
    }

    nbarrier_signal(b, flag, producers, consumers);
    nbarrier_wait(b); // consumers waiting for signal

    auto val = slm_block_load<int, VL>(off); // reading SLM
    // and storing it to output surface
    lsc_block_store<int, VL>(acc, off + i * SlmSize * sizeof(int), val);

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
    int etalon = 0;
    if (i < Size / 4)
      etalon = 0xdead0001;
    if (i >= Size / 2) {
      if (i < (7 * Size / 8)) {
        if (i < (5 * Size / 8))
          etalon = 0xdead0001;
        else
          etalon = 0xdead0002;
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
