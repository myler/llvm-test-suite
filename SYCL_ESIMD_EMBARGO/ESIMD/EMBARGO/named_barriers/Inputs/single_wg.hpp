using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned prods, unsigned cons, unsigned Size, typename AccessorTy>
ESIMD_INLINE void work(AccessorTy acc, cl::sycl::nd_item<1> ndi) {
  static_assert(cons >= prods, "Consumers must be greater than producers");
  static_assert(cons % prods == 0, "Consumers must be multiple of producers");

  constexpr unsigned scale = cons / prods;
  constexpr unsigned Threads = cons + prods;
  static_assert(Threads > 3, "Total number of threads must be greater than 3");

  constexpr unsigned bnum = 2; // 1 named barrier, id 0 reserved for unnamed
  constexpr unsigned bid = 1;

  // number of ints loaded/stored by each thread
  constexpr unsigned VL = Size / Threads;
  // number of ints stored to SLM by producer
  constexpr unsigned NUM = VL * (1 + scale);

  nbarrier_init<bnum>();

  unsigned int idx = ndi.get_local_id(0);
  unsigned int off = idx * VL * sizeof(int);

  bool is_producer = (idx % (scale + 1)) == scale;
  bool is_consumer = !is_producer;
  // only-consumer or only-producer modes
  unsigned int flag = is_producer ? 0x1 : 0x2;

  slm_init(Size * sizeof(int));
  slm_block_store(off, simd<int, VL>(0));
  barrier();

  if (is_producer) {
    unsigned int x = VL * (idx - scale);
    unsigned int p_off = x * sizeof(int);
    // each producer stores x4 data
    simd<int, NUM> init(0xdead0000 + x, 1);
    slm_block_store(p_off, init); // producers store data to SLM
  }

  // signaling after data stored
  nbarrier_signal(bid, flag, prods, cons);

  if (is_consumer)
    nbarrier_wait(bid); // consumers waiting for signal

  auto val = slm_block_load<int, VL>(off); // reading SLM
  lsc_block_store<int, VL>(acc, off, val); // and storing it to output surface
}

template <int case_num> class KernelID;

template <int case_num, unsigned prods, unsigned cons>
bool test() {
  constexpr unsigned Groups = 1;
  constexpr unsigned Threads = cons + prods;
  constexpr unsigned Size = 4 * Threads;

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
            work<prods, cons, Size>(acc, ndi);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return -1;
  }

  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 0xdead0000 + i;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }

  std::cout << "#" << case_num << (passed ? " Passed\n" : " FAILED\n");
  return passed;
}
