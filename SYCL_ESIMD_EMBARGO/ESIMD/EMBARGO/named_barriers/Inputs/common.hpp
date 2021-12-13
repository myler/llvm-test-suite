using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned Groups, unsigned Threads, unsigned Size> int test() {
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
      cgh.parallel_for<class Test>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            work<Groups, Threads, Size>(acc, ndi);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return -1;
  }

  bool passed = check(out);
  std::cout << (passed ? " Passed\n" : " FAILED\n");
  return passed ? 0 : 1;
}
