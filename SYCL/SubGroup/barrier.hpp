#include "helper.hpp"
#include <limits>
#include <numeric>
#include <sycl/sycl.hpp>

template <typename T, bool UseNewSyntax> class sycl_subgr;
using namespace sycl;
template <typename T, bool UseNewSyntax = false>
void check(queue &Queue, size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    std::vector<T> data(G);
    std::iota(data.begin(), data.end(), sizeof(T));
    buffer<T> addbuf(data.data(), range<1>(G));
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto addacc = addbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<sycl_subgr<T, UseNewSyntax>>(
          NdRange, [=](nd_item<1> NdItem) {
            ext::oneapi::sub_group SG = NdItem.get_sub_group();
            size_t lid = SG.get_local_id().get(0);
            size_t gid = NdItem.get_global_id(0);
            size_t SGoff = gid - lid;

            T res = 0;
            for (size_t i = 0; i <= lid; i++) {
              res += addacc[SGoff + i];
            }
            if constexpr (UseNewSyntax) {
              group_barrier(SG);
            } else {
              SG.barrier(access::fence_space::global_space);
            }
            addacc[gid] = res;
            if (NdItem.get_global_id(0) == 0)
              sgsizeacc[0] = SG.get_max_local_range()[0];
          });
    });
    auto addacc = addbuf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();

    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    T add = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        add = 0;
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      add += j + sizeof(T);
      exit_if_not_equal<T>(addacc[j], add, "barrier");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
