<<<<<<< HEAD
#define TM 32
#define TN 64
=======
#define TM 8
#define TN SG_SZ
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
#define TK 16

#define BF16_EPSILON 0.00781250

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
private:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / 2, N * 2> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<bfloat16, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

<<<<<<< HEAD
<<<<<<<< HEAD:SYCL/Matrix/Legacy/joint_matrix_bfloat16_impl.hpp
=======
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<bfloat16, TM, TK> sub_a(sg);
           // For B, since current implementation does not support non-packed
           // layout, users need to specify the updated VNNI sizes along with
           // the packed_b layout. By default, the layout is row_major and size
           // is (TK, TN).
           joint_matrix<bfloat16, TK, TN, matrix_layout::packed_b> sub_b(sg);
           joint_matrix<float, TM, TN> sub_c(sg);
<<<<<<< HEAD
========
           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
>>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391)):SYCL/Matrix/joint_matrix_bfloat16_32x64_impl.hpp
=======
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))

           joint_matrix_load(sg, sub_c,
                             accC.get_pointer() + (sg_startx * TM) * N +
                                 sg_starty / SG_SZ * TN,
<<<<<<< HEAD
<<<<<<<< HEAD:SYCL/Matrix/Legacy/joint_matrix_bfloat16_impl.hpp
=======
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
                             N, matrix_layout::row_major);
           for (int k = 0; k < K / TK; k += 1) { //
             joint_matrix_load(
                 sg, sub_a, accA.get_pointer() + (sg_startx * TM) * K + k * TK,
                 K, matrix_layout::row_major);
<<<<<<< HEAD
========
                             N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) { //
             joint_matrix_load(
                 sg, sub_a, accA.get_pointer() + (sg_startx * TM) * K + k * TK,
                 K);
>>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391)):SYCL/Matrix/joint_matrix_bfloat16_32x64_impl.hpp
=======
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
             // Assuming B data is already in VNNI format.
             joint_matrix_load(sg, sub_b,
                               accB.get_pointer() + (k * TK / 2) * (N * 2) +
                                   sg_starty / SG_SZ * TN * 2,
<<<<<<< HEAD
<<<<<<<< HEAD:SYCL/Matrix/Legacy/joint_matrix_bfloat16_impl.hpp
                               N * 2, matrix_layout::packed_b);
========
                               N * 2);
>>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391)):SYCL/Matrix/joint_matrix_bfloat16_32x64_impl.hpp
=======
                               N * 2, matrix_layout::packed_b);
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(sg, sub_c,
                              accC.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;
bfloat16 A[MATRIX_M][MATRIX_K];
bfloat16 B[MATRIX_K / 2][MATRIX_N * 2];
unsigned short Aref[MATRIX_M][MATRIX_K];
unsigned short Bref[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

float make_fp32(short x) {
  unsigned int y = x;
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

unsigned short make_bf16(float x) {
  int *res = reinterpret_cast<int *>(&x);
  *res = *res >> 16;
  return (unsigned short)*res;
}

void matrix_multiply_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) {
  // tiling
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        short *va = (short *)(A_mem + m * K + k);
        short *vb = (short *)(B_mem + k * N + n);
        float acc = *((float *)(C_mem + m * N + n));
        // FIXME: Should we do reduce-add in another version?
        for (int i = 0; i < 2; i++) {
          acc += (make_fp32(va[i]) * make_fp32(vb[i]));
        }
        *((float *)(C_mem + m * N + n)) = acc;
      }
    }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      // bfloat16 is created using unsigned short since conversion from float to
      // bfloat16 is not supported on the host side yet
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<<< HEAD:SYCL/Matrix/Legacy/joint_matrix_bfloat16_impl.hpp
      A[i][j] = bfloat16(1.0f * (i + j));
========
      A[i][j] = bfloat16::from_bits(make_bf16(1.0f * (i + j)));
>>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391)):SYCL/Matrix/joint_matrix_bfloat16_32x64_impl.hpp
=======
      A[i][j] = bfloat16::from_bits(make_bf16(1.0f * (i + j)));
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
=======
      A[i][j] = bfloat16(1.0f * (i + j));
>>>>>>> 2722bd134 ([SYCL][Matrix]update recent tests to use the new API and remove deprecated bfloat16::from_bits   (#1494))
      Aref[i][j] = make_bf16(1.0f * (i + j));
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
<<<<<<< HEAD
<<<<<<< HEAD
      B[i][j] = bfloat16(2.0f * i + 3.0f * j);
=======
      B[i][j] = bfloat16::from_bits((make_bf16(2.0f * i + 3.0f * j)));
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
=======
      B[i][j] = bfloat16(2.0f * i + 3.0f * j);
>>>>>>> 2722bd134 ([SYCL][Matrix]update recent tests to use the new API and remove deprecated bfloat16::from_bits   (#1494))
      Bref[i][j] = make_bf16(2.0f * i + 3.0f * j);
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1.0;
      D[i][j] = 1.0;
    }
  }

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);
  big_matrix<bfloat16, MATRIX_K / 2, MATRIX_N * 2> MB((bfloat16 *)&B);
  matrix_multiply(MC, MA, MB);
  matrix_multiply_ref((int32_t *)Aref, (int32_t *)Bref, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K / 2);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if ((fabs(C[i][j]) - fabs(D[i][j])) > BF16_EPSILON)
        res = false;
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
=======
  if (res)
    std::cout << "passed\n";
  else
    std::cout << "failed\n";
>>>>>>> cbbfcc6c1 ([SYCL] Add matrix tests that use the new API (unified API) (#1391))
=======
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
>>>>>>> 2722bd134 ([SYCL][Matrix]update recent tests to use the new API and remove deprecated bfloat16::from_bits   (#1494))
}
