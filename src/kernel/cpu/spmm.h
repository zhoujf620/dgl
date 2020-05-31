#ifndef DGL_KERNEL_CPU_SPMM_CUH_
#define DGL_KERNEL_CPU_SPMM_CUH_

#include <dgl/array.h>

namespace dgl {
namespace kernel {
namespace cpu {

/*
 * This func do the followings:
 *   1. Convert flattened index to multi-dimension index
 *      according to output shape (assume row-major).
 *   2. Convert multi-dimension index to flattened index for lhs.
 *   3. Convert multi-dimension index to flattened index for rhs.
 */
void UnravelRavel(
    const int64_t idx, const int ndim, const int64_t* out_shape, const int64_t* out_stride,
    const int64_t* lhs_shape, const int64_t* lhs_stride,
    const int64_t* rhs_shape, const int64_t* rhs_stride,
    int64_t *lhs_out, int64_t *rhs_out) {
  if (out_stride[0] == lhs_stride[0]) {
    for (int d = 0; d < ndim; ++d) {
      int64_t o_sh = out_shape[d];
      int64_t o_st = out_stride[d];
      int64_t rhs_sh = rhs_shape[d];
      int64_t rhs_st = rhs_stride[d];
      int64_t i = (idx / o_st) % o_sh;
      /*
       * Simplfied for rhs_out += min(i, rhs_sh - 1) * rhs_st;
       * rhs_sh be o_sh or 1
       */
      if (rhs_sh > i) {
        *rhs_out += i * rhs_st;
      }
    }
    *lhs_out = idx;
  } else {
    for (int d = 0; d < ndim; ++d) {
      int64_t o_sh = out_shape[d];
      int64_t o_st = out_stride[d];
      int64_t lhs_sh = lhs_shape[d];
      int64_t lhs_st = lhs_stride[d];

      int64_t i = (idx / o_st) % o_sh;
      /*
       * Simplfied for lhs_out += min(i, lhs_sh - 1) * lhs_st;
       * lhs_sh be o_sh or 1
       */
      if (lhs_sh > i) {
        *lhs_out += i * lhs_st;
      }
    }
    *rhs_out = idx;
  }
}

template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(const aten::CSRMatrix& csr,
                NDArray ufeat, NDArray efeat,
                NDArray out) {
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = Op::use_rhs ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = out->shape[1];
  DType* O = static_cast<DType*>(out->data);
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      DType accum = 0;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const DType* lhs_off = Op::use_lhs? X + cid * dim + k : nullptr;
        const DType* rhs_off = Op::use_rhs? W + edges[j] * dim + k : nullptr;
        accum += Op::Call(lhs_off, rhs_off);
      }
      out_off[k] = accum;
    }
  }
}


template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsr(const aten::CSRMatrix& csr,
                NDArray ufeat, NDArray efeat,
                NDArray out, NDArray argu, NDArray arge) {
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = Op::use_rhs ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = out->shape[1];
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    IdType* argx_off = argX + rid * dim;
    IdType* argw_off = argW + rid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      DType accum = Cmp::zero;
      IdType ax = 0, aw = 0;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const DType* lhs_off = Op::use_lhs? X + cid * dim + k : nullptr;
        const DType* rhs_off = Op::use_rhs? W + edges[j] * dim + k : nullptr;
        const DType val = Op::Call(lhs_off, rhs_off);
        if (Cmp::Call(accum, val)) {
          accum = val;
          if (Op::use_lhs)
            ax = cid;
          if (Op::use_rhs)
            aw = edges[j];
        }
      }
      out_off[k] = accum;
      if (Op::use_lhs)
        argx_off[k] = ax;
      if (Op::use_rhs)
        argw_off[k] = aw;
    }
  }
}

namespace op {
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off + *rhs_off;
  }
};

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off * *rhs_off;
  }
};

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(const DType* lhs_off, const DType* ) {
    return *lhs_off;
  }
};

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* , const DType* rhs_off) {
    return *rhs_off;
  }
};

template <typename DType>
struct Max {
  static constexpr DType zero = std::numeric_limits<DType>::lowest();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum < val;
  }
};

template <typename DType>
struct Min {
  static constexpr DType zero = std::numeric_limits<DType>::max();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum > val;
  }
};

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef dgl::kernel::cpu::op::Add<DType> Op;                  \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef dgl::kernel::cpu::op::Mul<DType> Op;                  \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_u") {                                  \
      typedef dgl::kernel::cpu::op::CopyLhs<DType> Op;              \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_e") {                                  \
      typedef dgl::kernel::cpu::op::CopyRhs<DType> Op;              \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

}  // namespace op

}  // namespace cpu
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_SPMM_CUH_
