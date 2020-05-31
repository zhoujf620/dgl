#ifndef DGL_KERNEL_CPU_SDDMM_CUH_
#define DGL_KERNEL_CPU_SDDMM_CUH_

#include <dgl/array.h>

namespace dgl {
namespace kernel {
namespace cpu {

template <typename IdType, typename DType, typename Op>
void SDDMMCsr(const aten::CSRMatrix& csr,
              NDArray ufeat, NDArray vfeat, NDArray out) {
  const bool has_idx = !aten::IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = has_idx? nullptr : static_cast<IdType*>(csr.data->data);
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* Y = Op::use_rhs? static_cast<DType*>(vfeat->data) : nullptr;
  const int64_t dim = out->shape[1];
  DType* O = static_cast<DType*>(out->data);
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      DType* out_off = O + (has_idx? edges[j] : j) * dim;
      for (int64_t k = 0; k < dim; ++k) {
        const DType* lhs_off = Op::use_lhs? X + rid * dim + k : nullptr;
        const DType* rhs_off = Op::use_rhs? Y + cid * dim + k : nullptr;
        out_off[k] = Op::Call(lhs_off, rhs_off);
      }
    }
  }
}

template <typename IdType, typename DType, typename Op>
void SDDMMCoo(const aten::COOMatrix& coo,
              NDArray ufeat, NDArray vfeat, NDArray out) {
  const bool has_idx = !aten::IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges = has_idx? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* Y = Op::use_rhs? static_cast<DType*>(vfeat->data) : nullptr;
  const int64_t dim = out->shape[1];
  DType* O = static_cast<DType*>(out->data);
  const int64_t nnz = coo.row->shape[0];
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx? edges[i] : i;
    DType* out_off = O + eid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const DType* lhs_off = Op::use_lhs? X + rid * dim + k : nullptr;
      const DType* rhs_off = Op::use_rhs? Y + cid * dim + k : nullptr;
      out_off[k] = Op::Call(lhs_off, rhs_off);
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

#endif  // DGL_KERNEL_CPU_SDDMM_CUH_
