#include "./spmm.cuh"
#include "./functor2.cuh"
#include <minigun/minigun.h>
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace kernel {

using minigun::advance::RuntimeConfig;

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef dgl::kernel::cuda::binary::Add<DType> Op;             \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef dgl::kernel::cuda::binary::Mul<DType> Op;             \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_u") {                                  \
      typedef dgl::kernel::cuda::binary::CopyU<DType> Op;           \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_e") {                                  \
      typedef dgl::kernel::cuda::binary::CopyE<DType> Op;           \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

template <int XPU, typename IdType, typename DType>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType> >(
          csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Max<IdType, DType> >(
          csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Min<IdType, DType> >(
          csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template <int XPU, typename IdType, typename DType>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const aten::COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Sum<IdType, DType, true> > (
          coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Max<IdType, DType, true> > (
          coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  }  else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Min<IdType, DType, true> > (
          coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template void SpMMCsr<kDLGPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void SpMMCoo<kDLGPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

namespace cusparse {

template <typename DType>
cusparseStatus_t Xcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA,
    const DType* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA,
    const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA,
    const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <typename DType>
void CusparseCsrmm2(
    const RuntimeConfig& rtcfg,
    const aten::CSRMatrix& csr,
    const DType* B_data, DType* C_data,
    int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix for node
  // feature tensor. However, since cusparse only supports column-major, while our tensor
  // is stored in row-major, the actual computation is:
  // C = trans(A x trans(B)).
  // Currently, we use cublasXgeam to implement transposition and allocate intermediate
  // workspace memory for this.
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(rtcfg.ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, rtcfg.stream));
  // allocate matrix for temporary transposed output
  DType* trans_out = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, m * n * sizeof(DType)));
  // all one data array
  DType* valptr = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, nnz * sizeof(DType)));
  utils::Fill<kDLGPU>(rtcfg.ctx, valptr, nnz, static_cast<DType>(1.));
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE,
      m, n, k, nnz, &alpha,
      descr, valptr,
      static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data),
      B_data, n, &beta, trans_out, m));
  device->FreeWorkspace(rtcfg.ctx, valptr);
  // transpose the output matrix
  if (!thr_entry->cublas_handle) {
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  }
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, rtcfg.stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n, m,
      &alpha, trans_out, m,
      &beta, nullptr, n,
      C_data, n));
  device->FreeWorkspace(rtcfg.ctx, trans_out);
}

}  // namespace cusparse

/*
template <int32_t, float>
void SpMMSumCsr(const aten::CSRMatrix& csr,
                NDArray ufeat, NDArray efeat,
                NDArray out) {
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out->ctx;
  DType *udata = static_cast<float*>(ufeat->data);
  DType *edata = static_cast<float*>(efeat->data);
  const int64_t len = out->shape[1];
  cusparse::CusparseCsrmm2(rtcfg, csr, edata, udata, len);
}

template <int32_t, double>
void SpMMSumCsr(const aten::CSRMatrix& csr,
                NDArray ufeat, NDArray efeat,
                NDArray out) {
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out->ctx;
  DType *udata = static_cast<double*>(ufeat->data);
  DType *edata = static_cast<double*>(efeat->data);
  const int64_t len = out->shape[1];
  cusparse::CusparseCsrmm2(rtcfg, csr, edata, udata, len);
}

template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(const aten::CSRMatrix& csr,
                NDArray ufeat, NDArray efeat,
                NDArray out) {
  
} 
*/


}  // namespace kernel
}  // namespace dgl
