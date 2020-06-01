/*
 *
 */

#ifndef DGL_KERNEL_CUDA_SDDMM_CUH_
#define DGL_KERNEL_CUDA_SDDMM_CUH_

#include "../utils.h"
#include "../binary_reduce_impl_decl.h"
#include "../binary_reduce.h"
#include "atomic.cuh"

namespace dgl {
namespace kernel {
namespace cuda {

template <typename T>
__device__ __forceinline__ T _ldg(T* addr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
}

template <typename Idx, typename DType,
          typename BinaryOp>
__global__ void SDDMMCooKernel(
  DType *ufeat, DType *vfeat, DType *out,
  Idx *row, Idx *col, Idx* edge_map,
  int64_t N, int64_t M, int64_t E, int64_t reduce_size,
  int64_t *ubcast_off, int64_t *vbcast_off,
  int64_t ufeat_len, int64_t vfeat_len, int64_t out_len) {
  // SDDMM with COO.
  const bool has_idx = edge_map;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = has_idx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    DType* lhsoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_len): nullptr;
    DType* rhsoff = BinaryOp::UseRhs() ? (vfeat + dst * vfeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      DType val = BinaryOp::Call(
          lhsoff + ubcast_off[tx] * reduce_size,
          rhsoff + vbcast_off[tx] * reduce_size,
          reduce_size, reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(Idx *array, Idx length, Idx eid) {
  Idx lo = 0, hi = length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (_ldg(array + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (_ldg(array + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp>
__global__ void SDDMMCsrKernel(
  DType *ufeat, DType *vfeat, DType *out,
  Idx *indptr, Idx *indices, Idx* edge_map,
  int64_t N, int64_t M, int64_t E, int64_t reduce_size,
  int64_t *ubcast_off, int64_t *vbcast_off,
  int64_t ufeat_len, int64_t vfeat_len, int64_t out_len) {
  // SDDMM with Csr.
  const bool has_idx = edge_map;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = BinarySearchSrc<Idx>(indptr, N, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = has_idx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    DType* lhsoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_len): nullptr;
    DType* rhsoff = BinaryOp::UseRhs() ? (vfeat + dst * vfeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      DType val = BinaryOp::Call(
          lhsoff + ubcast_off[tx] * reduce_size,
          rhsoff + vbcast_off[tx] * reduce_size,
          reduce_size, reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType, typename Op>
void SDDMMCoo(
    const dgl::aten::COOMatrix& coo,
    NDArray ufeat,
    NDArray vfeat,
    NDArray out) {
  Idx *row = static_cast<Idx*>(coo.row->data),
      *col = static_cast<Idx*>(coo.col->data),
      *edge_map = aten::IsNullArray(coo.data) ?
          nullptr : static_cast<Idx*>(coo.data->data);
  DType *ufeat_data = static_cast<DType*>(ufeat->data),
        *vfeat_data = static_cast<DType*>(vfeat->data),
        *out_data = static_cast<DType*>(out->data);
  cudaStream_t stream{nullptr};
  int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = 1, reduce_size = ufeat->shape[ufeat->ndim - 1];
  for (int64_t i = 1; i < ufeat->ndim; ++i)
    len *= ufeat->shape[i];
  const dim3 nblks(E, 1);
  const dim3 nthrs(1, 32);

  SDDMMCooKernel<Idx, DType, Op>
    <<<nblks, nthrs, 0, stream>>>(
      ufeat_data, vfeat_data, out_data,
      row, col, edge_map,
      N, M, E, reduce_size,
      ubcast_off, ebcast_off,
      len, len, len
    );
}

template <typename Idx, typename DType, typename Op>
void SDDMMCsr(
    const dgl::aten::CSRMatrix& csr,
    NDArray ufeat,
    NDArray vfeat,
    NDArray out) {
  Idx *indptr = static_cast<Idx*>(csr.indptr->data),
      *indices = static_cast<Idx*>(csr.indices->data),
      *edge_map = aten::IsNullArray(csr.data) ?
          nullptr : static_cast<Idx*>(csr.data->data);
  DType *ufeat_data = static_cast<DType*>(ufeat->data),
        *vfeat_data = static_cast<DType*>(vfeat->data),
        *out_data = static_cast<DType*>(out->data);
  cudaStream_t stream{nullptr};
  int64_t N = csr.num_rows, M = csr.num_cols, E = csr.indices->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = 1, reduce_size = ufeat->shape[ufeat->ndim - 1];
  for (int64_t i = 1; i < ufeat->ndim; ++i)
    len *= ufeat->shape[i];
  const dim3 nblks(E, 1);
  const dim3 nthrs(1, 32);

  SDDMMCsrKernel<Idx, DType, Op>
    <<<nblks, nthrs, 0, stream>>>(
      ufeat_data, vfeat_data, out_data,
      indptr, indices, edge_map,
      N, M, E, reduce_size,
      ubcast_off, ebcast_off,
      len, len, len
    );
}


}
}
}

#endif
