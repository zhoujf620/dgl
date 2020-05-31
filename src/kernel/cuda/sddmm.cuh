/*
 *
 */

#ifndef DGL_KERNEL_CUDA_SDDMM_CUH_
#define DGL_KERNEL_CUDA_SDDMM_CUH_

#include "../graph/unit_graph.h"
#include "../util.h"

namespace dgl {
namespace kernel {
namespace cuda {

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void SDDMMCooKernel(
  DType *ufeat, DType *vfeat, DType *out,
  Idx *row, Idx *col, Idx* edge_map,
  int64_t N, int64_t M, int64_t E, int64_t ndim,
  int64_t *ubcast_off, int64_t *vbcast_off,
  int64_t *ufeat_shp, int64_t *vfeat_shp, int64_t *out_shp,
  int64_t ufeat_len, int64_t vfeat_len, int64_t out_len) {
  // SDDMM with COO.
  const bool has_idx = !aten::IsNullArray(csr.data);
  const Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
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
          lhsoff + ubcast_off[tx] * BinaryOp::ReduceSize(ufeat_shp, ndim),
          rhsoff + vbcast_off[tx] * BinaryOp::ReduceSize(vfeat_shp, ndim));
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
          typename BinaryOp, typename ReduceOp>
__global__ void SDDMMCsrKernel(
  DType *ufeat, DType *vfeat, DType *out,
  Idx *indptr, Idx *indices, Idx* edge_map,
  int64_t N, int64_t M, int64_t E, int64_t ndim,
  int64_t *ubcast_off, int64_t *vbcast_off,
  int64_t *ufeat_shp, int64_t *vfeat_shp, int64_t *out_shp,
  int64_t ufeat_len, int64_t vfeat_len, int64_t out_len) {
  // SDDMM with Csr.
  const bool has_idx = !aten::IsNullArray(csr.data);
  const Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = BinarySearchSrc(indptr, N, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = has_idx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    DType* lhsoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_len): nullptr;
    DType* rhsoff = BinaryOp::UseRhs() ? (vfeat + dst * vfeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      DType val = BinaryOp::Call(
          lhsoff + ubcast_off[tx] * BinaryOp::ReduceSize(ufeat_shp, ndim),
          rhsoff + vbcast_off[tx] * BinaryOp::ReduceSize(vfeat_shp, ndim));
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void CudaSDDMMCoo(
    dgl::aten::COOMatrix coo,
    NDArray ufeat,
    NDArray vfeat,
    NDArray out) {
  // TODO(zihao)
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void CudaSDDMMCsr(
    dgl::aten::CSRMatrix csr,
    NDArray ufeat,
    NDArray vfeat,
    NDArray out) {
  // TODO(zihao)
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void CudaCallSDDMM(
  const UnitGraph* graph,
  NDArray ufeat,
  NDArray vfeat,
  NDArray out,
  SparseFormat preferred_format = SparseFormat::kCoo,
  ) {
  // TODO(zihao)
}


}
}
}

#endif