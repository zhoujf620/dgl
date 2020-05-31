/*
 *
 */

#ifndef DGL_KERNEL_CUDA_SPMM_CUH_
#define DGL_KERNEL_CUDA_SPMM_CUH_

#include "../../graph/unit_graph.h"
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

/*
 * This func do the followings:
 *   1. Convert flattened index to multi-dimension index
 *      according to output shape (assume row-major).
 *   2. Convert multi-dimension index to flattened index for lhs.
 *   3. Convert multi-dimension index to flattened index for rhs.
 */
__device__ __forceinline__ void UnravelRavel(
    const int64_t idx, const int ndim, const int64_t* out_shape, const int64_t* out_len,
    const int64_t* lhs_shape, const int64_t* lhs_stride,
    const int64_t* rhs_shape, const int64_t* rhs_stride,
    int64_t *lhs_out, int64_t *rhs_out) {
  if (out_len[0] == lhs_stride[0]) {
    for (int d = 0; d < ndim; ++d) {
      int64_t o_sh = out_shape[d];
      int64_t o_st = out_len[d];
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
      int64_t o_st = out_len[d];
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

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void SpMMCooKernel(
  DType *ufeat, DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  Idx *row, Idx *col, Idx* edge_map,
  int64_t N, int64_t M, int64_t E,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  const bool has_idx = edge_map;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = has_idx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    DType* uoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_len): nullptr;
    DType* eoff = BinaryOp::UseRhs() ? (efeat + eid * efeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    Idx* arguoff = (ReduceOp::RequireArg() && BinaryOp::UseLhs()) ? (arg_u + dst * out_len): nullptr;
    Idx* argeoff = (ReduceOp::RequireArg() && BinaryOp::UseRhs()) ? (arg_e + dst * out_len): nullptr;
    while (tx < out_len) {
      int64_t lhs_add = ubcast_off ? ubcast_off[tx] : tx;
      int64_t rhs_add = ebcast_off ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::Call(tx, outoff, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void ArgSpMMCooKernel(
  DType *ufeat, DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  Idx *row, Idx *col, Idx* edge_map,
  int64_t N, int64_t M, int64_t E,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
  const bool has_idx = edge_map;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = has_idx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    DType* uoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_len): nullptr;
    DType* eoff = BinaryOp::UseRhs() ? (efeat + eid * efeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::UseLhs() ? (arg_u + dst * out_len): nullptr;
    Idx* argeoff = BinaryOp::UseRhs() ? (arg_e + dst * out_len): nullptr;
    while (tx < out_len) {
      int64_t lhs_add = ubcast_off ? ubcast_off[tx] : tx;
      int64_t rhs_add = ebcast_off ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void SpMMCsrKernel(
  DType *ufeat, DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  Idx *indptr, Idx *indices, Idx *edge_map,
  int64_t N, int64_t M, int64_t E,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  const bool has_idx = edge_map;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < M) {
    const Idx dst = ty;
    for (Idx i = indptr[dst]; i < indptr[dst + 1]; ++i) {
      const Idx eid = has_idx ? _ldg(edge_map + i) : i;
      const Idx src = i;
      int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
      const int64_t stride_x = blockDim.x * gridDim.x;
      DType* uoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_len): nullptr;
      DType* eoff = BinaryOp::UseRhs() ? (efeat + eid * efeat_len): nullptr;
      DType* outoff = out + dst * out_len;
      Idx* arguoff = (ReduceOp::RequireArg() && BinaryOp::UseLhs()) ? (arg_u + dst * out_len): nullptr;
      Idx* argeoff = (ReduceOp::RequireArg() && BinaryOp::UseRhs()) ? (arg_e + dst * out_len): nullptr;
      while (tx < out_len) {
        int64_t lhs_add = ubcast_off ? ubcast_off[tx] : tx;
        int64_t rhs_add = ebcast_off ? ebcast_off[tx] : tx;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(tx, outoff, arguoff, argeoff, out, src, eid);
        tx += stride_x;
      }
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    dgl::aten::COOMatrix coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  Idx *row = static_cast<Idx*>(coo.row->data),
      *col = static_cast<Idx*>(coo.col->data),
      *edge_map = aten::IsNullArray(coo.data) ?
          nullptr : static_cast<Idx*>(coo.data->data);
  DType *ufeat_data = static_cast<DType*>(ufeat->data),
        *efeat_data = static_cast<DType*>(efeat->data),
        *out_data = static_cast<DType*>(out->data);
  Idx *argu_data = static_cast<Idx*>(argu->data),
      *arge_data = static_cast<Idx*>(arge->data);
  cudaStream_t stream{nullptr};
  int64_t N = coo.num_rows, M = coo.num_cols, E = efeat->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  // ComputeBcastOff(ubcast_off, ebast_off, info);
  int64_t len = 1;
  for (int64_t i = 1; i < ufeat->ndim; ++i)
    len *= ufeat->shape[i];
  const dim3 nblks(E, 1);
  const dim3 nthrs(1, 32);

  SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
    <<<nblks, nthrs, 0, stream>>>(
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      row, col, edge_map,
      N, M, E,
      ubcast_off, ebcast_off,
      len, len, len
    );
  if (ReduceOp::RequireArg()) {
    ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
      <<<nblks, nthrs, 0, stream>>>(
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map,
        N, M, E,
        ubcast_off, ebcast_off,
        len, len, len
      );
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMBcastCoo(
    dgl::aten::COOMatrix coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge,
    BcastInfo info) {
  Idx *row = static_cast<Idx*>(coo.row->data),
      *col = static_cast<Idx*>(coo.col->data),
      *edge_map = aten::IsNullArray(coo.data) ?
          nullptr : static_cast<Idx*>(coo.data->data);
  DType *ufeat_data = static_cast<DType*>(ufeat->data),
        *efeat_data = static_cast<DType*>(efeat->data),
        *out_data = static_cast<DType*>(out->data);
  Idx *argu_data = static_cast<Idx*>(argu->data),
      *arge_data = static_cast<Idx*>(arge->data);
  cudaStream_t stream{nullptr};
  int64_t N = coo.num_rows, M = coo.num_cols, E = efeat->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  // ComputeBcastOff(ubcast_off, ebast_off, info);
  int64_t ufeat_len = utils::Prod(info.lhs_shape);
  int64_t efeat_len = utils::Prod(info.rhs_shape);
  int64_t out_len = utils::Prod(info.out_shape);
  const dim3 nblks(E, 1);
  const dim3 nthrs(1, 32);

  SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
    <<<nblks, nthrs, 0, stream>>>(
      ufeat_data, efeat_data, out_data,
      row, col, edge_map,
      N, M, E,
      ubcast_off, ebcast_off,
      ufeat_len, efeat_len, out_len
    );
  if (ReduceOp::RequireArg()) {
    ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
      <<<nblks, nthrs, 0, stream>>>(
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map,
        N, M, E,
        ubcast_off, ebcast_off,
        ufeat_len, efeat_len, out_len
      );
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    dgl::aten::CSRMatrix csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  Idx *indptr = static_cast<Idx*>(csr.indptr->data),
      *indices = static_cast<Idx*>(csr.indices->data), 
      *edge_map = static_cast<Idx*>(csr.data->data);
  DType *ufeat_data = static_cast<DType*>(ufeat->data),
        *efeat_data = static_cast<DType*>(efeat->data),
        *out_data = static_cast<DType*>(out->data);
  Idx *argu_data = static_cast<Idx*>(argu->data),
      *arge_data = static_cast<Idx*>(arge->data);
  cudaStream_t stream{nullptr};
  int64_t N = csr.num_rows, M = csr.num_cols, E = efeat->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  // ComputeBcastOff(ubcast_off, ebast_off, info);
  int64_t len = 1;
  for (int64_t i = 1; i < ufeat->ndim; ++i)
    len *= ufeat->shape[i];
  const dim3 nblks(N, 1);
  const dim3 nthrs(1, 32);

  SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp>
    <<<nblks, nthrs, 0, stream>>>(
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      indptr, indices, edge_map,
      N, M, E,
      ubcast_off, ebcast_off,
      len, len, len
    );
}


}
}
}

#endif