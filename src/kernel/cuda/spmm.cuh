/*
 *
 */

#ifndef DGL_KERNEL_CUDA_SPMM_CUH_
#define DGL_KERNEL_CUDA_SPMM_CUH_

#include "../graph/unit_graph.h"
#include "../util.h"

namespace dgl {
namespace kernel {
namespace cuda {

/*
 * This func do the followings:
 *   1. Convert flattened index to multi-dimension index
 *      according to output shape (assume row-major).
 *   2. Convert multi-dimension index to flattened index for lhs.
 *   3. Convert multi-dimension index to flattened index for rhs.
 */
__device__ __forceinline__ void UnravelRavel(
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

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void SpMMCooKernel(
  DType *ufeat, DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  Idx *row, Idx *col, Idx* edge_map,
  int64_t N, int64_t M, int64_t E, int64_t ndim,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t *ufeat_shp, int64_t *efeat_shp, int64_t *out_shp,
  int64_t ufeat_stride, int64_t efeat_stride, int64_t out_stride) {
  // SPMM with COO.
  const Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = _ldg(edge_map + ty);
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    DType* uoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_stride): nullptr;
    DType* eoff = BinaryOp::UseRhs() ? (efeat + eid * efeat_stride): nullptr;
    DType* outoff = out + dst * out_stride;
    Idx* arguoff = (ReduceOp::RequireArg() && BinaryOp::UseLhs()) ? (arg_u + dst * out_stride): nullptr;
    Idx* argeoff = (ReduceOp::RequireArg() && BinaryOp::UseRhs()) ? (arg_e + dst * out_stride): nullptr;
    while (tx < out_stride) {
      DType val = BinaryOp::Call(uoff + ubcast_off[tx], eoff + ebcast_off[tx]);
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
  int64_t N, int64_t M, int64_t E, int64_t ndim,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t *ufeat_shp, int64_t *efeat_shp, int64_t *out_shp,
  int64_t ufeat_stride, int64_t efeat_stride, int64_t out_stride) {
  // SPMM with COO.
  const Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = _ldg(edge_map + ty);
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    DType* uoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_stride): nullptr;
    DType* eoff = BinaryOp::UseRhs() ? (efeat + eid * efeat_stride): nullptr;
    DType* outoff = out + dst * out_stride;
    Idx* arguoff = BinaryOp::UseLhs() ? (arg_u + dst * out_stride): nullptr;
    Idx* argeoff = BinaryOp::UseRhs() ? (arg_e + dst * out_stride): nullptr;
    while (tx < out_stride) {
      DType val = BinaryOp::Call(uoff + ubcast_off[tx], eoff + ebcast_off[tx]);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

__global__ void SpMMCsrKernel(
  DType *ufeat, DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  Idx *indptr, Idx *indices, Idx *edge_map,
  int64_t N, int64_t M, int64_t E, int64_t ndim,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t *ufeat_shp, int64_t *efeat_shp, int64_t *out_shp,
  int64_t ufeat_stride, int64_t efeat_stride, int64_t out_stride) {
  // SPMM with COO.
  const Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  Idx start = 0, end = indptr[dst];
  while (ty < M) {
    const Idx dst = ty;
    for (Idx i = indptr[dst]; i < indptr[dst + 1]; ++i) {
      const Idx eid = _ldg(edge_map + i);
      const Idx src = _ldg(indptr + i);
      int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
      const int64_t stride_x = blockDim.x * gridDim.x;
      DType* uoff = BinaryOp::UseLhs() ? (ufeat + src * ufeat_stride): nullptr;
      DType* eoff = BinaryOp::UseRhs() ? (efeat + eid * efeat_stride): nullptr;
      DType* outoff = out + dst * out_stride;
      Idx* arguoff = (ReduceOp::RequireArg() && BinaryOp::UseLhs()) ? (arg_u + dst * out_stride): nullptr;
      Idx* argeoff = (ReduceOp::RequireArg() && BinaryOp::UseRhs()) ? (arg_e + dst * out_stride): nullptr;
      while (tx < out_stride) {
        DType out = BinaryOp::Call(uoff + ubcast_off[tx], eoff + ebcast_off[tx]);
        ReduceOp::Call(tx, outoff, arguoff, argeoff, out, src, eid);
        tx += stride_x;
      }
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void CudaSpMMCoo(
    dgl::aten::COOMatrix coo,
    NDArray ufeat,
    NDArray efeat,
    NDArray out) {
  // TODO(zihao)
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void CudaSpMMCsr(
    dgl::aten::CSRMatrix csr,
    NDArray ufeat,
    NDArray efeat,
    NDArray out) {
  // TODO(zihao)
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void CudaCallSpMM(
  const UnitGraph* graph,
  NDArray ufeat,
  NDArray efeat,
  NDArray out,
  SparseFormat preferred_format = SparseFormat::kCsc,
  ) {
  // TODO(zihao)
}

namespace binary {
template <typename DType>
struct Add {
  static __device__ __forceinline__ void Call(
      DType *lhs, DType *rhs, int64_t len_lhs = 1, int64_t len_rhs = 1) {
    return lhs[0] + rhs[0];
  }
  static __device__ __forceinline__ bool UseLhs() {
    return true;
  }
  static __device__ __forceinline__ bool UseRhs() {
    return true;
  }
  static __device__ __forceinline__ bool ReduceDim() {
    return false;
  }
  static __device__ __forceinline__ int64_t ReduceSize(int64_t *feat_shp, int64_t ndim) {
    return 1;
  }
};

template <typename DType>
struct Mul {
  static __device__ __forceinline__ void Call(
      DType *lhs, DType *rhs, int64_t len_lhs = 1, int64_t len_rhs = 1) {
    return lhs[0] * rhs[0];
  }
  static __device__ __forceinline__ bool UseLhs() {
    return true;
  }
  static __device__ __forceinline__ bool UseRhs() {
    return true;
  }
  static __device__ __forceinline__ bool ReduceDim() {
    return false;
  }
  static __device__ __forceinline__ int64_t ReduceSize(int64_t *feat_shp, int64_t ndim) {
    return 1;
  }
};

template <typename DType>
struct CopyU {
  static __device__ __forceinline__ void Call(
      DType *lhs, DType *rhs, int64_t len_lhs = 1, int64_t len_rhs = 1) {
    return lhs[0];
  }
  static __device__ __forceinline__ bool UseLhs() {
    return true;
  }
  static __device__ __forceinline__ bool UseRhs() {
    return false;
  }
  static __device__ __forceinline__ bool ReduceDim() {
    return false;
  }
  static __device__ __forceinline__ int64_t ReduceSize(int64_t *feat_shp, int64_t ndim) {
    return 1;
  }
};

template <typename DType>
struct CopyE {
  static __device__ __forceinline__ void Call(
      DType *lhs, DType *rhs, int64_t len_lhs = 1, int64_t len_rhs = 1) {
    return rhs[0];
  }
  static __device__ __forceinline__ bool UseLhs() {
    return false;
  }
  static __device__ __forceinline__ bool UseRhs() {
    return true;
  }
  static __device__ __forceinline__ bool ReduceDim() {
    return false;
  }
  static __device__ __forceinline__ int64_t ReduceSize(int64_t *feat_shp, int64_t ndim) {
    return 1;
  }
};

template <typename DType>
struct Dot {
  static __device__ __forceinline__ void Call(
      DType *lhs, DType *rhs, int64_t len_lhs = 1, int64_t len_rhs = 1) {
    DType rst = static_cast<DType>(0);
    for (int64_t i = 0; i < max(len_lhs, len_rhs); ++i) {
      rst += lhs[min(i, len_lhs - 1)] * rhs[min(i, len_rhs - 1)];
    }
    return rst;
  }
  static __device__ __forceinline__ bool UseLhs() {
    return true;
  }
  static __device__ __forceinline__ bool UseRhs() {
    return true;
  }
  static __device__ __forceinline__ bool ReduceDim() {
    return true;
  }
  static __device__ __forceinline__ int64_t ReduceSize(int64_t *feat_shp, int64_t ndim) {
    return feat_shp[ndim - 1];
  }
};


}   // end of namespace binary

namespace reduce {
template <typename Idx,
          typename DType,
          bool atomic=false>
struct Sum {
  static __device__ __forceinline__ void Call(Idx fid,
    DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, Idx uid, Idx eid) {
    if (!atomic) {
      out_buf[fid] += val;
    } else {
      cuda::AtomicAdd(out_buf + fid, val);
    }
  }
  static __device__ __forceinline__ bool RequireArg() {
    return false;
  }
  static __device__ __forceinline__ void CallArg(Idx fid,
    DType *arg_u_buf, DType *arg_e_buf,
    DType val, DType val_ref, Idx uid, Idx eid) {
      // placeholder
    }
};

template <typename Idx,
          typename DType,
          bool atomic=false>
struct Max {
  static __device__ __forceinline__ void Call(Idx fid,
    DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, Idx uid, Idx eid) {
    if (!atomic) {
      Idx max_val = max(out_buf[fid], val);
      if (max_val == val) {
        out_buf[fid] = max_val;
        arg_u_buf[fid] = uid;
        arg_e_buf[fid] = eid;
      }
    } else {
      cuda::AtomicMax(out_buf + fid, val);
    }
  }
  static __device__ __forceinline__ bool RequireArg() {
    return true;
  }
  static __device__ __forceinline__ void CallArg(Idx fid,
    DType *arg_u_buf, DType *arg_e_buf,
    DType val, DType val_ref, Idx uid, Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf)
          arg_u_buf[fid] = uid; // TODO(zihao): select min?
        if (arg_e_buf)
          arg_e_buf[fid] = eid;
      }
    }
  }
};

template <typename Idx,
          typename DType,
          bool atomic=false>
struct Min {
  static __device__ __forceinline__ void Call(Idx fid,
    DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, Idx uid, Idx eid) {
    if (!atomic) {
      Idx min_val = min(out_buf[fid], val);
      if (min_val == val) {
        out_buf[fid] = min_val;
        arg_u_buf[fid] = uid;
        arg_e_buf[fid] = eid;
      }
    } else {
      cuda::AtomicMin(out_buf + fid, val);
    }
  }
  static __device__ __forceinline__ bool RequireArg() {
    return true;
  }
  static __device__ __forceinline__ void CallArg(Idx fid,
    DType *arg_u_buf, DType *arg_e_buf,
    DType val, DType val_ref, Idx uid, Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf)
          arg_u_buf[fid] = uid; // TODO(zihao): select min?
        if (arg_e_buf)
          arg_e_buf[fid] = eid;
      }
    }
  }
};

}   // end of namespace reduce


}
}
}

#endif