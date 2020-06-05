#include "./sddmm.cuh"
#include "./functor2.cuh"
#include <dgl/array.h>
#include "../binary_reduce.h"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace kernel {

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
      LOG(FATAL) << "Unsupported SpMM/SDDMM binary operator: " << op;     \
    }                                                               \
  } while (0)


template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
	if (op == "dot") {
		int64_t reduce_dim = ufeat->shape[ufeat->ndim - 1];
		if (reduce_dim <= 2)
			cuda::SDDMMDotCsr<IdType, DType, 1>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 4)
			cuda::SDDMMDotCsr<IdType, DType, 2>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 8)
			cuda::SDDMMDotCsr<IdType, DType, 4>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 16)
			cuda::SDDMMDotCsr<IdType, DType, 8>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 32)
			cuda::SDDMMDotCsr<IdType, DType, 16>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 64)
			cuda::SDDMMDotCsr<IdType, DType, 32>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 128)
			cuda::SDDMMDotCsr<IdType, DType, 64>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 256)
			cuda::SDDMMDotCsr<IdType, DType, 128>(csr, ufeat, vfeat, out);
		else if (reduce_dim <= 512)
			cuda::SDDMMDotCsr<IdType, DType, 256>(csr, ufeat, vfeat, out);
		else
			cuda::SDDMMDotCsr(IdType, DType, 512>(csr, ufeat, vfeat, out);	
	} else {
		SWITCH_OP(op, Op, {
			cuda::SDDMMCsr<IdType, DType, Op>(csr, ufeat, vfeat, out);
		});
	}
}

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
	if (op == "dot") {
		int64_t reduce_dim = ufeat->shape[ufeat->ndim - 1];
		if (reduce_dim <= 2)
			cuda::SDDMMDotCoo<IdType, DType, 1>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 4)
			cuda::SDDMMDotCoo<IdType, DType, 2>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 8)
			cuda::SDDMMDotCoo<IdType, DType, 4>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 16)
			cuda::SDDMMDotCoo<IdType, DType, 8>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 32)
			cuda::SDDMMDotCoo<IdType, DType, 16>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 64)
			cuda::SDDMMDotCoo<IdType, DType, 32>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 128)
			cuda::SDDMMDotCoo<IdType, DType, 64>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 256)
			cuda::SDDMMDotCoo<IdType, DType, 128>(coo, ufeat, vfeat, out);
		else if (reduce_dim <= 512)
			cuda::SDDMMDotCoo<IdType, DType, 256>(coo, ufeat, vfeat, out);
		else
			cuda::SDDMMDotCoo(IdType, DType, 512>(coo, ufeat, vfeat, out);	
	} else {
		SWITCH_OP(op, Op, {
			cuda::SDDMMCoo<IdType, DType, Op>(coo, ufeat, vfeat, out);
		});
	}
}

template <int XPU, typename IdType, typename DType>
void SDDMMBcastCsr(const std::string& op,
                   const BcastInfo& info,
                   const aten::CSRMatrix& csr,
                   NDArray ufeat,
                   NDArray vfeat,
                   NDArray out,
                   std::vector<NDArray> out_aux) {
  SWITCH_OP(op, Op, {
    cuda::SDDMMBcastCsr<IdType, DType, Op>(info, csr, ufeat, vfeat, out);
  });
}

template <int XPU, typename IdType, typename DType>
void SDDMMBcastCoo(const std::string& op,
                   const BcastInfo& info,
                   const aten::COOMatrix& coo,
                   NDArray ufeat,
                   NDArray vfeat,
                   NDArray out,
                   std::vector<NDArray> out_aux) {
  SWITCH_OP(op, Op, {
    cuda::SDDMMBcastCoo<IdType, DType, Op>(info, coo, ufeat, vfeat, out);
  });
}

template void SDDMMCsr<kDLGPU, int32_t, float>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLGPU, int64_t, float>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLGPU, int32_t, double>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLGPU, int64_t, double>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);

template void SDDMMCoo<kDLGPU, int32_t, float>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLGPU, int64_t, float>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLGPU, int32_t, double>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLGPU, int64_t, double>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);

template void SDDMMBcastCsr<kDLGPU, int32_t, float>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLGPU, int64_t, float>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLGPU, int32_t, double>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLGPU, int64_t, double>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);

template void SDDMMBcastCoo<kDLGPU, int32_t, float>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLGPU, int64_t, float>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLGPU, int32_t, double>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLGPU, int64_t, double>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out, std::vector<NDArray> out_aux);

}  // namespace kernel
}  // namespace dgl
