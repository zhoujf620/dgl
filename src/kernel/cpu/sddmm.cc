#include "./sddmm.h"
#include <dgl/array.h>

namespace dgl {
namespace kernel {

template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
  if (op == "dot") {
    cpu::SDDMMDotCsr<IdType, DType>(csr, ufeat, vfeat, out);
  } else {
    SWITCH_OP(op, Op, {
      cpu::SDDMMCsr<IdType, DType, Op>(csr, ufeat, vfeat, out);
    });
  }
}

template void SDDMMCsr<kDLCPU, int32_t, float>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLCPU, int64_t, float>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLCPU, int32_t, double>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLCPU, int64_t, double>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
  if (op == "dot") {
    cpu::SDDMMDotCoo<IdType, DType>(coo, ufeat, vfeat, out);
  } else {
    SWITCH_OP(op, Op, {
      cpu::SDDMMCoo<IdType, DType, Op>(coo, ufeat, vfeat, out);
    });
  }
}

template void SDDMMCoo<kDLCPU, int32_t, float>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLCPU, int64_t, float>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLCPU, int32_t, double>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLCPU, int64_t, double>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template <int XPU, typename IdType, typename DType>
void SDDMMBcastCsr(const std::string& op,
                   const BcastInfo& info,
                   const aten::CSRMatrix& csr,
                   NDArray ufeat,
                   NDArray vfeat,
                   NDArray out,
                   std::vector<NDArray> out_aux) {
  SWITCH_OP(op, Op, {
    cpu::SDDMMBcastCsr<IdType, DType, Op>(info, csr, ufeat, vfeat, out);
  });
}

template void SDDMMBcastCsr<kDLCPU, int32_t, float>(
    const std::string& op, const BcastInfo& info, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLCPU, int64_t, float>(
    const std::string& op, const BcastInfo& info, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLCPU, int32_t, double>(
    const std::string& op, const BcastInfo& info, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLCPU, int64_t, double>(
    const std::string& op, const BcastInfo& info, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template <int XPU, typename IdType, typename DType>
void SDDMMBcastCoo(const std::string& op,
                   const BcastInfo& info,
                   const aten::COOMatrix& coo,
                   NDArray ufeat,
                   NDArray vfeat,
                   NDArray out,
                   std::vector<NDArray> out_aux) {
  SWITCH_OP(op, Op, {
    cpu::SDDMMBcastCoo<IdType, DType, Op>(info, coo, ufeat, vfeat, out);
  });
}

template void SDDMMBcastCoo<kDLCPU, int32_t, float>(
    const std::string& op, const BcastInfo& info, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLCPU, int64_t, float>(
    const std::string& op, const BcastInfo& info, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLCPU, int32_t, double>(
    const std::string& op, const BcastInfo& info, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLCPU, int64_t, double>(
    const std::string& op, const BcastInfo& info, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

}  // namespace kernel
}  // namespace dgl
