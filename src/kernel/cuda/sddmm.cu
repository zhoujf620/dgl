//#include "./sddmm.cuh"
#include <dgl/array.h>
#include "../binary_reduce.h"

namespace dgl {
namespace kernel {

template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray efeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray efeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template <int XPU, typename IdType, typename DType>
void SDDMMBcastCsr(const std::string& op,
                   const BcastInfo& info,
                   const aten::CSRMatrix& csr,
                   NDArray ufeat,
                   NDArray efeat,
                   NDArray out,
                   std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template <int XPU, typename IdType, typename DType>
void SDDMMBcastCoo(const std::string& op,
                   const BcastInfo& info,
                   const aten::COOMatrix& coo,
                   NDArray ufeat,
                   NDArray efeat,
                   NDArray out,
                   std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template void SDDMMCsr<kDLGPU, int32_t, float>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLGPU, int64_t, float>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLGPU, int32_t, double>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLGPU, int64_t, double>(
    const std::string& op, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void SDDMMCoo<kDLGPU, int32_t, float>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLGPU, int64_t, float>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLGPU, int32_t, double>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLGPU, int64_t, double>(
    const std::string& op, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void SDDMMBcastCsr<kDLGPU, int32_t, float>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLGPU, int64_t, float>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLGPU, int32_t, double>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCsr<kDLGPU, int64_t, double>(
    const std::string& op, const BcastInfo&, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void SDDMMBcastCoo<kDLGPU, int32_t, float>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLGPU, int64_t, float>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLGPU, int32_t, double>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMBcastCoo<kDLGPU, int64_t, double>(
    const std::string& op, const BcastInfo&, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

}  // namespace kernel
}  // namespace dgl
