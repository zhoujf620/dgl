//#include "./spmm.cuh"
#include <dgl/array.h>

namespace dgl {
namespace kernel {

template <int XPU, typename IdType, typename DType>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template <int XPU, typename IdType, typename DType>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const aten::COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
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

}  // namespace kernel
}  // namespace dgl
