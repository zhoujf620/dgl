/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/kernel2.cc
 * \brief New kernels
 */
#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace kernel {
namespace {

// Check whether the given arguments have the same context.
inline void CheckCtx(
    const DLContext& ctx,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (aten::IsNullArray(arrays[i]))
      continue;
    CHECK_EQ(ctx, arrays[i]->ctx)
      << "Expected device context " << ctx << ". But got "
      << arrays[i]->ctx << " for " << names[i] << ".";
  }
}

}  // namespace


DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelUOpESum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    CheckCtx(graph->Context(), {X, Y, Z}, {"U_data", "E_data", "Out"});
  });

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelCopyUSum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {X, Z}, {"U_data", "Out"});
  });

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelCopyESum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray Y = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {Y, Z}, {"E_data", "Out"});
  });

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelUOpEMax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    NDArray argX = args[5];
    NDArray argY = args[6];
    CheckCtx(graph->Context(), {X, Y, Z, argX, argY},
        {"U_data", "E_data", "Out", "U_index", "E_index"});
  });

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelUOpEMin")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    NDArray argX = args[5];
    NDArray argY = args[6];
    CheckCtx(graph->Context(), {X, Y, Z, argX, argY},
        {"U_data", "E_data", "Out", "U_index", "E_index"});
  });

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelUOpV")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    CheckCtx(graph->Context(), {X, Y, Z}, {"U_data", "V_data", "Out"});
  });

}  // namespace kernel
}  // namespace dgl
