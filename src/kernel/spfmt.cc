/*!
 *  Copyright (c) 2020 by Contributors
 * \file spfmt.cc
 * \brief Allowed sparse format controller interface.
 */

#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dgl/spfmt.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

namespace dgl{

using runtime::DGLArgs;
using runtime::DGLRetValue;

DGL_REGISTER_GLOBAL("spfmt._CAPI_SetKernelFormat")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string format = args[0];
    LOG(INFO) << "Set global sparse format to " << format;
    GlobalSparseFormat::Get()->SetFormat(format);
  });

}