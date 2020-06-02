/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/spmat.h
 * \brief Global control of allowed sparse format.
 */

#ifndef DGL_SPFMT_H_
#define DGL_SPFMT_H_

#include <dgl/array.h>
#include <dmlc/thread_local.h>

namespace dgl{

class GlobalSparseFormat{
 public:
  GlobalSparseFormat(): fmt(SparseFormat::kAny) {}
  static GlobalSparseFormat *Get() {
    static GlobalSparseFormat fmt;
    return &fmt;
  }
  SparseFormat GetFormat() const {
    return fmt;
  }
  void SetFormat(const std::string& fmt_str) {
    this->fmt = ParseSparseFormat(fmt_str);
  }
 private:
  SparseFormat fmt;
};

} // namespace dgl

#endif