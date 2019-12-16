// pybind/matrix/matrix_common_pybind.cc

// Copyright 2019   Daniel Povey
//           2019   Dongji Gao
//           2019   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "matrix/matrix_common_pybind.h"

#include "matrix/matrix-common.h"

using namespace kaldi;

void pybind_matrix_common(py::module& m) {
  py::enum_<MatrixResizeType>(m, "MatrixResizeType", py::arithmetic(),
                              "Matrix initialization policies")
      .value("kSetZero", kSetZero, "Set to zero")
      .value("kUndefined", kUndefined, "Leave undefined")
      .value("kCopyData", kCopyData, "Copy any previously existing data")
      .export_values();

  py::enum_<MatrixStrideType>(m, "MatrixStrideType", py::arithmetic(),
                              "Matrix stride policies")
      .value("kDefaultStride", kDefaultStride,
             "Set to a multiple of 16 in bytes")
      .value("kStrideEqualNumCols", kStrideEqualNumCols,
             "Set to the number of columns")
      .export_values();
}
