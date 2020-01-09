// pybind/cudamatrix/cu_matrix_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

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

#include "cudamatrix/cu_matrix_pybind.h"

#include "cudamatrix/cu-matrix.h"
#include "dlpack/dlpack_pybind.h"

using namespace kaldi;

void pybind_cu_matrix(py::module& m) {
  {
    using PyClass = CuMatrixBase<float>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "FloatCuMatrixBase", "Matrix for CUDA computing")
        .def("NumRows", &PyClass::NumRows, "Return number of rows")
        .def("NumCols", &PyClass::NumCols, "Return number of columns")
        .def("Stride", &PyClass::Stride, "Return stride")
        .def("ApplyExp", &PyClass::ApplyExp)
        .def("SetZero", &PyClass::SetZero)
        .def("Set", &PyClass::Set, py::arg("value"))
        .def("Add", &PyClass::Add, py::arg("value"))
        .def("Scale", &PyClass::Scale, py::arg("value"))
        .def("__getitem__",
             [](const PyClass& m, std::pair<ssize_t, ssize_t> i) {
               return m(i.first, i.second);
             });
  }

  {
    using PyClass = CuMatrix<float>;
    py::class_<PyClass, CuMatrixBase<float>>(m, "FloatCuMatrix")
        .def(py::init<>())
        .def(py::init<MatrixIndexT, MatrixIndexT, MatrixResizeType,
                      MatrixStrideType>(),
             py::arg("rows"), py::arg("cols"),
             py::arg("resize_type") = kSetZero,
             py::arg("MatrixStrideType") = kDefaultStride)
        .def(py::init<const MatrixBase<float>&, MatrixTransposeType>(),
             py::arg("other"), py::arg("trans") = kNoTrans)
        .def("to_dlpack",
             [](py::object obj) { return CuMatrixToDLPack(&obj); });
  }
  {
    using PyClass = CuSubMatrix<float>;
    py::class_<PyClass, CuMatrixBase<float>>(m, "FloatCuSubMatrix");
  }

  py::class_<DLPackCuSubMatrix<float>, CuSubMatrix<float>>(
      m, "DLPackFloatCuSubMatrix")
      .def("from_dlpack",
           [](py::capsule* capsule) { return CuSubMatrixFromDLPack(capsule); },
           py::return_value_policy::take_ownership);
}
