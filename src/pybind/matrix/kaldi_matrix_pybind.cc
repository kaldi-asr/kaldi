// pybind/matrix/kaldi_matrix_pybind.cc

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

#include "matrix/kaldi_matrix_pybind.h"

#include "dlpack/dlpack_pybind.h"
#include "matrix/kaldi-matrix.h"

using namespace kaldi;

void pybind_kaldi_matrix(py::module& m) {
  py::class_<MatrixBase<float>,
             std::unique_ptr<MatrixBase<float>, py::nodelete>>(
      m, "FloatMatrixBase",
      "Base class which provides matrix operations not involving resizing\n"
      "or allocation.   Classes Matrix and SubMatrix inherit from it and take "
      "care of allocation and resizing.")
      .def("NumRows", &MatrixBase<float>::NumRows, "Return number of rows")
      .def("NumCols", &MatrixBase<float>::NumCols, "Return number of columns")
      .def("Stride", &MatrixBase<float>::Stride, "Return stride")
      .def("__repr__",
           [](const MatrixBase<float>& b) -> std::string {
             std::ostringstream str;
             b.Write(str, false);
             return str.str();
           })
      .def("__getitem__",
           [](const MatrixBase<float>& m, std::pair<ssize_t, ssize_t> i) {
             return m(i.first, i.second);
           })
      .def("__setitem__",
           [](MatrixBase<float>& m, std::pair<ssize_t, ssize_t> i, float v) {
             m(i.first, i.second) = v;
           })
      .def("numpy", [](py::object obj) {
        auto* m = obj.cast<MatrixBase<float>*>();
        return py::array_t<float>(
            {m->NumRows(), m->NumCols()},                  // shape
            {sizeof(float) * m->Stride(), sizeof(float)},  // stride in bytes
            m->Data(),                                     // ptr
            obj);  // it will increase the reference count of **this** matrix
      });

  py::class_<Matrix<float>, MatrixBase<float>>(m, "FloatMatrix",
                                               pybind11::buffer_protocol())
      .def_buffer([](const Matrix<float>& m) -> pybind11::buffer_info {
        return pybind11::buffer_info(
            (void*)m.Data(),  // pointer to buffer
            sizeof(float),    // size of one scalar
            pybind11::format_descriptor<float>::format(),
            2,                           // num-axes
            {m.NumRows(), m.NumCols()},  // buffer dimensions
            {sizeof(float) * m.Stride(),
             sizeof(float)});  // stride for each index (in chars)
      })
      .def(py::init<>())
      .def(py::init<const MatrixIndexT, const MatrixIndexT, MatrixResizeType,
                    MatrixStrideType>(),
           py::arg("row"), py::arg("col"), py::arg("resize_type") = kSetZero,
           py::arg("stride_type") = kDefaultStride)
      .def(py::init<const MatrixBase<float>&, MatrixTransposeType>(),
           py::arg("M"), py::arg("trans") = kNoTrans)
      .def("Read", &Matrix<float>::Read, "allows resizing", py::arg("is"),
           py::arg("binary"), py::arg("add") = false)
      .def("to_dlpack", [](py::object obj) { return MatrixToDLPack(&obj); });

  py::class_<SubMatrix<float>, MatrixBase<float>>(m, "FloatSubMatrix")
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<float>::format()) {
          KALDI_ERR << "Expected format: "
                    << py::format_descriptor<float>::format() << "\n"
                    << "Current format: " << info.format;
        }
        if (info.ndim != 2) {
          KALDI_ERR << "Expected dim: 2\n"
                    << "Current dim: " << info.ndim;
        }

        // numpy is row major by default, so we use strides[0]
        return new SubMatrix<float>(reinterpret_cast<float*>(info.ptr),
                                    info.shape[0], info.shape[1],
                                    info.strides[0] / sizeof(float));
      }));

  py::class_<DLPackSubMatrix<float>, SubMatrix<float>>(m,
                                                       "DLPackFloatSubMatrix")
      .def("from_dlpack",
           [](py::capsule* capsule) { return SubMatrixFromDLPack(capsule); },
           py::return_value_policy::take_ownership);

  py::class_<Matrix<double>, std::unique_ptr<Matrix<double>, py::nodelete>>(
      m, "DoubleMatrix",
      "This bind is only for internal use, e.g. by OnlineCmvnState.")
      .def(py::init<const Matrix<float>&>(), py::arg("src"));
}
