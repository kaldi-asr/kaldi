// pybind/kaldi_pybind.cc

// Copyright 2019   Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "matrix/kaldi-matrix.h"

using namespace kaldi;

PYBIND11_MODULE(kaldi_pybind, m) {
  m.doc() = "pybind11 binding of some things from kaldi's src/matrix directory. "
      "Source is in $(KALDI_ROOT)/src/pybind/matrix-lib.cc";

  py::enum_<MatrixResizeType>(m, "MatrixResizeType", py::arithmetic(), "Matrix initialization policies")
      .value("kSetZero", kSetZero, "Set to zero")
      .value("kUndefined", kUndefined, "Leave undefined")
      .value("kCopyData", kCopyData, "Copy any previously existing data")
      .export_values();


  py::class_<Vector<float> >(m, "FloatVector", pybind11::buffer_protocol())
      .def_buffer([](const Vector<float> &v) -> pybind11::buffer_info {
    return pybind11::buffer_info(
        (void*)v.Data(),
        sizeof(float),
        pybind11::format_descriptor<float>::format(),
        1, // num-axes
        { v.Dim() },
        { 4 }); // strides (in chars)
        })
      .def("Dim", &Vector<float>::Dim, "Return the dimension of the vector")
      .def("__repr__",
           [] (const Vector<float> &a) -> std::string {
             std::ostringstream str;  a.Write(str, false); return str.str();
           })
      .def(py::init<const MatrixIndexT, MatrixResizeType>(),
           py::arg("size"), py::arg("resize_type") = kSetZero);

}







