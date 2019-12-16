// pybind/matrix/vector_pybind.cc

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

#include "matrix/vector_pybind.h"

#include "matrix/kaldi-vector.h"

using namespace kaldi;

void pybind_vector(py::module& m) {
  py::class_<VectorBase<float>,
             std::unique_ptr<VectorBase<float>, py::nodelete>>(
      m, "FloatVectorBase",
      "Provides a vector abstraction class.\n"
      "This class provides a way to work with vectors in kaldi.\n"
      "It encapsulates basic operations and memory optimizations.")
      .def("Read", &VectorBase<float>::Read,
           "Reads from C++ stream (option to add to existing contents).\n"
           "Throws exception on failure",
           py::arg("in"), py::arg("binary"), py::arg("add") = false)
      .def("Write", &VectorBase<float>::Write,
           "Writes to C++ stream (option to write in binary).", py::arg("out"),
           py::arg("binary"))
      .def("Dim", &VectorBase<float>::Dim,
           "Returns the  dimension of the vector.")
      .def("__repr__",
           [](const VectorBase<float>& a) -> std::string {
             std::ostringstream str;
             a.Write(str, false);
             return str.str();
           })
      .def("__getitem__",
           [](const VectorBase<float>& m, int i) { return m(i); })
      .def("__setitem__",
           [](VectorBase<float>& m, int i, float v) { m(i) = v; });

  py::class_<Vector<float>, VectorBase<float>>(m, "FloatVector",
                                               pybind11::buffer_protocol())
      .def_buffer([](const Vector<float>& v) -> pybind11::buffer_info {
        return pybind11::buffer_info(
            (void*)v.Data(), sizeof(float),
            pybind11::format_descriptor<float>::format(),
            1,                // num-axes
            {v.Dim()}, {4});  // strides (in chars)
      })
      .def(py::init<const MatrixIndexT, MatrixResizeType>(), py::arg("size"),
           py::arg("resize_type") = kSetZero);

  py::class_<SubVector<float>, VectorBase<float>>(m, "FloatSubVector")
      .def("__init__", [](SubVector<float>& m, py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<float>::format()) {
          KALDI_ERR << "Expected format: "
                    << py::format_descriptor<float>::format() << "\n"
                    << "Current format: " << info.format;
        }
        if (info.ndim != 1) {
          KALDI_ERR << "Expected dim: 1\n"
                    << "Current dim: " << info.ndim;
        }
        new (&m)
            SubVector<float>(reinterpret_cast<float*>(info.ptr), info.shape[0]);
      });
}
