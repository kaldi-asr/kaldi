// pybind/matrix/kaldi_vector_pybind.cc

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

#include "matrix/kaldi_vector_pybind.h"

#include "dlpack/dlpack_pybind.h"
#include "matrix/kaldi-vector.h"

using namespace kaldi;

void pybind_kaldi_vector(py::module& m) {
  py::class_<VectorBase<float>,
             std::unique_ptr<VectorBase<float>, py::nodelete>>(
      m, "FloatVectorBase",
      "Provides a vector abstraction class.\n"
      "This class provides a way to work with vectors in kaldi.\n"
      "It encapsulates basic operations and memory optimizations.")
      .def("Dim", &VectorBase<float>::Dim,
           "Returns the dimension of the vector.")
      .def("__repr__",
           [](const VectorBase<float>& v) -> std::string {
             std::ostringstream str;
             v.Write(str, false);
             return str.str();
           })
      .def("__getitem__",
           [](const VectorBase<float>& v, int i) { return v(i); })
      .def("__setitem__",
           [](VectorBase<float>& v, int i, float val) { v(i) = val; })
      .def("numpy", [](py::object obj) {
        auto* v = obj.cast<VectorBase<float>*>();
        return py::array_t<float>(
            {v->Dim()},       // shape
            {sizeof(float)},  // stride in bytes
            v->Data(),        // ptr
            obj);  // it will increase the reference count of **this** vector
      });

  py::class_<Vector<float>, VectorBase<float>>(m, "FloatVector",
                                               py::buffer_protocol())
      .def_buffer([](const Vector<float>& v) -> py::buffer_info {
        return py::buffer_info((void*)v.Data(), sizeof(float),
                               py::format_descriptor<float>::format(),
                               1,  // num-axes
                               {v.Dim()},
                               {sizeof(float)});  // strides (in chars)
      })
      .def(py::init<const MatrixIndexT, MatrixResizeType>(), py::arg("size"),
           py::arg("resize_type") = kSetZero)
      .def("to_dlpack", [](py::object obj) { return VectorToDLPack(obj); });

  py::class_<SubVector<float>, VectorBase<float>>(m, "FloatSubVector")
      .def(py::init([](py::buffer b) {
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
        return new SubVector<float>(reinterpret_cast<float*>(info.ptr),
                                    info.shape[0]);
      }));

  py::class_<DLPackSubVector<float>, SubVector<float>>(m,
                                                       "DLPackFloatSubVector")
      .def("from_dlpack",
           [](py::capsule* capsule) { return SubVectorFromDLPack(capsule); },
           py::return_value_policy::take_ownership);
}
