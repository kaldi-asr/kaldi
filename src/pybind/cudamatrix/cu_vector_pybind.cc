// pybind/cudamatrix/cu_vector_pybind.cc

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

#include "cudamatrix/cu_vector_pybind.h"

#include "cudamatrix/cu-vector.h"
#include "dlpack/dlpack_pybind.h"

using namespace kaldi;

void pybind_cu_vector(py::module& m) {
  {
    using PyClass = CuVectorBase<float>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "FloatCuVectorBase", "Vector for CUDA computing")
        .def("Dim", &PyClass::Dim, "Dimensions")
        .def("SetZero", &PyClass::SetZero)
        .def("Set", &PyClass::Set, py::arg("value"))
        .def("Add", &PyClass::Add, py::arg("value"))
        .def("Scale", &PyClass::Scale, py::arg("value"))
        .def("__getitem__", [](const PyClass& v, int i) { return v(i); });
  }
  {
    using PyClass = CuVector<float>;
    py::class_<PyClass, CuVectorBase<float>>(m, "FloatCuVector")
        .def(py::init<>())
        .def(py::init<MatrixIndexT, MatrixResizeType>(), py::arg("dim"),
             py::arg("MatrixResizeType") = kSetZero)
        .def(py::init<const VectorBase<float>&>(), py::arg("v"))
        .def("to_dlpack",
             [](py::object obj) { return CuVectorToDLPack(&obj); });
  }
  {
    using PyClass = CuSubVector<float>;
    py::class_<PyClass, CuVectorBase<float>>(m, "FloatCuSubVector");
  }
  py::class_<DLPackCuSubVector<float>, CuSubVector<float>>(
      m, "DLPackFloatCuSubVector")
      .def("from_dlpack",
           [](py::capsule* capsule) { return CuSubVectorFromDLPack(capsule); },
           py::return_value_policy::take_ownership);
}
