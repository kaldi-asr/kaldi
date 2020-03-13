// pybind/nnet3/nnet_normalize_component_pybind.cc

// Copyright 2020   JD AI, Beijing, China (author: Lu Fan)

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

#include "nnet3/nnet_normalize_component_pybind.h"

#include "nnet3/nnet-normalize-component.h"

using namespace kaldi::nnet3;

void pybind_nnet_normalize_component(py::module& m) {
  using PyClass = kaldi::nnet3::BatchNormComponent;
  py::class_<PyClass, Component>(m, "BatchNormComponent")
      .def("Mean", &PyClass::Mean)
      .def("Var", &PyClass::Var)
      .def("Count", &PyClass::Count)
      .def("Eps", &PyClass::Eps)
      .def("SetTestMode", &PyClass::SetTestMode, py::arg("test_mode"))
      .def("Offset", &PyClass::Offset, py::return_value_policy::reference)
      .def("Scale", overload_cast_<>()(&PyClass::Scale, py::const_),
           py::return_value_policy::reference);
}
