// pybind/nnet3/nnet_component_itf_pybind.cc

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

#include "nnet3/nnet_component_itf_pybind.h"

#include "nnet3/nnet-component-itf.h"

using namespace kaldi::nnet3;

void pybind_nnet_component_itf(py::module& m) {
  using PyClass = Component;
  py::class_<PyClass>(m, "Component",
                   "Abstract base-class for neural-net components.")
      .def("Type", &PyClass::Type,
           "Returns a string such as \"SigmoidComponent\", describing the "
           "type of the object.")
      .def("Info", &PyClass::Info,
           "Returns some text-form information about this component, for "
           "diagnostics. Starts with the type of the component.  E.g. "
           "\"SigmoidComponent dim=900\", although most components will have "
           "much more info.")
      .def_static("NewComponentOfType", &PyClass::NewComponentOfType,
                  py::return_value_policy::take_ownership);
}
