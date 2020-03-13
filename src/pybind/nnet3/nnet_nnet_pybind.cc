// pybind/nnet3/nnet_nnet_pybind.cc

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

#include "nnet3/nnet_nnet_pybind.h"

#include "nnet3/nnet-nnet.h"

using namespace kaldi;
using namespace kaldi::nnet3;

void pybind_nnet_nnet(py::module& m) {
  using PyClass = kaldi::nnet3::Nnet;
  auto nnet = py::class_<PyClass>(
      m, "Nnet",
      "This function can be used either to initialize a new Nnet from a "
      "config file, or to add to an existing Nnet, possibly replacing "
      "certain parts of it.  It will die with error if something went wrong. "
      "Also see the function ReadEditConfig() in nnet-utils.h (it's made a "
      "non-member because it doesn't need special access).");
  nnet.def(py::init<>())
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"))
      .def("GetComponentNames", &PyClass::GetComponentNames,
           "returns vector of component names (needed by some parsing code, "
           "for instance).",
           py::return_value_policy::reference)
      .def("GetComponentName", &PyClass::GetComponentName,
           py::arg("component_index"))
      .def("Info", &PyClass::Info,
           "returns some human-readable information about the network, "
           "mostly for debugging purposes. Also see function NnetInfo() in "
           "nnet-utils.h, which prints out more extensive infoformation.")
      .def("NumComponents", &PyClass::NumComponents)
      .def("NumNodes", &PyClass::NumNodes)
      .def("GetComponent", (Component * (PyClass::*)(int32)) & PyClass::GetComponent,
           py::arg("c"), py::return_value_policy::reference);
}
