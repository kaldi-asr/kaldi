// pybind/nnet3/nnet_simple_component_pybind.cc

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

#include "nnet3/nnet_simple_component_pybind.h"

#include "nnet3/nnet-simple-component.h"

using namespace kaldi::nnet3;

void pybind_nnet_simple_component(py::module& m) {
  using FAC = FixedAffineComponent;
  py::class_<FAC, Component>(m, "FixedAffineComponent")
      .def("LinearParams", &FAC::LinearParams,
           py::return_value_policy::reference)
      .def("BiasParams", &FAC::BiasParams, py::return_value_policy::reference);

  using LC = LinearComponent;
  py::class_<LC, Component>(m, "LinearComponent")
      .def("Params", overload_cast_<>()(&LC::Params2, py::const_),
           py::return_value_policy::reference);

  using AC = AffineComponent;
  py::class_<AC, Component>(m, "AffineComponent")
      .def("LinearParams", overload_cast_<>()(&AC::LinearParams, py::const_),
           py::return_value_policy::reference)
      .def("BiasParams", overload_cast_<>()(&AC::BiasParams, py::const_),
           py::return_value_policy::reference);

  using NGAC = NaturalGradientAffineComponent;
  py::class_<NGAC, AC>(m, "NaturalGradientAffineComponent");
}
