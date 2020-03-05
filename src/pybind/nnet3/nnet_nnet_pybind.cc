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

#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-convolutional-component.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-normalize-component.h"
#include "nnet3/nnet-simple-component.h"

using namespace kaldi;
using namespace kaldi::nnet3;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

void pybind_nnet_nnet(py::module& m) {
  using Comp = kaldi::nnet3::Component;
  py::class_<Comp>(m, "Component",
                   "Abstract base-class for neural-net components.")
      .def("Type", &Comp::Type,
           "Returns a string such as \"SigmoidComponent\", describing the "
           "type of the object.")
      .def("Info", &Comp::Info,
           "Returns some text-form information about this component, for "
           "diagnostics. Starts with the type of the component.  E.g. "
           "\"SigmoidComponent dim=900\", although most components will have "
           "much more info.")
      .def_static("NewComponentOfType", &Comp::NewComponentOfType,
                  py::return_value_policy::take_ownership);

  using BNC = kaldi::nnet3::BatchNormComponent;
  py::class_<BNC>(m, "BatchNormComponent")
      .def("Type", &BNC::Type)
      .def("Offset", &BNC::Offset)
      .def("Scale", overload_cast_<>()(&BNC::Scale, py::const_));

  using FAC = kaldi::nnet3::FixedAffineComponent;
  py::class_<FAC>(m, "FixedAffineComponent")
      .def("Type", &FAC::Type)
      .def("LinearParams", &FAC::LinearParams)
      .def("BiasParams", &FAC::BiasParams);

  using LC = kaldi::nnet3::LinearComponent;
  py::class_<LC>(m, "LinearComponent")
      .def("Type", &LC::Type)
      .def("Params", overload_cast_<>()(&LC::Params, py::const_));

  using NGAC = kaldi::nnet3::NaturalGradientAffineComponent;
  py::class_<NGAC>(m, "NaturalGradientAffineComponent")
      .def("Type", &NGAC::Type)
      .def("LinearParams", overload_cast_<>()(&NGAC::LinearParams, py::const_))
      .def("BiasParams", overload_cast_<>()(&NGAC::BiasParams, py::const_));

  using AC = kaldi::nnet3::AffineComponent;
  py::class_<AC>(m, "AffineComponent")
      .def("Type", &AC::Type)
      .def("LinearParams", overload_cast_<>()(&AC::LinearParams, py::const_))
      .def("BiasParams", overload_cast_<>()(&AC::BiasParams, py::const_));

  using TC = kaldi::nnet3::TdnnComponent;
  py::class_<TC>(m, "TdnnComponent")
      .def("Type", &TC::Type)
      .def("LinearParams", &TC::LinearParams)
      .def("BiasParams", &TC::BiasParams);

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
      .def("GetComponent", (Comp * (PyClass::*)(int32)) & PyClass::GetComponent,
           py::arg("c"), py::return_value_policy::reference);
}
