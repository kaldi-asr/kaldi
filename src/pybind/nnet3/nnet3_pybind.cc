// pybind/nnet3/nnet3_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)
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

#include "nnet3/nnet3_pybind.h"

#include "nnet3/nnet_chain_example_pybind.h"
#include "nnet3/nnet_common_pybind.h"
#include "nnet3/nnet_component_itf_pybind.h"
#include "nnet3/nnet_convolutional_component_pybind.h"
#include "nnet3/nnet_example_pybind.h"
#include "nnet3/nnet_nnet_pybind.h"
#include "nnet3/nnet_normalize_component_pybind.h"
#include "nnet3/nnet_simple_component_pybind.h"

void pybind_nnet3(py::module& _m) {
  py::module m = _m.def_submodule("nnet3", "nnet3 pybind for Kaldi");

  pybind_nnet_common(m);
  pybind_nnet_component_itf(m);
  pybind_nnet_convolutional_component(m);
  pybind_nnet_example(m);
  pybind_nnet_chain_example(m);
  pybind_nnet_nnet(m);
  pybind_nnet_normalize_component(m);
  pybind_nnet_simple_component(m);
}
