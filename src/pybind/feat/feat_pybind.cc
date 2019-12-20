// pybind/feat/feat_pybind.cc

// Copyright 2019   Microsoft Corporation (author: Xingyu Na)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/feat_pybind.h"

#include "feat/feature_pybind.h"
#include "feat/wave_reader_pybind.h"

void pybind_feat(py::module& _m) {
  py::module m = _m.def_submodule("feat", "feat pybind for Kaldi");

  pybind_wave_reader(m);
  pybind_feature(m);
}
