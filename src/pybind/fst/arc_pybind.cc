// pybind/fst/arc_pybind.cc

// Copyright 2019-2020   Mobvoi AI Lab, Beijing, China
//                       (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

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

#include "fst/arc_pybind.h"

#include "fst/arc.h"

void pybind_arc(py::module& m) {
  // (fangjun): we have wrapped fst::TropicalWeight
  // in fst/weight_pybind.cc
  pybind_arc_impl<fst::TropicalWeight>(m, "StdArc");
}
