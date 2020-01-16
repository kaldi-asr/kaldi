// pybind/util/util_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

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

#include "util/util_pybind.h"

#include "util/kaldi_holder_pybind.h"
#include "util/kaldi_io_pybind.h"
#include "util/parse_options_pybind.h"
#include "util/table_types_pybind.h"

void pybind_util(py::module& m) {
  pybind_table_types(m);
  pybind_kaldi_io(m);
  pybind_kaldi_holder(m);
  pybind_parse_options(m);
}
