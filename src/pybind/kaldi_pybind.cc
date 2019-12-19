// pybind/kaldi_pybind.cc

// Copyright 2019   Daniel Povey
//           2019   Dongji Gao
//           2019   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

// See ../../COPYING for clarification regarding multiple authors
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

#include "pybind/kaldi_pybind.h"

#include <string>

#include "matrix/matrix_common_pybind.h"
#include "matrix/matrix_pybind.h"
#include "matrix/vector_pybind.h"
#include "util/table_types_pybind.h"
#include "feat/wave_reader_pybind.h"

void pybind_matrix(py::module& m);
PYBIND11_MODULE(kaldi_pybind, m) {
  m.doc() =
      "pybind11 binding of some things from kaldi's "
      "src/matrix and src/util directories. "
      "Source is in $(KALDI_ROOT)/src/pybind/kaldi_pybind.cc";

  pybind_matrix_common(m);
  pybind_matrix(m);
  pybind_vector(m);
  pybind_table_types(m);
  pybind_wave_reader(m);
}
