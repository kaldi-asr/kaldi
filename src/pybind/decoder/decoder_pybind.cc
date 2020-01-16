// pybind/decoder/decoder_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China
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

#include "decoder/decoder_pybind.h"

#include "decoder/decodable_matrix_pybind.h"
#include "decoder/decoder_wrappers_pybind.h"
#include "decoder/lattice_faster_decoder_pybind.h"

void pybind_decoder(py::module& m) {
  pybind_lattice_faster_decoder(m);
  pybind_decodable_matrix(m);
  pybind_decoder_wrappers(m);
}
