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
#include "decoder/lattice_faster_decoder_pybind.h"

void pybind_decoder(py::module& kaldi_m) {
  py::module decoder_m = kaldi_m.def_submodule("decoder", "pybind for decoder");

  pybind_lattice_faster_decoder(kaldi_m, decoder_m);
  pybind_decodable_matrix(kaldi_m);
}
