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

#include "chain/chain_pybind.h"
#include "ctc/ctc_pybind.h"
#include "cudamatrix/cudamatrix_pybind.h"
#include "decoder/decoder_pybind.h"
#include "dlpack/dlpack_pybind.h"
#include "feat/feat_pybind.h"
#include "feat/wave_reader_pybind.h"
#include "fst/fst_pybind.h"
#include "hmm/hmm_pybind.h"
#include "itf/itf_pybind.h"
#include "lat/lat_pybind.h"
#include "matrix/matrix_pybind.h"
#include "nnet3/nnet3_pybind.h"
#include "util/util_pybind.h"

PYBIND11_MODULE(kaldi_pybind, m) {
  m.doc() = "pybind11 binding of some things from kaldi";

  pybind_matrix(m);
  pybind_itf(m);
  pybind_util(m);
  pybind_feat(m);

  pybind_fst(m);
  pybind_lat(m);
  pybind_chain(m);
  pybind_nnet3(m);

  pybind_dlpack(m);

  pybind_cudamatrix(m);

  pybind_decoder(m);
  pybind_hmm(m);

  pybind_ctc(m);

  void test_dlpack(py::module & m);  // forward declaration
  test_dlpack(m);
}
