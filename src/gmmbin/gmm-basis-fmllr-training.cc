// gmmbin/gmm-basis-fmllr-training.cc

// Copyright 2012  Carnegie Mellon University (author: Yajie Miao)

// See ../../COPYING for clarification regarding multiple authors
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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/fmllr-diag-gmm.h"
#include "transform/basis-fmllr-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate fMLLR basis representation. Reads a set of gradient scatter\n"
        "accumulations. Outputs basis matrices.\n"
        "Usage: gmm-basis-fmllr-training [options] <model-in> <basis-wspecifier>"
         "<accs-in1> <accs-in2> ...\n";

    bool binary_write = true;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    string
        model_rxfilename = po.GetArg(1),
        basis_wspecifier = po.GetArg(2);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    BasisFmllrAccus basis_accs(am_gmm.Dim());
    int num_accs = po.NumArgs() - 2;

    for (int i = 3, max = po.NumArgs(); i <= max; ++i) {
      std::string accs_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(accs_in_filename, &binary_read);
      basis_accs.Read(ki.Stream(), binary_read, true /* add read values*/);
    }

    // Estimate the basis matrices
    BasisFmllrEstimate basis_est(am_gmm.Dim());
    basis_est.EstimateFmllrBasis(am_gmm, basis_accs);
    {
      Output ko(basis_wspecifier, binary_write);
      basis_est.WriteBasis(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Summed " << num_accs << " gradient scatter stats";
    KALDI_LOG << "Generate " << basis_est.basis_size_ << " bases, written to "
    		  << basis_wspecifier;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

