// sgmmbin/sgmm-init.cc

// Copyright 2009-2011   Saarland University
// Author:  Arnab Ghoshal

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

#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize an SGMM from a trained full-covariance UBM and a specified"
        " model topology.\n"
        "Usage: sgmm-init [options] <am-gmm-in> <ubm-in> <sgmm-out>\n";

    bool binary = false;
    int32 phn_space_dim = 0, spk_space_dim = 0;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("phn-space-dim", &phn_space_dim, "Phonetic space dimension.");
    po.Register("spk-space-dim", &spk_space_dim, "Speaker space dimension.");


    po.Read(argc, argv);
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        ubm_in_filename = po.GetArg(2),
        sgmm_out_filename = po.GetArg(3);

    kaldi::AmDiagGmm am_gmm;
    kaldi::TransitionModel trans_model;
    {
      bool binary_read;
      kaldi::Input is(model_in_filename, &binary_read);
      trans_model.Read(is.Stream(), binary_read);
      am_gmm.Read(is.Stream(), binary_read);
    }

    kaldi::FullGmm ubm;
    {
      bool binary_read;
      kaldi::Input is(ubm_in_filename, &binary_read);
      ubm.Read(is.Stream(), binary_read);
    }

    kaldi::AmSgmm sgmm;
    sgmm.InitializeFromFullGmm(ubm, am_gmm.NumPdfs(), phn_space_dim,
                               spk_space_dim);
    sgmm.ComputeNormalizers();

    {
      kaldi::Output os(sgmm_out_filename, binary);
      trans_model.Write(os.Stream(), binary);
      sgmm.Write(os.Stream(), binary, kaldi::kSgmmWriteAll);
    }

    KALDI_LOG << "Written model to " << sgmm_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


