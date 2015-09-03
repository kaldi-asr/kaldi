// sgmmbin/init-ubm.cc

// Copyright 2009-2011   Saarland University
// Author:  Arnab Ghoshal

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/kaldi-io.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    typedef kaldi::BaseFloat BaseFloat;

    const char *usage =
        "Cluster the Gaussians in a diagonal-GMM acoustic model\n"
        "to a single full-covariance or diagonal-covariance GMM.\n"
        "Usage: init-ubm [options] <model-file> <state-occs> <gmm-out>\n";

    bool binary_write = true, fullcov_ubm = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("fullcov-ubm", &fullcov_ubm, "Write out full covariance UBM.");
    kaldi::UbmClusteringOptions ubm_opts;
    ubm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    ubm_opts.Check();
    
    std::string model_in_filename = po.GetArg(1),
        occs_in_filename = po.GetArg(2),
        gmm_out_filename = po.GetArg(3);

    kaldi::AmDiagGmm am_gmm;
    kaldi::TransitionModel trans_model;
    {
      bool binary_read;
      kaldi::Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    kaldi::Vector<BaseFloat> state_occs;
    state_occs.Resize(am_gmm.NumPdfs());
    {
      bool binary_read;
      kaldi::Input ki(occs_in_filename, &binary_read);
      state_occs.Read(ki.Stream(), binary_read);
    }

    kaldi::DiagGmm ubm;
    ClusterGaussiansToUbm(am_gmm, state_occs, ubm_opts, &ubm);
    if (fullcov_ubm) {
      kaldi::FullGmm full_ubm;
      full_ubm.CopyFromDiagGmm(ubm);
      kaldi::Output ko(gmm_out_filename, binary_write);
      full_ubm.Write(ko.Stream(), binary_write);
    } else {
      kaldi::Output ko(gmm_out_filename, binary_write);
      ubm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written UBM to " << gmm_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


