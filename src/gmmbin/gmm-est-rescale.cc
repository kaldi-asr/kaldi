// gmmbin/gmm-est-rescale.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "gmm/indirect-diff-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Do \"re-scaling\" re-estimation of GMM-based model\n"
        " (this update changes the model as features change, but preserves\n"
        "  the difference between the model and the features, to keep\n"
        "  the effect of any prior discriminative training).  Used in fMPE.\n"
        "  Does not update the transitions or weights.\n"
        "Usage: gmm-est-rescale [options] <model-in> <old-stats-in> <new-stats-in> <model-out>\n"
        "e.g.: gmm-est-rescale 1.mdl old.acc new.acc 2.mdl\n";
    
    bool binary_write = true;
    MleDiagGmmOptions opts; // Not passed to command-line-- just a mechanism to
    // ensure our options have the same default values as those ones.
    BaseFloat min_variance = opts.min_variance;
    BaseFloat min_gaussian_occupancy = opts.min_gaussian_occupancy;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("min-variance", &min_variance,
                "Variance floor (absolute variance).");
    po.Register("min-gaussian-occupancy", &min_gaussian_occupancy,
                "Minimum occupancy to update a Gaussian.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        old_stats_rxfilename = po.GetArg(2),
        new_stats_rxfilename = po.GetArg(3),
        model_wxfilename = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    AccumAmDiagGmm old_gmm_accs, new_gmm_accs;
    {
      Vector<double> transition_accs;
      bool binary;
      Input ki(old_stats_rxfilename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      old_gmm_accs.Read(ki.Stream(), binary, true);
    }
    {
      Vector<double> transition_accs;
      bool binary;
      Input ki(new_stats_rxfilename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      new_gmm_accs.Read(ki.Stream(), binary, true);
    }

    DoRescalingUpdate(old_gmm_accs, new_gmm_accs,
                      min_variance, min_gaussian_occupancy,
                      &am_gmm);
    
    {
      Output ko(model_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Rescaled model and wrote to " << model_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


