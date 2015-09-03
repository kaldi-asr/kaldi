// gmmbin/gmm-get-stats-deriv.cc

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
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "gmm/indirect-diff-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    MleDiagGmmOptions gmm_opts;
    
    const char *usage =
        "Get statistics derivative for GMM models\n"
        "(used in fMPE/fMMI feature-space discriminative training)\n"
        "Usage:  gmm-get-stats-deriv [options] <model-in> <num-stats-in>"
        " <den-stats-in> <ml-stats-in> <deriv-out>\n"
        "e.g. (for fMMI/fBMMI): gmm-get-stats-deriv 1.mdl 1.acc 2.mdl\n";
    
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

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        num_stats_rxfilename = po.GetArg(2),
        den_stats_rxfilename = po.GetArg(3),
        ml_stats_rxfilename = po.GetArg(4),
        deriv_wxfilename = po.GetArg(5);
        
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    Vector<double> transition_accs; // Reuse this for all transition accs we
    // read, as it's not needed.
    AccumAmDiagGmm num_stats, den_stats, ml_stats;
    {
      bool binary_read;
      Input ki(num_stats_rxfilename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read);
      num_stats.Read(ki.Stream(), binary_read, false);
    }
    {
      bool binary_read;
      Input ki(den_stats_rxfilename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read);
      den_stats.Read(ki.Stream(), binary_read, false);
    }
    {
      bool binary_read;
      Input ki(ml_stats_rxfilename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read);
      ml_stats.Read(ki.Stream(), binary_read, false);
    }

    AccumAmDiagGmm model_deriv; // Use GMM accumulators to represent
    // derivative of discriminative objective function w.r.t.
    // accumulated stats.
        
    GetStatsDerivative(am_gmm, num_stats, den_stats, ml_stats,
                       min_variance, min_gaussian_occupancy,
                       &model_deriv);

    WriteKaldiObject(model_deriv, deriv_wxfilename, binary_write);
    
    KALDI_LOG << "Computed model derivative and wrote it to "
              << deriv_wxfilename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


