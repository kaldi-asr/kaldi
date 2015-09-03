// gmmbin/gmm-diff-accs.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#include "util/common-utils.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute difference between accumulators\n"
        "Usage: gmm-diff-accs [options] plus-stats minus-stats diff-stats-out\n";
        
    bool binary = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string plus_stats_rxfilename = po.GetArg(1),
        minus_stats_rxfilename = po.GetArg(2),
        diff_stats_wxfilename = po.GetArg(3);

    kaldi::Vector<double> transition_accs_plus;
    kaldi::AccumAmDiagGmm gmm_accs_plus;
    {
      bool binary_read;
      kaldi::Input ki(plus_stats_rxfilename, &binary_read);
      transition_accs_plus.Read(ki.Stream(), binary_read, false);
      gmm_accs_plus.Read(ki.Stream(), binary_read, false);
    }
    kaldi::Vector<double> transition_accs_minus;
    kaldi::AccumAmDiagGmm gmm_accs_minus;
    {
      bool binary_read;
      kaldi::Input ki(minus_stats_rxfilename, &binary_read);
      transition_accs_minus.Read(ki.Stream(), binary_read, false);
      gmm_accs_minus.Read(ki.Stream(), binary_read, false);
    }

    double tot_count_before = gmm_accs_plus.TotStatsCount();
    // subtract accs.
    {
      transition_accs_plus.AddVec(-1.0, transition_accs_minus);
      gmm_accs_plus.Add(-1.0, gmm_accs_minus);
    }
    double tot_count_after = gmm_accs_plus.TotStatsCount();
    
    // Write out the accs
    {
      kaldi::Output ko(diff_stats_wxfilename, binary);
      transition_accs_plus.Write(ko.Stream(), binary);
      gmm_accs_plus.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Differenced stats, count of first stats was "
              << tot_count_before << ", count of difference was "
              << tot_count_after;
    KALDI_LOG << "Wrote stats to " << diff_stats_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


