// bin/loglike-to-post.cc

// Copyright 2015  Vimal Manohar (Johns Hopkins University)

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"

/* Convert a matrix of log-likelihoods to posteriors */

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert a matrix of log-likelihoods (e.g. from gmm-compute-loglikes) to posteriors\n"
        "Usage:  loglikes-to-post [options] <loglikes-matrix-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " gmm-compute-loglikes [args] | loglike-to-post ark:- ark:1.post\n";
    
    ParseOptions po(usage);
    
    BaseFloat min_post = 0.01;
    bool random_prune = true; // preserve expectations.

    po.Register("min-post", &min_post, "Minimum posterior we will output (smaller "
                "ones are pruned).  Also see --random-prune");
    po.Register("random-prune", &random_prune, "If true, prune posteriors with a "
                "randomized method that preserves expectations.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string loglikes_rspecifier = po.GetArg(1);
    std::string posteriors_wspecifier = po.GetArg(2);

    int32 num_done = 0;
    SequentialBaseFloatMatrixReader loglikes_reader(loglikes_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !loglikes_reader.Done(); loglikes_reader.Next()) {
      num_done++;
      const Matrix<BaseFloat> &loglikes = loglikes_reader.Value();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post(loglikes.NumRows());
      for (int32 i = 0; i < loglikes.NumRows(); i++) {
        Vector<BaseFloat> row(SubVector<BaseFloat>(loglikes, i));
        row.ApplySoftMax();
        for (int32 j = 0; j < row.Dim(); j++) {
          BaseFloat p = row(j);
          if (p >= min_post) {
            post[i].push_back(std::make_pair(j, p));
          } else if (random_prune && (p / min_post) >= RandUniform()) {
            post[i].push_back(std::make_pair(j, min_post));
          }
        }
      }
      posterior_writer.Write(loglikes_reader.Key(), post);
    }
    KALDI_LOG << "Converted " << num_done << " log-likes matrices to posteriors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
