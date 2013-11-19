// bin/logprob-to-post.cc

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"

/* Convert a matrix of log-probabilities 
   to something of type Posterior, i.e. for each utterance, a
   vector<vector<pair<int32, BaseFloat> > >, which is a sparse representation
   of the probabilities.
   To avoid getting very tiny values making it non-sparse, we support
   thresholding, and this can either be done as a simple threshold, or (the
   default) a pseudo-random thing where you preserve the expectation, e.g.
   if the threshold is 0.01 and the value is 0.001, it will be zero with
   probability 0.9 and 0.01 with probability 0.1.
*/

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert a matrix of log-probabilities (e.g. from nnet-logprob) to posteriors\n"
        "Usage:  logprob-to-post [options] <logprob-matrix-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " nnet-logprob [args] | logprob-to-post ark:- ark:1.post\n"
        "Caution: in this particular example, the output would be posteriors of pdf-ids,\n"
        "rather than transition-ids (c.f. post-to-pdf-post)\n";
    
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

    std::string logprob_rspecifier = po.GetArg(1);
    std::string posteriors_wspecifier = po.GetArg(2);

    int32 num_done = 0;
    SequentialBaseFloatMatrixReader logprob_reader(logprob_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !logprob_reader.Done(); logprob_reader.Next()) {
      num_done++;
      const Matrix<BaseFloat> &logprobs = logprob_reader.Value();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post(logprobs.NumRows());
      for (int32 i = 0; i < logprobs.NumRows(); i++) {
        SubVector<BaseFloat> row(logprobs, i);
        for (int32 j = 0; j < row.Dim(); j++) {
          BaseFloat p = exp(row(j));
          if (p >= min_post) {
            post[i].push_back(std::make_pair(j, p));
          } else if (random_prune && (p / min_post) >= RandUniform()) {
            post[i].push_back(std::make_pair(j, min_post));
          }
        }
      }
      posterior_writer.Write(logprob_reader.Key(), post);
    }
    KALDI_LOG << "Converted " << num_done << " log-prob matrices to posteriors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


