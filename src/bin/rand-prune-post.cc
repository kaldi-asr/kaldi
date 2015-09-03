// bin/rand-prune-post.cc

// Copyright 2009-2011  Microsoft Corporation

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



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Randomized pruning of posteriors less than threshold\n"
        "Note: for posteriors derived from alignments, threshold must be\n"
        "greater than one, or this will have no effect (speedup factor will\n"
        "be roughly the same as the threshold)\n"
        "Usage:  rand-prune-post [options] <rand-prune-value> <posteriors-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " rand-prune-post 5.0 ark:- ark:-\n";

    ParseOptions po(usage);
        
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rand_prune_str = po.GetArg(1),
        posteriors_rspecifier = po.GetArg(2),
        posteriors_wspecifier = po.GetArg(3);

    BaseFloat rand_prune = 0.0;
    if (!ConvertStringToReal(rand_prune_str, &rand_prune) || rand_prune < 0.0)
      KALDI_ERR << "Invalid rand_prune parameter: expected float, got \""
                 << rand_prune_str << '"';
    
    int32 num_posteriors = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      num_posteriors++;
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      const Posterior &posterior = posterior_reader.Value();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      if (rand_prune == 0.0) {
        posterior_writer.Write(posterior_reader.Key(), posterior);
      } else {
        Posterior new_post(posterior.size());
        for (size_t i = 0; i < posterior.size(); i++) {
          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first;
            BaseFloat weight = RandPrune(posterior[i][j].second, rand_prune);
            if (weight != 0.0)
              new_post[i].push_back(std::make_pair(tid, weight));
          }
        }
        posterior_writer.Write(posterior_reader.Key(), new_post);
      }
    }
    KALDI_LOG << "rand-prune-post: processed " << num_posteriors << " posteriors.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


