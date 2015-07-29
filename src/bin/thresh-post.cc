// bin/thresh-post.cc

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Down-weight posteriors that are lower than a supplied confidence threshold.\n"
        "(for those below the weight, rather than set to zero we downweight according\n"
        "to the --weight option)\n"
        "\n"
        "Usage:  thresh-post [options] <posteriors-rspecifier> <posteriors-wspecifier>\n"
        "e.g.: thresh-post --threshold=0.9 --scale=0.1 ark:- ark:-\n";

    ParseOptions po(usage);

    BaseFloat threshold = 0.9;
    BaseFloat scale = 0.1;

    po.Register("threshold", &threshold, "Threshold below which we down-weight posteriors.");
    po.Register("scale", &scale, "Scale which we apply to posteriors below the threshold.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string posteriors_rspecifier = po.GetArg(1),
        posteriors_wspecifier = po.GetArg(2);

    KALDI_ASSERT(threshold < 1.0 && threshold >= 0.0 && scale >= 0.0 && scale <= 1.0);
    
    int32 num_posteriors = 0;
    double total_weight_in = 0.0, total_weight_out = 0.0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      num_posteriors++;
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      const Posterior &posterior = posterior_reader.Value();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior new_post(posterior.size());
      for (size_t i = 0; i < posterior.size(); i++) {
        for (size_t j = 0; j < posterior[i].size(); j++) {
          int32 tid = posterior[i][j].first;
          double weight = posterior[i][j].second;
          total_weight_in += weight;
          if (weight < threshold) weight *= scale;
          total_weight_out += weight;
          if (weight != 0.0)
            new_post[i].push_back(std::make_pair(tid, static_cast<BaseFloat>(weight)));
        }
      }
      posterior_writer.Write(posterior_reader.Key(), new_post);
    }
    KALDI_LOG << "thresh-post: thresholded " << num_posteriors 
              << " posteriors, reduced them by a factor of "
              << (total_weight_out/total_weight_in) << " on average.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


