// bin/scale-post.cc

// Copyright 2009-2011 Chao Weng 

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "scale the posterior according to num&den score for MCE\n"
        "Usage: score-post [option] numscore-rspecifier denscore-rspecifier\n"
        "unscaled-post-rspecifier scaled-post-wspecifier\n";
     
    ParseOptions po(usage);
    kaldi::BaseFloat mce_alpha = 1.0, mce_beta = 0.0; 
    po.Register("mce-alpha", &mce_alpha, "alpha parameter for sigmoid");
    po.Register("mce-beta", &mce_beta, "beta parameter for sigmoid");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string num_score_rspecifier = po.GetArg(1);
    std::string den_score_rspecifier = po.GetArg(2);
    std::string post_rspecifier = po.GetArg(3);
    std::string post_wspecifier = po.GetArg(4);
  
    kaldi::RandomAccessBaseFloatReader num_score_reader(num_score_rspecifier);
    kaldi::RandomAccessBaseFloatReader den_score_reader(den_score_rspecifier); 
    kaldi::SequentialPosteriorReader posterior_reader(post_rspecifier);
    kaldi::PosteriorWriter posterior_writer(post_wspecifier); 

    int32 num_scaled = 0, num_no_score = 0;  
   
    for(; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      kaldi::Posterior posterior = posterior_reader.Value();
      posterior_reader.FreeCurrent();
      if (!num_score_reader.HasKey(key) || !den_score_reader.HasKey(key)) {
        num_no_score++;
      } else {
        //calculate the sigmoid scaling factor for MCE
        // = sigmoid(num_score - den_score)
        BaseFloat num_score = num_score_reader.Value(key);
        BaseFloat den_score = den_score_reader.Value(key);
        BaseFloat score_difference = mce_alpha * (num_score - den_score) + mce_beta;
        BaseFloat sigmoid_difference = 1.0 / (1.0 + exp(score_difference)); 
        BaseFloat post_scale = mce_alpha * sigmoid_difference * (1 - sigmoid_difference);
        for (size_t i = 0; i < posterior.size(); i++) {
          for (size_t j = 0; j < posterior[i].size(); j++) {
            posterior[i][j].second = posterior[i][j].second * post_scale;  	 
          }
        }
        num_scaled++; 
      }
      posterior_writer.Write(key, posterior);
    }
    KALDI_LOG << "Scale " << num_scaled << " posteriors;  " << num_no_score
              << " had no scores.";
    return (num_scaled != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

