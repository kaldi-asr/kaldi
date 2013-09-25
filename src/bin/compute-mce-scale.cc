// bin/compute-mce-scale.cc

// Copyright 2009-2011 Chao Weng 

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "compute the scale of MCE, which is used to scale posteriors\n"
        "Usage: compute-mce-scale [option] num-score-rspecifier "
        "den-score-rspecifier out-scale-wspecifier\n";
    
    ParseOptions po(usage);
    kaldi::BaseFloat mce_alpha = 1.0, mce_beta = 0.0;
    po.Register("mce-alpha", &mce_alpha, "alpha parameter for sigmoid");
    po.Register("mce-beta", &mce_beta, "beta parameter for sigmoid");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string num_score_rspecifier = po.GetArg(1);
    std::string den_score_rspecifier = po.GetArg(2);
    std::string scale_wspecifier = po.GetArg(3);

    kaldi::SequentialBaseFloatReader num_score_reader(num_score_rspecifier);
    kaldi::RandomAccessBaseFloatReader den_score_reader(den_score_rspecifier);
    kaldi::BaseFloatWriter scale_writer(scale_wspecifier);
   
    int32 num_scaled = 0, num_no_score = 0;
    double tot_sigmoid = 0.0;
    
    for (; !num_score_reader.Done(); num_score_reader.Next()) {
      std::string key = num_score_reader.Key();
      kaldi::BaseFloat num_score = num_score_reader.Value();
      num_score_reader.FreeCurrent();
      if (!den_score_reader.HasKey(key)) {
        num_no_score++;
      } else {
        // calculate the sigmoid scaling factor for MCE
        // Note: the derivative is:
        //  \alpha * sigmoid(num - den) * (1 - sigmoid(num - den))
        // but the make the scale be:
        //  4 * sigmoid(num - den) * (1 - sigmoid(num - den))
        // which is just multiplying by 4/alpha; this means
        // that the maximum value the scale can have is 1, which
        // means it's more comparable with MMI/MPE.
        BaseFloat den_score = den_score_reader.Value(key);
        BaseFloat score_difference = mce_alpha * (num_score - den_score) + mce_beta;
        BaseFloat sigmoid_difference = 1.0 / (1.0 + exp(score_difference));
        // It might be more natural to make the scale
        //
        BaseFloat scale = 4.0 * sigmoid_difference * (1 - sigmoid_difference);
        scale_writer.Write(key, scale);
        num_scaled++;
        tot_sigmoid += sigmoid_difference;
      }    
    }
    KALDI_LOG << num_scaled << " scales generated; " << num_no_score
              << " had no num/den scores.";
    KALDI_LOG << "Overall MCE objective function per utterance is "
              << (tot_sigmoid/num_scaled) << " over "
              << num_scaled << " utterance.  [Note: should go down]";
    return (num_scaled != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
