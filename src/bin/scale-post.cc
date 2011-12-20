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
        "scale the posterior \n"
        "Usage: score-post scale-rspecifier post-rspecifier post-wspecifier\n";
    
    ParseOptions po(usage); 
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string scale_rspecifier = po.GetArg(1);
    std::string post_rspecifier = po.GetArg(2);
    std::string post_wspecifier = po.GetArg(3);
  
    kaldi::RandomAccessBaseFloatReader scale_reader(scale_rspecifier);
    kaldi::SequentialPosteriorReader posterior_reader(post_rspecifier);
    kaldi::PosteriorWriter posterior_writer(post_wspecifier); 

    int32 num_scaled = 0, num_no_scale = 0;  
   
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      kaldi::Posterior posterior = posterior_reader.Value();
      posterior_reader.FreeCurrent();
      if (!scale_reader.HasKey(key)) {
        num_no_scale++;
      } else {
        BaseFloat post_scale = scale_reader.Value(key);
        for (size_t i = 0; i < posterior.size(); i++) {
          for (size_t j = 0; j < posterior[i].size(); j++) {
            posterior[i][j].second = posterior[i][j].second * post_scale;  	 
          }
        }
        num_scaled++; 
        posterior_writer.Write(key, posterior);
      }
    }
    KALDI_LOG << "Scale " << num_scaled << " posteriors;  " << num_no_scale
              << " had no scales.";
    return (num_scaled != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

