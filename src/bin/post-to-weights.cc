// bin/post-to-weights.cc

// Copyright 2009-2011 Chao Weng  Microsoft Corporation

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
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "Turn posteriors into per-frame weights (typically most useful after\n"
        "weight-silence-post, to get silence weights)\n"
        "See also: weight-silence-post, post-to-pdf-post, post-to-phone-post\n"
        "post-to-feats, get-post-on-ali\n"
        "Usage: post-to-weights <post-rspecifier> <weights-wspecifier>\n";
    
    ParseOptions po(usage); 
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string post_rspecifier = po.GetArg(1),
        weights_wspecifier = po.GetArg(2);

    SequentialPosteriorReader posterior_reader(post_rspecifier);
    BaseFloatVectorWriter weights_writer(weights_wspecifier); 
    
    int32 num_done = 0;
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      const Posterior &posterior = posterior_reader.Value();
      int32 num_frames = static_cast<int32>(posterior.size());
      Vector<BaseFloat> weights(num_frames);
      for (int32 i = 0; i < num_frames; i++) {
        BaseFloat sum = 0.0;
        for (size_t j = 0; j < posterior[i].size(); j++)
          sum += posterior[i][j].second;
        weights(i) = sum;
      }
      weights_writer.Write(key, weights);
      num_done++;
    }
    KALDI_LOG << "Done converting " << num_done << " posteriors to weights.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

