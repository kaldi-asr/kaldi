// bin/post-to-bits.cc

// Copyright 2015   Vimal Manohar

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
    typedef TableWriter<BasicVectorVectorHolder<bool> >  BooleanVectorVectorWriter;

    const char *usage =
        "Turn posteriors representing quantized feats into into boolean vector.\n"
        "Usage: post-to-bits <post-rspecifier> <bits-wspecifier>\n"
        " e.g.: post-to-bits ark:post.1.ark ark:bits.1.ark\n"
        "See also: quantize-feats, post-to-weights\n";
    
    int32 num_bins = -1;

    ParseOptions po(usage); 

    po.Register("num-bins", &num_bins, 
                "Number of bins used for each dimension");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string post_rspecifier = po.GetArg(1),
                bits_wspecifier = po.GetArg(2);

    SequentialPosteriorReader posterior_reader(post_rspecifier);
    BooleanVectorVectorWriter bits_writer(bits_wspecifier);
    
    int32 num_done = 0;
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      const Posterior &posterior = posterior_reader.Value();
      int32 num_frames = static_cast<int32>(posterior.size());
      std::vector<std::vector<bool> > bits(num_frames);
      for (int32 i = 0; i < num_frames; i++) {
        bits[i].resize(posterior[i].size() * num_bins);
        for (size_t j = 0; j < posterior[i].size(); j++) {
          if (posterior[i][j].second != 1.0)
            KALDI_ERR << "Posterior " << posterior[i][j].second << " is not 1. "
                      << "These posteriors do not represent quantized feats.";
          if (posterior[i][j].first >= posterior[i].size() * num_bins)
            KALDI_ERR << "bit index " << posterior[i][j].first << " does not "
                      << "match the specified number of bins " << num_bins
                      << "; bit index must be less than " 
                      << posterior[i].size() * num_bins << ".";
          bits[i][posterior[i][j].first] = true;
        }
      }
      bits_writer.Write(key, bits);
      num_done++;
    }
    KALDI_LOG << "Done converting " << num_done << " quantized feats to bits.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


