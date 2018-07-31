// bin/weight-post.cc

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

    int32 length_tolerance = 2;

    const char *usage =
        "Takes archives (typically per-utterance) of posteriors and per-frame weights,\n"
        "and weights the posteriors by the per-frame weights\n"
        "\n"
        "Usage: weight-post <post-rspecifier> <weights-rspecifier> <post-wspecifier>\n";

    ParseOptions po(usage);

    po.Register("length-tolerance", &length_tolerance,
                "Tolerate this many frames of length mismatch between "
                "posteriors and weights");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string post_rspecifier = po.GetArg(1),
        weights_rspecifier = po.GetArg(2),
        post_wspecifier = po.GetArg(3);

    SequentialPosteriorReader posterior_reader(post_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    PosteriorWriter post_writer(post_wspecifier);

    int32 num_done = 0, num_err = 0;

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      Posterior post  = posterior_reader.Value();
      if (!weights_reader.HasKey(key)) {
        KALDI_WARN << "No weights for utterance " << key;
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &weights = weights_reader.Value(key);
      if (std::abs(weights.Dim() - static_cast<int32>(post.size())) > 
          length_tolerance) {
        KALDI_WARN << "Weights for utterance " << key
                   << " have wrong size, " << weights.Dim()
                   << " vs. " << post.size();
        num_err++;
        continue;
      }
      for (size_t i = 0; i < post.size(); i++) {
        if (weights(i) == 0.0) post[i].clear();
        for (size_t j = 0; j < post[i].size(); j++)
          post[i][j].second *= i < weights.Dim() ? weights(i) : 0.0;
      }
      post_writer.Write(key, post);
      num_done++;
    }
    KALDI_LOG << "Scaled " << num_done << " posteriors; errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

