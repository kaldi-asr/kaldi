// bin/copy-post.cc

// Copyright 2011-2012 Johns Hopkins University (Author: Daniel Povey)  Chao Weng

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
        "Copy archives of posteriors, with optional scaling\n"
        "\n"
        "Usage: copy-post <post-rspecifier> <post-wspecifier>\n"
        "See also: post-to-weights, scale-post, sum-post, weight-post ...\n";

    BaseFloat scale = 1.0;
    ParseOptions po(usage);
    po.Register("scale", &scale, "Scale for posteriors");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string post_rspecifier = po.GetArg(1),
        post_wspecifier = po.GetArg(2);

    kaldi::SequentialPosteriorReader posterior_reader(post_rspecifier);
    kaldi::PosteriorWriter posterior_writer(post_wspecifier);

    int32 num_done = 0;

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();

      if (scale != 1.0) {
        kaldi::Posterior posterior = posterior_reader.Value();
        ScalePosterior(scale, &posterior);
        posterior_writer.Write(key, posterior);
      } else {
        posterior_writer.Write(key, posterior_reader.Value());
      }
      num_done++;
    }
    KALDI_LOG << "Done copying " << num_done << " posteriors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
