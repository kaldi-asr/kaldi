// nnetbin/feat-to-post.cc

// Copyright 2014       Brno University of Technology (Author: Karel Vesely)

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

/** @brief Converts features into posterior format, which is the generic
 *  format of NN training targets in 'nnet1'. */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Convert features into posterior format, which is the generic format \n"
      "of NN training targets in Karel's nnet1 tools.\n"
      "(speed is not an issue for reasonably low NN-output dimensions)\n"
      "Usage:  feat-to-post [options] feat-rspecifier posteriors-wspecifier\n"
      "e.g.:\n"
      " feat-to-post scp:feats.scp ark:feats.post\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feats_rspecifier = po.GetArg(1);
    std::string posteriors_wspecifier = po.GetArg(2);

    int32 num_done = 0;
    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !feats_reader.Done(); feats_reader.Next()) {
      num_done++;
      const Matrix<BaseFloat> &mat = feats_reader.Value();
      int32 num_frames = mat.NumRows(),
        num_dims = mat.NumCols();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post(num_frames);
      // Fill posterior with matrix values,
      for (int32 f = 0; f < num_frames; f++) {
        for (int32 d = 0; d < num_dims; d++) {
          post[f].push_back(std::make_pair(d, mat(f, d)));
        }
        KALDI_ASSERT(post[f].size() == num_dims);
      }
      // Store
      posterior_writer.Write(feats_reader.Key(), post);
    }
    KALDI_LOG << "Converted " << num_done << " alignments.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


