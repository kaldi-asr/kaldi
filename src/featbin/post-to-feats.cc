// featbin/post-to-feats.cc

// Copyright 2016 Brno University of Technology (Author: Karel Vesely)

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
#include "matrix/kaldi-matrix.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Convert posteriors to features\n"
        "\n"
        "Usage: post-to-feats [options] <in-rspecifier> <out-wspecifier>\n"
        " or: post-to-feats [options] <in-rxfilename> <out-wxfilename>\n"
        "e.g.: post-to-feats --post-dim=50 ark:post.ark ark:feat.ark\n"
        "See also: post-to-weights feat-to-post, append-vector-to-feats, append-post-to-feats\n";

    ParseOptions po(usage);

    bool binary = true;
    po.Register("binary", &binary, "If true, output files in binary "
                "(only relevant for single-file operation, i.e. no tables)");

    int32 post_dim = 0;
    po.Register("post-dim", &post_dim, "Dimensionality of the posteriors.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (post_dim == 0) {
      KALDI_ERR << "You have to set the dimensionality of posteriors "
                   "with '--post-dim=D'";
    }

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL)
        != kNoRspecifier) {
      // We're operating on tables, e.g. archives.

      string post_rspecifier = po.GetArg(1);
      SequentialPosteriorReader post_reader(post_rspecifier);

      string wspecifier = po.GetArg(2);
      BaseFloatMatrixWriter feat_writer(wspecifier);

      int32 num_done = 0, num_err = 0;
      // Main loop
      for (; !post_reader.Done(); post_reader.Next()) {
        string utt = post_reader.Key();
        KALDI_VLOG(2) << "Processing utterance " << utt;

        const Posterior &post(post_reader.Value());

        Matrix<BaseFloat> output;
        PosteriorToMatrix(post, post_dim, &output);

        feat_writer.Write(utt, output);
        num_done++;
      }
      KALDI_LOG << "Done " << num_done << " utts, errors on "
                << num_err;

      return (num_done == 0 ? -1 : 0);
    } else {
      // We're operating on rxfilenames|wxfilenames, most likely files.
      Posterior post;
      bool binary_in;
      Input ki(po.GetArg(1), &binary_in);
      ReadPosterior(ki.Stream(), binary_in, &post);

      Matrix<BaseFloat> output;
      PosteriorToMatrix(post, post_dim, &output);

      std::string output_wxfilename = po.GetArg(3);
      WriteKaldiObject(output, output_wxfilename, binary);
      KALDI_LOG << "Wrote posteriors as feature-matrix to " << output_wxfilename;
      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
