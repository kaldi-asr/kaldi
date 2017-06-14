// featbin/append-post-to-feats.cc

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

namespace kaldi {

void AppendPostToFeats(const Matrix<BaseFloat> &in,
                       const Posterior &post,
                       const int32 post_dim,
                       Matrix<BaseFloat> *out) {
  // Check inputs,
  if (in.NumRows() != post.size()) {
    KALDI_WARN << "Mismatch of length!"
               << " features: " << in.NumRows()
               << " posteriors: " << post.size();
  }
  KALDI_ASSERT(in.NumRows() == post.size());
  // Convert post to Matrix,
  Matrix<BaseFloat> post_mat;
  PosteriorToMatrix(post, post_dim, &post_mat);
  // Build the output matrix,
  out->Resize(in.NumRows(), in.NumCols() + post_dim);
  out->ColRange(0, in.NumCols()).CopyFromMat(in);
  out->ColRange(in.NumCols(), post_mat.NumCols()).CopyFromMat(post_mat);
}

}  // namespace kaldi,

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Append posteriors to features\n"
        "\n"
        "Usage: append-post-to-feats [options] <in-rspecifier1> <in-rspecifier2> <out-wspecifier>\n"
        " or: append-post-to-feats [options] <in-rxfilename1> <in-rxfilename2> <out-wxfilename>\n"
        "e.g.: append-post-to-feats --post-dim=50 ark:input.ark scp:post.scp ark:output.ark\n"
        "See also: paste-feats, concat-feats, append-vector-to-feats\n";

    ParseOptions po(usage);

    bool binary = true;
    po.Register("binary", &binary, "If true, output files in binary "
                "(only relevant for single-file operation, i.e. no tables)");

    int32 post_dim = 0;
    po.Register("post-dim", &post_dim, "Dimensionality of the posteriors.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
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

      string feat_rspecifier = po.GetArg(1);
      SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);

      string post_rspecifier = po.GetArg(2);
      RandomAccessPosteriorReader post_reader(post_rspecifier);

      string wspecifier = po.GetArg(3);
      BaseFloatMatrixWriter feat_writer(wspecifier);

      int32 num_done = 0, num_err = 0;
      // Main loop
      for (; !feat_reader.Done(); feat_reader.Next()) {
        string utt = feat_reader.Key();
        KALDI_VLOG(2) << "Processing utterance " << utt;

        const Matrix<BaseFloat> &feats(feat_reader.Value());

        if (!post_reader.HasKey(utt)) {
          KALDI_WARN << "Could not read posteriors for utterance " << utt;
          num_err++;
          continue;
        }
        const Posterior &post(post_reader.Value(utt));

        Matrix<BaseFloat> output;
        AppendPostToFeats(feats, post, post_dim, &output);
        feat_writer.Write(utt, output);
        num_done++;
      }
      KALDI_LOG << "Done " << num_done << " utts, errors on "
                << num_err;

      return (num_done == 0 ? -1 : 0);
    } else {
      // We're operating on rxfilenames|wxfilenames, most likely files.
      Matrix<BaseFloat> mat;
      ReadKaldiObject(po.GetArg(1), &mat);

      Posterior post;
      bool binary_in;
      Input ki(po.GetArg(2), &binary_in);
      ReadPosterior(ki.Stream(), binary_in, &post);

      Matrix<BaseFloat> output;
      AppendPostToFeats(mat, post, post_dim, &output);

      std::string output_wxfilename = po.GetArg(3);
      WriteKaldiObject(output, output_wxfilename, binary);
      KALDI_LOG << "Wrote appended features to " << output_wxfilename;
      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
