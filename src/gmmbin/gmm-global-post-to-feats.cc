// gmmbin/gmm-global-post-to-feats.cc

// Copyright 2016 Brno University of Technology (Author: Karel Vesely)
//           2016 Vimal Manohar

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
#include "gmm/diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Convert GMM global posteriors to features\n"
        "\n"
        "Usage: gmm-global-post-to-feats [options] <model-rxfilename|model-rspecifier> <in-rspecifier> <out-wspecifier>\n"
        "e.g.: gmm-global-post-to-feats ark:1.gmm ark:post.ark ark:feat.ark\n"
        "See also: post-to-feats --post-dim, post-to-weights feat-to-post, append-vector-to-feats, append-post-to-feats\n";

    ParseOptions po(usage);
    std::string utt2spk_rspecifier;

    po.Register("utt2spk", &utt2spk_rspecifier, 
                "rspecifier for utterance to speaker map for reading "
                "per-speaker GMM models");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
      post_rspecifier = po.GetArg(2),
      feat_wspecifier = po.GetArg(3);
    
    DiagGmm diag_gmm;
    RandomAccessDiagGmmReaderMapped *gmm_reader = NULL;
    SequentialPosteriorReader post_reader(post_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL)
        != kNoRspecifier) {  // We're operating on tables, e.g. archives.
      gmm_reader = new RandomAccessDiagGmmReaderMapped(model_in_filename,
                                                       utt2spk_rspecifier);
    } else {
      ReadKaldiObject(model_in_filename, &diag_gmm);
    }

    int32 num_done = 0, num_err = 0;
    
    for (; !post_reader.Done(); post_reader.Next()) {
      const std::string &utt = post_reader.Key();

      const DiagGmm *gmm = &diag_gmm;
      if (gmm_reader) {
        if (!gmm_reader->HasKey(utt)) {
          KALDI_WARN << "Could not find GMM model for utterance " << utt;
          num_err++;
          continue;
        }
        gmm = &(gmm_reader->Value(utt));
      }
      
      int32 post_dim = gmm->NumGauss();

      const Posterior &post = post_reader.Value();

      Matrix<BaseFloat> output;
      PosteriorToMatrix(post, post_dim, &output);

      feat_writer.Write(utt, output);
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utts, errors on "
              << num_err;

    return (num_done == 0 ? -1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
