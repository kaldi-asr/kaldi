// gmmbin/gmm-basis-fmllr-accs-gpost.cc

// Copyright 2012  Carnegie Mellon University (author: Yajie Miao)
//           2014  Guoguo Chen

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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/fmllr-diag-gmm.h"
#include "transform/basis-fmllr-diag-gmm.h"
#include "hmm/posterior.h"

namespace kaldi {
void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const GaussPost &gpost,
                            const TransitionModel &trans_model,
                            const AmDiagGmm &am_gmm,
                            FmllrDiagGmmAccs *spk_stats) {
  for (size_t i = 0; i < gpost.size(); i++) {
    for (size_t j = 0; j < gpost[i].size(); j++) {
      int32 pdf_id = gpost[i][j].first;
      const Vector<BaseFloat> & posterior(gpost[i][j].second);
      spk_stats->AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                          feats.Row(i), posterior);
    }
  }
}


}

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Accumulate gradient scatter from training set, either per utterance or \n"
        "for the supplied set of speakers (spk2utt option). Reads Gaussian-level \n"
        "posterior to accumulate fMLLR stats for each speaker/utterance. Writes \n"
        "gradient scatter matrix.\n"
        "Usage: gmm-basis-fmllr-accs-gpost [options] <model-in> <feature-rspecifier>"
        "<post-rspecifier> <accs-wspecifier>\n";

    bool binary_write = true;
    string spk2utt_rspecifier;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    string
        model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        gpost_rspecifier = po.GetArg(3),
        accs_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    RandomAccessGaussPostReader gpost_reader(gpost_rspecifier);
    BasisFmllrAccus basis_accs(am_gmm.Dim());

    int32 num_done = 0, num_no_post = 0, num_other_error = 0;
    if (spk2utt_rspecifier != "") {  // per-speaker mode
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      int32 num_spk = 0;
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        FmllrDiagGmmAccs spk_stats(am_gmm.Dim());
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_other_error++;
            continue;
          }
          if (!gpost_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_post++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const GaussPost &gpost = gpost_reader.Value(utt);
          if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
            KALDI_WARN << "GaussPost has wrong size " << (gpost.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          AccumulateForUtterance(feats, gpost, trans_model, am_gmm, &spk_stats);

          num_done++;
        }  // end looping over all utterances of this speaker
        basis_accs.AccuGradientScatter(spk_stats);
        num_spk++;
      }  // end looping over speakers
      KALDI_LOG << "Accumulate statistics from " << num_spk << " speakers";

    } else {  // per-utterance mode
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!gpost_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posts for utterance "
                     << utt;
          num_no_post++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const GaussPost &gpost = gpost_reader.Value(utt);

        if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
          KALDI_WARN << "GaussPost has wrong size " << (gpost.size())
                     << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }
        // Accumulate stats for this utterance
        FmllrDiagGmmAccs utt_stats(am_gmm.Dim());
        AccumulateForUtterance(feats, gpost, trans_model, am_gmm, &utt_stats);
        num_done++;

        basis_accs.AccuGradientScatter(utt_stats);
      } // end looping over all utterances
    }
    // Write out accumulations
    {
      Output ko(accs_wspecifier, binary_write);
      basis_accs.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Done " << num_done << " files, " << num_no_post
              << " with no posts, " << num_other_error << " with other errors.";
    KALDI_LOG << "Written gradient scatter to " << accs_wspecifier;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

