// gmmbin/gmm-est-fmllr.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2014  Guoguo Chen

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
#include "hmm/posterior.h"

namespace kaldi {
void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmDiagGmm &am_gmm,
                            FmllrDiagGmmAccs *spk_stats) {
  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
  for (size_t i = 0; i < post.size(); i++) {
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 pdf_id = pdf_post[i][j].first;
      spk_stats->AccumulateForGmm(am_gmm.GetPdf(pdf_id),
                                  feats.Row(i),
                                  pdf_post[i][j].second);
    }
  }
}


}

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate global fMLLR transforms, either per utterance or for the supplied\n"
        "set of speakers (spk2utt option).  Reads posteriors (on transition-ids).  Writes\n"
        "to a table of matrices.\n"
        "Usage: gmm-est-fmllr [options] <model-in> "
        "<feature-rspecifier> <post-rspecifier> <transform-wspecifier>\n";

    ParseOptions po(usage);
    FmllrOptions fmllr_opts;
    string spk2utt_rspecifier;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    fmllr_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    string
        model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        post_rspecifier = po.GetArg(3),
        trans_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    RandomAccessPosteriorReader post_reader(post_rspecifier);

    double tot_impr = 0.0, tot_t = 0.0;

    BaseFloatMatrixWriter transform_writer(trans_wspecifier);

    int32 num_done = 0, num_no_post = 0, num_other_error = 0;
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        FmllrDiagGmmAccs spk_stats(am_gmm.Dim(), fmllr_opts);
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_other_error++;
            continue;
          }
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_post++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != feats.NumRows()) {
            KALDI_WARN << "Posterior vector has wrong size " << (post.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          AccumulateForUtterance(feats, post, trans_model, am_gmm, &spk_stats);

          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
          transform.SetUnit();
          spk_stats.Update(fmllr_opts, &transform, &impr, &spk_tot_t);
          transform_writer.Write(spk, transform);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from fMLLR is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!post_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posts for utterance "
                     << utt;
          num_no_post++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const Posterior &post = post_reader.Value(utt);

        if (static_cast<int32>(post.size()) != feats.NumRows()) {
          KALDI_WARN << "Posterior has wrong size " << (post.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }
        num_done++;

        FmllrDiagGmmAccs spk_stats(am_gmm.Dim(), fmllr_opts);

        AccumulateForUtterance(feats, post, trans_model, am_gmm,
                               &spk_stats);

        BaseFloat impr, utt_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
          transform.SetUnit();
          spk_stats.Update(fmllr_opts, &transform, &impr, &utt_tot_t);
          transform_writer.Write(utt, transform);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from fMLLR is "
                  << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.";
        tot_impr += impr;
        tot_t += utt_tot_t;
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_post
              << " with no posts, " << num_other_error << " with other errors.";
    KALDI_LOG << "Overall fMLLR auxf impr per frame is "
              << (tot_impr / tot_t) << " over " << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

