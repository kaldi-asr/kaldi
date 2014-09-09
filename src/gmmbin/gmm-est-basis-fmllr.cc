// gmmbin/gmm-est-basis-fmllr.cc

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
        "Perform basis fMLLR adaptation in testing stage, either per utterance or\n"
        "for the supplied set of speakers (spk2utt option). Reads posterior to\n"
        "accumulate fMLLR stats for each speaker/utterance. Writes to a table of\n"
        "matrices.\n"
        "Usage: gmm-est-basis-fmllr [options] <model-in> <basis-rspecifier> <feature-rspecifier> "
        "<post-rspecifier> <transform-wspecifier>\n";

    ParseOptions po(usage);
    BasisFmllrOptions basis_fmllr_opts;
    string spk2utt_rspecifier;
    string weights_out_filename;

    po.Register("spk2utt", &spk2utt_rspecifier, "Rspecifier for speaker to "
                "utterance-list map");
    po.Register("write-weights", &weights_out_filename, "File to write base "
                    "weights to.");

    basis_fmllr_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    string
        model_rxfilename = po.GetArg(1),
        basis_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        post_rspecifier = po.GetArg(4),
        trans_wspecifier = po.GetArg(5);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    BasisFmllrEstimate basis_est;
    ReadKaldiObject(basis_rspecifier, &basis_est);
    
    RandomAccessPosteriorReader post_reader(post_rspecifier);

    double tot_impr = 0.0, tot_t = 0.0;

    BaseFloatMatrixWriter transform_writer(trans_wspecifier);
    BaseFloatVectorWriter weights_writer;
    if (!weights_out_filename.empty()) {
      weights_writer.Open(weights_out_filename);
    }

    int32 num_done = 0, num_no_post = 0, num_other_error = 0;
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

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

        double impr, spk_tot_t; int32 wgt_size;
        {
          // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim() + 1);
          transform.SetUnit();
          Vector<BaseFloat> weights;  // size will be adjusted
          impr = basis_est.ComputeTransform(spk_stats, &transform,
                                            &weights, basis_fmllr_opts);
          spk_tot_t = spk_stats.beta_;
          wgt_size = weights.Dim();
          transform_writer.Write(spk, transform);
          // Optionally write out the base weights
          if (!weights_out_filename.empty() && weights.Dim() > 0)
              weights_writer.Write(spk, weights);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from Basis fMLLR is "
                  << (impr / spk_tot_t) << ", over " << spk_tot_t << " frames, "
                  << "the top " << wgt_size << " basis elements have been used";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!post_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posts for utterance " << utt;
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

        FmllrDiagGmmAccs spk_stats(am_gmm.Dim());
        AccumulateForUtterance(feats, post, trans_model, am_gmm, &spk_stats);
        num_done++;

        BaseFloat impr, utt_tot_t; int32 wgt_size;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
          transform.SetUnit();
          Vector<BaseFloat> weights(am_gmm.Dim() * (am_gmm.Dim() + 1)); // size will be adjusted
          impr = basis_est.ComputeTransform(spk_stats, &transform,
                                            &weights, basis_fmllr_opts);
          utt_tot_t = spk_stats.beta_;
          wgt_size = weights.Dim();
          transform_writer.Write(utt, transform);
          // Optionally write out the base weights
          if (!weights_out_filename.empty() && weights.Dim() > 0)
            weights_writer.Write(utt, weights);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from Basis fMLLR is "
                  << (impr / utt_tot_t) << ", over " << utt_tot_t << " frames, "
                  << "the top " << wgt_size << " basis elements have been used";
        tot_impr += impr;
        tot_t += utt_tot_t;
      }  // end looping over all the utterances
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_post
              << " with no posts, " << num_other_error << " with other errors.";
    KALDI_LOG << "Overall fMLLR auxf-impr per frame is "
              << (tot_impr / tot_t) << " over " << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

