// sgmm2bin/sgmm2-est-fmllr.cc

// Copyright 2009-2012  Saarland University   Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
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
#include "sgmm2/am-sgmm2.h"
#include "sgmm2/fmllr-sgmm2.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

namespace kaldi {

void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Matrix<BaseFloat> &transformed_feats, // if already fMLLR
                            const std::vector<std::vector<int32> > &gselect,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmSgmm2 &am_sgmm,
                            BaseFloat logdet,
                            Sgmm2PerSpkDerivedVars *spk_vars,
                            FmllrSgmm2Accs *spk_stats) {
  kaldi::Sgmm2PerFrameDerivedVars per_frame_vars;

  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
  for (size_t t = 0; t < post.size(); t++) {
    // per-frame vars only used for computing posteriors... use the
    // transformed feats for this, if available.
    am_sgmm.ComputePerFrameVars(transformed_feats.Row(t), gselect[t],
                                *spk_vars, &per_frame_vars);
    

    for (size_t j = 0; j < pdf_post[t].size(); j++) {
      int32 pdf_id = pdf_post[t][j].first;
      Matrix<BaseFloat> posteriors;
      am_sgmm.ComponentPosteriors(per_frame_vars, pdf_id,
                                  spk_vars, &posteriors);
      posteriors.Scale(pdf_post[t][j].second);
      spk_stats->AccumulateFromPosteriors(am_sgmm, *spk_vars, feats.Row(t),
                                          gselect[t], posteriors, pdf_id);
    }
  }
}

}  // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate FMLLR transform for SGMMs, either per utterance or for the "
        "supplied set of speakers (with spk2utt option).\n"
        "Reads state-level posteriors. Writes to a table of matrices.\n"
        "--gselect option is mandatory.\n"
        "Usage: sgmm2-est-fmllr [options] <model-in> <feature-rspecifier> "
        "<post-rspecifier> <mats-wspecifier>\n";
    
    ParseOptions po(usage);
    string spk2utt_rspecifier, spkvecs_rspecifier, fmllr_rspecifier,
        gselect_rspecifier;
    BaseFloat min_count = 100;
    Sgmm2FmllrConfig fmllr_opts;
    
    po.Register("spk2utt", &spk2utt_rspecifier,
                "File to read speaker to utterance-list map from.");
    po.Register("spkvec-min-count", &min_count,
                "Minimum count needed to estimate speaker vectors");
    po.Register("spk-vecs", &spkvecs_rspecifier,
                "Speaker vectors to use during aligment (rspecifier)");
    po.Register("input-fmllr", &fmllr_rspecifier,
                "Initial FMLLR transform per speaker (rspecifier)");
    po.Register("gselect", &gselect_rspecifier,
                "Precomputed Gaussian indices (rspecifier)");
    fmllr_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        post_rspecifier = po.GetArg(3),
        fmllr_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmSgmm2 am_sgmm;
    Sgmm2FmllrGlobalParams fmllr_globals;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
      fmllr_globals.Read(ki.Stream(), binary);
    }
    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect option is required.";
    
    RandomAccessPosteriorReader post_reader(post_rspecifier);
    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatMatrixReader fmllr_reader(fmllr_rspecifier);

    BaseFloatMatrixWriter fmllr_writer(fmllr_wspecifier);

    int32 dim = am_sgmm.FeatureDim();
    FmllrSgmm2Accs spk_stats;
    spk_stats.Init(dim, am_sgmm.NumGauss());
    Matrix<BaseFloat> fmllr_xform(dim, dim + 1);
    BaseFloat logdet = 0.0;
    double tot_impr = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_err = 0;
    std::vector<std::vector<int32> > empty_gselect;

    if (!spk2utt_rspecifier.empty()) {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        spk_stats.SetZero();
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();

        Sgmm2PerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(spk)) {
            spk_vars.SetSpeakerVector(spkvecs_reader.Value(spk));
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << spk;
            num_err++;
            continue;
          }
        }  // else spk_vars is "empty"

        if (fmllr_reader.IsOpen()) {
          if (fmllr_reader.HasKey(spk)) {
            fmllr_xform.CopyFromMat(fmllr_reader.Value(spk));
            logdet = fmllr_xform.Range(0, dim, 0, dim).LogDet();
          } else {
            KALDI_WARN << "Cannot find FMLLR transform for " << spk;
            fmllr_xform.SetUnit();
            logdet = 0.0;
          }
        } else {
          fmllr_xform.SetUnit();
          logdet = 0.0;
        }

        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          if (!post_reader.HasKey(utt) ||
              post_reader.Value(utt).size() != feats.NumRows()) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt
                       << " (or wrong size).";
            num_err++;
            continue;
          }
          const Posterior &post = post_reader.Value(utt);
          if (!gselect_reader.HasKey(utt) ||
              gselect_reader.Value(utt).size() != feats.NumRows()) {
            KALDI_WARN << "Did not find gselect info for utterance " << utt
                       << " (or wrong size).";
            num_err++;
            continue;
          }
          const std::vector<std::vector<int32> > &gselect =
              gselect_reader.Value(utt);
          
          Matrix<BaseFloat> transformed_feats(feats);
          for (int32 r = 0; r < transformed_feats.NumRows(); r++) {
            SubVector<BaseFloat> row(transformed_feats, r);
            ApplyAffineTransform(fmllr_xform, &row);
          }
          AccumulateForUtterance(feats, transformed_feats, gselect,
                                 post, trans_model, am_sgmm,
                                 logdet, &spk_vars, &spk_stats);
          num_done++;
        }  // end looping over all utterances of the current speaker
        
        BaseFloat impr, spk_frame_count;
        // Compute the FMLLR transform and write it out.
        spk_stats.Update(am_sgmm, fmllr_globals, fmllr_opts, &fmllr_xform,
                         &spk_frame_count, &impr);
        fmllr_writer.Write(spk, fmllr_xform);
        tot_impr += impr;
        tot_t += spk_frame_count;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        const Matrix<BaseFloat> &feats = feature_reader.Value();

        if (!post_reader.HasKey(utt) ||
            post_reader.Value(utt).size() != feats.NumRows()) {
          KALDI_WARN << "Did not find posteriors for utterance " << utt
                     << " (or wrong size).";
          num_err++;
          continue;
        }
        const Posterior &post = post_reader.Value(utt);
        if (!gselect_reader.HasKey(utt) ||
            gselect_reader.Value(utt).size() != feats.NumRows()) {
          KALDI_WARN << "Did not find gselect info for utterance " << utt
                     << " (or wrong size).";
          num_err++;
          continue;
        }
        const std::vector<std::vector<int32> > &gselect =
            gselect_reader.Value(utt);
        
        if (fmllr_reader.IsOpen()) {
          if (fmllr_reader.HasKey(utt)) {
            fmllr_xform.CopyFromMat(fmllr_reader.Value(utt));
            logdet = fmllr_xform.Range(0, dim, 0, dim).LogDet();
          } else {
            KALDI_WARN << "Cannot find FMLLR transform for " << utt;
            fmllr_xform.SetUnit();
            logdet = 0.0;
          }
        } else {
          fmllr_xform.SetUnit();
          logdet = 0.0;
        }
        
        Matrix<BaseFloat> transformed_feats(feats);
        for (int32 r = 0; r < transformed_feats.NumRows(); r++) {
          SubVector<BaseFloat> row(transformed_feats, r);
          ApplyAffineTransform(fmllr_xform, &row);
        }
        
        Sgmm2PerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.SetSpeakerVector(spkvecs_reader.Value(utt));
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
            num_err++;
            continue;
          }
        }  // else spk_vars is "empty"

        spk_stats.SetZero();

        AccumulateForUtterance(feats, transformed_feats, gselect,
                               post, trans_model, am_sgmm,
                               logdet, &spk_vars, &spk_stats);
        num_done++;
        
        BaseFloat impr, spk_frame_count;
        // Compute the FMLLR transform and write it out.
        spk_stats.Update(am_sgmm, fmllr_globals, fmllr_opts, &fmllr_xform,
                         &spk_frame_count, &impr);
        fmllr_writer.Write(utt, fmllr_xform);
        tot_impr += impr;
        tot_t += spk_frame_count;
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err << " with errors.";
    KALDI_LOG << "Overall auxf impr per frame is " << (tot_impr / tot_t)
              << " per frame, over " << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

