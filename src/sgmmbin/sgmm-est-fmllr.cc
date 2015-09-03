// sgmmbin/sgmm-est-fmllr.cc

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
#include "sgmm/am-sgmm.h"
#include "sgmm/fmllr-sgmm.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

namespace kaldi {

void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Matrix<BaseFloat> &transformed_feats, // if already fMLLR
                            const std::vector<std::vector<int32> > &gselect,
                            const SgmmGselectConfig &sgmm_config,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmSgmm &am_sgmm,
                            const SgmmPerSpkDerivedVars &spk_vars,
                            BaseFloat logdet,
                            FmllrSgmmAccs *spk_stats) {
  kaldi::SgmmPerFrameDerivedVars per_frame_vars;

  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
  for (size_t t = 0; t < post.size(); t++) {
    std::vector<int32> this_gselect;
    if (!gselect.empty()) {
      KALDI_ASSERT(t < gselect.size());
      this_gselect = gselect[t];
    } else  {
      am_sgmm.GaussianSelection(sgmm_config, feats.Row(t), &this_gselect);
    }
    // per-frame vars only used for computing posteriors... use the
    // transformed feats for this, if available.
    am_sgmm.ComputePerFrameVars(transformed_feats.Row(t), this_gselect, spk_vars,
                                0.0 /*fMLLR logdet*/, &per_frame_vars);


    for (size_t j = 0; j < pdf_post[t].size(); j++) {
      int32 pdf_id = pdf_post[t][j].first;
      Matrix<BaseFloat> posteriors;
      am_sgmm.ComponentPosteriors(per_frame_vars, pdf_id,
                                  &posteriors);
      posteriors.Scale(pdf_post[t][j].second);
      spk_stats->AccumulateFromPosteriors(am_sgmm, spk_vars, feats.Row(t),
                                          this_gselect,
                                          posteriors, pdf_id);
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
        "Usage: sgmm-est-fmllr [options] <model-in> <feature-rspecifier> "
        "<post-rspecifier> <mats-wspecifier>\n";

    ParseOptions po(usage);
    string spk2utt_rspecifier, spkvecs_rspecifier, fmllr_rspecifier,
        gselect_rspecifier;
    BaseFloat min_count = 100;
    SgmmFmllrConfig fmllr_opts;
    SgmmGselectConfig sgmm_opts;
    
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
    sgmm_opts.Register(&po);
    
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
    AmSgmm am_sgmm;
    SgmmFmllrGlobalParams fmllr_globals;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
      fmllr_globals.Read(ki.Stream(), binary);
    }

    RandomAccessPosteriorReader post_reader(post_rspecifier);
    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatMatrixReader fmllr_reader(fmllr_rspecifier);

    BaseFloatMatrixWriter fmllr_writer(fmllr_wspecifier);

    int32 dim = am_sgmm.FeatureDim();
    FmllrSgmmAccs spk_stats;
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

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(spk)) {
            spk_vars.v_s = spkvecs_reader.Value(spk);
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
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != feats.NumRows()) {
            KALDI_WARN << "posterior vector has wrong size " << (post.size())
                       << " vs. " << (feats.NumRows());
            num_err++;
            continue;
          }

          bool have_gselect  = !gselect_rspecifier.empty()
              && gselect_reader.HasKey(utt)
              && gselect_reader.Value(utt).size() == feats.NumRows();
          if (!gselect_rspecifier.empty() && !have_gselect)
            KALDI_WARN << "No Gaussian-selection info available for utterance "
                       << utt << " (or wrong size)";
          const std::vector<std::vector<int32> > *gselect =
              (have_gselect ? &gselect_reader.Value(utt) : &empty_gselect);
          
          Matrix<BaseFloat> transformed_feats(feats);
          for (int32 r = 0; r < transformed_feats.NumRows(); r++) {
            SubVector<BaseFloat> row(transformed_feats, r);
            ApplyAffineTransform(fmllr_xform, &row);
          }
          AccumulateForUtterance(feats, transformed_feats, *gselect, sgmm_opts,
                                 post, trans_model, am_sgmm, spk_vars,
                                 logdet, &spk_stats);
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
        if (!post_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posts for utterance "
                     << utt;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
            num_err++;
            continue;
          }
        }  // else spk_vars is "empty"

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

        const Posterior &post = post_reader.Value(utt);

        if (static_cast<int32>(post.size()) != feats.NumRows()) {
          KALDI_WARN << "post has wrong size " << (post.size())
              << " vs. " << (feats.NumRows());
          num_err++;
          continue;
        }
        spk_stats.SetZero();

        Matrix<BaseFloat> transformed_feats(feats);
        for (int32 r = 0; r < transformed_feats.NumRows(); r++) {
          SubVector<BaseFloat> row(transformed_feats, r);
          ApplyAffineTransform(fmllr_xform, &row);
        }
        bool have_gselect  = !gselect_rspecifier.empty()
            && gselect_reader.HasKey(utt)
            && gselect_reader.Value(utt).size() == feats.NumRows();
        if (!gselect_rspecifier.empty() && !have_gselect)
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
        const std::vector<std::vector<int32> > *gselect =
            (have_gselect ? &gselect_reader.Value(utt) : &empty_gselect);
        
        AccumulateForUtterance(feats, transformed_feats, *gselect, sgmm_opts,
                               post, trans_model, am_sgmm, spk_vars,
                               logdet, &spk_stats);
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

