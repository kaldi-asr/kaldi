// sgmm2bin/sgmm2-est-spkvecs.cc

// Copyright 2009-2012  Saarland University  Microsoft Corporation
//                      Johns Hopkins University (Author: Daniel Povey)
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
#include "sgmm2/estimate-am-sgmm2.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

namespace kaldi {

void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmSgmm2 &am_sgmm,
                            const vector< vector<int32> > &gselect,
                            Sgmm2PerSpkDerivedVars *spk_vars,
                            MleSgmm2SpeakerAccs *spk_stats) {
  kaldi::Sgmm2PerFrameDerivedVars per_frame_vars;

  KALDI_ASSERT(gselect.size() == feats.NumRows());
  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
  for (size_t i = 0; i < post.size(); i++) {
    am_sgmm.ComputePerFrameVars(feats.Row(i), gselect[i],
                                *spk_vars, &per_frame_vars);
    
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 pdf_id = pdf_post[i][j].first;
      spk_stats->Accumulate(am_sgmm, per_frame_vars, pdf_id,
                            pdf_post[i][j].second, spk_vars);
    }
  }
}

}  // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate SGMM speaker vectors, either per utterance or for the "
        "supplied set of speakers (with spk2utt option).\n"
        "Reads Gaussian-level posteriors. Writes to a table of vectors.\n"
        "Usage: sgmm2-est-spkvecs [options] <model-in> <feature-rspecifier> "
        "<post-rspecifier> <vecs-wspecifier>\n"
        "note: --gselect option is required.";
    
    ParseOptions po(usage);
    string gselect_rspecifier, spk2utt_rspecifier, spkvecs_rspecifier;
    BaseFloat min_count = 100;
    BaseFloat rand_prune = 1.0e-05;

    po.Register("gselect", &gselect_rspecifier,
                "rspecifier for precomputed per-frame Gaussian indices from.");
    po.Register("spk2utt", &spk2utt_rspecifier,
        "File to read speaker to utterance-list map from.");
    po.Register("spkvec-min-count", &min_count,
        "Minimum count needed to estimate speaker vectors");
    po.Register("rand-prune", &rand_prune, "Pruning threshold for posteriors");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors to use during aligment (rspecifier)");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect option is mandatory.";
    
    string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        post_rspecifier = po.GetArg(3),
        vecs_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmSgmm2 am_sgmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }
    MleSgmm2SpeakerAccs spk_stats(am_sgmm, rand_prune);

    RandomAccessPosteriorReader post_reader(post_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);

    BaseFloatVectorWriter vecs_writer(vecs_wspecifier);

    double tot_impr = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_err = 0;
    std::vector<std::vector<int32> > empty_gselect;

    if (!spk2utt_rspecifier.empty()) {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        spk_stats.Clear();
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();

        Sgmm2PerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(spk)) {
            spk_vars.SetSpeakerVector(spkvecs_reader.Value(spk));
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for speaker " << spk
                       << ", not processing this speaker.";
            num_err++; // standard Kaldi behavior is to not process data
            // when errors like this happen, as it's generally a script error;
            continue;
          }
        }  // else spk_vars is "empty"

        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
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
            KALDI_WARN << "Posterior vector has wrong size " << (post.size())
                       << " vs. " << (feats.NumRows());
            num_err++;
            continue;
          }
          if (!gselect_reader.HasKey(utt) ||
              gselect_reader.Value(utt).size() != feats.NumRows()) {
            KALDI_WARN << "No Gaussian-selection info available for utterance "
                       << utt << " (or wrong size)";
            num_err++;
            continue;
          }
          const std::vector<std::vector<int32> > &gselect =
              gselect_reader.Value(utt);
          
          AccumulateForUtterance(feats, post, trans_model, am_sgmm,
                                 gselect, &spk_vars, &spk_stats);
          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_sgmm.SpkSpaceDim(), kSetZero);
          if (spk_vars.GetSpeakerVector().Dim() != 0)
            spk_vec.CopyFromVec(spk_vars.GetSpeakerVector());
          spk_stats.Update(am_sgmm, min_count, &spk_vec, &impr, &spk_tot_t);
          vecs_writer.Write(spk, spk_vec);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from speaker vector is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();        
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        if (!post_reader.HasKey(utt) ||
            post_reader.Value(utt).size() != feats.NumRows()) {
          KALDI_WARN << "Did not find posts for utterance "
                     << utt << " (or wrong size).";
          num_err++;
          continue;
        }
        const Posterior &post = post_reader.Value(utt);

        Sgmm2PerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.SetSpeakerVector(spkvecs_reader.Value(utt));
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for utterance " << utt
                       << ", not processing it.";
            num_err++;
            continue;
          }
        }  // else spk_vars is "empty"
        
        num_done++;

        if (!gselect_reader.HasKey(utt) ||
            gselect_reader.Value(utt).size() != feats.NumRows()) {
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
          num_err++;
          continue;
        }
        const std::vector<std::vector<int32> > &gselect =
            gselect_reader.Value(utt);

        spk_stats.Clear();
        
        AccumulateForUtterance(feats, post, trans_model, am_sgmm,
                               gselect, &spk_vars, &spk_stats);

        BaseFloat impr, utt_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_sgmm.SpkSpaceDim(), kSetZero);
          if (spk_vars.GetSpeakerVector().Dim() != 0)
            spk_vec.CopyFromVec(spk_vars.GetSpeakerVector());
          spk_stats.Update(am_sgmm, min_count, &spk_vec, &impr, &utt_tot_t);
          vecs_writer.Write(utt, spk_vec);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from speaker vectors is "
                  << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.";
        tot_impr += impr;
        tot_t += utt_tot_t;
      }
    }

    KALDI_LOG << "Overall auxf impr per frame is "
              << (tot_impr / tot_t) << " over " << tot_t << " frames.";
    KALDI_LOG << "Done " << num_done << " files, " << num_err << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

