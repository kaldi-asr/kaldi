// sgmmbin/sgmm-est-spkvecs-gpost.cc

// Copyright 2009-2011   Saarland University;  Microsoft Corporation

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
#include "sgmm/estimate-am-sgmm.h"
#include "hmm/transition-model.h"

namespace kaldi {

void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const SgmmGauPost &gpost,
                            const TransitionModel &trans_model,
                            const AmSgmm &am_sgmm,
                            const SgmmPerSpkDerivedVars &spk_vars,
                            MleSgmmSpeakerAccs *spk_stats) {
  kaldi::SgmmPerFrameDerivedVars per_frame_vars;

  for (size_t i = 0; i < gpost.size(); i++) {
    am_sgmm.ComputePerFrameVars(feats.Row(i),
                                gpost[i].gselect, spk_vars, 0.0,
                                &per_frame_vars);

    for (size_t j = 0; j < gpost[i].tids.size(); j++) {
      int32 pdf_id = trans_model.TransitionIdToPdf(gpost[i].tids[j]);
      spk_stats->AccumulateFromPosteriors(am_sgmm, per_frame_vars,
                                          gpost[i].posteriors[j], pdf_id);
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
        "Usage: sgmm-est-spkvecs-gpost [options] <model-in> <feature-rspecifier> "
        "<gpost-rspecifier> <vecs-wspecifier>\n";

    ParseOptions po(usage);
    string spk2utt_rspecifier, spkvecs_rspecifier;
    BaseFloat min_count = 100;
    BaseFloat rand_prune = 1.0e-05;

    po.Register("spk2utt", &spk2utt_rspecifier,
        "File to read speaker to utterance-list map from.");
    po.Register("spkvec-min-count", &min_count,
        "Minimum count needed to estimate speaker vectors");
    po.Register("rand-prune", &rand_prune, "Randomized pruning parameter for posteriors (more->faster).");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors to use during aligment (rspecifier)");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        gpost_rspecifier = po.GetArg(3),
        vecs_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmSgmm am_sgmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }
    MleSgmmSpeakerAccs spk_stats(am_sgmm, rand_prune);

    RandomAccessSgmmGauPostReader gpost_reader(gpost_rspecifier);

    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);

    BaseFloatVectorWriter vecs_writer(vecs_wspecifier);

    double tot_impr = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_no_gpost = 0, num_other_error = 0;

    if (!spk2utt_rspecifier.empty()) {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        spk_stats.Clear();
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(spk)) {
            spk_vars.v_s = spkvecs_reader.Value(spk);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << spk;
          }
        }  // else spk_vars is "empty"

        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            continue;
          }
          if (!gpost_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_gpost++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const SgmmGauPost &gpost = gpost_reader.Value(utt);
          if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
            KALDI_WARN << "gpost vector has wrong size " << (gpost.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          AccumulateForUtterance(feats, gpost, trans_model, am_sgmm, spk_vars, &spk_stats);
          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_sgmm.SpkSpaceDim(), kSetZero);
          if (spk_vars.v_s.Dim() != 0) spk_vec.CopyFromVec(spk_vars.v_s);
          spk_stats.Update(min_count, &spk_vec, &impr, &spk_tot_t);
          vecs_writer.Write(spk, spk_vec);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from speaker vector is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.\n";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!gpost_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posts for utterance "
                     << utt;
          num_no_gpost++;
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
          }
        }  // else spk_vars is "empty"
        const SgmmGauPost &gpost = gpost_reader.Value(utt);

        if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
          KALDI_WARN << "gpost has wrong size " << (gpost.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }
        num_done++;

        spk_stats.Clear();

        AccumulateForUtterance(feats, gpost, trans_model, am_sgmm, spk_vars, &spk_stats);

        BaseFloat impr, utt_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_sgmm.SpkSpaceDim(), kSetZero);
          if (spk_vars.v_s.Dim() != 0) spk_vec.CopyFromVec(spk_vars.v_s);
          spk_stats.Update(min_count, &spk_vec, &impr, &utt_tot_t);
          vecs_writer.Write(utt, spk_vec);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from speaker vectors is "
                  << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.";
        tot_impr += impr;
        tot_t += utt_tot_t;
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_gpost
              << " with no gposts, " << num_other_error << " with other errors.";
    KALDI_LOG << "Overall auxf impr per frame is " << (tot_impr / tot_t)
              << " over " << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

