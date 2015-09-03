// sgmmbin/sgmm-acc-stats-gpost.cc

// Copyright 2009-2012   Saarland University (Author: Arnab Ghoshal)
//                       Microsoft Corporation;
//                       Johns Hopkins University (Author: Daniel Povey)

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
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "sgmm/estimate-am-sgmm.h"




int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate stats for SGMM training, given Gaussian-level posteriors\n"
        "Usage: sgmm-acc-stats-gpost [options] <model-in> <feature-rspecifier> "
        "<gpost-rspecifier> <stats-out>\n"
        "e.g.: sgmm-acc-stats-gpost 1.mdl 1.ali scp:train.scp ark, s, cs:- 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string spkvecs_rspecifier, utt2spk_rspecifier;
    std::string update_flags_str = "vMNwcSt";
    BaseFloat rand_prune = 1.0e-05;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("rand-prune", &rand_prune, "Pruning threshold for posteriors");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to update: subset of vMNwcS.");
    po.Read(argc, argv);

    kaldi::SgmmUpdateFlagsType acc_flags = StringToSgmmUpdateFlags(update_flags_str);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        gpost_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    // Initialize the readers before the model, as this can avoid
    // crashes on systems with low virtual memory.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessSgmmGauPostReader gpost_reader(gpost_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    if (acc_flags & kaldi::kSgmmTransitions)
      trans_model.InitStats(&transition_accs);
    MleAmSgmmAccs sgmm_accs(rand_prune);
    sgmm_accs.ResizeAccumulators(am_sgmm, acc_flags);

    double tot_t = 0.0;
    kaldi::SgmmPerFrameDerivedVars per_frame_vars;

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!gpost_reader.HasKey(utt)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const SgmmGauPost &gpost = gpost_reader.Value(utt);

        if (gpost.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (gpost.size()) <<
              " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
            num_other_error++;
            continue;
          }
        }  // else spk_vars is "empty"

        num_done++;
        BaseFloat tot_weight = 0.0;

        for (size_t i = 0; i < gpost.size(); i++) {
          const std::vector<int32> &gselect = gpost[i].gselect;
          am_sgmm.ComputePerFrameVars(mat.Row(i), gselect, spk_vars, 0.0,
                                      &per_frame_vars);

          for (size_t j = 0; j < gpost[i].tids.size(); j++) {
            int32 tid = gpost[i].tids[j],  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);

            BaseFloat weight = gpost[i].posteriors[j].Sum();
            if (acc_flags & kaldi::kSgmmTransitions)
              trans_model.Accumulate(weight, tid, &transition_accs);
            sgmm_accs.AccumulateFromPosteriors(am_sgmm, per_frame_vars,
                                               gpost[i].posteriors[j],
                                               spk_vars.v_s,
                                               pdf_id, acc_flags);
            tot_weight += weight;
          }
        }

        sgmm_accs.CommitStatsForSpk(am_sgmm, spk_vars.v_s);  // no harm doing it per utterance.

        tot_t += tot_weight;
        if (num_done % 50 == 0)
          KALDI_LOG << "Processed " << num_done << " utterances";
      }
    }
    KALDI_LOG << "Overall number of frames is " << tot_t;

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    {
      Output ko(accs_wxfilename, binary);
      // TODO(arnab): Ideally, we shouldn't be writing transition accs if not
      // asked for, but that will complicate reading later. To be fixed?
      transition_accs.Write(ko.Stream(), binary);
      sgmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


