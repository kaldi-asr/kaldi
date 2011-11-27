// sgmmbin/sgmm-acc-stats.cc

// Copyright 2009-2011   Saarland University
// Author:  Arnab Ghoshal

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
        "Accumulate stats for SGMM training.\n"
        "Usage: sgmm-acc-stats [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <stats-out>\n"
        "e.g.: sgmm-acc-stats 1.mdl 1.ali scp:train.scp 'ark:ali-to-post 1.ali ark:-|' 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    std::string update_flags_str = "vMNwcSt";
    BaseFloat rand_prune = 1.0e-05;
    SgmmGselectConfig sgmm_opts;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices (rspecifier)");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("rand-prune", &rand_prune, "Pruning threshold for posteriors");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to update: subset of vMNwcS.");
    sgmm_opts.Register(&po);
    po.Read(argc, argv);

    kaldi::SgmmUpdateFlagsType acc_flags = StringToSgmmUpdateFlags(update_flags_str);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    // Initialize the readers before the model, as the model can
    // be large, and we don't want to call fork() after reading it if
    // virtual memory may be low.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);
    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);
    
    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_sgmm.Read(is.Stream(), binary);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    MleAmSgmmAccs sgmm_accs(rand_prune);
    sgmm_accs.ResizeAccumulators(am_sgmm, acc_flags);

    double tot_like = 0.0;
    double tot_t = 0;

    kaldi::SgmmPerFrameDerivedVars per_frame_vars;

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!posteriors_reader.HasKey(utt)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(utt);

        bool have_gselect  = !gselect_rspecifier.empty()
            && gselect_reader.HasKey(utt)
            && gselect_reader.Value(utt).size() == mat.NumRows();
        if (!gselect_rspecifier.empty() && !have_gselect)
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
        std::vector<std::vector<int32> > empty_gselect;
        const std::vector<std::vector<int32> > *gselect =
            (have_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (posterior.size()) <<
              " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        string utt_or_spk;
        if (utt2spk_rspecifier.empty())  utt_or_spk = utt;
        else {
          if (!utt2spk_reader.HasKey(utt)) {
            KALDI_WARN << "Utterance " << utt << " not present in utt2spk map; "
                       << "skipping this utterance.";
            num_other_error++;
            continue;
          } else {
            utt_or_spk = utt2spk_reader.Value(utt);
          }
        }

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt_or_spk)) {
            spk_vars.v_s = spkvecs_reader.Value(utt_or_spk);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt_or_spk;
          }
        }  // else spk_vars is "empty"

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        for (size_t i = 0; i < posterior.size(); i++) {
          std::vector<int32> this_gselect;
          if (!gselect->empty()) this_gselect = (*gselect)[i];
          else am_sgmm.GaussianSelection(sgmm_opts, mat.Row(i), &this_gselect);
          am_sgmm.ComputePerFrameVars(mat.Row(i), this_gselect, spk_vars, 0.0,
                                      &per_frame_vars);

          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;
            trans_model.Accumulate(weight, tid, &transition_accs);
            tot_like_this_file += sgmm_accs.Accumulate(am_sgmm, per_frame_vars,
                                                       spk_vars.v_s, pdf_id,
                                                       weight, acc_flags)
                                                       * weight;
            tot_weight += weight;
          }
        }

        sgmm_accs.CommitStatsForSpk(am_sgmm, spk_vars.v_s);  // no harm doing it per utterance.
        
        KALDI_VLOG(2) << "Average like for this file is "
                      << (tot_like_this_file/tot_weight) << " over "
                      << tot_weight <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += tot_weight;
        if (num_done % 50 == 0) {
          KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
                    << utt << " avg. like is "
                    << (tot_like_this_file/tot_weight)
                    << " over " << tot_weight <<" frames.";
        }
      }
    }
    KALDI_LOG << "Overall like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      sgmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


