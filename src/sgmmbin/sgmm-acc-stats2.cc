// sgmmbin/sgmm-acc-stats2.cc

// Copyright 2009-2012   Saarland University (Author:  Arnab Ghoshal),
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
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate numerator and denominator stats for discriminative training\n"
        "of SGMMs (input is posteriors of mixed sign)\n"
        "Usage: sgmm-acc-stats2 [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <num-stats-out> <den-stats-out>\n"
        "e.g.: sgmm-acc-stats2 1.mdl 1.ali scp:train.scp ark:1.posts num.acc den.acc\n";

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
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to accumulate "
                "stats for: subset of vMNwcS.");
    sgmm_opts.Register(&po);

    po.Read(argc, argv);

    kaldi::SgmmUpdateFlagsType acc_flags = StringToSgmmUpdateFlags(update_flags_str);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        num_accs_wxfilename = po.GetArg(4),
        den_accs_wxfilename = po.GetArg(5);
    

    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    // Initialize the readers before the model, as the model can
    // be large, and we don't want to call fork() after reading it if
    // virtual memory may be low.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
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

    Vector<double> num_transition_accs, den_transition_accs;
    if (acc_flags & kaldi::kSgmmTransitions) {
      trans_model.InitStats(&num_transition_accs);
      trans_model.InitStats(&den_transition_accs);
    }
    MleAmSgmmAccs num_sgmm_accs(rand_prune), den_sgmm_accs(rand_prune);
    num_sgmm_accs.ResizeAccumulators(am_sgmm, acc_flags);
    den_sgmm_accs.ResizeAccumulators(am_sgmm, acc_flags);   

    double tot_like = 0.0, tot_weight = 0.0, tot_abs_weight = 0.0;
    int64 tot_frames = 0;

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

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
            continue;
            num_other_error++;
          }
        }  // else spk_vars is "empty"

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight_this_file = 0.0,
            tot_abs_weight_this_file = 0.0;
        
        for (size_t i = 0; i < posterior.size(); i++) {
          std::vector<int32> this_gselect;
          if (!gselect->empty()) this_gselect = (*gselect)[i];
          else am_sgmm.GaussianSelection(sgmm_opts, mat.Row(i), &this_gselect);
          am_sgmm.ComputePerFrameVars(mat.Row(i), this_gselect, spk_vars, 0.0,
                                      &per_frame_vars);

          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second,
                abs_weight = std::abs(weight);
            
            if (acc_flags & kaldi::kSgmmTransitions) {
              trans_model.Accumulate(abs_weight, tid,  weight > 0 ?
                                     &num_transition_accs : &den_transition_accs);
            }
            tot_like_this_file +=
                (weight > 0 ? num_sgmm_accs : den_sgmm_accs).Accumulate(
                    am_sgmm, per_frame_vars, spk_vars.v_s, pdf_id,
                    abs_weight, acc_flags)
                * weight;
            tot_weight_this_file += weight;
            tot_abs_weight_this_file += abs_weight;
          }
        }
        num_sgmm_accs.CommitStatsForSpk(am_sgmm, spk_vars.v_s);  // no harm doing it per utterance.
        den_sgmm_accs.CommitStatsForSpk(am_sgmm, spk_vars.v_s);
        
        tot_like += tot_like_this_file;
        tot_weight += tot_weight_this_file;
        tot_abs_weight += tot_abs_weight_this_file;
        tot_frames += posterior.size();
        if (num_done % 50 == 0)
          KALDI_LOG << "Processed " << num_done << " utterances.";
      }
    }
    KALDI_LOG << "Overall weighted acoustic likelihood per frame was "
              << (tot_like/tot_frames) << " over " << tot_frames << " frames; "
              << "average weight per frame is " << (tot_weight/tot_frames)
              << ", average abs(weight) per frame is "
              << (tot_abs_weight/tot_frames);
    
    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";
    
    {
      Output ko(num_accs_wxfilename, binary);
      // TODO(arnab): Ideally, we shouldn't be writing transition accs if not
      // asked for, but that will complicate reading later. To be fixed?
      num_transition_accs.Write(ko.Stream(), binary);
      num_sgmm_accs.Write(ko.Stream(), binary);
    }
    {
      Output ko(den_accs_wxfilename, binary);
      den_transition_accs.Write(ko.Stream(), binary);
      den_sgmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


