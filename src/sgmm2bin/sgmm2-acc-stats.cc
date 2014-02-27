// sgmm2bin/sgmm2-acc-stats.cc

// Copyright 2009-2012   Saarland University (Author:  Arnab Ghoshal),
//                       Johns Hopkins University (Author:  Daniel Povey)
//                2014   Guoguo Chen

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
#include "sgmm2/am-sgmm2.h"
#include "hmm/transition-model.h"
#include "sgmm2/estimate-am-sgmm2.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate stats for SGMM training.\n"
        "Usage: sgmm2-acc-stats [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <stats-out>\n"
        "e.g.: sgmm2-acc-stats --gselect=ark:gselect.ark 1.mdl 1.ali scp:train.scp 'ark:ali-to-post 1.ali ark:-|' 1.acc\n"
        "(note: gselect option is mandatory)\n";
        
    ParseOptions po(usage);
    bool binary = true;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    std::string update_flags_str = "vMNwcSt";
    BaseFloat rand_prune = 1.0e-05;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices (rspecifier)");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("rand-prune", &rand_prune, "Pruning threshold for posteriors");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to accumulate "
                "stats for: subset of vMNwcS.");

    po.Read(argc, argv);

    kaldi::SgmmUpdateFlagsType acc_flags = StringToSgmmUpdateFlags(update_flags_str);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect option is mandatory.";
    
    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    int32 num_done = 0, num_err = 0;
    Vector<double> transition_accs;
    MleAmSgmm2Accs sgmm_accs(rand_prune);

    { // this anonymous scope is to ensure deallocation of unnecessary stuff
      // while we're writing out the accs, which could be a long time for large
      // models.
      
      // Initialize the readers before the model, as the model can
      // be large, and we don't want to call fork() after reading it if
      // virtual memory may be low.
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
      RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
      RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                             utt2spk_rspecifier);
      RandomAccessTokenReader utt2spk_map(utt2spk_rspecifier);
      
      AmSgmm2 am_sgmm;
      TransitionModel trans_model;
      {
        bool binary;
        Input ki(model_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_sgmm.Read(ki.Stream(), binary);
      }


      trans_model.InitStats(&transition_accs);
      sgmm_accs.ResizeAccumulators(am_sgmm, acc_flags, (spkvecs_rspecifier!=""));

      double tot_like = 0.0;
      double tot_t = 0;

      kaldi::Sgmm2PerFrameDerivedVars per_frame_vars;
      std::string cur_spk;
      Sgmm2PerSpkDerivedVars spk_vars;
              
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        std::string spk = utt;
        if (!utt2spk_rspecifier.empty()) {
          if (!utt2spk_map.HasKey(utt)) {
            KALDI_WARN << "utt2spk map does not have value for " << utt
                       << ", ignoring this utterance.";
            continue;
          } else { spk = utt2spk_map.Value(utt); }
        }

        if (spk != cur_spk && cur_spk != "")
          sgmm_accs.CommitStatsForSpk(am_sgmm, spk_vars);        
        
        if (spk != cur_spk || spk_vars.Empty()) {
          spk_vars.Clear();
          if (spkvecs_reader.IsOpen()) {
            if (spkvecs_reader.HasKey(utt)) {
              spk_vars.SetSpeakerVector(spkvecs_reader.Value(utt));
              am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
            } else {
              KALDI_WARN << "Cannot find speaker vector for " << utt;
              num_err++;
              continue;
            }
          } // else spk_vars is "empty"
        }
        
        cur_spk = spk;
        
        const Matrix<BaseFloat> &features = feature_reader.Value();
        if (!posteriors_reader.HasKey(utt) ||
            posteriors_reader.Value(utt).size() != features.NumRows()) {
          KALDI_WARN << "No posterior info available for utterance "
                     << utt << " (or wrong size)";
          num_err++;
          continue;
        }
        const Posterior &posterior = posteriors_reader.Value(utt);
      
        if (!gselect_reader.HasKey(utt)
            && gselect_reader.Value(utt).size() != features.NumRows()) {
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
          num_err++;
        }
        const std::vector<std::vector<int32> > &gselect =
            gselect_reader.Value(utt);

        num_done++;
      
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        Posterior pdf_posterior;
        ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
        for (size_t i = 0; i < posterior.size(); i++) {
          am_sgmm.ComputePerFrameVars(features.Row(i), gselect[i], spk_vars,
                                      &per_frame_vars);
          // Accumulates for SGMM.
          for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
            int32 pdf_id = pdf_posterior[i][j].first;
            BaseFloat weight = pdf_posterior[i][j].second;
            tot_like_this_file += sgmm_accs.Accumulate(am_sgmm, per_frame_vars,
                                                       pdf_id, weight, &spk_vars)
                * weight;
            tot_weight += weight;
          }

          // Accumulates for transitions.
          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first;
            BaseFloat weight = posterior[i][j].second;
            trans_model.Accumulate(weight, tid, &transition_accs);
          }
        }
        
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
      sgmm_accs.CommitStatsForSpk(am_sgmm, spk_vars); // commit stats for
      // last speaker.
      
      KALDI_LOG << "Overall like per frame (Gaussian only) = "
                << (tot_like/tot_t) << " over " << tot_t << " frames.";

      KALDI_LOG << "Done " << num_done << " files, " << num_err
                << " with errors.";
    } 

    {
      Output ko(accs_wxfilename, binary);
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


