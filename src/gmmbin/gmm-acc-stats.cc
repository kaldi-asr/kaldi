// gmmbin/gmm-acc-stats.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate stats for GMM training (reading in posteriors).\n"
        "Usage:  gmm-acc-stats [options] <model-in> <feature-rspecifier>"
        "<posteriors-rspecifier> <stats-out>\n"
        "e.g.: \n"
        " gmm-acc-stats 1.mdl scp:train.scp ark:1.post 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string update_flags_str = "mvwt"; // note: t is ignored, we acc
    // transition stats regardless.
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters will be "
                "updated: subset of mvwt.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);


    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    AccumAmDiagGmm gmm_accs;
    gmm_accs.Init(am_gmm, StringToGmmFlags(update_flags_str));

    double tot_like = 0.0;
    double tot_t = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!posteriors_reader.HasKey(key)) {
        KALDI_WARN << "Could not find posteriors for utterance " << key;
        num_err++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);

        if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
          KALDI_WARN << "Posterior vector has wrong size " 
                     << (posterior.size()) << " vs. "
                     << (mat.NumRows());
          num_err++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        Posterior pdf_posterior;
        ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
        for (size_t i = 0; i < posterior.size(); i++) {
          // Accumulates for GMM.
          for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
            int32 pdf_id = pdf_posterior[i][j].first;
            BaseFloat weight = pdf_posterior[i][j].second;
            tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, mat.Row(i), pdf_id, weight)
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
        if (num_done % 50 == 0) {
          KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
                    << key << " avg. like is " << (tot_like_this_file/tot_weight)
                    << " over " << tot_weight <<" frames.";
        }
        tot_like += tot_like_this_file;
        tot_t += tot_weight;
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";
    
    KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


