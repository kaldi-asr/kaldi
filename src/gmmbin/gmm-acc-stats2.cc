// gmmbin/gmm-acc-stats2.cc

// Copyright 2009-2012  Johns Hopkins University (Author: Daniel Povey)

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
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Accumulate stats for GMM training (from posteriors)\n"
        "This version writes two accumulators (e.g. num and den),\n"
        "and puts the positive accumulators in num, negative in den\n"
        "Usage:  gmm-acc-stats2 [options] <model> <feature-rspecifier>"
        "<posteriors-rspecifier> <num-stats-out> <den-stats-out>\n"
        "e.g.:\n"
        "gmm-acc-stats 1.mdl \"$feats\" ark:1.post 1.num_acc 1.den_acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string update_flags_str = "mvwt"; // note: t is ignored, we acc
    // transition stats regardless.
    po.Register("binary", &binary, "Write stats in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters to "
                "update: subset of mvwt.");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        num_accs_wxfilename = po.GetArg(4),
        den_accs_wxfilename = po.GetArg(5);

    
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }
    
    Vector<double> num_trans_accs, den_trans_accs;
    trans_model.InitStats(&num_trans_accs);
    trans_model.InitStats(&den_trans_accs);
    AccumAmDiagGmm num_gmm_accs, den_gmm_accs;
    num_gmm_accs.Init(am_gmm, StringToGmmFlags(update_flags_str));
    den_gmm_accs.Init(am_gmm, StringToGmmFlags(update_flags_str));

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);


    BaseFloat tot_like = 0.0, tot_weight = 0.0;
    // tot_like is total weighted likelihood (note: weighted
    // by both +ve and -ve numbers)
    // tot_t is total weight in posteriors (will often be about zero).
    int64 tot_frames = 0.0; 
    
    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!posteriors_reader.HasKey(key)) {
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

        BaseFloat tot_like_this_file = 0.0, tot_weight_this_file = 0.0;

        num_done++;
        for (size_t i = 0; i < posterior.size(); i++) {
          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;
            trans_model.Accumulate(fabs(weight), tid,
                                   (weight > 0.0 ?
                                    &num_trans_accs : &den_trans_accs));
            tot_like_this_file +=
                (weight > 0.0 ? &num_gmm_accs : &den_gmm_accs) ->
                AccumulateForGmm(am_gmm, mat.Row(i), pdf_id, fabs(weight)) * weight;
            tot_weight_this_file += weight;
          }
        }
        tot_like += tot_like_this_file;
        tot_weight += tot_weight_this_file;
        tot_frames += static_cast<int32>(posterior.size());
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " had errors.";
    
    KALDI_LOG << "Overall weighted acoustic likelihood per frame was "
              << (tot_like/tot_frames) << " over " << tot_frames << " frames;"
              << " average weight per frame was " << (tot_weight / tot_frames);

    {
      Output ko(num_accs_wxfilename, binary);
      num_trans_accs.Write(ko.Stream(), binary);
      num_gmm_accs.Write(ko.Stream(), binary);
    }
    {
      Output ko(den_accs_wxfilename, binary);
      den_trans_accs.Write(ko.Stream(), binary);
      den_gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
