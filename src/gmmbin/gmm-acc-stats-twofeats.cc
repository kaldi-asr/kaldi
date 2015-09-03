// gmmbin/gmm-acc-stats-twofeats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Guoguo Chen
//                2014  Johns Hopkins University (author: Daniel Povey)

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
  try {
    const char *usage =
        "Accumulate stats for GMM training, computing posteriors with one set of features\n"        
        "but accumulating statistics with another.\n"
        "First features are used to get posteriors, second to accumulate stats\n"        
        "Usage:  gmm-acc-stats-twofeats [options] <model-in> <feature1-rspecifier> <feature2-rspecifier> <posteriors-rspecifier> <stats-out>\n"
        "e.g.: \n"
        " gmm-acc-stats-twofeats 1.mdl 1.ali scp:train.scp scp:train_new.scp ark:1.ali 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature1_rspecifier = po.GetArg(2),
        feature2_rspecifier = po.GetArg(3),
        posteriors_rspecifier = po.GetArg(4),
        accs_wxfilename = po.GetArg(5);

    using namespace kaldi;
    typedef kaldi::int32 int32;

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
    int32 new_dim = 0;
    AccumAmDiagGmm gmm_accs;
    // will initialize once we know new_dim.

    double tot_like = 0.0;
    double tot_t = 0.0;

    SequentialBaseFloatMatrixReader feature1_reader(feature1_rspecifier);
    RandomAccessBaseFloatMatrixReader feature2_reader(feature2_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    int32 num_done = 0, num_no2ndfeats = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature1_reader.Done(); feature1_reader.Next()) {
      std::string key = feature1_reader.Key();
      if (!feature2_reader.HasKey(key)) {
        KALDI_WARN << "For utterance " << key << ", second features not present.";
        num_no2ndfeats ++;
      } else if (!posteriors_reader.HasKey(key)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat1 = feature1_reader.Value();
        const Matrix<BaseFloat> &mat2 = feature2_reader.Value(key);
        KALDI_ASSERT(mat1.NumRows() == mat2.NumRows());
        if (new_dim == 0) {
          new_dim = mat2.NumCols();
          gmm_accs.Init(am_gmm, new_dim, kGmmAll);
        }
        const Posterior &posterior = posteriors_reader.Value(key);

        if (posterior.size() != mat1.NumRows()) {
          KALDI_WARN << "Posteriors has wrong size "<< (posterior.size()) << " vs. "<< (mat1.NumRows());
          num_other_error++;
          continue;
        }
        if (mat1.NumRows() != mat2.NumRows()) {
          KALDI_WARN << "Features have mismatched numbers of frames "
                     << mat1.NumRows() << " vs. " << mat2.NumRows();
          num_other_error++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0,
            tot_weight_this_file = 0.0;

        Posterior pdf_posterior;
        ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
        for (size_t i = 0; i < posterior.size(); i++) {
          // Accumulates for GMM.
          for (size_t j = 0; j <pdf_posterior[i].size(); j++) {
            int32 pdf_id = pdf_posterior[i][j].first;
            BaseFloat weight = pdf_posterior[i][j].second;
            tot_like_this_file += weight *
                gmm_accs.AccumulateForGmmTwofeats(am_gmm,
                                                  mat1.Row(i),
                                                  mat2.Row(i),
                                                  pdf_id,
                                                  weight);
            tot_weight_this_file += weight;
          }

          // Accumulates for transitions.
          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first;
            BaseFloat weight = posterior[i][j].second;
            trans_model.Accumulate(weight, tid, &transition_accs);
          }
        }
        KALDI_LOG << "Average like for this file is "
                  << (tot_like_this_file/tot_weight_this_file) << " over "
                  << tot_weight_this_file <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += tot_weight_this_file;
        if (num_done % 10 == 0)
          KALDI_LOG << "Avg like per frame so far is " << (tot_like/tot_t);
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_no2ndfeats
              << " with no second features, " << num_other_error
              << " with other errors.";

    KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


