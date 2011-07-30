// gmmbin/gmm-acc-hlda.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "transform/hlda.h"




int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate HLDA statistics\n"
        "Usage:  gmm-acc-hlda [options] <model-in> <orig-transform-in> <orig-feature-rspecifier> <posteriors-rspecifier> <stats-out>\n"
        "Note: orig-transform-in must be the current truncated HLDA transform (e.g. from LDA)."
        "e.g.: \n"
        " gmm-acc-hlda 1.mdl 1.hlda \"ark:splice-feats scp:train.scp |\" ark:1.post 1.hacc\n";

    ParseOptions po(usage);
    bool binary = false;
    BaseFloat speedup = 1.0;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("speedup", &speedup, "Proportion of data to accumulate full HLDA stats with");
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        cur_transform_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        posteriors_rspecifier = po.GetArg(4),
        accs_wxfilename = po.GetArg(5);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_gmm.Read(is.Stream(), binary);
    }

    Matrix<BaseFloat> cur_transform;
    {
      bool binary;
      Input is(cur_transform_filename, &binary);
      cur_transform.Read(is.Stream(), binary);
      KALDI_ASSERT(cur_transform.NumRows() == am_gmm.Dim() &&
                   "Transform num-rows must match model dim (need truncated transform)");
    }
    HldaAccsDiagGmm hlda_accs(am_gmm, cur_transform.NumCols(), speedup);

    double tot_like = 0.0;
    double tot_t = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!posteriors_reader.HasKey(key)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);

        // Compute transformed features (need them in order to compute
        // Gaussian-level posteriors)
        Matrix<BaseFloat> transformed_mat(mat.NumRows(), am_gmm.Dim());
        transformed_mat.AddMatMat(1.0, mat, kNoTrans, cur_transform, kTrans, 0.0);

        if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
          KALDI_WARN << "Posterior vector has wrong size "<< (posterior.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        for (size_t i = 0; i < posterior.size(); i++) {
          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;

            Vector<BaseFloat> posteriors;
            const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
            tot_like_this_file +=
                weight * gmm.ComponentPosteriors(transformed_mat.Row(i),
                                                 &posteriors);
            tot_weight += weight;
            posteriors.Scale(weight);
            hlda_accs.AccumulateFromPosteriors(pdf_id,
                                               gmm,
                                               mat.Row(i),
                                               posteriors);
          }
        }
        KALDI_LOG << "Average like for this file is "
                  << (tot_like_this_file/tot_weight) << " over "
                  << tot_weight <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += tot_weight;
        if (num_done % 10 == 0)
          KALDI_LOG << "Avg like per frame so far is "
                    << (tot_like/tot_t);
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";
    
    {
      Output ko(accs_wxfilename, binary);
      hlda_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


