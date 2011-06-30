// gmmbin/gmm-post-to-gpost.cc

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


int main(int argc, char *argv[])
{
  using namespace kaldi;
  try {
    const char *usage =
        "Convert state-level posteriors to Gaussian-level posteriors\n"
        "Usage:  gmm-post-to-gpost [options] <model-in> <feature-rspecifier> <posteriors-rspecifier> "
        "<gpost-wspecifier>\n"
        "e.g.: \n"
        " gmm-post-to-gpost 1.mdl scp:train.scp ark:1.post ark:1.gpost\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        gpost_wspecifier = po.GetArg(4);

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

    double tot_like = 0.0;
    double tot_t = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    GauPostWriter gpost_writer(gpost_wspecifier);

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!posteriors_reader.HasKey(key)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);
        GauPost gpost(posterior.size());

        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Posterior vector has wrong size "<< (posterior.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        for (size_t i = 0; i < posterior.size(); i++) {
          gpost[i].resize(posterior[i].size());
          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;
            const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
            gpost[i][j].first = tid;
            BaseFloat like =
                gmm.ComponentPosteriors(mat.Row(i), &(gpost[i][j].second));
            gpost[i][j].second.Scale(weight);
            tot_like_this_file += like * weight;
            tot_weight += weight;
          }
        }
        KALDI_VLOG(1) << "Average like for this file is "
                      << (tot_like_this_file/tot_weight) << " over "
                      << tot_weight <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += tot_weight;
        gpost_writer.Write(key, gpost);
      }
    }
    KALDI_LOG << "Num frames " << tot_t
              << ", avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << '\n';

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    KALDI_LOG << "Done converting post to gpost\n";
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


