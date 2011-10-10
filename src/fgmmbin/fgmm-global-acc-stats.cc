// fgmmbin/fgmm-global-acc-stats.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    typedef kaldi::BaseFloat BaseFloat;

    const char *usage =
        "Accumulate stats for training a full-covariance GMM.\n"
        "Usage:  fgmm-global-acc-stats [options] <model-in> <feature-rspecifier> "
        "<stats-out>\n"
        "e.g.: fgmm-global-acc-stats 1.mdl scp:train.scp 1.acc\n";

    kaldi::ParseOptions po(usage);
    bool binary = true;
    int32 diag_gmm_nbest = 0;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("diag-gmm-nbest", &diag_gmm_nbest, "If nonzero, prune likelihood computation withdiagonal version of GMM, to this many indices.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        accs_wxfilename = po.GetArg(3);

    kaldi::FullGmm fgmm;
    {
      bool binary_read;
      kaldi::Input is(model_filename, &binary_read);
      fgmm.Read(is.Stream(), binary_read);
    }
    kaldi::DiagGmm dgmm;
    if (diag_gmm_nbest != 0)
      dgmm.CopyFromFullGmm(fgmm);

    kaldi::AccumFullGmm gmm_accs;
    gmm_accs.Resize(fgmm, kaldi::kGmmAll);

    double tot_like = 0.0;
    double tot_frames = 0.0;

    kaldi::SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    int32 num_done = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const kaldi::Matrix<kaldi::BaseFloat> &mat = feature_reader.Value();
      kaldi::BaseFloat file_like = 0.0, file_frames = mat.NumRows();

      for (int32 i = 0; i < file_frames; ++i) {
        if (diag_gmm_nbest == 0 || diag_gmm_nbest >= fgmm.NumGauss()) {
          file_like += gmm_accs.AccumulateFromFull(fgmm, mat.Row(i), 1.0);
        } else {
          kaldi::Vector<BaseFloat> loglikes;
          dgmm.LogLikelihoods(mat.Row(i), &loglikes);
          kaldi::Vector<BaseFloat> loglikes_copy(loglikes);
          int32 ngauss = fgmm.NumGauss();
          BaseFloat *ptr = loglikes_copy.Data();
          std::nth_element(ptr, ptr+ngauss-diag_gmm_nbest, ptr+ngauss);
          BaseFloat thresh = ptr[ngauss-diag_gmm_nbest];
          kaldi::Vector<BaseFloat> full_loglikes(ngauss, kaldi::kUndefined);
          full_loglikes.Set(-std::numeric_limits<BaseFloat>::infinity());
          for (int32 g = 0; g < ngauss; g++)
            if (loglikes(g) >= thresh)
              full_loglikes(g) = fgmm.ComponentLogLikelihood(mat.Row(i), g);
          file_like += full_loglikes.ApplySoftMax();
          gmm_accs.AccumulateFromPosteriors(mat.Row(i), full_loglikes);
        }
      }
      KALDI_VLOG(1) << "File '" << key << "': Average likelihood = "
                    << (file_like/file_frames) << " over "
                    << file_frames <<" frames.";
      tot_like += file_like;
      tot_frames += file_frames;
      num_done++;
      if (num_done % 100 == 0) {
        KALDI_VLOG(2) << "Average likelihood per frame after " << num_done
                      << " files is " << (tot_like/tot_frames);
      }
    }
    KALDI_LOG << "Done " << num_done << " files.";
    KALDI_LOG << "Overall likelihood per "
              << "frame = " << (tot_like/tot_frames) << " over " << tot_frames
              << "frames.";

    {
      kaldi::Output ko(accs_wxfilename, binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs to " << accs_wxfilename;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
