// nnet/nnet-pdf-prior.cc

// Copyright 2013  Brno University of Technology (Author: Karel Vesely);
//                 Arnab Ghoshal

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

#include "nnet/nnet-pdf-prior.h"

namespace kaldi {
namespace nnet1 {

PdfPrior::PdfPrior(const PdfPriorOptions &opts)
    : prior_scale_(opts.prior_scale) {
  if (opts.class_frame_counts == "") {
    // class_frame_counts is empty, the PdfPrior is deactivated...
    // (for example when 'nnet-forward' generates bottleneck features)
    return;
  }

  KALDI_LOG << "Computing pdf-priors from : " << opts.class_frame_counts;

  Vector<double> frame_counts, rel_freq, log_priors;
  {
    Input in;
    in.OpenTextMode(opts.class_frame_counts);
    frame_counts.Read(in.Stream(), false);
    in.Close();
  }

  // get relative frequencies,
  rel_freq = frame_counts;
  rel_freq.Scale(1.0/frame_counts.Sum());

  // get the log-prior,
  log_priors = rel_freq;
  log_priors.Add(1e-20);
  log_priors.ApplyLog();

  // Make the priors for classes with low counts +inf (i.e. -log(0))
  // such that the classes have 0 likelihood (i.e. -inf log-likelihood).
  // We use sqrt(FLT_MAX) instead of -kLogZeroFloat to prevent NANs
  // from appearing in computation.
  int32 num_floored = 0;
  for (int32 i = 0; i < log_priors.Dim(); i++) {
    if (rel_freq(i) < opts.prior_floor) {
      log_priors(i) = sqrt(FLT_MAX);
      num_floored++;
    }
  }
  KALDI_LOG << "Floored " << num_floored << " pdf-priors "
            << "(hard-set to " << sqrt(FLT_MAX)
            << ", which disables DNN output when decoding)";

  // sanity check,
  KALDI_ASSERT(KALDI_ISFINITE(log_priors.Sum()));

  // push to GPU,
  log_priors_ = Vector<BaseFloat>(log_priors);
}


void PdfPrior::SubtractOnLogpost(CuMatrixBase<BaseFloat> *llk) {
  if (log_priors_.Dim() == 0) {
    KALDI_ERR << "--class-frame-counts is empty: Cannot initialize priors "
              << "without the counts.";
  }
  if (log_priors_.Dim() != llk->NumCols()) {
    KALDI_ERR << "Dimensionality mismatch,"
              << " class_frame_counts " << log_priors_.Dim()
              << " pdf_output_llk " << llk->NumCols();
  }
  llk->AddVecToRows(-prior_scale_, log_priors_);
}

}  // namespace nnet1
}  // namespace kaldi
