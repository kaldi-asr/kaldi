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
    //Empty file with counts is not an error, 
    //there are cases when PdfPrior is not active 
    //(example: nnet-forward over feature transform, bn-feature extractor)
    return;

    //KALDI_ERR << "--class-frame-counts is empty: Cannot initialize priors "
    //          << "without the counts.";
  }

  Vector<double> tmp_priors;
  KALDI_LOG << "Computing pdf-priors from : " << opts.class_frame_counts;

  {
    Input in;
    in.OpenTextMode(opts.class_frame_counts);
    tmp_priors.Read(in.Stream(), false);
    in.Close();
  }

  int32 prior_dim = tmp_priors.Dim();
  Vector<BaseFloat> tmp_mask(prior_dim, kSetZero);
  int32 num_cutoff = 0;
  for (int32 i = 0; i < prior_dim; i++) {
    if (tmp_priors(i) < opts.prior_cutoff) {
      tmp_priors(i) = opts.prior_cutoff;
      tmp_mask(i) = FLT_MAX/2;  // not using -kLogZeroFloat to prevent NANs
      num_cutoff++;
    }
  }
  if (num_cutoff > 0) {
    KALDI_WARN << num_cutoff << " out of " << prior_dim << " classes have counts"
               << " lower than " << opts.prior_cutoff;
  }

  double sum = tmp_priors.Sum();
  tmp_priors.Scale(1.0 / sum);
  tmp_priors.ApplyLog();
  for (int32 i = 0; i < prior_dim; i++) {
    KALDI_ASSERT(tmp_priors(i) != kLogZeroDouble);
  }

  // Make the priors for classes with low counts +inf (i.e. -log(0)) such that
  // the classes have 0 likelihood (i.e. -inf log-likelihood). We use FLT_MAX/2
  // instead of -kLogZeroFloat to prevent NANs from appearing in computation.
  Vector<BaseFloat> tmp_priors_f(tmp_priors);
  tmp_priors_f.AddVec(1.0, tmp_mask);

  // push priors to GPU
  log_priors_.Resize(prior_dim);
  log_priors_.CopyFromVec(tmp_priors_f);
}


void PdfPrior::SubtractOnLogpost(CuMatrixBase<BaseFloat> *llk) {
  if(log_priors_.Dim() == 0) {
    KALDI_ERR << "--class-frame-counts is empty: Cannot initialize priors "
              << "without the counts.";
  }
  if(log_priors_.Dim() != llk->NumCols()) {
    KALDI_ERR << "Dimensionality mismatch,"
              << " class_frame_counts " << log_priors_.Dim()
              << " pdf_output_llk " << llk->NumCols();
  }
  llk->AddVecToRows(-prior_scale_, log_priors_);
}

}  // namespace nnet1
}  // namespace kaldi
