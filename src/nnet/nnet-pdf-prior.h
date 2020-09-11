// nnet/nnet-pdf-prior.h

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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

#ifndef KALDI_NNET_NNET_PDF_PRIOR_H_
#define KALDI_NNET_NNET_PDF_PRIOR_H_

#include <cfloat>
#include <string>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

namespace kaldi {
namespace nnet1 {

struct PdfPriorOptions {
  std::string class_frame_counts;
  BaseFloat prior_scale;
  BaseFloat prior_floor;

  PdfPriorOptions():
    class_frame_counts(""),
    prior_scale(1.0),
    prior_floor(1e-10)
  { }

  void Register(OptionsItf *opts) {
    opts->Register("class-frame-counts", &class_frame_counts,
                   "Vector with frame-counts of pdfs to compute log-priors."
                   " (priors are typically subtracted from log-posteriors"
                   " or pre-softmax activations)");
    opts->Register("prior-scale", &prior_scale,
                   "Scaling factor to be applied on pdf-log-priors");
    opts->Register("prior-floor", &prior_floor,
                   "Flooring constant for prior probability "
                   "(i.e. label rel. frequency)");
  }
};

class PdfPrior {
 public:
  /// Initialize pdf-prior from options
  explicit PdfPrior(const PdfPriorOptions &opts);

  /// Subtract pdf priors from log-posteriors to get pseudo log-likelihoods
  void SubtractOnLogpost(CuMatrixBase<BaseFloat> *llk);

 private:
  BaseFloat prior_scale_;
  CuVector<BaseFloat> log_priors_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(PdfPrior);
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_PDF_PRIOR_H_
