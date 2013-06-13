// nnet/nnet-pdf-prior.h

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

#include <cfloat>

namespace kaldi {

struct PdfPriorOptions {
  std::string class_frame_counts;
  BaseFloat prior_scale;
  double prior_floor;
  
  PdfPriorOptions() : class_frame_counts(""), 
                      prior_scale(1.0),
                      prior_floor(1e-10) {};

  void Register(ParseOptions *po) {
    po->Register("class-frame-counts", &class_frame_counts, 
                 "Vector with frame-counts of pdfs to compute log-priors."
                 " (priors are typically subtracted from log-posteriors"
                 " or pre-softmax activations)");
    po->Register("prior-scale", &prior_scale, 
                 "Scaling factor to be applied on pdf-log-priors");
    po->Register("prior-floor", &prior_floor, 
                 "Floor applied to the priors (prevents amplification of noise coming from NN-outputs with no data)");
  }
};

class PdfPrior {
 public:
  /**
   * Initialize pdf-prior from options
   */
  PdfPrior(const PdfPriorOptions &opts) {
    //make local copy of opts
    opts_ = opts;
    //initialize the class
    if(opts_.class_frame_counts == "") {
      KALDI_LOG << "Pdf-priors are not computed, --class-frame-counts is empty";
    } else {
      //show what we are doing..
      KALDI_LOG << "Computing pdf-priors from : " << opts_.class_frame_counts;
      //read vector with frame-counts
      Vector<double> tmp_priors;
      { 
        Input in;
        in.OpenTextMode(opts_.class_frame_counts);
        tmp_priors.Read(in.Stream(), false);
        in.Close();
        { //warn for hard zeros and avoid negative values
          int32 num_hard_zero = 0, num_neg_val = 0;
          for(int32 i=0; i<tmp_priors.Dim(); i++) {
            if(tmp_priors(i) == 0.0) num_hard_zero++;
            if(tmp_priors(i) < 0.0) num_neg_val++;
          }
          if(num_hard_zero) {
            KALDI_WARN << "--class-frame-counts contains " << num_hard_zero << " hard zeros" 
                       << " (" << opts_.class_frame_counts << ")";
          }
          if(num_neg_val) {
            KALDI_ERR << "--class-frame-counts contains " << num_neg_val << " negative values"
                      << " (" << opts_.class_frame_counts << ")";
          }
        }
      }
      //normalize
      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0/sum);
      //apply flooring to the priors, so that noise coming from NN-outputs which had 
      //no training data is not amplified to the extent that decoding breaks.
      int32 prior_dim = tmp_priors.Dim();
      { 
        int32 num_floored = 0;
        for(int32 i=0; i<prior_dim; i++) {
          if(tmp_priors(i) < opts_.prior_floor) {
            tmp_priors(i) = opts_.prior_floor; num_floored++;
          }
        }
        if(num_floored) {
          KALDI_LOG << "floored " << num_floored << "/" << prior_dim 
                    << " priors based on --class-frame-counts " 
                    << opts_.class_frame_counts;
        }
      }
      //apply log
      tmp_priors.ApplyLog();
      for(int32 i=0; i<prior_dim; i++) {
        KALDI_ASSERT(tmp_priors(i) != kLogZeroDouble);
      }
      //push priors to GPU
      Vector<BaseFloat> tmp_priors_f(tmp_priors);
      log_priors_.Resize(prior_dim);
      log_priors_.CopyFromVec(tmp_priors_f);
    }
  }

  /**
   * Subtract pdf priors on matrix with log-posteriors.
   * (ie. convert log-posteriors to log-likelihoods)
   */
  void SubtractOnLogpost(CuMatrix<BaseFloat> *llk) {
    if(opts_.class_frame_counts != "") {
      llk->AddVecToRows(-opts_.prior_scale, log_priors_);
    } else {
      KALDI_ERR << "Cannot subtract log-prior, --class-frame-count not set";
    }
  }

 private:
  PdfPriorOptions opts_;
  CuVector<BaseFloat> log_priors_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(PdfPrior);
};
  
}// namespace kaldi

#endif
