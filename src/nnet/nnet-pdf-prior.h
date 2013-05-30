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
  
  PdfPriorOptions() : class_frame_counts(""), 
                      prior_scale(1.0) {};

  void Register(ParseOptions *po) {
    po->Register("class-frame-counts", &class_frame_counts, 
                 "Vector with frame-counts of pdfs to compute log-priors."
                 " (priors are typically subtracted from log-posteriors"
                 " or pre-softmax activations)");
    po->Register("prior-scale", &prior_scale, 
                 "Scaling factor to be applied on pdf-log-priors");
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
      }
      //normalize
      { 
        BaseFloat sum = tmp_priors.Sum();
        tmp_priors.Scale(1.0/sum);
        { //make sure we don't get hard zeros as input to log
          tmp_priors.Add(DBL_MIN); 
          sum = tmp_priors.Sum();
          tmp_priors.Scale(1.0/sum);
          for(int32 i=0; i<tmp_priors.Dim(); i++) {
            KALDI_ASSERT(tmp_priors(i) > 0.0);
          }
        }
      }
      //apply log
      tmp_priors.ApplyLog();
      for(int32 i=0; i<tmp_priors.Dim(); i++) {
        KALDI_ASSERT(tmp_priors(i) != kLogZeroDouble);
      }
      //push priors to GPU
      Vector<BaseFloat> tmp_priors_f(tmp_priors);
      log_priors_.Resize(tmp_priors.Dim());
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
