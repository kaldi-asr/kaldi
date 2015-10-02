// sgmm2/am-sgmm2-project.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_SGMM2_AM_SGMM2_PROJECT_H_
#define KALDI_SGMM2_AM_SGMM2_PROJECT_H_

#include <vector>
#include <queue>

#include "sgmm2/am-sgmm2.h"

namespace kaldi {

class Sgmm2Project {
  // This class essentially functions as a namespace for some functions;
  // it's a friend of AmSgmm.h.  It relates to "predictive" SGMMs.  This
  // hasn't been written up yet.  We don't make any functions const or
  // static, because there are no member variables.
 public:

  // If inv_lda_mllt is the matrix that projects from the space the SGMM is
  // in, typically back to the spliced-MFCC space, and begin_dim and end_dim
  // represent the range of dims we want to model, then "projection" will be
  // a matrix, applied *after* the "inv_lda_mllt" matrix, that projects from
  // the raw splice-MFCC features to the space we want to model.  This matrix
  // is of dimension e.g. 40 x 117, and omits the space that the model's states
  // all treat the same.
  void ComputeProjection(const AmSgmm2 &sgmm,
                         const Matrix<BaseFloat> &inv_lda_mllt,
                         int32 begin_dim,
                         int32 end_dim, // last dim plus one that we keep.
                         Matrix<BaseFloat> *projection);

  // This function applies the feature-space projection to the SGMM.
  // The matrix "total_projection" is the product of the "projection" matrix
  // of ComputeProjection times the "inv_lda_mllt" matrix.  It actually
  // projects from a larger dimension than the current SGMM.  We treat
  // the means as if extended with zeros, and the covariances as if
  // extended with a unit matrix.
  void ApplyProjection(const Matrix<BaseFloat> &total_projection,
                       AmSgmm2 *sgmm);
                         
 private:
  // Computes statistics for LDA, in the SGMM's feature space.
  // This only needs to be approximate, so we use stats based
  // on the means in the UBM.
  void ComputeLdaStats(const FullGmm &full_ubm,
                       SpMatrix<double> *between_covar,
                       SpMatrix<double> *within_covar);

  void ProjectVariance (const Matrix<double> &total_projection,
                        bool inverse,
                        SpMatrix<double> *variance);
  
  void ProjectVariance (const Matrix<double> &total_projection,
                        bool inverse,
                        SpMatrix<float> *variance);
  
  void ComputeLdaTransform(const SpMatrix<double> &B,
                           const SpMatrix<double> &W,
                           int32 dim_to_retain, 
                           Matrix<double> *Projection);
  
};



} // end namespace kaldi

#endif  // KALDI_SGMM2_AM_SGMM2_PROJECT_H_
