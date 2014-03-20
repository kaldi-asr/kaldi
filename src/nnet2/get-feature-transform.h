// nnet2/get-feature-transform.h

// Copyright 2009-2011  Jan Silovsky
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_GET_FEATURE_TRANSFORM_H_
#define KALDI_NNET2_GET_FEATURE_TRANSFORM_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "transform/lda-estimate.h"

namespace kaldi {

// This file is modified from transform/lda-estimate.h
// It contains a class intended to be used in preconditioning
// data for neural network training.

struct FeatureTransformEstimateOptions {
  bool remove_offset;
  int32 dim;
  BaseFloat within_class_factor;
  BaseFloat max_singular_value;
  FeatureTransformEstimateOptions(): remove_offset(true), dim(200),
                                     within_class_factor(0.001), max_singular_value(5.0) { }
  
  void Register(OptionsItf *po) {
    po->Register("remove-offset", &remove_offset, "If true, output an affine "
                 "transform that makes the projected data mean equal to zero.");
    po->Register("dim", &dim, "Dimension to project to with LDA");
    po->Register("within-class-factor", &within_class_factor, "If 1.0, do "
                 "conventional LDA where the within-class variance will be "
                 "unit in the projected space.  May be set to less than 1.0, "
                 "which scales the features to have less variance, particularly "
                 "for dimensions where between-class variance is small. ");
    po->Register("max-singular-value", &max_singular_value, "If >0, maximum "
                 "allowed singular value of final transform (they are floored "
                 "to this)");
  }    
};

/** Class for computing a feature transform used in neural-networks.
 */
class FeatureTransformEstimate: public LdaEstimate {
 public:

  /// Estimates the LDA transform matrix m.  If Mfull != NULL, it also outputs
  /// the full matrix (without dimensionality reduction), which is useful for
  /// some purposes.  If opts.remove_offset == true, it will output both matrices
  /// with an extra column which corresponds to mean-offset removal (the matrix
  /// should be multiplied by the feature with a 1 appended to give the correct
  /// result, as with other Kaldi transforms.)
  /// "within_cholesky" is a pointer to an SpMatrix that, if non-NULL, will
  /// be set to the Cholesky factor of the within-class covariance matrix.
  /// This is used for perturbing features.
  void Estimate(const FeatureTransformEstimateOptions &opts,
                Matrix<BaseFloat> *M,
                TpMatrix<BaseFloat> *within_cholesky) const;
 protected:
  static void EstimateInternal(const FeatureTransformEstimateOptions &opts,
                               const SpMatrix<double> &total_covar,
                               const SpMatrix<double> &between_covar,
                               const Vector<double> &mean,
                               Matrix<BaseFloat> *M,
                               TpMatrix<BaseFloat> *C);
};


class FeatureTransformEstimateMulti: public FeatureTransformEstimate {
 public:
  /// This is as FeatureTransformEstimate, but for use in
  /// nnet-get-feature-transform-multi.cc, see the usage message
  /// of that program for a description of what it does.
  void Estimate(const FeatureTransformEstimateOptions &opts,
                const std::vector<std::vector<int32> > &indexes,
                Matrix<BaseFloat> *M) const;

 private:
  void EstimateTransformPart(const FeatureTransformEstimateOptions &opts,
                             const std::vector<int32> &indexes,
                             const SpMatrix<double> &total_covar,
                             const SpMatrix<double> &between_covar,
                             const Vector<double> &mean,
                             Matrix<BaseFloat> *M) const;
};



}  // End namespace kaldi

#endif  // KALDI_NNET2_GET_FEATURE_TRANSFORM_H_

