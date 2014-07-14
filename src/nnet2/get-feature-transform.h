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

/**
   @file
   This file is modified from transform/lda-estimate.h
   It contains a class intended to be used in preconditioning
   data for neural network training.  See the documentation for class
   FeatureTransformEstimate for more details.
*/

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

/**
     Class for computing a feature transform used for preconditioning of the
     training data in neural-networks.

     By preconditioning here, all we really mean is an affine transform of the
     input data-- say if we set up the classification as going from vectors x_i
     to labels y_i, then this would be a linear transform on X, so we replace
     x_i with x'_i = A x_i + b.  The statistics we use to obtain this transform
     are the within-class and between class variance statistics, and the global
     data mean, that we would use to estimate LDA.  When designing this, we had
     a few principles in mind:
        - We want to remove the global mean of the input features (this is
          well established, I think there is a paper by LeCun explaining why
          this is a good thing).
        - We would like the transform to make the training process roughly
          invariant to linear transformations of the input features, meaning
          that whatever linear transformation you apply prior to this transform,
          it should 'undo' it.
        - We want directions in which there is a lot of between-class variance
          to be given a higher variance than directions that have mostly
          within-class variance-- it has been our experience that these
          'nuisance directions' will interfere with the training if they are
          given too large a scaling.
     It is essential to our method that the number of classes is higher than
     the dimension of the input feature space, which is normal for speech
     recognition tasks (~5000 > ~250).

     Basically our method is as follows:

       - First subtract the mean.
       - Get the within-class and between-class stats, as for LDA.
       - Normalize the space as for LDA, so that the within-class covariance
         matrix is unit and the between-class covariance matrix is diagonalized
       - At this stage, if the user asked for dimension reduction then
         reduce the dimension by taking out dimensions with least between-class
         variance [note: the current scripts do not do this by default]
       - Apply a transform that reduces the variance of dimensions
         with low between-class variance, as we'll describe below.
       - Finally, do an SVD of the resulting transform, A = U S V^T, apply a
         maximum to the diagonal elements of the matrix S (e.g. 5.0), and
         reconstruct A' = U S' V^T; this is the final transform.  The point of
         this stage is to stop the transform from 'blowing up' any dimensions of
         the space excessively; this stage was introduced in response to a
         problem we encountered at one point, and I think normally not very many
         dimensions of S end up getting floored.

      We need to explain the step that applies the dimension-specific scaling,
      which we described above as, "Apply a transform that reduces the variance
      of dimensions with low between-class variance".  For a particular
      dimension, let the between-class diagonal covariance element be \lambda_i,
      and the within-class diagonal covariance is 1 at this point (since we
      have normalized the within-class covariance to unity); hence, the total
      variance is \lambda_i + 1.
      Below, "within-class-factor" is a constant that we set by default to
      0.001.  We scale the i'th dimension of the features by:
      
         \f$  sqrt( (within-class-factor + \lambda_i) / (1 + \lambda_i) ) \f$
           
      If \lambda_i >> 1, this scaling factor approaches 1 (we don't need to
      scale up dimensions with high between-class variance as they already
      naturally have a higher variance than other dimensions.  As \lambda_i
      becomes small, this scaling factor approaches sqrt(within-class-factor),
      so dimensions with very small between-class variance get assigned a small
      variance equal to within-class-factor, and for dimensions with
      intermediate between-class variance, they end up with a variance roughly
      equal to \lambda_i: consider that the variance was originally (1 +
      \lambda_i), so by scaling the features by approximately sqrt((\lambda_i) /
      (1 + \lambda_i)), the variance becomes approximately \lambda_i [this is
      clear after noting that the variance gets scaled by the square of the
      feature scale].      
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

