// transform/differentiable-fmllr.h

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_
#define KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"
#include "matrix/matrix-functions.h"

namespace kaldi {


namespace differentiable_transform {


// This header contains some utilities for implementing differentiable fMLLR.
// Since it is fairly complicated, we aren't putting all the implementation
// details in class FmllrTransform (in differentiable-transform.h), but
// segregating most of the technical stuff to this file.  This also
// allows us to separate out the testing of individual components.
// The reference for things in this header is
// http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf.
// The notation we are using corresponds to the notation used in
// the "Summary" section of that document.



/**
   With reference to the notation in
     http://www.danielpovey.com/files/2018_differentiable_fmllr.pdf,
   this class implements the operation that takes G and K as input (and the
   count gamma), and produces A.  This has been separated into its own object
   for purposes of testability.
 */


struct CoreFmllrEstimatorOptions {

  // singular_value_relative_floor is floor that we apply on the
  // singular values of the inputs G and K, to ensure that no NaN's are
  // generated in the forward pass and to prevent the derivatives
  // in the backprop from becoming undefined.  It affects both
  // the forward and backward computations.  A warning will be printed
  // if this floor actually had an effect.
  // Must be greater than zero (to avoid the possibility of generating
  // NaN's).
  BaseFloat singular_value_relative_floor;

  CoreFmllrEstimatorOptions():
      singular_value_relative_floor(0.001) { }
};


class CoreFmllrEstimator {
 public:
  /**
     Constructor.  Does not do any real work.  This class will store
     references/pointers to G, K and A, so you need to make sure that
     those quantities exist for the lifetime of this object.

       @param [in] opts  Options class; see its definition for details.  Will be copied
                      in the constructor.
       @param [in]  gamma  The total data-count (often this will be the number of frames).
       @param [in]  G  A symmetric matrix containing the quadratic
                       stats for estimating A.  This the sum of outer products
                       of the input features, after mean subtraction, and
                       weighted by the inverse-variance factor s_i.  Must be
                       positive definite for this computation to be well
                       defined.
       @param [in] K   A matrix containing the linear stats for estimating A.
                       This is a sum of outer products of the means with the
                       input features, with mean subtraction and inverse-variance
                       weighting.  Must not have more than one zero singular value
                       for this computation to be well defined.
       @param [in] A   We mark this as an input parameter but it is the location
                       where the output of this computation will be placed when
                       you call Forward().  May be undefined (e.g., NaN) on
                       entry.  You must not change the value of A between
                       calling Forward() and calling Backward().

                       TODO: introduc
   */
  CoreFmllrEstimator(const CoreFmllrEstimatorOptions &opts,
                     BaseFloat gamma,
                     const MatrixBase<BaseFloat> &G,
                     const MatrixBase<BaseFloat> &K,
                     MatrixBase<BaseFloat> *A);

  /**
     Does the forward pass of estimation.  Writes to the location
     'A' that was passed to the constructor.

     Returns the objective-function improvement per frame, as compared
     with what the objective-function would be with unit A.  This equals
     the total objective function improvement divided by gamma.
   */
  BaseFloat Forward();


  /**
     Does the backward pass.
       @param [in] A_deriv  The derivative of the objective
           function (say, f) w.r.t. the output A (which was passed as a
           pointer to the constructor).
       @param [out] G_deriv  A pointer to a location where the
           derivative df/dG will be written.  Will be added to, so
           should contain zero (or some other defined value)
           at input.
       @param [out] K_deriv  A pointer to a location where the
           derivative df/dK will be written (so the i,j'th
           element is the derivative w.r.t. the i,j'th element
           of the input matrix K.
  */
  void Backward(const MatrixBase<BaseFloat> &A_deriv,
                Matrix<BaseFloat> *G_deriv,
                Matrix<BaseFloat> *K_deriv);

 private:
  // Computes H = G^{-0.5}
  void ComputeH();
  // Compute L = K H
  void ComputeL();
  // Compute B = F(L), where F is the
  // function that takes the singular values of L, puts them through the function
  // f(lamba) = (lambda + sqrt(lambda^2 + 4 gamma)) / 2.
  void ComputeB();
  // Computes A = B H.
  void ComputeA();


  // Backprops through the operation "A = B H".  B_deriv and H_deriv
  // must be free of NaN and inf on entry.
  void BackpropA(const MatrixBase<BaseFloat> &A_deriv,
                 MatrixBase<BaseFloat> *B_deriv,
                 MatrixBase<BaseFloat> *H_deriv);

  // Backprops through the function "L = K H"..
  // K_deriv must be free of NaN and inf on entry, but otherwise
  // its value is ignored.  H_deriv is added to by this function.
  void BackpropL(const MatrixBase<BaseFloat> &L_deriv,
                 MatrixBase<BaseFloat> *K_deriv,
                 MatrixBase<BaseFloat> *H_deriv);

  // returns the objective-function change (vs. A being the unit matrix) from
  // this estimation.
  BaseFloat ComputeObjfChange();

  CoreFmllrEstimatorOptions opts_;
  BaseFloat gamma_;
  const MatrixBase<BaseFloat> &G_;
  const MatrixBase<BaseFloat> &K_;
  MatrixBase<BaseFloat> *A_;

  // H = G^{-0.5} is symmetric.
  Matrix<BaseFloat> H_;
  // L = K H.
  Matrix<BaseFloat> L_;
  // B = F(L) is the result of applying SvdRescaler with
  // the function f(lambda) = ((lambda + sqrt(lambda^2 + 4 gamma)) / 2)
  Matrix<BaseFloat> B_;

  // Object that helps us to compute, and to backprop through the
  // computation of, H = G^{-0.5}.
  SvdRescaler G_rescaler_;

  // Object that helps us to compute, and to backprop through the computation
  // of: B = F(L), where F is the function that takes the singular values of L,
  // puts them through the function f(lamba) = (lambda + sqrt(lambda^2 + 4
  // gamma)) / 2.
  SvdRescaler L_rescaler_;

};


} // namespace differentiable_transform
} // namespace kaldi

#endif  // KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_
