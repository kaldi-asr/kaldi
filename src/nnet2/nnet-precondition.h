// nnet2/nnet-precondition.h

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_PRECONDITION_H_
#define KALDI_NNET2_NNET_PRECONDITION_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"

#include <iostream>

namespace kaldi {
namespace nnet2 {


/**
  The function PreconditionDirections views the input R as
  a set of directions or gradients, each row r_i being one of the
  directions.  For each i it constructs a preconditioning matrix
  G_i formed from the *other* i's, using the formula:

  G_i = (\lambda I + (1/(N-1)) \sum_{j \neq i} r_j r_j^T)^{-1},

  where N is the number of rows in R.  This can be seen as a kind
  of estimated Fisher matrix that has been smoothed with the
  identity to make it invertible.  We recommend that you set
  \lambda using:
    \lambda = \alpha/(N D) trace(R^T, R)
  for some small \alpha such as \alpha = 0.1.  However, we leave
  this to the caller because there are reasons relating to
  unbiased-ness of the resulting stochastic gradient descent, why you
  might want to set \lambda using "other" data, e.g. a previous
  minibatch.

  The output of this function is a matrix P, each row p_i of
  which is related to r_i by:
    p_i = G_i r_i
  Here, p_i is preconditioned by an estimated Fisher matrix
  in such a way that it's suitable to be used as an update direction.

 */
void PreconditionDirections(const CuMatrixBase<BaseFloat> &R,
                            double lambda,
                            CuMatrixBase<BaseFloat> *P);

/**
   This wrapper for PreconditionDirections computes lambda
   using \lambda = \alpha/(N D) trace(R^T, R), and calls
   PreconditionDirections. */
void PreconditionDirectionsAlpha(
    const CuMatrixBase<BaseFloat> &R,
    double alpha,
    CuMatrixBase<BaseFloat> *P);

/**
   This wrapper for PreconditionDirections computes lambda
   using \lambda = \alpha/(N D) trace(R^T, R), and calls
   PreconditionDirections.  It then rescales *P so that
   its 2-norm is the same as that of R. */
void PreconditionDirectionsAlphaRescaled(
    const CuMatrixBase<BaseFloat> &R,
    double alpha,
    CuMatrixBase<BaseFloat> *P);
  
                           

} // namespace nnet2
} // namespace kaldi


#endif
