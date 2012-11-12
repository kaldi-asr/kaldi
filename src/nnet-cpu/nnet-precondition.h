// nnet-cpu/nnet-precondition.h

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_PRECONDITION_H_
#define KALDI_NNET_CPU_PRECONDITION_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

#include <iostream>

namespace kaldi {

/*
  Notes for an idea on preconditioning.
  update is of form:
     params += learning_rate * input_row * output_deriv'
  want to precondition by fisher-like matrix in each of (the input dim and the
  output dim).
  [note: in this method we'll pretend the chunk-weights are all one.
   It shouldn't really matter, it's only preconditioning.]

   The first observation is, if we do:

    params += learning_rate * S * input_row * output_deriv' * T

   for any positive definite S and T that we choose (well, perhaps we have
   to ensure their eigenvalues are bounded in some way, but we'll bother with
   that later),  then we'll still get convergence.  But S and T cannot be
   functions of the current sample, the one that creates "input_row" and
   "output_deriv", or this introduces a bias.

   We can view it as a preconditioning of the vectorized form of the
   transformation matrix.

   For a Fisher-like preconditioning, we can precondition using
   the inverse of the scatter of the other features in the batch.
   For the input_row, call this r_j.

   Let the total scatter be

    S =  \sum_n r_n r_n^T
  where the sum is taken over the minibatch, and
   S_n = S - r_n  r_n^T
  i.e. the scatter with this sample removed.
  Let F_n be the normalized version of this, dividing by the #samples.
   F_n = 1/(N-1) S_n
  where N is the minibatch size (so N-1 is excluding the current sample).
 We're going to want to invert F_n, so we need to make it positive definite.

  We're going to define G_n as a smoothed form of the estimated Fisher matrix
  for this batch:
   G_n = F_n + \lambda_n I
  where I is the identity.  A suitable formula for \lambda_n is to define
  a small constant \alpha (say, \alpha=0.01), and let
  
   \lambda_n =  (\alpha/dim(F)) trace(F_n) .
  
  This is an easy way to set it.  Let's define P_n as the inverse of G_n.  This
  is what we'll be multiplying the input values by:

    P_n = G_n^{-1} = (F_n + \lambda_n I)^{-1}

  First, let's define an uncorrected "global" Fisher matrix
    F = (1/(N-1)) S_n,
  and G = F^{-1}, and
      \lambda = (\alpha/dim(F)) trace(F).
              = (\alpha/dim(F)) \sum_n r_n^T r_n.
  If we let R be the matrix each of whose rows is one of the r_n,
  then
    S = R^T R, and
   F = 1/(N-1) R^T R

           G = (F + \lambda I)^{-1}
             = (1/(N-1) R^T R + \lambda I)^{-1}
Using the Woodbury formula,
     G  = (1/\lambda) I  - (1+1/\lambda^2) R^T M R
where
  M = ((N-1) I + 1/\lambda R R^T)^{-1}
(and this inversion for M is actually done as an inversion, in a lower
 dimension such as 250, versus the actual dimension which might be 1000).

Let's assume \lambda is a constant, i.e. there is no \lambda_n.
We can get it from the previous minibatch.

 We want to compute

    G_n = F_n^{-1} = (F - 1/(N-1) r_n r_n^T)^{-1}

 and using the Sherman-Morrison formula, this may be written as:

   G_n = G  +  \alpha_n q_n q_n^T

 where q_n = G r_n, and

 \alpha_n =  1/( (N-1) (1 - 1/(N-1) r_n^T q_n) )
          =  1 / (N - 1 - r_n^T q_n)

  We'll want to compute this efficiently.  For each r_n we'll want to compute

 p_n =  G_n r_n

 which will correspond to the direction we update in.
 We'll use

  p_n = G r_n + \alpha_n q_n q_n^T r_n

  and since q_n = G r_n, both terms in this equation point in
  the same direction, and we can write this as:

  p_n = \beta_n q_n,

  where, defining \gamma_n = r_n^T q_n, we have

  \beta_n = 1 + \gamma_n \alpha_n 
          = 1  +  \gamma_n / ((N-1) (1 - \gamma_n/(N-1)))
          = 1  +  \gamma_n / (N - 1 - \gamma_n)
  

   SUMMARY:
   let the input features (we can extend these with a 1 for the bias term) be
   a matrix R, each row of which corresponds to a training example r_n

   The dimension of R is N x D, where N is the minibatch size and D is the
   dimension of the input to this layer of the network.

   We'll be computing a matrix P, each row p_n of which will be the corresponding
   row r_n of R, multiplied by a positive definite preconditioning matrix G_n.
   [we can check that for each i, p_n^T r_n >= 0].
   The following computation obtains P:

   C <-- 3/4.  # C is a constant that determines when to use the Morrison-Woodbury formula
               # or to do direct inversion.  It needs to be tuned empirically based on speed,
               # if we plan to use minibatch sizes about equal to the dimension of
               # the hidden layers.
   
   \lambda <-- (\alpha/D) \trace(R R^T).   # 0 < \alpha <= 1 is a global constant, e.g.
                                           # \alpha = 0.1, but should try different
                                           # values, this will be important (note: if the
                                           # minibatch size is >= the dimension (N >= D),
                                           # then we can let \alpha be quite small, e.g.
                                           # 0.001.

   if N >= C D, then
     # compute G by direct inversion.
     G <-- (\lambda I  +  1/(N-1) R^T R)^{-1}
   else   # number of samples is less than dimension, use
          # morrison-Woodbury formula, it's more efficient.
      M <-- ((N-1) I + 1/\lambda R R^T)^{-1}
      G <-- 1/\lambda I  -  (1 + 1/\lambda^2) R^T M R
   fi

   Let

   Q <-- R G.

   Here, we're right multiplying each row r_n of r by the symmetric matrix G, to get
   the corresponding row q_n of q.  Note: in practice Q will be the same memory as P.
   Next we work out for each n:
     \gamma_n = r_n^T q_n     # This should be nonnegative!  Check this.
      \beta_n = 1  +  \gamma_n / (N - 1 - \gamma_n)  # This should be positive; check this.
  For each n, we'll do (for the corresponding rows of P and Q):
     p_n <-- \beta_n q_n.
  In practice, we'd do this computation in-place, with P and Q using the
  same memory.

  If we're being paranoid, we should verify that

   p_n = (\lambda I  +  1/(N-1) \sum_{m != n} r_n r_n^T)^{-1} r_n.

  This is exact mathematically, but there could be differences due to roundoff,
  and if \alpha is quite small, these differences could be substantial.

  

 */



} // namespace kaldi


#endif
