// nnet2/nnet-precondition-online.h

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_PRECONDITION_ONLINE_H_
#define KALDI_NNET2_NNET_PRECONDITION_ONLINE_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"

#include <iostream>

namespace kaldi {
namespace nnet2 {


/**
   It will help to first try to understand ./nnet-precondition.h before
   reading this comment and trying to understand what's going on here.  The motivation
   for this method was that the code in nnet-precondition.h was too slow when
   implemented on CUDA cards, it was taking well over half the time.  The problem
   is that algorithms like Cholesky decomposition and triangular solvers, that
   were used in that method, are not as parallelizable as matrix multiplication.
   The method in nnet-precondition.h involved inverting symmetric matrices whose
   dimension was the number of frames in a minibatch.  

   Our method here aims to reduce the dimension in which we have to do things
   like inversion.  (In fact, for CUDA implementation we'll deal with small matrices
   like 10x10 for which it's faster to move them to the CPU and invert them there,
   and then move them back).

   Firstly, for each affine layer, we treat it for purposes of this code as
   a linear layer on an input that has been extended by a 1.  This code does not
   even see the bias as a separate thing.
   
   The basic aim (just like nnet-precondition.h) is to deal with a factored form
   of a Fisher matrix (outer product of derivatives), where for each affine
   layer we have one such matrix that is in the space of the input to the layer
   and another that is in the space of the derivatives at the output of the
   layer.  There are reasons from information geometry to believe that it's a
   good thing to multiply the derivatives by the inverse of the Fisher matrix,
   so that's basically what we'll do, except we have to worry about things like
   smoothing estimates of the Fisher matrix, and putting in an overall scaling
   factor so we still have a reasonable learning rate.

   In the previous code (nnet-precondition.h), we got the Fisher matrix on the
   input side from the scatter of inputs and on the output side from the scatter
   of derivatives of the output.  We didn't have to invert matrices of dimension
   like 2048 (a typical hidden-layer dim) because via the use of various matrix
   equalities we did operations on the dim of (minibatch size), based on the
   cross-products of gradients.

   Below, assume just for the sake of exposition that the matrix we're training
   goes from (hidden-layer-dim + 1) to (hidden-layer-dim), and
   (hidden-layer-dim) is 2048, and the minibatch-size is 255.  In fact this is a
   special case as the input and output dims of the affind component don't have
   to be equal, we'll gloss over those details.  Let's consider the
   preconditioning of the input values (the 2049-dimensional input vectors); the
   preconditioning of the output derivatives happens in the same way.

   We choose a rank, say R = 10, and we have a 10 x 2049 dimensional matrix.
   Call this N, but it changes on each minibatch, so we'll call it N_i, where
   i is the iteration number. (later we'll come to how we implement this for
   multi-core).  On each iteration, let the matrix of input-values be M_i,
   which is a 256 x 2049 dimensional matrix.  
   
   We'll update N_i on each iteration by doing:

          O_i =  N_i (M_i^T M_i)
          P_i = O_i + \eta_i N_i 
      N_{i+1} = orthogonalize(P_i)

   where "orthogonalize" means orthogonalizing the rows of the matrix they have unit
   norm and are orthogonal to each other; if P_i is not full rank (which should
   be very rare), we can just use random values.  So the N_i will always have
   orthonormal rows.  We choose the \eta_i with:
      \eta_i = \eta sqrt( tr(O_i O_i^T) / tr(N_i N_i^T) ) = \eta sqrt( tr(O_i O_i^T) / 10),
   for a globally chosen \eta >= 0 (e.g. something like 2) which the larger it is, the
   more "inertia" the N_i have (they'll change more slowly)
   
   Now, the N_i is in some sense an arbitrary orthogonal matrix but what is
   important is that it is "enriched" in the directions in which the Fisher
   matrix has large eigenvalues; in fact, we can show for large \eta (and
   assuming the parameters are constant) that it will approach the top
   eigenvalues of the Fisher matrix.
   
   To discuss the preconditioning method, first imagine we have an orthogonal
   matrix Q_i of dimension 2049 x 2049, which is formed from N_i by choosing
   arbitrary extra rows orthogonal to the first rows and to each other.
   Project the gradients M_i with Q_i, forming
        M'_i = M_i Q_i^T.
   [for each vector, it would be m'_i = Q_i m_i, but the m_i are the rows of
    M_i so we have to transpose the equation].

   The preconditioner acts in this transformed space, and is of the form
     diag(G_i, I) where G is of dimension 10 x 10 and I is the
   identity matrix of dimension (2049 - 10).  So it only "does something"
   to the first 10 dimensions.  In fact, the eigenvalues of G_i will almost
   certainly all be less than one, so the transform will make those dimensions
   smaller.

   Now we describe what G_i is.  First, let F_i be the Fisher matrix,
   limited to those ten dimensions, and estimated from the current
   minibatch.  We'll have
      G_i = \beta_i F_i^{-1}
   and we'll work out the constant \beta_i.  We want to choose 
   \beta_id in such a way that the identity matrix (I) that we apply
   to the "remaining dimensions" will be a reasonable approximation to beta_i
   times the inverse Fisher on those dimensions.  Thus, we're applying
   a matrix \beta_i diag(F_i^{-1}, 1/\beta_i I), where the matrix
   in the diag( ... ) expression is an approximation to the inverse
   Fisher matrix, and the \beta_i can just be viewed as part of the learning
   rate.  We could of course remove the factor \beta_i, but this could lead
   to problems early on in training when some parameters are zero.

   We have a unit-matrix approximation to the "remaining" dimensions
   of F_i.  That is: we compute the trace of (the fisher from dimension
   11 to 2049) and divide by (2049 - 10), and that be \beta_i.  So
   we approximate full-fisher(i) = diag(F_i, \beta_i I).
   Then the inverse-Fisher will be diag(F_i^{-1}, 1/\beta_i I).
   Then we (quite arbitrarily) multiply by \beta_i so the matrix we use
   as a preconditioner is diag(\beta_i F_i^{-1}, I).

   Now, as to the actual computation... it's important to never explicitly
   construct Q_i.  

  Input: we have N_i (orthogonal) and M_i.

  L_i will be M_i multiplied by the preconditioner.  This will be:

   L_i = M_i Q_i^T diag(\beta_i F_i^{-1}, I) Q_i

      =  M_i Q_i^T (I + diag(I - \beta_i F_i^{-1}, 0)) Q_i
      =  M_i - M_i N_i^T (I - \beta_i F_i^{-1}) N_i

 First: what is F_i ?  
       F_i = (1/256) N_i M_i^T M_i N_i^T
  This is the Fisher matrix projected with the basis N_i.  The 1/256 is dividing
  by the minibatch size.  And let
     \beta_i = (1/(2049 - 10)) (1/256) (tr(M_i^T M) - tr(F_i) )
  (So \beta_i is the average diagonal element of (M_i^T M projected to
   the remaining dimensions of Q_i), i.e. the average diagonal of
   the remaining dimensions of the Fisher).

  Now, as to the computation of the "next" F_i...
  Note: above, while computing F_i, we computed the following expression, which
  we'll give a name here:
     O_i = N_i M_i^T M_i
  Define
     X_i = O_i O_i^T.
  We can compute \eta_i  = \eta sqrt( tr(O_i O_i^T) / 10 ) = \eta sqrt(trace(X_i) / 10).
  Then we can compute P_i = O_i + \eta_i N_i.
  We need to orthogonalize P_i.  We can do this as follows.  We can compute the matrix
  of inner products of its rows:
    Y_i =(def) P_i P_i^T.
  Now, Y_i = (O_i + \eta_i N_i) (\eta_i O_i + N_i)^T.
           =  X_i + \eta_i^2 I + \eta_i O_i N_i^T + \eta_i N_i O_i^T
           =  X_i + \eta_i^2 I + (2 * \eta_i * 256) F_i
 (using the fact that F_i = (1/256) N_i M_i^T M_i N_i = (1/256) O_i N_i^T, and 
   that F_i is symmetric).

  Note: if N_i has full row rank rank (which it is, by induction) then P_i also
  will have full row rank.  The basic proof would be that P_i equals N_i times a
  strictly positive definite matrix, namely (\eta_i I + M_i^T M_i).  We could
  probably even get a bound on the condition of P_i.  

  OK, we have Y_i, so we need any matrix that makes Y_i unit.  If we do the
  Cholesky decomposition
    Y_i = C_i C_i^T, then (C_i^{-1} Y_i C_i^{-T}) = I, so (C_i^{-1} P_i P_i^T C_i^{-T}) = I,
  so (C_i^{-1} P_i) is orthogonal.  So to get N_{i+1} we simply do

     N_{i+1} = C_i^{-1} P_i

     *****************
  OK, we'll now describe, compactly, one iteration of the algorithm in which we
  "precondition" a minibatch of vectors
     M_i \in Re^{B \times D},
  where B is the minibatch size and D is the dimension of the vectors (e.g. 2049).
  N_i is an input and we output N_{i+1}.  (We assume that N_i has already been
  initialized to the correct dimension, randomly if necessary).

  Let R be the "rank of the correction", so
     N_i \in \Re^{R \times D}.

  If B < 2 * R, return without doing anything and print a warning (maybe a partial
    minibatch, at the end).

 Let    NMT_i = N_i M_i^T             # dimension R by B.
 Let      O_i = NMT_i   M_i.          # note: O_i has same dimension as N_i: R by D.
 Let      F_i = (1/B) O_i N_i^T       # F_i \in Re^{R, R} is the low-dimension Fisher matrix: F_i = (1/B) N_i M_i^T M_i N_i
 Let      t_f = Trace(F_i) 
 Let      t_m = Trace(M_i^T M)        # i.e. sumsq of elements of M_i
 Let  \beta_i = (t_m - B t_f) / ((D - R) * B)     # \beta_i is the average diagonal element of
                                                  # the "rejected dimensions" of the Fisher matrix,
                                                  # i.e. not in the chosen subspace.
 Let F_i_inv = (F_i + (\epsilon t_f/R + \delta) I)^{-1}
                                                # e.g. with \epsilon = 1.0e-4, \delta = 1.0e-10.  The epsilon and delta
                                                # are really there
                                                # just to prevent crashes if the matrix is nearly singular, or exactly zero.

                                                # A note on the next line: we expect \beta_i F_i_inv to have
                                                # eigenvalues less than one.
 Let      L_i = M_i  + NMT_i^T (\beta_i F_i_inv - I) N_i  # L is the "preconditioned" version of M_i; it's an output.

 Let        X_i = O_i O_i^T
 Let     \eta_i = max( \eta sqrt(Trace(O_i O_i^T) / R), \delta)  # e.g. delta = 1.0e-10.
 Let        Y_i = X_i + \eta_i^2 I + (2 * \eta_i * B) F_i
    # Note: it's almost inconceivable that the Cholesky below or the inversion should fail; it
    # would be a kind of perverse coincidence.  (Y_i must be +ve semidefinite but
    # will almost surely be +ve definite).  If it happens we could just leave N_i at
    # the same value it had previously.
Do Cholesky Y_i = C_i C_i^T.
 Let        P_i = O_i + \eta_i N_i.
 Let    C_i_inv = C_i^{-1} 
 Let    N_{i+1} = C_i_inv P_i
   # check: N_{i+1} is orthogonal?

     *****************
     */

/*
   The following describes compactly how the function's behavior is defined
   (ignoring small terms that are introduced to ensure invertibility of certain
   quantities); this description is supplied for testing purposes, it doesn't
   correspond to how the computation is done.
   
   Inputs: N_i \in \Re^{R times D}, M \in \Re^{B \times D}.  Require N_i N_i^T = I.

   Let T_i \in \Re^{(D - R) times D} be chosen such that T_i T_i^T = I and T_i N_i^T = 0,
     i.e. a matrix such that Q_i = [ N_i
                                     T_i ] is orthogonal.
   
   Let F_i = (N_i M^T M N_i).
   Let \beta_i = (trace(T_i M^T M T_i)) / ((D - R) * B).
   Let Finv = diag( \beta_i F_i^{-1} , I ), where I is in dimension (D - R).  Finv is
       our Fisher-inverse approximation in the space projected by Q_i.
   Output: M <-- M Q_i^T Finv Q_i
   Next we update N_i:
      O_i <-- N_i M_i^T M_i  # this is proportional to N_i times the Fisher matrix.
   \eta_i = \eta sqrt(tr(O_i O_i^T) / tr(N_i N_i^T) )
      P_i = O_i + \eta_i N_i
# now orthogonalize P_i: let Y_i = P_i P_i^T, do Cholesky Y_i = C C^T, then do      
  N_{i+1} = C^{-1} P_i.  
*/
   

/** In this function, N is the orthogonal (R x D) matrix which is updated
   each time and which controls the preconditioning.  M is a (B x D) matrix
   where B is the batch size and D is the dimension of the problem (e.g.
   a hidden-layer dimension or a hidden-layer dimension plus on for the input).
   R (the #rows of N) is the rank ofthe Fisher matrix, which is an important
   parameter.  lambda controls how fast we update M.  A suitable value is,
   say, 0.25.  The algorithm should not be too sensitive to this.

   If the "first_time" option is true, the function will first initialize N to a
   random matrix with orthonormal rows, and will then call itself once with a
   small eta (e.g. 0.001, just nonzero enough to handle zero M without making a
   singular N), and discard the resulting M, in order to get a more reasonable
   value for N.  */
void PreconditionDirectionsOnline(BaseFloat eta,
                                  bool first_time,
                                  CuMatrixBase<BaseFloat> *N,
                                  CuMatrixBase<BaseFloat> *M);
                            


} // namespace nnet2
} // namespace kaldi


#endif
