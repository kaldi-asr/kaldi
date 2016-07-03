// nnet3/natural-gradient-online.h

// Copyright 2013-2015   Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NATURAL_GRADIENT_ONLINE_H_
#define KALDI_NNET3_NATURAL_GRADIENT_ONLINE_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "thread/kaldi-mutex.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {


/**
   Keywords for search: natural gradient, naturalgradient, NG-SGD

   This method is explained in the paper
   "Parallel training of DNNs with Natural Gradient and Parameter Averaging"
   by D. Povey, X. Zhang and S. Khudanpur, ICLR Workshop, 2015, where
   it is referred to as online NG-SGD.  Note that the method exported
   from this header is just the core of the algorithm, and some outer-level parts
   of it are implemented in class NaturalGradientAffineComponent.

  The rest of this extended comment describes the way we keep updated an estimate
  of the inverse of a scatter matrix, in an online way.  This is the same as the
  estimation of one of the A or B quantities in the paper.  This comment is slightly
  redundant with the paper- actually it precedes the paper- but we keep it in case it
  is useful in understanging our method.

  We consider the problem of doing online estimation of a (scaled-identity plus low-rank)
  approximation of a Fisher matrix... since the Fisher matrix is a scatter of vector-valued derivatives
  and we will be given the derivatives (or at least terms in a factorization of the derivatives
  which need not concern us right now), we can just think of the present task as being
  the online accumulation of a (low-rank plus scaled-identity) approximation to a variance
  of a distribution with mean zero.

  Later on we'll think about how to get easy access to the inverse of this approximate
  variance, which is what we really need.

  Our approximation to the Fisher matrix (the scatter of derivatives) will be of the following form
  (and just think of this as an approximate variance matrix of some arbitrary quantities).

     F_t =(def) R_t^T D_t R_t + \rho_t I

  (t is the minibatch index), where R_t is an R by D matrix with orthonormal
  rows (1 <= R < D is our chosen rank), D_t is a positive-definite diagonal matrix, and
  \rho_t > 0.  Suppose the dimension of F_t is D.  Let the vectors whose variance
  we are approximating be provided in minibatches of size M (M can vary from
  iteration to iteration, but it won't vary in the normal case, so we omit the
  subscript t).  The batch of gradients is given as X_t \in Re^{M \times D},
  i.e. each row is one of the vectors whose scatter we're estimating.  On the
  t'th iteration, define the scatter S_t of the input vectors X_t as:

     S_t =(def) 1/N X_t^T X_t           (eqn:St)

  (where N is the minibatch size).  Be careful not to confuse the rank R with
  with input X_t (we would typeface X_t in bold if this were not plain text, to
  make the distinction clearer).  We want F_t to approach some kind of
  time-weighted average of the S_t quantities, to the extent permitted by the
  limitation of the rank R.  We want the F_t quantities to stay "fresh" (since
  we'll be doing this in a SGD context and the parameters will be slowly
  changing).  We use a constant 0 < \eta < 1 to control the updating rate.  Our
  update for R_t is based on the power method.  Define the smoothed scatter

   T_t =(def) \eta S_t + (1-\eta) F_t

  we'll use this in place of the observed scatter S_t, to slow down the update.
  Defining

   Y_t =(def) R_t T_t

  which can be expanded as follows:
       Y_t = R_t ( \eta S_t + (1-\eta) F_t )
           = R_t ( \eta S_t + (1-\eta) (R_t^T D_t R_t + \rho_t I) )
           = R_t ( \eta S_t + (1-\eta) (R_t^T D_t R_t + \rho_t I) )
           = \eta R_t S_t + (1-\eta) (D_t + \rho_t I) R_t

  It is useful to think of Y_t as having each of the top eigenvectors of the
  scatter scaled by the corresponding eigenvalue \lambda_i.
  We compute the following R by R matrix:
    Z_t =(def) Y_t Y_t^T
  and do the symmetric eigenvalue decomposition
    Z_t = U_t C_t U_t^T
  where C_t is diagonal and U_t orthogonal; the diagonal elements of C_t will be
  positive (since \rho_t > 0, T_t is positive definite; since R_t has full row rank
  and T_t is positive definite, Y_t has full row rank; hence Z_t is positive definite).
  The diagonal elements of C_t can be thought of as corresponding to the squares of
  our current estimate of the top eigenvalues of the scatter matrix.
  [we should check that no element of C_t is <= 0.]

  It is easy to show that C_t^{-0.5} U_t^T Z_t U_t C_t^{-0.5} = I, so
     (C_t^{-0.5} U_t^T Y_t) (Y_t^T U_t C_t^{-0.5}) = I.  Define
    R_{t+1} =(def) C_t^{-0.5} U_t^T Y_t

  and it's clear that R_{t+1} R_{t+1}^T = I.
  We will set
     D_{t+1} =(def) C_t^{0.5} - \rho_{t+1} I             (eqn:dt1)

  which ensures that for each row r of R_{t+1}, the variance of our scatter
  matrix F_{t+1} will be the square root of the corresponding diagonal element
  of C_t.  This makes sense because, as we have pointed out, the diagonal
  elements of C_t can be thought of as corresponding to squared eigenvalues.
  But a proper treatment of this would require convergence analysis that would
  get quite complicated.  We will choose \rho_{t+1} in order to ensure that
  tr(F_{t+1}) = tr(T_t).

  For any t,
     tr(F_t) = D \rho_t + tr(D_t)
     tr(T_t) = \eta tr(S_t) + (1-\eta) tr(F_t)
             = \eta tr(S_t) + (1-\eta) (D \rho_t + tr(D_t))
  Expanding out D_{t+1} from (eqn:dt1) in the expression for tr(F_{t+1}) below:
      tr(F_{t+1})  = D \rho_{t+1} +  tr(D_{t+1})
      tr(F_{t+1})  = D \rho_{t+1} +  tr(C_t^{0.5} - \rho_{t+1} I)
                   = (D - R) \rho_{t+1} + tr(C_t^{0.5})
   and equating tr(F_{t+1}) with T_t (since F_{t+1} is supposed to be a low-rank
   approximation to T_t), we have
                          tr(F_{t+1}) = tr(T_t)
  (D - R) \rho_{t+1} + tr(C_t^{0.5})  = \eta tr(S_t) + (1-\eta) (D \rho_t + tr(D_t))

  Solving for \rho_{t+1},
       \rho_{t+1} = 1/(D - R) (\eta tr(S_t) + (1-\eta)(D \rho_t + tr(D_t)) - tr(C_t^{0.5})).   (eqn:rhot1)

  Note that it is technically possible that diagonal elements of
  of D_{t+1} may be negative, but we can still show that F_{t+1} is strictly
  positive definite if F_t was strictly positive definite.

  If the quantities for which we are computing the Fisher matrix are all zero
  for some, reason, the sequence of F_t will geometrically approach zero, which
  would cause problems with inversion; to prevent this happening, after setting
  D_{t+1} and \rho_{t+1} as above, we floor \rho_{t+1} to a small value (like
  1.0e-10).

  OK, we have described the updating of R_t, D_t and \rho_t.  Next, we need to
  figure out how to efficiently multiply by the inverse of F_t.  Our experience
  from working with the old preconditioning method was that it's best not to use
  the inverse of the Fisher matrix itself, but a version of the Fisher matrix
  that's smoothed with some constant times the identity.  Below, (\alpha is a
  configuration value, e.g. 4.0 seemed to work well).  The following formula is
  designed to ensure that the smoothing varies proportionally with the scale of F_t:

        G_t =(def) F_t +  \alpha/D tr(F_t) I
            =     R_t^T D_t R_t + (\rho_t + \alpha/D tr(F_t)) I
            =     R_t^T D_t R_t + \beta_t I
  where
    \beta_t =(def) \rho_t + \alpha/D tr(F_t)
            =      \rho_t(1+\alpha) + \alpha/D tr(D_t)       (eqn:betat2)

  Define
     \hat{X}_t =(def)  \beta_t X_t G_t^{-1}.
  the factor of \beta_t is inserted arbitrarily as it just happens to be convenient
  to put unit scale on X_t in the formula for \hat{X}_t; it will anyway be canceled out
  in the next step.  Then our final preconditioned minibatch of vectors is:
     \bar{X}_t = \gamma_t \hat{X}_t
  where
     \gamma_t = sqrt(tr(X_t X_t^T)  / tr(\hat{X}_t \hat{X}_t^T).
  The factor of \gamma ensures that \bar{X}_t is scaled to have the same overall
  2-norm as the input X_t.  We found in previous versions of this method that this
  rescaling was helpful, as otherwise there are certain situations (e.g. at the
  start of training) where the preconditioned derivatives can get very large.  Note
  that this rescaling introduces a small bias into the training, because now the
  scale applied to a given sample depends on that sample itself, albeit in an
  increasingly diluted way as the minibatch size gets large.

  To efficiently compute G_t^{-1}, we will use the Woodbury matrix identity.
  Writing the Woodbury formula for the symmetric case,
    (A + U D U^T)^{-1} = A^{-1} - A^{-1} U (D^{-1} + U^T A^{-1} U)^{-1} U^T A^{-1}
  Substituting A = \beta_t I, D = D_t and U = R_t^T, this becomes
       G_t^{-1} = 1/\beta_t I - 1/\beta_t^2 R_t^T (D_t^{-1} + 1/\beta_t I)^{-1} R_t
                = 1/\beta_t (I - R_t^T E_t R_t)
  where
        E_t =(def)  1/\beta_t (D_t^{-1} + 1/\beta_t I)^{-1},         (eqn:etdef)
  so
    e_{tii} =   1/\beta_t * 1/(1/d_{tii} + 1/\beta_t)                (eqn:tii)
            =   1/(\beta_t/d_{tii} + 1)

  We would like an efficient-to-compute expression for \hat{X}_t, without too many separate
  invocations of kernels on the GPU.
     \hat{X}_t = \beta_t X_t G_t^{-1}
         = X_t - X_t R_t^T E_t R_t
  For efficient operation on the GPU, we want to reduce the number of high-dimensional
  operations that we do (defining "high-dimension" as anything involving D or M, but not
  R, since R is likely small, such as 20).  We define
     W_t =(def)  E_t^{0.5} R_t.
  We will actually be storing W_t on the GPU rather than R_t, in order to reduce the
  number of operations on the GPU.  We can now write:

        \hat{X}_t = X_t - X_t W_t^T W_t       (eqn:pt2)

  The following, which we'll compute on the GPU, are going to be useful in computing
  quantities like Z_t:

     H_t =(def) X_t W_t^T     (dim is N by R)
     J_t =(def) H_t^T X_t     (dim is R by D)
         =      W_t X_t^T X_t
     K_t =(def) J_t J_t^T     (dim is R by R, symmetric).. transfer this to CPU.
     L_t =(def) H_t^T H_t     (dim is R by R, symmetric).. transfer this to CPU.
         =      W_t X_t^T X_t W_t^T
     Note: L_t may also be computed as
     L_t = J_t W_t^T
     which may be more efficient if D < N.

  Note: after we have computed H_t we can directly compute
     \hat{X}_t = X_t - H_t W_t

  We need to determine how Y_t and Z_t relate to the quantities we just defined.
  First, we'll expand out H_t, J_t, K_t and L_t in terms of the more fundamental quantities.
     H_t = X_t R_t^T E_t^{0.5}
     J_t = E_t^{0.5} R_t X_t^T X_t
     K_t = E_t^{0.5} R_t X_t^T X_t X_t^T X_t R_t^T E_t^{0.5}
     L_t = E_t^{0.5} R_t X_t^T X_t R_t^T E_t^{0.5}

  we wrote above that
      Y_t = \eta R_t S_t + (1-\eta) (D_t + \rho_t I) R_t
  so
      Y_t = \eta/N R_t X_t^T X_t   + (1-\eta) (D_t + \rho_t I) R_t
          = \eta/N E_t^{-0.5} J_t  + (1-\eta) (D_t + \rho_t I) R_t     (eqn:yt)
  We will expand Z_t using the expression for Y_t in the line above:
      Z_t = Y_t Y_t^T
          =  (\eta/N)^2 E_t^{-0.5} J_t J_t^T E_t^{-0.5}
            +(\eta/N)(1-\eta) E_t^{-0.5} J_t R_t^T (D_t + \rho_t I)
            +(\eta/N)(1-\eta) (D_t + \rho_t I) R_t J_t^T E_t^{-0.5}
            +(1-\eta)^2 (D_t + \rho_t I)^2
          = (\eta/N)^2 E_t^{-0.5} K_t E_t^{-0.5}
           +(\eta/N)(1-\eta) E_t^{-0.5} L_t E_t^{-0.5} (D_t + \rho_t I)
           +(\eta/N)(1-\eta) (D_t + \rho_t I) E_t^{-0.5} L_t E_t^{-0.5}
           +(1-\eta)^2 (D_t + \rho_t I)^2                              (eqn:Zt)
  We compute Z_t on the CPU using the expression above, and then do the symmetric
  eigenvalue decomposition (also on the CPU):
      Z_t = U_t C_t U_t^T.
  and we make sure the eigenvalues are sorted from largest to smallest, for
  reasons that will be mentioned later.

  Mathematically, no diagonal element of C_t can be less than (1-\eta)^2
  \rho_t^2, and since negative or zero elements of C_t would cause us a problem
  later, we floor C_t to this value.  (see below regarding how we ensure R_{t+1}
  has orthonormal rows).

  We will continue the discussion below regarding what we do with C_t and U_t.
  Next, we need to digress briefly and describe how to compute
  tr(\hat{X}_t \hat{X}_t^T) and tr(X_t X_t^2), since these appear in expressions for
  \gamma_t (needed to produce the output \bar{X}_t), and for \rho_{t+1}.  It happens
  that we need, for purposes of appying "max_change" in the neural net code, the
  squared 2-norm of each row of the output \bar{X}_t.  In order to be able to compute
  \gamma_t, it's most convenient to compute this squared row-norm for each row
  of \hat{X}_t, as a vector, to compute tr(\hat{X}_t \hat{X}_t^2) from this vector as its sum, and
  to then work back to compute tr(X_t X_t^2) from the relation between \hat{X}_t and
  X_t.  We can then scale the row-norms we computed for \hat{X}_t, so they apply to
  \bar{X}_t.

  For current purposes, you can imagine that we computed tr(\hat{X}_t \hat{X}_t^T) directly.
  Using (from eqn:pt2)
      \hat{X}_t = X_t - X_t W_t^T W_t,
  we can expand tr(\hat{X}_t \hat{X}_t^T) as:
   tr(\hat{X}_t \hat{X}_t^T) = tr(X_t X_t^T) + tr(X_t W_t^T W_t W_t^T W_t X_t^T)
                  - 2 tr(X_t W_t^T W_t X_t^T)
                 = tr(X_t X_t^T) + tr(W_t X_t^T X_t W_t^T W_t W_t^T)
                  - 2 tr(W_t X_t^T X_t W_t^T)
                 = tr(X_t X_t^T) + tr(L_t W_t W_t^T) - 2 tr(L_t)
                 = tr(X_t X_t^T) + tr(L_t E_t) - 2 tr(L_t)
  and all quantities have already been computed (or are quick to compute, such as
  the small traces on the right), except tr(X_t X_t^T), so we can write

    tr(X_t X_t^T) = tr(\hat{X}_t \hat{X}_t^T) - tr(L_t E_t) + 2 tr(L_t)
  and the above expression can be used to obtain tr(X_t X_t^2).
  We can then do
     \gamma_t <-- sqrt(tr(X_t X_t^T)  / tr(\hat{X}_t \hat{X}_t^T)).
  (or one if the denominator is zero), and then
      \bar{X}_t <-- \gamma_t \hat{X}_t
  We can then output the per-row squared-l2-norms of Q by scaling those we
  computed from P by \gamma_t^2.

  OK, the digression on how to compute \gamma_t and tr(X_t X_t^T) is over.
  We now return to the computation of R_{t+1}, W_{t+1}, \rho_{t+1}, D_{t+1} and E_{t+1}.

  We found above in (eqn:rhot1)
     \rho_{t+1} = 1/(D - R) (\eta tr(S_t) + (1-\eta)(D \rho_t + tr(D_t)) - tr(C_t^{0.5})).
  Expanding out S_t from its definition in (eqn:St),
     \rho_{t+1} = 1/(D - R) (\eta/N tr(X_t X_t^T) + (1-\eta)(D \rho_t + tr(D_t)) - tr(C_t^{0.5})).
  We can compute this directly as all the quantities involved are already known
  or easy to compute.
  Next, from (eqn:dt1), we compute
     D_{t+1} = C_t^{0.5} - \rho_{t+1} I
  At this point if \rho_{t+1} is smaller than some small value \epsilon, e.g. 1.0e-10, we
  set it to \epsilon; as mentioned, we do this to stop F_t approaching zero if all inputs
  are zero.  Next, if any diagonal element D_{t+1,i,i} has absolute value less
  than \epsilon, we set it to +\epsilon.  This is to ensure that diagonal
  elements of E are never zero, which would cause problems.

  Next, we compute (from eqn:betat2, eqn:etdef, eqn:tii),
        \beta_{t+1} = \rho_{t+1} (1+\alpha) + \alpha/D tr(D_{t+1})
            E_{t+1} = 1/\beta_{t+1} (D_{t+1}^{-1} + 1/\beta_{t+1} I)^{-1},
 i.e.:      e_{tii} = 1/(\beta_{t+1}/d_{t+1,ii} + 1)

 We'll want to store D_{t+1}.  We next want to compute W_{t+1}.

  Before computing W_{t+1}, we need to find an expression for
     R_{t+1} = C_t^{-0.5} U_t^T Y_t
   Expanding out Y_t using the expression in (eqn:yt),
     R_{t+1} = C_t^{-0.5} U_t^T  (\eta/N E_t^{-0.5} J_t  + (1-\eta) (D_t + \rho_t I) R_t)
             =  (\eta/N C_t^{-0.5} U_t^T E_t^{-0.5})  J_t
               +((1-\eta) C_t^{-0.5} U_t^T (D_t + \rho_t I) E_t^{-0.5}) W_t

   What we actually want is W_{t+1} = E_{t+1}^{0.5} R_{t+1}:
     W_{t+1} = (\eta/N E_{t+1}^{0.5} C_t^{-0.5} U_t^T E_t^{-0.5}) J_t
              +((1-\eta) E_{t+1}^{0.5} C_t^{-0.5} U_t^T (D_t + \rho_t I) E_t^{-0.5}) W_t
   and to minimize the number of matrix-matrix multiplies we can factorize this as:
     W_{t+1} = A_t B_t
        A_t = (\eta/N) E_{t+1}^{0.5} C_t^{-0.5} U_t^T E_t^{-0.5}
        B_t = J_t + (1-\eta)/(\eta/N) (D_t + \rho_t I) W_t
   [note: we use the fact that (D_t + \rho_t I) and E_t^{-0.5} commute because
    they are diagonal].

  A_t is computed on the CPU and transferred from there to the GPU, B_t is
  computed on the PGU, and the multiplication of A_t with B_t is done on the GPU.

   * Keeping R_t orthogonal *

   Our method requires the R_t matrices to be orthogonal (which we define to
   mean that R_t R_t^T = I).  If roundoff error causes this equality to be
   significantly violated, it could cause a problem for the stability of our
   method.  We now address our method for making sure that the R_t values stay
   orthogonal.  We do this in the algorithm described above, after creating
   W_{t+1}.  This extra step is only executed if the condition number of C_t
   (i.e. the ratio of its largest to smallest diagonal element) exceeds a
   specified threshold, such as 1.0e+06 [this is tested before applying the
   floor to C_t].  The threshold was determined empirically by finding the
   largest value needed to ensure a certain level of orthogonality in R_{t+1}.
   For purposes of the present discussion, since R_{t+1} is not actually stored,
   define it as E_{t+1}^{-0.5} W_{t+1}.  Define the following (and we will
   just use t instead of t+1 below, as all quantities have the same subscript):

      O_t =(def) R_t R_t^T
          =  E_t^{-0.5} W_t W_t^T E_t^{-0.5}

   (and we would compute this by computing W_t W_t^T on the GPU, transferring
   it to the CPU, and doing the rest there).  If O_t is not sufficiently close
   to the unit matrix, we can re-orthogonalize as follows:
   Do the Cholesky decomposition
      O_t = C C^T
   Clearly C^{-1} O_t C^{-T} = I, so if we correct R_t with:
      R_t <-- C^{-1} R_t
   we can ensure orthogonality.  If R_t's first k rows are orthogonal, this
   transform will not affect them, because of its lower-triangular
   structure... this is good because (thanks to the eigenvalue sorting), the
   larger eigenvectors are first and it is more critical to keep them pointing
   in the same direction.  Any loss of orthogonality will be dealt with by
   modifying the smaller eigenvectors.
   As a modification to W_t, this would be:
      W_t <-- (E_t^{0.5} C^{-1} E_t^{-0.5}) W_t,
   and the matrix in parentheses is computed on the CPU, transferred to the
   GPU, and the multiplication is done there.


   * Initialization *

   Now, a note on what we do on time t = 0, i.e. for the first minibatch.  We
   initialize X_0 to the top R eigenvectors of 1/N X_0 X_0^T, where N is the
   minibatch size (num-rows of R0).  If L is the corresponding RxR diagonal
   matrix of eigenvalues, then we will set D_0 = L - \rho_0 I.  We set \rho_0
   to ensure that
                      tr(F_0) = 1/N tr(X_0 X_0^T),
           tr(D_0) - \rho_0 D = 1/N tr(X_0 X_0^T),
  tr(L) + \rho_0 R - \rho_0 D = 1/N tr(X_0 X_0^T)
                       \rho_0 = (1/N tr(X_0 X_0^T) - tr(L)) / (D - R)

   We then floor \rho_0 to \epsilon (e.g. 1.0e-10) and also floor the
   diagonal elements of D_0 to \epsilon; this ensures that we won't
   crash for zero inputs.

   A note on multi-threading.  This technique was really designed for use
   with a GPU, where we won't have multi-threading, but we want it to work
   also on a CPU, where we may have multiple worker threads.
   Our approach is as follows (we do this when we're about to start updating
   the parameters R_t, D_t, \rho_t and derived quantities):

    For time t > 0 (where the matrices are already initialized), before starting
    the part of the computation that updates the parameters (R_t, D_t, \rho_t and
    derived quantities), we try to lock a mutex that guards the OnlinePreconditioner.
    If we can lock it right away, we go ahead and do the update, but if not,
    we just abandon the attempt to update those quantities.

    We will have another mutex to ensure that when we access quantities like
    W_t, \rho_t they are all "in sync" (and we don't access them while they are
    being written by another thread).  This mutex will only be locked for short
    periods of time.

   Note: it might be a good idea to make sure that the R_t still retain orthonormal
   rows even in the presence of roundoff, without errors accumulating.  My instinct
   is that this isn't going to be a problem.
 */


class OnlineNaturalGradient {
 public:
  OnlineNaturalGradient();

  void SetRank(int32 rank);
  void SetUpdatePeriod(int32 update_period);
  // num_samples_history is a time-constant (in samples) that determines eta.
  void SetNumSamplesHistory(BaseFloat num_samples_history);
  void SetAlpha(BaseFloat alpha);
  void TurnOnDebug() { self_debug_ = true; }
  BaseFloat GetNumSamplesHistory() const { return num_samples_history_; }
  BaseFloat GetAlpha() const { return alpha_; }
  int32 GetRank() const { return rank_; }
  int32 GetUpdatePeriod() const { return update_period_; }

  // The "R" pointer is both the input (R in the comment) and the output (P in
  // the comment; equal to the preconditioned directions before scaling by
  // gamma).  If the pointer "row_prod" is supplied, it's set to the inner product
  // of each row of the preconditioned directions P, at output, with itself.
  // You would need to apply "scale" to R and "scale * scale" to row_prod, to
  // get the preconditioned directions; we don't do this ourselves, in order to
  // save CUDA calls.
  void PreconditionDirections(CuMatrixBase<BaseFloat> *R,
                              CuVectorBase<BaseFloat> *row_prod,
                              BaseFloat *scale);

  // Copy constructor.
  explicit OnlineNaturalGradient(const OnlineNaturalGradient &other);
  // Assignent operator
  OnlineNaturalGradient &operator = (const OnlineNaturalGradient &other);
 private:

  // This does the work of PreconditionDirections (the top-level
  // function handles some multithreading issues and then calls this function).
  // Note: WJKL_t (dimension 2*R by D + R) is [ W_t L_t; J_t K_t ].
  void PreconditionDirectionsInternal(const int32 t,
                                      const BaseFloat rho_t,
                                      const Vector<BaseFloat> &d_t,
                                      CuMatrixBase<BaseFloat> *WJKL_t,
                                      CuMatrixBase<BaseFloat> *X_t,
                                      CuVectorBase<BaseFloat> *row_prod,
                                      BaseFloat *scale);

  void ComputeEt(const VectorBase<BaseFloat> &d_t,
                 BaseFloat beta_t,
                 VectorBase<BaseFloat> *e_t,
                 VectorBase<BaseFloat> *sqrt_e_t,
                 VectorBase<BaseFloat> *inv_sqrt_e_t) const;

  void ComputeZt(int32 N,
                 BaseFloat rho_t,
                 const VectorBase<BaseFloat> &d_t,
                 const VectorBase<BaseFloat> &inv_sqrt_e_t,
                 const MatrixBase<BaseFloat> &K_t,
                 const MatrixBase<BaseFloat> &L_t,
                 SpMatrix<double> *Z_t) const;
  // Computes W_{t+1}.  Overwrites J_t.
  void ComputeWt1(int32 N,
                  const VectorBase<BaseFloat> &d_t,
                  const VectorBase<BaseFloat> &d_t1,
                  BaseFloat rho_t,
                  BaseFloat rho_t1,
                  const MatrixBase<BaseFloat> &U_t,
                  const VectorBase<BaseFloat> &sqrt_c_t,
                  const VectorBase<BaseFloat> &inv_sqrt_e_t,
                  const CuMatrixBase<BaseFloat> &W_t,
                  CuMatrixBase<BaseFloat> *J_t,
                  CuMatrixBase<BaseFloat> *W_t1) const;

  // This function is called if C_t has high condition number; it makes sure
  // that R_{t+1} is orthogonal.  See the section in the extended comment above
  // on "keeping R_t orthogonal".
  void ReorthogonalizeXt1(const VectorBase<BaseFloat> &d_t1,
                          BaseFloat rho_t1,
                          CuMatrixBase<BaseFloat> *W_t1,
                          CuMatrixBase<BaseFloat> *temp_W,
                          CuMatrixBase<BaseFloat> *temp_O);

  void Init(const CuMatrixBase<BaseFloat> &R0);

  // Initialize to some small 'default' values, called from Init().  Init() then
  // does a few iterations of update with the first batch's data to give more
  // reasonable values.
  void InitDefault(int32 D);

  // initializes R, which is assumed to have at least as many columns as rows,
  // to a specially designed matrix with orthonormal rows, that has no zero rows
  // or columns.
  static void InitOrthonormalSpecial(CuMatrixBase<BaseFloat> *R);

  // Returns the learning rate eta as the function of the number of samples
  // (actually, N is the number of vectors we're preconditioning, which due to
  // context is not always exactly the same as the number of samples).  The
  // value returned depends on num_samples_history_.
  BaseFloat Eta(int32 N) const;

  // called if self_debug_ = true, makes sure the members satisfy certain
  // properties.
  void SelfTest() const;

  // Configuration values:

  // The rank of the correction to the unit matrix (e.g. 20).
  int32 rank_;

  // After a few initial iterations of updating whenever we can, we start only
  // updating the Fisher-matrix parameters every "update_period_" minibatches;
  // this saves time.
  int32 update_period_;

  // num_samples_history_ determines the value of eta, which in turn affects how
  // fast we update our estimate of the covariance matrix.  We've done it this
  // way in order to make it easy to have a single configuration value that
  // doesn't have to be changed when we change the minibatch size.
  BaseFloat num_samples_history_;

  // alpha controls how much we smooth the Fisher matrix with the unit matrix.
  // e.g. alpha = 4.0.
  BaseFloat alpha_;

  // epsilon is an absolute floor on the unit-matrix scaling factor rho_t in our
  // Fisher estimate, which we set to 1.0e-10.  We don't actually make this
  // configurable from the command line.  It's needed to avoid crashes on
  // all-zero inputs.
  BaseFloat epsilon_;

  // delta is a relative floor on the unit-matrix scaling factor rho_t in our
  // Fisher estimate, which we set to 1.0e-05: this is relative to the largest
  // value of D_t.  It's needed to control roundoff error.  We apply the same
  // floor to the eigenvalues in D_t.
  BaseFloat delta_;

  // t is a counter that measures how many updates we've done.
  int32 t_;

  // This keeps track of how many minibatches we've skipped updating the parameters,
  // since the most recent update; it's used in enforcing "update_period_", which
  // is a mechanism to avoid spending too much time updating the subspace (which can
  // be wasteful).
  int32 num_updates_skipped_;

  // If true, activates certain checks.
  bool self_debug_;

  CuMatrix<BaseFloat> W_t_;
  BaseFloat rho_t_;
  Vector<BaseFloat> d_t_;


  // Used to prevent parameters being read or written in an inconsistent state.
  Mutex read_write_mutex_;

  // This mutex is used to control which thread gets to update the
  // parameters, in multi-threaded code.
  Mutex update_mutex_;
};

} // namespace nnet3
} // namespace kaldi


#endif
