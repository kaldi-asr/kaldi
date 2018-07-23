// cudamatrix/cu-math.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (Author: David Snyder)

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



#ifndef KALDI_CUDAMATRIX_CU_MATH_H_
#define KALDI_CUDAMATRIX_CU_MATH_H_
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-device.h"
#include "base/timer.h"

namespace kaldi {

namespace cu {

/// RegularizeL1 is a gradient step with l1 regularization added to the
/// gradient.  We don't let the value cross over zero from positive to negative
/// or vice versa, in a single step.  If an element tries to cross zero and is
/// stopped, we zero the gradient.  (Dan: not sure why).
template<typename Real>
void RegularizeL1(CuMatrixBase<Real> *weight, CuMatrixBase<Real> *gradient,
                  Real l1_penalty, Real learning_rate);

/// Copies a permutation of src into tgt. The row permutation is specified in
/// copy_from_idx such that src.Row(copy_from_idx[r]) == tgt.Row(r). The
/// dimensions of copy_from_idx must be equivalent to the number of rows in
/// tgt and src and all elements in the vector must be in [0, src.numRows()-1].
template<typename Real>
void Randomize(const CuMatrixBase<Real> &src,
               const CuArray<int32> &copy_from_idx,
               CuMatrixBase<Real> *tgt);

/// Splice concatenates frames of src as specified in frame_offsets into tgt.
/// The dimensions of tgt must be equivalent to the number of rows in src
/// and it must be that tgt.NumColumns == src.NumColumns * frame_offsets.Dim().
/// As a result, tgt(i, k*n_cols + j) == src(i + frame_offsets[k], j) for the
/// general case where i in [0..src.NumRows()-1],
/// k in [0..frame_offsets.Dim()-1], j in [0..src.NumRows()-1]
/// and n_cols = src.NumColumns(). If i + frame_offsets[k] is greater than the
/// number of rows in src or less than 0 than the right side of the equation
/// is replaced by src(src.NumRows()-1, j) or src(0, j) respectively, to avoid
/// an index out of bounds.
template<typename Real>
void Splice(const CuMatrixBase<Real> &src,
            const CuArray<int32> &frame_offsets,
            CuMatrixBase<Real> *tgt);

/// Copies elements from src into tgt as given by copy_from_indices.
/// The matrices src and tgt must have the same dimensions and
/// the dimension of copy_from_indices must equal the number of columns
/// in the src matrix. As a result, tgt(i, j) == src(i, copy_from_indices[j]).
/// Also see CuMatrix::CopyCols(), which is more general.
template<typename Real>
void Copy(const CuMatrixBase<Real> &src,
          const CuArray<int32> &copy_from_indices,
          CuMatrixBase<Real> *tgt);


/// This function requires that src and dest have the same dimension and epsilon
/// > 0.  It copies src to dest while ensuring that the values are bounded away
/// from zero by at least epsilon:
/// \code
///  y =  x if fabs(x) >= epsilon;
///       epsilon if 0 <= x < epsilon;
///       -epsilon if -epsilon < x < 0.
/// \endcode
template <typename Real>
void EnsureNonzero(const CuMatrixBase<Real> &src,
                   Real epsilon,
                   CuMatrixBase<Real> *dest);

/// Vector version of EnsureNonzero, see matrix version for documentation.
template <typename Real>
void EnsureNonzero(const CuVectorBase<Real> &src,
                   Real epsilon,
                   CuVectorBase<Real> *dest);

/**
 this is a special-purpose function used by class LstmNonlinearityComponent,
 to do its forward propagation.  It computes the core part of the LSTM nonlinearity.
 Refer to class LstmNonlinearityComponent in ../nnet3/nnet-simple-component.h
 for more context.

 @param [in]  input  A matrix, of dimension N by 5C (i.e. its num-cols must be
                     a multiple of 5).  The column-space is interpreted as 5
                     consecutive blocks, each of dimension C, which we name:
                     (i_part, f_part, c_part, o_part, c_{t-1}).
                     This function will also accept input of dimension N by 5C + 3,
                     and the three final elements will be used as scaling factors
                     on i_t, f_t and o_t (useful as per-frame dropout masks).
 @param [in] params  A matrix, of dimension 3 by C, with rows containing the three
                     diagonal parameter matrices used in LSTMs, namely
                     w_{ic}, w_{fc} and w_{oc}.
 @param [out] output A matrix, of dimension N by 2C.  The quantities c_t and m_t
                     respectively are put there (in two blocks of column-dimension C),
                     according to the following equations:

                     i_t = Sigmoid(i_part + w_{ic}*c_{t-1})
                     f_t = Sigmoid(f_part + w_{fc}*c_{t-1})
                     c_t = f_t*c_{t-1} + i_t * Tanh(c_part)
                     o_t = Sigmoid(o_part + w_{oc}*c_t)
                     m_t = o_t * Tanh(c_t)

                     Note on dropout: if the dropout mask is provided, let the
                     mask values be i_t_mask, f_t_mask and o_t_mask (for each
                     matrix row, these are scalars while i_t, f_t and o_t are of
                     dimension C, because this is 'per-frame' dropout as described in
                     http://www.danielpovey.com/files/2017_interspeech_dropout.pdf).
                     Then the modification to the equations above consists of
                     replacing 'i_t' with 'i_t_mask * i_t' in the RHS of the equations
                     above, and the same type of change for f_t and o_t.
 */
template<typename Real>
void ComputeLstmNonlinearity(const CuMatrixBase<Real> &input,
                             const CuMatrixBase<Real> &params,
                             CuMatrixBase<Real> *output);
// This is a version of ComputeLstmNonlinearity that only uses the CPU
// even if a GPU is available. It's made available for testing purposes.
template<typename Real>
void CpuComputeLstmNonlinearity(const MatrixBase<Real> &input,
                                const MatrixBase<Real> &params,
                                MatrixBase<Real> *output);


/**
   This function does the 'backward' pass corresponding to the function
   ComputeLstmNonlinearity.  It's a little more complicated than you might
   expect because of the 'self-repair' mechanism that we use to prevent the
   sigmoid and tanh nonlinearities oversaturating,  and because of the
   average-activation and average-derivative stats that we store for these
   nonlinearites (these stats are used both to control the self-repair
   mechanism, and for diagnostic purposes).

   Because the forward pass computes various intermediate values that are not
   output, this function actually has to do the same computations as the
   forward pass before it actually does the backprop.


 @param [in]  input  The same as in ComputeLstmNonlinearity().
                     A matrix, of dimension N by 5C (i.e. its num-cols must be
                     a multiple of 5).  The column-space is interpreted as 5
                     consecutive blocks, each of dimension C, which we name:
                     (i_part, f_part, c_part, o_part, c_{t-1}).
                     This function will also accept input of dimension N by 5C + 3,
                     and the three final elements will be interpreted as scaling factors
                     on i_t, f_t and o_t (useful as per-frame dropout masks).
 @param [in] params  The same as in ComputeLstmNonlinearity().
                     A matrix, of dimension 3 by C, with rows containing the three
                     diagonal parameter matrices used in LSTMs, namely
                     w_{ic}, w_{fc} and w_{oc}.
 @param [in] output_deriv
                     A matrix, of dimension N by 2C, containing the derivative of the
                     objective function we're backpropagating, w.r.t. the quantities
                     c_t and m_t (in two blocks of column-dimension C).
 @param [in] deriv_sum_in
                     This is used in the self-repair code to identify oversaturated
                     nonlinearities.  It is a matrix, of dimension 5 by C, corresponding
                     to the totals of the derivatives of the 5 sigmoid and tanh
                     nonlinearities, in they order they appear in the equations
                     in the documentation of ComputeLstmNonlinearity() Rspectively,
                     they appear in the equations for (i_t, f_t, c_t, o_t, m_t).
                     This will be divided by 'count_in' to get the average derivative
                     value so far, for each of the nonlinearities.
 @param [in] self_repair_config
                     A vector of dimension 10, containing the configuration of the self-repair
                     to be used for the 5 nonlinearities.  The first 5 elements are the
                     self_repair_lower_threshold values (typically 0.05 for sigmoid and 0.2
                     for tanh), and the next 5 elements are the corresponding
                     self-repair-scales (typically 10^-5).
 @param [in] count_in  The data-count that corresponds to the stats in 'deriv_sum_in'
                     at entry to the function.  This function should tolerate the count
                     being zero (in that case, it is free to do the self-repair or not,
                     as this should only happen on the 1st minibatch of each training job).
 @param [out] input_deriv
                     May be NULL; if not, this function writes, to this
                     location, the backpropagated derivative of the objective
                     function w.r.t. the 'input' matrix.  This matrix should
                     have the same dimension as 'input'.  In addition to the
                     regular backpropagated derivative, the output will include
                     small values relating to 'self-repair'.  If the input
                     is of column-dimension  5C + 3 (i.e. we are using dropout
                     masks), the derivatives w.r.t. the dropout masks will not
                     be set; they will retain their value prior to this
                     function call.
 @param [out] params_deriv
                     May be NULL; if not, this is where this function *writes*
                     [not adds] the backpropagated derivative of the objective
                     function w.r.t. 'params'; it should have the same dimension
                     as 'params' (3 by C).  (This matrix will then be processed
                     by the natural gradient code and added to the appropriate
                     copy of the parameter matrix, outside this function).
 @param [out] value_sum_out
                     Must be NULL if params_deriv is NULL; if not, a matrix of
                     dimension 5 by C.  This function *adds* to this location
                     the total value of each of the sigmoid/tanh nonlinearities
                     that it computes (this is for diagnostic purposes).
 @param [out] deriv_sum_out
                     Must be NULL if params_deriv is NULL; if not, a matrix of
                     dimension 5 by C; this function *adds* to this location the
                     total of the derivative of each of the sigmoid/tanh
                     nonlinearities that it computes (this is for diagnostic
                     purposes and to control the self-repair).  This function
                     should tolerate the case when 'deriv_sum_out' points to the
                     same data as 'deriv_sum_in'.
 @param [out] self_repair_sum_out
                     Must be NULL if params_deriv is NULL; if not, a matrix of
                     dimension 5 by C; this function *writes* to this location
                     the sum of the number of times the self-repair code was
                     activated (integer values 0 <= k <= N).  This will be
                     processed outside this function into self-repair stats for
                     diagnostics.
*/

template<typename Real>
void BackpropLstmNonlinearity(const CuMatrixBase<Real> &input,
                              const CuMatrixBase<Real> &params,
                              const CuMatrixBase<Real> &output_deriv,
                              const CuMatrixBase<double> &deriv_sum_in,
                              const CuVectorBase<Real> &self_repair_config,
                              double count_in,
                              CuMatrixBase<Real> *input_deriv,
                              CuMatrixBase<Real> *params_deriv,
                              CuMatrixBase<double> *value_sum_out,
                              CuMatrixBase<double> *deriv_sum_out,
                              CuMatrixBase<Real> *self_repair_sum_out);
// This is a version of BackpropLstmNonlinearity that only uses the CPU
// even if a GPU is available. It's made available for testing purposes.
template<typename Real>
void CpuBackpropLstmNonlinearity(const MatrixBase<Real> &input,
                                 const MatrixBase<Real> &params,
                                 const MatrixBase<Real> &output_deriv,
                                 const MatrixBase<double> &deriv_sum_in,
                                 const VectorBase<Real> &self_repair_config,
                                 double count_in,
                                 MatrixBase<Real> *input_deriv,
                                 MatrixBase<Real> *params_deriv,
                                 MatrixBase<double> *value_sum_out,
                                 MatrixBase<double> *deriv_sum_out,
                                 MatrixBase<Real> *self_repair_sum_out);

/// Normalize nonlinearity modifies the vector of activations
/// by scaling it so that the root-mean-square equals 1.0.
///
/// The output y_i = scale * x_i,
/// and we want to RMS value of the y_i to equal target_rms,
/// so y^t y = D * target_rms^2 (if y is one row of the input).
/// we need to have scale = 1.0 / sqrt(x^t x / (D * target_rms^2)).
/// there is also flooring involved, to avoid division-by-zero
/// problems.  It's important for the backprop, that the floor's
/// square root is exactly representable as float.
/// If add_log_stddev_ is true, log(max(epsi, sqrt(x^t x / D)))
/// is an extra dimension of the output.
template<typename Real>
void NormalizePerRow(const CuMatrixBase<Real>& in, const Real target_rms,
                     const bool add_log_stddev, CuMatrixBase<Real>* out);

// A note on the derivative of NormalizeComponent...
// let both row_in and row_out be vectors of dimension D.
// Let p = row_in^T row_in / (D * target_rms^2), and let
// f = 1.0 / sqrt(max(kSquaredNormFloor, p)), and we compute row_out as:
// row_out = f row_in.
// Suppose we have a quantity deriv_out which is the derivative
// of the objective function w.r.t. row_out.  We want to compute
// deriv_in which is the derivative of the objective function w.r.t.
// row_in.  Let the objective function be F.  One term is obvious: we have
// deriv_in = f deriv_out + ....
// next we have to take into account the derivative that gets back-propagated
// through f.  Obviously, dF/df = deriv_out^T row_in.
// And df/dp = (p <= kSquaredNormFloor ? 0.0 : -0.5 p^{-1.5}) = (f == 1.0 / sqrt(kSquaredNormFloor) ? 0.0 : -0.5 f^3),
// and dp/d(row_in) = 2/(D * target_rms^2) row_in. [it's vector_valued].
// So this term in dF/d(row_in) equals:
// dF/df df/dp dp/d(row_in)   =    2/(D * target_rms^2) (f == 1.0 / sqrt(kSquaredNormFloor)  ? 0.0 : -0.5 f^3) (deriv_out^T row_in) row_in
// So
// deriv_in = f deriv_out + (f == 1.0 ? 0.0 : -f^3  / (D * target_rms^2) ) (deriv_out^T row_in) row_in
//  if add_log_stddev_ true, the deriv_in has another term as
// dF/dx_i = dF/df . df/dx_i => df/dx_i = x_i/(x^T x)
template<typename Real>
void DiffNormalizePerRow(const CuMatrixBase<Real> &in_value,
                         const CuMatrixBase<Real> &out_deriv,
                         const Real target_rms, const bool add_log_stddev,
                         CuMatrixBase<Real>* in_deriv);


} // namespace cu
} // namespace kaldi


#endif
