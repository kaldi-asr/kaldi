// nnet3/attention.h

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)
//                      Hossein Hadian

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

#ifndef KALDI_NNET3_ATTENTION_H_
#define KALDI_NNET3_ATTENTION_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "itf/options-itf.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "nnet3/nnet-common.h"
#include "nnet3/convolution.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {
namespace attention {

/// @file  attention.h
///
/// This file contains the lower-level interface for self-attention.
/// This is a form of self-attention, inspired by Google's paper
/// "Attention is all you need", but implemented in a way that's more
/// obviously suitable for speech tasks.  The main difference is that
/// instead of taking as input *all frames* from the previous layer,
/// we accept a limited grid of frames (so the left-context and
/// right-context are finite).  Also time-encoding is handled in a different
/// way-- we encode the time as a relative offset.



// Our attention is "multi-head", like in Google's paper.  Note: we're basically
// implementing multi-head attention as a fixed nonlinearity, with the actual
// parameters relegated to the previous layer.  That is, the attention layer
// won't have any parameters of its own, but the parameters of the preceding
// layer will be interpretable as the parameters.  It doesn't change what's
// computed, it just affects how the neural net is divided into components.
//
//  * Basic restricted self-attention (without positional encoding).
//
// To explain what's going on, we start with the simplest form of attention:
// single-head, and no positional encoding, but with restricted context.  For purposes
// of exposition we assume that the time offsets we need form a contiguous
// range, i.e. with time-stride == 1; the code does have the notion of a stride (you'll
// see later).
//
// Using notation similar to the Google paper, suppose we have a time-sequence
// of inputs, and the inputs are (keys, values and queries):
//
//   k_t, v_t, q_t
//
// where k_t and q_t are vectors of dimension 'key_dim' and v_t is a vector
// of dimension 'value_dim' (you may choose to make this the same as key_dim, but
// that isn't a constraint).

// Let's make num_left_inputs and num_right_inputs be the number of
// left-context and right-context frames required, and for some t,
// let input_indexes(t) be the set
//  [ t - num_left_inputs, t - num_left_inputs + 1, ... t + num_right_inputs].
// To evaluate the output (which we'll write u_t), we need the query
// value q_t, plus the keys and values k_s and v_s for all s in input_indexes(t).
// If the inputs are not available for some subset of input_indexes(t),
// we just let them be zeros; the network can learn to ignore them if it wants,
// but making them zeros is simpler to implement.
//
//
// Anyway, the output u_t (without positional encoding yet) is:
//
//  u_t := \sum_{s in input_indexes(t)}  Z_t exp(q_t . k_s) v_s
//
// where Z_t is 1/(\sum_s exp(q_t . k_s)).  We'll handle scaling
// issues (the 1/sqrt(dim) factor in the Google paper) later on,
// by scaling the keys.
//
//
// * Positional encoding
// We now explain how we include positional encoding in the model.
//
//
// Let context_dim = 1 + num_left_inputs + num_right_inputs.
// Let v be a vector, and let the function Extend(v, o) (where
// 0 <= o < context_dim) extend v with extra dimensions
// that encode the time-offset.  To be precise, we have
//
//  Extend(v, o) = Append(v, u_o)
//
// where u_o is a unit vector of dimension context_dim that is nonzero in the
// o'th dimension (assuming zero-based indexing).
//
// So when we add the positional encoding (and the scale on the keys), we replace
// the equation:
//  u_t := \sum_{s in input_indexes(t)}  Z_t exp(q_t . k_s) v_s
// with:
//  u_t := \sum_{s in input_indexes(t)}  Z_t exp(alpha q_t . Extend(key-scale * k_s, s - t + num_left_inputs)) Extend(v_s, s - t + num_left_inputs)
//
// (we won't actually physically extend the vectors as we compute this,
// we'll do it a different way, but it's equivalent to what we wrote
// above. The dimension of q_t is key_dim + context_dim, and the dimension
// of the output u_t is value_dim + context_dim.
//
//
// * Multi-head attention
//
// The attention component if we had a single head, would have an input dimension
// of (2*key_dim + context_dim + value_dim), which would be interpreted
// as Append(k_t, q_t, v_t), of dimensions respectively
// (key_dim, key_dim + context_dim, value_dim).  It would have an output
// dimension of value_dim + context_dim.
//
// In any case, the multi-head version has input and output dimension that
// are larger by a factor of 'num_heads', and which is equivalent to
// several single-head components appended together.
//
//
//
//  * The actual calculation
//
// Let's assume that we might have multiple independent sequences; we'll
// call this 'num_images' because we're borrowing certain structures from
// the convolution code.

// The input is formatted as a matrix, whose NumRows() could be written as
// num_images * num_t_in, where num_t_in is the number of distinct input 't'
// values, and whose output is num_images * num_t_out.  To keep it simple we'll
// explain this under the assumption that we don't have any 't' stride in the
// attention (t_stride == 1 in the code), and that num_heads == 1; both of
// those are fairly simple modifications to the basic scheme.
// The image (normally 'n') index has a higher stride than the 't' index in
// both the input and the output.  We assume that there is 'enough'
// context of the input to compute all required offsets of the output.
//
// Define the intermediate quantity b_{t,o}, which you can think of
// as the input to softmax; the index 't' is the output time-index
// index at the output, and o ranges from 0 to context_dim - 1.
//
//    b_{t,o} =  q_t . Extend(key-scale * k_{t + o - num_left_inputs}, o)
//
// To get rid of the Extend() expressions, define sub-ranges of q_t by
// writing q_t = Append(r_t, s_t) where r_t is of dimension 'key_dim'
// and s_t is of dimension context_dim.
//
//    b_{t,o} =   s_{t,o}  +  key-scale (r_t . k_{t+o-num_left_inputs})  [eqn:b]
//
// The 'b' quantity is the input to the softmax.  Define
//     c_t = Softmax(b_t)
// so \sum_o c_{t,o} = 1.0.  These are the weights on the values.
//
//
//  The output can be written as:
//        u_t :=  \sum_o c_{t,o} Extend(v_{t+o-num_left_inputs}, o)
//  but we can write this in a form more suitable for computation as:
//     u_t :=  Append(\sum_o c_{t,o} v_{t+o-num_left_inputs},  c_t)     [eqn:u]
//
//
//  * Implementation
//
// The most time-consuming parts of this computation, we imagine, would be the
// dot-products in [eqn:b] and the weighted sum (\sum_o) in [eqn:u].  They have
// an awkward band-diagonal structure that would not be particularly convenient
// to implement using CUBLAS and the like; I don't believe the relevant operations
// exist in the BLAS interface, at least for [eqn:u].
//
// In the future I hope to implement this with block-wise matrix
// multiplies-- imagine covering the band-diagonal part of a matrix with
// rectangular blocks in such a way that all the nonzero elements are covered,
// but the blocks might go over the zero parts a bit.   This could be done with
// Or perhaps we can figure out how to implement the block-diagonal matrix
// multiplies in CUDA.



/**
   This function is a utility function that is at the core of how we implement
   attention.  It may in future need to be renamed and possibly moved into the
   cudamatrix directory and implemented in CUDA.  The current implementation is
   quite inefficient.  We can also consider doing a complete redesign of how the
   implementation works, such that this function doesn't exist at all; or we
   could have a batched version of this function that would operate on a batch
   of A, B and C at once (or a "strided, batched" version where the difference
   between the members of the batch is expressed as a stride).

   This function implements a special operation that you could view as some kind
   of matrix multiplication where only a band of the product is retained.

   The inputs A and B must have the same number of columns
   (A.NumCols() == B.NumCols()), and A and C must have the same
   number of rows (A.NumRows() == C->NumRows()).  The number of
   rows of B must exceed the number of rows of A.  Define
      num_extra_rows = B.NumRows() - A.NumRows().
   Then C.NumCols() - 1 must divide num_extra_rows.
   Define
      row_shift = num_extra_rows / (C.NumCols() - 1).

   This function implements:
      (*C)(i, j) = alpha * VecVec(A.Row(i), B.Row(i + j * row_shift))
 */
void GetAttentionDotProducts(BaseFloat alpha,
                             const CuMatrixBase<BaseFloat> &A,
                             const CuMatrixBase<BaseFloat> &B,
                             CuMatrixBase<BaseFloat> *C);


/**
   This function is related to GetAttentionDotProducts(); it
   is used in scaling the values by the softmax scales, and
   in backprop.

   We have put the A, B and C in an unusual order here in order
   to make clearer the relationship with GetAttentionDotProducts().
   The matrices have the same relationship in terms of their
   dimensions, as A, B and C do in GetAttentionDotProducts().

   This function implements:

     A->Row(i) += \sum_j alpha * C(i, j) * B.Row(i + j * row_shift).
 */
void ApplyScalesToOutput(BaseFloat alpha,
                         const CuMatrixBase<BaseFloat> &B,
                         const CuMatrixBase<BaseFloat> &C,
                         CuMatrixBase<BaseFloat> *A);


/**
   This function is related to GetAttentionDotProducts(); it
   is used in backprop.

   We have put the A, B and C in an unusual order here in order
   to make clearer the relationship with GetAttentionDotProducts().
   The matrices have the same relationship in terms of their
   dimensions, as A, B and C do in GetAttentionDotProducts().

   This function implements:

     B->Row(i + j * row_shift) += alpha * C(i, j) * A.Row(i).
 */
void ApplyScalesToInput(BaseFloat alpha,
                        const CuMatrixBase<BaseFloat> &A,
                        const CuMatrixBase<BaseFloat> &C,
                        CuMatrixBase<BaseFloat> *B);



/**
   This is a higher-level interface to the attention code.
   Read the extended comment in the file nnet3/attention.h for context.

     @param [in] key_scale   Scale on the non-context part of the keys.
     @param [in] keys       Matrix whose rows contains the keys, dimension is
                            num-input-rows by key-dim.
     @param [in] queries    Matrix whose rows contains the queries, dimension
                            is num-output-rows by query-dim, where query-dim
                            == key-dim + context-dim.
                            num-output-rows - num-input-rows must be a multiple
                            of context-dim - 1 (we'll 'shift' the keys by multiples
                            of 0, n, 2n, ... (context-dim - 1) * n.
     @param [in] values     Values to average at the output, of dimension
                            num-input-rows by value-dim.  [we may add context
                            information to these averages if required, see comment
                            for 'output'].
     @param [out] c         Expected to be finite at entry (no infs or nan's);
                            at exit this will contain the output of the softmax.
                            Must be of dimension num-output-rows by context-dim.
     @param [out] output    The output of the attention mechanism will be *added*
                            to this location.  Dimension must be num-output-rows
                            by either value-dim, or value-dim + context-dim.  To
                            the first 'value-dim' columns of this will be added
                            the weighted combination of 'values', weighted by
                            the corresponding weights of 'c' (e.g. the first
                            column of 'c' scaling the first 'output-dim' rows of
                            'values', then the next column of 'c' scaling the
                            submatrix of 'values' shifted by 'n', and so on.
                            If the output->NumCols() is value-dim + context-dim,
                            'c' will be added to the remaining columns of
                            'output'.
 */
void AttentionForward(BaseFloat key_scale,
                      const CuMatrixBase<BaseFloat> &keys,
                      const CuMatrixBase<BaseFloat> &queries,
                      const CuMatrixBase<BaseFloat> &values,
                      CuMatrixBase<BaseFloat> *c,
                      CuMatrixBase<BaseFloat> *output);

/** Performs the backward pass corresponding to 'AttentionForward',
    propagating the derivative back to the keys, queries and values.

    The interface should be easy to understand with reference
    to AttentionForward(), so we won't document it, except to note
    that 'keys_deriv', 'queries_deriv' and 'values_deriv' are
    *added to*, not set, by this function.
 */
void AttentionBackward(BaseFloat key_scale,
                       const CuMatrixBase<BaseFloat> &keys,
                       const CuMatrixBase<BaseFloat> &queries,
                       const CuMatrixBase<BaseFloat> &values,
                       const CuMatrixBase<BaseFloat> &c,
                       const CuMatrixBase<BaseFloat> &output_deriv,
                       CuMatrixBase<BaseFloat> *keys_deriv,
                       CuMatrixBase<BaseFloat> *queries_deriv,
                       CuMatrixBase<BaseFloat> *values_deriv);






} // namespace attention
} // namespace nnet3
} // namespace kaldi


#endif
