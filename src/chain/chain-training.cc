// chain/chain-training.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)
//                2018   Hossein Hadian

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

#include "chain/chain-training.h"
#include "chain/chain-kernels-ansi.h"
#include "chain/chain-numerator.h"
#include "chain/chain-generic-numerator.h"
#include "chain/chain-denominator.h"

namespace kaldi {
namespace chain {


/**
   This is a rather special-purpose function which adds something to
   the derivative in order to encourage the value to stay within
   a specified range.  This is something we use in chain training
   in order to encourage the nnet outputs to stay within the
   range [-30, 30] (needed because we don't do the forward-backward
   denominator computation in log space).

   It's very similar to l2 regularization but only applied once you depart
   the range [-limit, limit].

   Basically, this function does as follows:

     (*out_deriv)(i,j) +=   0                                if   -limit <= in_value(i,j) <= limit
                            (-limit - in_value(i,j)) * scale if  in_value(i,j) < -limit
                            (limit - in_value(i,j)) * scale  if  in_value(i,j) > limit
   If limit were zero, this would be the same as l2 regularization with scale 'scale'.
 */
static void PenalizeOutOfRange(const CuMatrixBase<BaseFloat> &in_value,
                               BaseFloat limit,
                               BaseFloat scale,
                               CuMatrixBase<BaseFloat> *out_deriv) {
  KALDI_ASSERT(SameDim(in_value, *out_deriv) && limit > 0 && scale >= 0);
  if (scale == 0)
    return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(in_value.NumCols(), CU2DBLOCK),
                 n_blocks(in_value.NumRows(), CU2DBLOCK));
    cuda_penalize_out_of_range(dimGrid, dimBlock, limit, scale,
                               in_value.Data(), in_value.Dim(),
                               out_deriv->Stride(), out_deriv->Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    int32 num_rows = in_value.NumRows(),
        num_cols = in_value.NumCols();
    for (int32 r = 0; r < num_rows; r++) {
      const BaseFloat *in_row_data =  in_value.RowData(r);
      BaseFloat *out_row_data = out_deriv->RowData(r);
      for (int32 c = 0; c < num_cols; c++) {
        BaseFloat val = in_row_data[c];
        if (val < -limit) {
          out_row_data[c] -= scale * (val + limit);
        } else if (val > limit) {
          out_row_data[c] -= scale * (val - limit);
        }
      }
    }
  }
}


void ComputeChainObjfAndDerivE2e(const ChainTrainingOptions &opts,
                                 const DenominatorGraph &den_graph,
                                 const Supervision &supervision,
                                 const CuMatrixBase<BaseFloat> &nnet_output,
                                 BaseFloat *objf,
                                 BaseFloat *l2_term,
                                 BaseFloat *weight,
                                 CuMatrixBase<BaseFloat> *nnet_output_deriv,
                                 CuMatrix<BaseFloat> *xent_output_deriv) {
  NVTX_RANGE(__func__);
  BaseFloat num_logprob_weighted, den_logprob_weighted;
  bool denominator_ok = true;
  bool numerator_ok = true;
  *weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;

  if (nnet_output_deriv != NULL)
    nnet_output_deriv->SetZero();

  if (nnet_output_deriv != NULL && RandInt(0, 1) == 0) {
    // Only do this about every other frame, for efficiency; we'll multiply the
    // scale by 2 to compensate.  See docs for the function, for its purpose.
    PenalizeOutOfRange(nnet_output, 30.0,
                       2.0 * opts.out_of_range_regularize,
                       nnet_output_deriv);
  }

  { // Doing the denominator first helps to reduce the maximum
    // memory use, as we can set 'xent_deriv' to nonempty after
    // we've freed the memory in this object.
    DenominatorComputation denominator(opts, den_graph,
                                       supervision.num_sequences,
                                       nnet_output);

    den_logprob_weighted = supervision.weight * denominator.Forward();
    if (nnet_output_deriv)
      denominator_ok = denominator.Backward(-supervision.weight,
                                nnet_output_deriv);
  }

  if (xent_output_deriv != NULL) {
    // the reason for kStrideEqualNumCols is so that we can share the memory
    // block with the memory that was used for exp_nnet_output_transposed_ from
    // chain-denominator.cc, which has just been freed; it also uses the
    // kStrideEqualNumCols arg (its shape is the transpose of this matrix's
    // shape).
    xent_output_deriv->Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                              kSetZero, kStrideEqualNumCols);
  }


  {
    GenericNumeratorComputation numerator(opts.numerator_opts,
                                          supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, as well as the returned logprob.
    if (xent_output_deriv) {
      numerator_ok = numerator.ForwardBackward(&num_logprob_weighted,
                                               xent_output_deriv);
      if (numerator_ok && nnet_output_deriv)
        nnet_output_deriv->AddMat(1.0, *xent_output_deriv);
    } else if (nnet_output_deriv) {
      numerator_ok = numerator.ForwardBackward(&num_logprob_weighted,
                                               nnet_output_deriv);
    } else {
      num_logprob_weighted = numerator.ComputeObjf();
    }
    if (!numerator_ok)
        KALDI_WARN << "Numerator forward-backward failed.";
  }
  numerator_ok = numerator_ok &&
                 (num_logprob_weighted - num_logprob_weighted == 0);

  *objf = num_logprob_weighted - den_logprob_weighted;
  if (!((*objf) - (*objf) == 0) || !denominator_ok || !numerator_ok) {
    // inf or NaN detected, or denominator computation returned false.
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    if (xent_output_deriv)
      xent_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*objf)
               << " and denominator computation (if done) returned "
               << std::boolalpha << denominator_ok
               << " and numerator computation returned "
               << std::boolalpha << numerator_ok
               << ", setting objective function to " << default_objf
               << " per frame.";
    *objf  = default_objf * *weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1 && nnet_output_deriv != NULL && RandInt(0, 10) == 0) {
    int32 tot_frames = nnet_output_deriv->NumRows(),
 frames_per_sequence = supervision.frames_per_sequence,
       num_sequences = supervision.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }

  *l2_term = 0.0;
  if (opts.l2_regularize != 0.0 && numerator_ok) {  // we should have some derivs to include a L2 term
    // compute the l2 penalty term and its derivative
    BaseFloat scale = supervision.weight * opts.l2_regularize;
    *l2_term = -0.5 * scale * TraceMatMat(nnet_output, nnet_output, kTrans);
    if (nnet_output_deriv)
      nnet_output_deriv->AddMat(-1.0 * scale, nnet_output);
  }
}


void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *l2_term,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv,
                              CuMatrix<BaseFloat> *xent_output_deriv) {
  NVTX_RANGE(__func__);
  if (!supervision.e2e_fsts.empty()) {
    ComputeChainObjfAndDerivE2e(opts, den_graph, supervision,
                                nnet_output, objf, l2_term,
                                weight, nnet_output_deriv, xent_output_deriv);
    return;
  }

  BaseFloat num_logprob_weighted, den_logprob_weighted;
  bool ok = true;
  if (nnet_output_deriv != NULL)
    nnet_output_deriv->SetZero();

  { // Doing the denominator first helps to reduce the maximum
    // memory use, as we can set 'xent_deriv' to nonempty after
    // we've freed the memory in this object.
    DenominatorComputation denominator(opts, den_graph,
                                       supervision.num_sequences,
                                       nnet_output);

    den_logprob_weighted = supervision.weight * denominator.Forward();
    if (nnet_output_deriv)
      ok = denominator.Backward(-supervision.weight,
                                nnet_output_deriv);
  }

  if (nnet_output_deriv != NULL && RandInt(0, 1) == 0) {
    // Only do this about every other frame, for efficiency; we'll multiply the
    // scale by 2 to compensate.  See docs for the function, for its purpose.
    PenalizeOutOfRange(nnet_output, 30.0,
                       2.0 * opts.out_of_range_regularize,
                       nnet_output_deriv);
  }

  if (xent_output_deriv != NULL) {
    // the reason for kStrideEqualNumCols is so that we can share the memory
    // block with the memory that was used for exp_nnet_output_transposed_ from
    // chain-denominator.cc, which has just been freed; it also uses the
    // kStrideEqualNumCols arg (its shape is the transpose of this matrix's
    // shape).
    xent_output_deriv->Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                              kSetZero, kStrideEqualNumCols);
  }


  {
    NumeratorComputation numerator(supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, as well as the returned logprob.
    num_logprob_weighted = numerator.Forward();

    if (xent_output_deriv) {
      numerator.Backward(xent_output_deriv);
      if (nnet_output_deriv)
        nnet_output_deriv->AddMat(1.0, *xent_output_deriv);
    } else if (nnet_output_deriv) {
      numerator.Backward(nnet_output_deriv);
    }
  }

  *objf = num_logprob_weighted - den_logprob_weighted;
  *weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;
  if (!((*objf) - (*objf) == 0) || !ok) {
    // inf or NaN detected, or denominator computation returned false.
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    if (xent_output_deriv)
      xent_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*objf)
               << " and denominator computation (if done) returned "
               << std::boolalpha << ok
               << ", setting objective function to " << default_objf
               << " per frame.";
    *objf  = default_objf * *weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1 && nnet_output_deriv != NULL && RandInt(0, 10) == 0) {
    int32 tot_frames = nnet_output_deriv->NumRows(),
 frames_per_sequence = supervision.frames_per_sequence,
       num_sequences = supervision.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }

  if (opts.l2_regularize == 0.0) {
    *l2_term = 0.0;
  } else {
    // compute the l2 penalty term and its derivative
    BaseFloat scale = supervision.weight * opts.l2_regularize;
    *l2_term = -0.5 * scale * TraceMatMat(nnet_output, nnet_output, kTrans);
    if (nnet_output_deriv)
      nnet_output_deriv->AddMat(-1.0 * scale, nnet_output);
  }
}


}  // namespace chain
}  // namespace kaldi
