// nnet3/attention.cc

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/attention.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {
namespace attention {


void GetAttentionDotProducts(BaseFloat alpha,
                             const CuMatrixBase<BaseFloat> &A,
                             const CuMatrixBase<BaseFloat> &B,
                             CuMatrixBase<BaseFloat> *C) {
  KALDI_ASSERT(A.NumCols() == B.NumCols() &&
               A.NumRows() == C->NumRows());
  int32 num_output_rows = A.NumRows(),
      input_num_cols = A.NumCols(),
      num_extra_rows = B.NumRows() - A.NumRows(),
      context_dim = C->NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C->NumCols(),
                             C->NumRows());
  for (int32 o = 0; o < context_dim; o++) {
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    c_col.AddDiagMatMat(alpha, A, kNoTrans, B_part, kTrans, 0.0);
  }
  C->CopyFromMat(Ctrans, kTrans);
}

void ApplyScalesToOutput(BaseFloat alpha,
                         const CuMatrixBase<BaseFloat> &B,
                         const CuMatrixBase<BaseFloat> &C,
                         CuMatrixBase<BaseFloat> *A) {
  KALDI_ASSERT(A->NumCols() == B.NumCols() &&
               A->NumRows() == C.NumRows());
  int32 num_output_rows = A->NumRows(),
      input_num_cols = A->NumCols(),
      num_extra_rows = B.NumRows() - A->NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C, kTrans);
  for (int32 o = 0; o < context_dim; o++) {
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    A->AddDiagVecMat(alpha, c_col, B_part, kNoTrans, 1.0);
  }
}

void ApplyScalesToInput(BaseFloat alpha,
                        const CuMatrixBase<BaseFloat> &A,
                        const CuMatrixBase<BaseFloat> &C,
                        CuMatrixBase<BaseFloat> *B) {
  KALDI_ASSERT(A.NumCols() == B->NumCols() &&
               A.NumRows() == C.NumRows());
  int32 num_output_rows = A.NumRows(),
      input_num_cols = A.NumCols(),
      num_extra_rows = B->NumRows() - A.NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C, kTrans);
  for (int32 o = 0; o < context_dim; o++) {
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(*B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    B_part.AddDiagVecMat(alpha, c_col, A, kNoTrans, 1.0);
  }
}

void AttentionForward(BaseFloat key_scale,
                      const CuMatrixBase<BaseFloat> &keys,
                      const CuMatrixBase<BaseFloat> &queries,
                      const CuMatrixBase<BaseFloat> &values,
                      CuMatrixBase<BaseFloat> *c,
                      CuMatrixBase<BaseFloat> *output) {
  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
      context_dim = queries.NumCols() - key_dim,
      value_dim = values.NumCols();
  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows > num_output_rows &&
               context_dim > 0 &&
               (num_input_rows - num_output_rows) % (context_dim - 1) == 0 &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(c->NumRows() == num_output_rows &&
               c->NumCols() == context_dim);
  KALDI_ASSERT(output->NumRows() == num_output_rows &&
               (output->NumCols() == value_dim ||
                output->NumCols() == value_dim + context_dim));

  CuSubMatrix<BaseFloat> queries_key_part(
      queries, 0, num_output_rows,
      0, key_dim),
      queries_context_part(
          queries, 0, num_output_rows,
          key_dim, context_dim);

  GetAttentionDotProducts(key_scale,
                          queries_key_part,
                          keys, c);
  // think of 'queries_context_part' as a position-dependent bias term.
  c->AddMat(1.0, queries_context_part);
  // compute the soft-max function.  Up till this point, 'c'
  // actually contained what in attention.h we called 'b', which is
  // the input to the softmax.
  c->SoftMaxPerRow(*c);


  // the part of the output that is weighted
  // combinations of the input values.
  CuSubMatrix<BaseFloat> output_values_part(
      *output, 0, num_output_rows, 0, value_dim);

  ApplyScalesToOutput(1.0, values, *c, &output_values_part);


  if (output->NumCols() == value_dim + context_dim) {
    CuSubMatrix<BaseFloat> output_context_part(
        *output, 0, num_output_rows, value_dim, context_dim);
    output_context_part.CopyFromMat(*c);
  }
}

void AttentionBackward(BaseFloat key_scale,
                       const CuMatrixBase<BaseFloat> &keys,
                       const CuMatrixBase<BaseFloat> &queries,
                       const CuMatrixBase<BaseFloat> &values,
                       const CuMatrixBase<BaseFloat> &c,
                       const CuMatrixBase<BaseFloat> &output_deriv,
                       CuMatrixBase<BaseFloat> *keys_deriv,
                       CuMatrixBase<BaseFloat> *queries_deriv,
                       CuMatrixBase<BaseFloat> *values_deriv) {

  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
      context_dim = queries.NumCols() - key_dim,
      value_dim = values.NumCols();
  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows > num_output_rows &&
               context_dim > 0 &&
               (num_input_rows - num_output_rows) % (context_dim - 1) == 0 &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(SameDim(keys, *keys_deriv) &&
               SameDim(queries, *queries_deriv) &&
               SameDim(values, *values_deriv));

  KALDI_ASSERT(c.NumRows() == num_output_rows &&
               c.NumCols() == context_dim);
  KALDI_ASSERT(output_deriv.NumRows() == num_output_rows &&
               (output_deriv.NumCols() == value_dim ||
                output_deriv.NumCols() == value_dim + context_dim));

  CuMatrix<BaseFloat> c_deriv(num_output_rows, context_dim,
                              kUndefined);

  CuSubMatrix<BaseFloat> output_values_part_deriv(
      output_deriv, 0, num_output_rows, 0, value_dim);
  // This is the backprop w.r.t. the forward-pass statement:
  // ApplyScalesToOutput(1.0, values, *c, &output_values_part);
  GetAttentionDotProducts(1.0, output_values_part_deriv,
                          values, &c_deriv);

  if (output_deriv.NumCols() == value_dim + context_dim) {
    CuSubMatrix<BaseFloat> output_deriv_context_part(
        output_deriv, 0, num_output_rows, value_dim, context_dim);
    // this is the backprop w.r.t. the
    // forward-pass statement: output_context_part.CopyFromMat(*c);
    c_deriv.AddMat(1.0, output_deriv_context_part);
  }

  // Propagate the derivatives back through the softmax nonlinearity,
  // in-place; this is the backprop w.r.t. the statement
  // 'c->SoftMaxPerRow(*c);'.  From this point on, c_deriv actually
  // contains the derivative to the pre-softmax values which we call
  // 'b' in the math.
  c_deriv.DiffSoftmaxPerRow(c, c_deriv);


  CuSubMatrix<BaseFloat> queries_key_part(
      queries, 0, num_output_rows,
      0, key_dim),
      queries_key_part_deriv(
          *queries_deriv, 0, num_output_rows,
          0, key_dim),
      queries_context_part_deriv(
          *queries_deriv, 0, num_output_rows,
          key_dim, context_dim);

  // Below is the backprop corresponding to the forward-propagation command:
  // c->AddMat(1.0, queries_context_part)
  queries_context_part_deriv.AddMat(1.0, c_deriv);

  // The following statement is the part of the backprop w.r.t. the
  // statement:
  // GetAttentionDotProducts(key_scale, queries_key_part, keys, c);
  // which propagates the derivative back to 'queries_key_part'.
  ApplyScalesToOutput(key_scale, keys, c_deriv, &queries_key_part_deriv);

  // The following statement is the part of the backprop w.r.t. the
  // statement:
  // GetAttentionDotProducts(key_scale, queries_key_part, keys, c);
  // which propagates the derivative back to 'keys'.
  ApplyScalesToInput(key_scale, queries_key_part, c_deriv, keys_deriv);

  // The followign statement is the part of the backprop w.r.t.
  // the statement:
  // ApplyScalesToOutput(1.0, values, *c, &output_values_part);
  // which propagates the derivative back to 'values'.
  ApplyScalesToInput(1.0, output_values_part_deriv, c,  values_deriv);
}

} // namespace attention
} // namespace nnet3
} // namespace kaldi
