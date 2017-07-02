// nnet3/attention-test.cc

// Copyright 2017    Johns Hopkins University (author:  Hossein Hadian)

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

#include "nnet3/attention.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet3 {
namespace attention {


// (*C)(i, j) = alpha * VecVec(A.Row(i), B.Row(i + j * row_shift))
void GetAttentionDotProductsSimple(BaseFloat alpha,
                                   const CuMatrixBase<BaseFloat> &A,
                                   const CuMatrixBase<BaseFloat> &B,
                                   CuMatrixBase<BaseFloat> *C) {
  KALDI_ASSERT(A.NumCols() == B.NumCols() &&
               A.NumRows() == C->NumRows());
  int32 input_num_cols = A.NumCols(),
      num_extra_rows = B.NumRows() - A.NumRows(),
      context_dim = C->NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  for (int32 i = 0; i < C->NumRows(); i++) {
    for (int32 j = 0; j < C->NumCols(); j++) {
      (*C)(i, j) = 0.0;
      for (int32 k = 0; k < input_num_cols; k++) {
        (*C)(i, j) += alpha * A(i, k) * B(i + (j * row_shift), k);
      }
    }
  }
}

//     A->Row(i) += \sum_k alpha * C(i, k) * B.Row(i + k * row_shift).
void ApplyScalesToOutputSimple(BaseFloat alpha,
                               const CuMatrixBase<BaseFloat> &B,
                               const CuMatrixBase<BaseFloat> &C,
                               CuMatrixBase<BaseFloat> *A) {
  KALDI_ASSERT(A->NumCols() == B.NumCols() &&
               A->NumRows() == C.NumRows());
  int32 num_extra_rows = B.NumRows() - A->NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  for (int32 i = 0; i < A->NumRows(); i++) {
    for (int32 j = 0; j < A->NumCols(); j++) {
      for (int32 k = 0; k < context_dim; k++) {
        (*A)(i, j) += alpha * C(i, k) * B(i + (k * row_shift), j);
      }
    }
  }
}

//     B->Row(i + j * row_shift) += alpha * C(i, j) * A.Row(i).
void ApplyScalesToInputSimple(BaseFloat alpha,
                              const CuMatrixBase<BaseFloat> &A,
                              const CuMatrixBase<BaseFloat> &C,
                              CuMatrixBase<BaseFloat> *B) {
  KALDI_ASSERT(A.NumCols() == B->NumCols() &&
               A.NumRows() == C.NumRows());
  int32 num_extra_rows = B->NumRows() - A.NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  for (int32 i = 0; i < A.NumRows(); i++) {
    for (int32 j = 0; j < A.NumCols(); j++) {
      for (int32 k = 0; k < context_dim; k++) {
        (*B)(i + (k * row_shift), j) += alpha * C(i, k) * A(i, j);
      }
    }
  }
}

void UnitTestAttentionDotProductAndAddScales() {
  int32 output_num_rows = RandInt(1, 50), input_num_cols = RandInt(1, 10),
      row_shift = RandInt(1, 5), context_dim = RandInt(2, 5),
      num_extra_rows = (context_dim - 1) * row_shift,
      input_num_rows = output_num_rows + num_extra_rows;
  BaseFloat alpha = 0.25 * RandInt(1, 5);
  CuMatrix<BaseFloat> A(output_num_rows, input_num_cols),
      B(input_num_rows, input_num_cols),
      C(output_num_rows, context_dim);

  B.SetRandn();
  C.SetRandn();
  A.Set(0.0);
  CuMatrix<BaseFloat> A2(A);
  ApplyScalesToOutput(alpha, B, C, &A);
  ApplyScalesToOutputSimple(alpha, B, C, &A2);
  AssertEqual(A, A2);

  CuMatrix<BaseFloat> C2(C);
  GetAttentionDotProductsSimple(alpha, A, B, &C);
  GetAttentionDotProducts(alpha, A, B, &C2);
  AssertEqual(C, C2);

  CuMatrix<BaseFloat> B2(B);
  ApplyScalesToInput(alpha, A, C, &B);
  ApplyScalesToInputSimple(alpha, A, C, &B2);
  AssertEqual(B, B2);
}

void UnitTestAttention() {
  UnitTestAttentionDotProductAndAddScales();
}


} // namespace attention
} // namespace nnet3
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  using namespace kaldi::nnet3::attention;
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
    for (int32 i = 0; i < 5; i++) {
      UnitTestAttention();
    }
  }
}
