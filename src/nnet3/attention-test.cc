// nnet3/attention-test.cc

// Copyright      2017  Hossein Hadian
//                2017  Johns Hopkins University (author: Daniel Povey)

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

void TestAttentionForwardBackward() {
  BaseFloat key_scale = 0.5 * RandInt(1, 3);
  BaseFloat epsilon = 1.0e-03;
  int32 test_dim = 3;
  bool output_context = (RandInt(0, 1) == 0);
  int32 output_num_rows = RandInt(1, 50),
      value_dim = RandInt(10, 30), key_dim = RandInt(10, 30),
      row_shift = RandInt(1, 5), context_dim = RandInt(2, 5),
      num_extra_rows = (context_dim - 1) * row_shift,
      input_num_rows = output_num_rows + num_extra_rows,
      query_dim = key_dim + context_dim;
  CuMatrix<BaseFloat> keys(input_num_rows, key_dim),
      queries(output_num_rows, query_dim),
      values(input_num_rows, value_dim),
      C(output_num_rows, context_dim),
      output(output_num_rows, value_dim + (output_context ? context_dim : 0));


  keys.SetRandn();
  queries.SetRandn();
  values.SetRandn();


  AttentionForward(key_scale, keys, queries, values, &C, &output);

  CuMatrix<BaseFloat> keys_deriv(input_num_rows, key_dim),
      queries_deriv(output_num_rows, query_dim),
      values_deriv(input_num_rows, value_dim),
      output_deriv(output_num_rows, output.NumCols());

  output_deriv.SetRandn();

  AttentionBackward(key_scale, keys, queries, values, C,
                    output_deriv, &keys_deriv, &queries_deriv,
                    &values_deriv);

  BaseFloat objf_baseline = TraceMatMat(output_deriv, output, kTrans);




  {  // perturb the values and see if the objf changes as predicted.
    Vector<BaseFloat> predicted_vec(test_dim), observed_vec(test_dim);
    for (int32 i = 0; i < test_dim; i++) {
      CuMatrix<BaseFloat> values2(input_num_rows, value_dim);
      values2.SetRandn();
      values2.Scale(epsilon);
      BaseFloat predicted_delta_objf = TraceMatMat(values_deriv, values2, kTrans);
      values2.AddMat(1.0, values);

      output.SetZero();
      AttentionForward(key_scale, keys, queries, values2, &C, &output);
      BaseFloat objf2 = TraceMatMat(output_deriv, output, kTrans),
          observed_delta_objf = objf2 - objf_baseline;
      KALDI_LOG << "Changing values: predicted objf change is "
                << predicted_delta_objf << ", observed objf change is "
                << observed_delta_objf;
      predicted_vec(i) = predicted_delta_objf;
      observed_vec(i) = observed_delta_objf;
    }
    KALDI_ASSERT(predicted_vec.ApproxEqual(observed_vec, 0.1));
  }

  {  // perturb the keys and see if the objf changes as predicted.
    Vector<BaseFloat> predicted_vec(test_dim), observed_vec(test_dim);
    for (int32 i = 0; i < test_dim; i++) {
      CuMatrix<BaseFloat> keys2(input_num_rows, key_dim);
      keys2.SetRandn();
      keys2.Scale(epsilon);
      BaseFloat predicted_delta_objf = TraceMatMat(keys_deriv, keys2, kTrans);
      keys2.AddMat(1.0, keys);

      output.SetZero();
      AttentionForward(key_scale, keys2, queries, values, &C, &output);
      BaseFloat objf2 = TraceMatMat(output_deriv, output, kTrans),
          observed_delta_objf = objf2 - objf_baseline;
      KALDI_LOG << "Changing keys: predicted objf change is "
                << predicted_delta_objf << ", observed objf change is "
                << observed_delta_objf;
      predicted_vec(i) = predicted_delta_objf;
      observed_vec(i) = observed_delta_objf;
    }
    KALDI_ASSERT(predicted_vec.ApproxEqual(observed_vec, 0.1));
  }


  {  // perturb the queries and see if the objf changes as predicted.
    Vector<BaseFloat> predicted_vec(test_dim), observed_vec(test_dim);
    for (int32 i = 0; i < test_dim; i++) {
      CuMatrix<BaseFloat> queries2(output_num_rows, query_dim);
      queries2.SetRandn();
      queries2.Scale(epsilon);
      BaseFloat predicted_delta_objf = TraceMatMat(queries_deriv, queries2, kTrans);
      queries2.AddMat(1.0, queries);

      output.SetZero();
      AttentionForward(key_scale, keys, queries2, values, &C, &output);
      BaseFloat objf2 = TraceMatMat(output_deriv, output, kTrans),
          observed_delta_objf = objf2 - objf_baseline;
      KALDI_LOG << "Changing queries: predicted objf change is "
                << predicted_delta_objf << ", observed objf change is "
                << observed_delta_objf;
      predicted_vec(i) = predicted_delta_objf;
      observed_vec(i) = observed_delta_objf;
    }
    KALDI_ASSERT(predicted_vec.ApproxEqual(observed_vec, 0.1));
  }
}

void UnitTestAttention() {
  UnitTestAttentionDotProductAndAddScales();
  TestAttentionForwardBackward();
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
