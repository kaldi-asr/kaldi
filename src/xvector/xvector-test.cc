// ivector/xvector-test.cc

// Copyright 2016  David Snyder

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

#include "xvector/xvector.h"
#include "util/kaldi-io.h"
#include "cudamatrix/cu-matrix-lib.h"

namespace kaldi {
BaseFloat TestSimilarityScore(const CuVector<BaseFloat> &v,
  const CuVector<BaseFloat> &w, const CuSpMatrix<BaseFloat> &S,
  BaseFloat b);

void TestGetDeriv(const CuVector<BaseFloat> &v,
    const CuVector<BaseFloat> &w, const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, bool is_same, BaseFloat similarity_score,
    CuVector<BaseFloat> *deriv_v, CuVector<BaseFloat> *deriv_w,
    CuVector<BaseFloat> *deriv_S, BaseFloat *deriv_b);

void TestComputeXvectorObjfAndDeriv(
    const CuMatrixBase<BaseFloat> &xvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S, BaseFloat *deriv_b, BaseFloat *tot_objf,
    BaseFloat *tot_weight);

bool TestXvectorExtractorDerivative(BaseFloat perturb_delta) {
  int32 xvector_dim = RandInt(4, 100),
        num_rows = 2 * RandInt(2, 10); // The number of rows must be even
                                       // and greater than 2.
  int32 num_rows_subset = RandInt(1, num_rows);
  CuSpMatrix<BaseFloat> S(xvector_dim);
  S.SetRandn();
  // Necessary to keep the similarity scores from getting too large or small.
  S.Scale(1.0e-01);
  BaseFloat b = RandInt(-100, 100) / 10.0,
            tot_weight,
            tot_objf,
            deriv_b;
  int32 S_dim = S.NumCols() * (S.NumCols() + 1) / 2;
  CuMatrix<BaseFloat> xvector_pairs(num_rows, xvector_dim, kSetZero),
                      deriv_xvector(num_rows, xvector_dim, kSetZero);
  CuVector<BaseFloat> deriv_S(S_dim, kSetZero);
  xvector_pairs.SetRandn();
  ComputeXvectorObjfAndDeriv(xvector_pairs, S, b, &deriv_xvector,
    &deriv_S, &deriv_b, NULL, &tot_objf, &tot_weight);
  CuVector<BaseFloat> deriv_xvector_vec(xvector_dim);

  // Sum over the derivatives for xvector input.
  deriv_xvector_vec.AddRowSumMat(1.0, deriv_xvector.RowRange(0, num_rows_subset),
                                 0.0);
  BaseFloat l2_xvector = 0,
            l2_S = 0,
            l2_b = 0;

  // Compare the xvector derivatives calculated above with a numerical
  // approximation.
  for (int32 i = 0; i < xvector_dim; i++) {
    CuMatrix<BaseFloat> xvector_pairs_p(xvector_pairs);
    CuMatrix<BaseFloat> xvector_pairs_n(xvector_pairs);
    for (int32 j = 0; j < num_rows_subset; j++) {
      xvector_pairs_p(j, i) += perturb_delta;
      xvector_pairs_n(j, i) += -perturb_delta;
    }
    BaseFloat tot_objf_p,
        tot_objf_n;
    ComputeXvectorObjfAndDeriv(xvector_pairs_p, S, b, NULL,
      NULL, NULL, NULL, &tot_objf_p, &tot_weight);
    ComputeXvectorObjfAndDeriv(xvector_pairs_n, S, b, NULL,
      NULL, NULL, NULL, &tot_objf_n, &tot_weight);
    BaseFloat delta = (tot_objf_p  - tot_objf_n)
      * 1.0 / (2.0 * perturb_delta);
    l2_xvector += pow(deriv_xvector_vec(i) - delta, 2);
  }

  // Compare the S derivative calculated above with a numerical
  // approximation.
  for (int32 i = 0; i < S_dim; i++) {
    CuSpMatrix<BaseFloat> S_p(S);
    CuSpMatrix<BaseFloat> S_n(S);
    CuSubVector<BaseFloat> S_p_vec(S_p.Data(), S_dim);
    CuSubVector<BaseFloat> S_n_vec(S_n.Data(), S_dim);
    S_p_vec(i) += perturb_delta;
    S_n_vec(i) += -perturb_delta;
    BaseFloat tot_objf_p,
              tot_objf_n;
    ComputeXvectorObjfAndDeriv(xvector_pairs, S_p, b, NULL,
      NULL, NULL, NULL, &tot_objf_p, &tot_weight);
    ComputeXvectorObjfAndDeriv(xvector_pairs, S_n, b, NULL,
      NULL, NULL, NULL, &tot_objf_n, &tot_weight);
    BaseFloat delta = (tot_objf_p  - tot_objf_n)
      * 1.0 / (2.0 * perturb_delta);
    l2_S += pow(deriv_S(i) - delta, 2);
  }

  // Compare the b derivative calculated above with a numerical
  // approximation.
  BaseFloat b_p = b + perturb_delta;
  BaseFloat b_n = b - perturb_delta;
  BaseFloat tot_objf_p;
  BaseFloat tot_objf_n;
  ComputeXvectorObjfAndDeriv(xvector_pairs, S, b_p, NULL,
    NULL, NULL, NULL, &tot_objf_p, &tot_weight);
  ComputeXvectorObjfAndDeriv(xvector_pairs, S, b_n, NULL,
    NULL, NULL, NULL, &tot_objf_n, &tot_weight);
  BaseFloat delta = (tot_objf_p  - tot_objf_n)
                    * 1.0 / (2.0 * perturb_delta);
  l2_b = pow(deriv_b - delta, 2);
  KALDI_ASSERT(l2_xvector < 1.0e-03);
  KALDI_ASSERT(l2_S <  1.0e-03);
  KALDI_ASSERT(l2_b < 1.0e-03);
  return true;
}

bool TestXvectorComputeObjf() {
  int32 xvector_dim = RandInt(4, 100),
      num_rows = 2 * RandInt(2, 10); // The number of rows must be even
                                       // and greater than 2.
  CuSpMatrix<BaseFloat> S(xvector_dim);
  S.SetRandn();
  // Necessary to keep the similarity scores from getting too large or small.
  S.Scale(1.0e-01);
  BaseFloat b = RandInt(-200, 200) / 10.0,
            tot_weight,
            tot_weight_test,
            tot_objf,
            tot_objf_test,
            deriv_b,
            deriv_b_test;
  int32 S_dim = S.NumCols() * (S.NumCols() + 1) / 2;
  CuMatrix<BaseFloat> xvector_pairs(num_rows, xvector_dim, kSetZero),
                      deriv_xvector(num_rows, xvector_dim, kSetZero),
                      deriv_xvector_test(num_rows, xvector_dim, kSetZero);
  CuVector<BaseFloat> deriv_S(S_dim, kSetZero),
                      deriv_S_test(S_dim, kSetZero);
  xvector_pairs.SetRandn();

  ComputeXvectorObjfAndDeriv(xvector_pairs, S, b, &deriv_xvector,
    &deriv_S, &deriv_b, NULL, &tot_objf, &tot_weight);
  TestComputeXvectorObjfAndDeriv(xvector_pairs, S, b, &deriv_xvector_test,
    &deriv_S_test, &deriv_b_test, &tot_objf_test, &tot_weight_test);

  CuVector<BaseFloat> deriv_xvector_vec(xvector_dim);
  deriv_xvector_vec.AddRowSumMat(1.0, deriv_xvector, 0.0);
  CuVector<BaseFloat> deriv_xvector_vec_test(xvector_dim);
  deriv_xvector_vec_test.AddRowSumMat(1.0, deriv_xvector_test, 0.0);
  KALDI_ASSERT(deriv_xvector.ApproxEqual(deriv_xvector_test, 0.01));

  // Verify that the objfs are the same.
  KALDI_ASSERT(ApproxEqual(tot_objf, tot_objf_test, 0.001));

  // Also verify that the gradients are the same.
  for (int32 i = 0; i < deriv_xvector_vec.Dim(); i++)
    KALDI_ASSERT(ApproxEqual(deriv_xvector_vec(i),
    deriv_xvector_vec_test(i), 0.001));

  // Verify that the S derivates are the same.
  for (int32 i = 0; i < deriv_S.Dim(); i++)
    KALDI_ASSERT(ApproxEqual(deriv_S(i), deriv_S_test(i), 0.001));

  // Verify that the b derivates are the same.
  KALDI_ASSERT(ApproxEqual(deriv_b, deriv_b_test, 0.001));
  return true;
}

void TestComputeXvectorObjfAndDeriv(
    const CuMatrixBase<BaseFloat> &xvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S, BaseFloat *deriv_b, BaseFloat *tot_objf,
    BaseFloat *tot_weight) {

  int32 N = xvector_pairs.NumRows();
  BaseFloat same_objf = 0,
            diff_objf = 0;
  BaseFloat K = 1.0 / (N - 2.0);
  (*deriv_b) = 0;
  // Handle portion of the objf corresponding to pairs of xvectors
  // from the same classes.
  for (int32 i = 0; i < N/2; i++) {
    const CuVector<BaseFloat> &v(xvector_pairs.Row(2 * i)),
                              &w(xvector_pairs.Row(2 * i + 1));
    CuVector<BaseFloat> deriv_v,
                        deriv_w,
                        deriv_S_part;
    BaseFloat similarity_score = TestSimilarityScore(v, w, S, b),
              deriv_b_part = 0;
    same_objf += Log(1 + Exp(-similarity_score));
    TestGetDeriv(v, w, S, b, true, similarity_score, &deriv_v,
     &deriv_w, &deriv_S_part, &deriv_b_part);
    deriv_xvector->Row(2 * i).AddVec(1.0, deriv_v);
    deriv_xvector->Row(2 * i + 1).AddVec(1.0, deriv_w);
    deriv_S->AddVec(1.0, deriv_S_part);
    (*deriv_b) += deriv_b_part;
  }

  // Handle portion of the objf corresponding to pairs of xvectors
  // from different classes.
  for (int32 i = 0; i < N; i++) {
    for (int32 j = 2 * std::ceil((i + 1) / 2.0); j < N; j++) {
      const CuVector<BaseFloat> &v(xvector_pairs.Row(i)),
                                &w(xvector_pairs.Row(j));
      CuVector<BaseFloat> deriv_v,
                          deriv_w,
                          deriv_S_part;
      BaseFloat similarity_score = TestSimilarityScore(v, w, S, b),
              deriv_b_part = 0;
      diff_objf += Log(1 + Exp(similarity_score));
      TestGetDeriv(v, w, S, b, false, similarity_score, &deriv_v,
        &deriv_w, &deriv_S_part, &deriv_b_part);
      deriv_xvector->Row(i).AddVec(K, deriv_v);
      deriv_xvector->Row(j).AddVec(K, deriv_w);
      deriv_S->AddVec(K, deriv_S_part);
      (*deriv_b) += K * deriv_b_part;
    }
  }
  // Scale the same and different portions of the objective function
  // so that both contribute a weight of N.
  (*tot_objf) = -same_objf - K * diff_objf;
  (*tot_weight) = N;
}


void TestGetDeriv(const CuVector<BaseFloat> &v,
    const CuVector<BaseFloat> &w, const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, bool is_same, BaseFloat similarity_score,
    CuVector<BaseFloat> *deriv_v, CuVector<BaseFloat> *deriv_w,
    CuVector<BaseFloat> *deriv_S, BaseFloat *deriv_b) {
  int32 d = is_same ? 1 : -1,
        S_dim = S.NumCols() * (S.NumCols() + 1) / 2;
  deriv_v->Resize(v.Dim(), kSetZero);
  deriv_w->Resize(v.Dim(), kSetZero);
  deriv_S->Resize(S_dim, kSetZero);

  // This scalar is common to the different derivatives.
  BaseFloat deriv_coef = -d * Exp(-1 * d * similarity_score)
    / (1 + Exp(-1 * d * similarity_score));

  // Handle derivative with respect to v and w.
  deriv_v->CopyFromVec(w);
  deriv_w->CopyFromVec(v);
  deriv_v->AddSpVec(2.0, S, v, -1.0);
  deriv_w->AddSpVec(2.0, S, w, -1.0);
  deriv_v->Scale(deriv_coef);
  deriv_w->Scale(deriv_coef);

  // Handle derivative with respect to S.
  CuSpMatrix<BaseFloat> deriv_S_mat(S.NumCols(), kSetZero);
  deriv_S_mat.AddVec2(2.0, v);
  deriv_S_mat.AddVec2(2.0, w);
  for (int32 i = 0; i < S.NumCols(); i++)
    deriv_S_mat(i, i) = 0.5 * deriv_S_mat(i, i);
  CuSubVector<BaseFloat> deriv_S_vec(deriv_S_mat.Data(), S_dim);
  deriv_S->AddVec(deriv_coef, deriv_S_vec);

  // Handle derivative with respect to b.
  (*deriv_b) = -deriv_coef;
}

BaseFloat TestSimilarityScore(const CuVector<BaseFloat> &v,
  const CuVector<BaseFloat> &w, const CuSpMatrix<BaseFloat> &S,
  BaseFloat b) {
  CuVector<BaseFloat> Sv(v.Dim());
  Sv.AddSpVec(1.0, S, v, 0);
  CuVector<BaseFloat> Sw(w.Dim());
  Sw.AddSpVec(1.0, S, w, 0);
  BaseFloat L = VecVec(v, w) - VecVec(v, Sv) - VecVec(w, Sw) + b;
  return L;
}

void UnitTestXvectorExtractor() {
  if (!TestXvectorComputeObjf())
    KALDI_ERR << "Xvector objf test failed";
  if (!TestXvectorExtractorDerivative(1.0e-02) &&
     !TestXvectorExtractorDerivative(1.0e-03) &&
     !TestXvectorExtractorDerivative(1.0e-04) &&
     !TestXvectorExtractorDerivative(1.0e-05))
    KALDI_ERR << "Xvector derivative test failed";
}

} // namespace kaldi

int main() {
  using namespace kaldi;
  for (int32 i = 0; i < 2; i++) {
#if HAVE_CUDA == 1
    if (i == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    UnitTestXvectorExtractor();
  }
  std::cout << "Xvector tests succeeded.\n";
  return 0;
}
