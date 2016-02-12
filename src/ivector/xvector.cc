// ivector/xvector.cc

// Copyright 2016     Daniel Povey
//                    David Snyder

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

#include "ivector/xvector.h"

namespace kaldi {

void ComputeXvectorObjfAndDeriv(
    const CuMatrixBase<BaseFloat> &xvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S_and_b, BaseFloat *tot_objf,
    BaseFloat *tot_weight) {

  int32 N = xvector_pairs.NumRows();
  BaseFloat same_objf = 0,
            diff_objf = 0;
  BaseFloat K = 1.0 / (N - 2.0);
  int32 S_dim = S.NumCols() * (S.NumCols() + 1) / 2;
  CuMatrix<BaseFloat> tmp_deriv(N, xvector_pairs.NumCols()
                                + S_dim + 1, kSetZero);
  // Handle portion of the objf corresponding to pairs of xvectors
  // from the same classes.
  for (int32 i = 0; i < N/2; i++) {
    const CuVector<BaseFloat> &v(xvector_pairs.Row(2 * i)),
                              &w(xvector_pairs.Row(2 * i + 1));
    CuVector<BaseFloat> deriv_v,
                        deriv_w,
                        deriv_S_and_b_part;
    BaseFloat similarity_score = SimilarityScore(v, w, S, b);
    same_objf += Log(1 + Exp(-similarity_score));
    GetDeriv(v, w, S, b, true, similarity_score, &deriv_v,
     &deriv_w, &deriv_S_and_b_part);
    deriv_xvector->Row(2 * i).AddVec(1.0, deriv_v);
    deriv_xvector->Row(2 * i + 1).AddVec(1.0, deriv_w);
    deriv_S_and_b->AddVec(1.0, deriv_S_and_b_part);
  }

  // Handle portion of the objf corresponding to pairs of xvectors
  // from different classes.
  for (int32 i = 0; i < N; i++) {
    for (int32 j = 2 * std::ceil((i + 1) / 2.0); j < N; j++) {
      const CuVector<BaseFloat> &v(xvector_pairs.Row(i)),
                                &w(xvector_pairs.Row(j));
      CuVector<BaseFloat> deriv_v,
                          deriv_w,
                          deriv_S_and_b_part;
      BaseFloat similarity_score = SimilarityScore(v, w, S, b);
      diff_objf += Log(1 + Exp(similarity_score));
      GetDeriv(v, w, S, b, false, similarity_score, &deriv_v,
        &deriv_w, &deriv_S_and_b_part);
      deriv_xvector->Row(i).AddVec(K, deriv_v);
      deriv_xvector->Row(j).AddVec(K, deriv_w);
      deriv_S_and_b->AddVec(K, deriv_S_and_b_part);
    }
  }
  // Scale the same and different portions of the objective function
  // so that both contribute a weight of N.
  (*tot_objf) = same_objf + K * diff_objf;
  (*tot_weight) = 2 * N;
}

void GetDeriv(const CuVector<BaseFloat> &v,
    const CuVector<BaseFloat> &w, const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, bool is_same, BaseFloat similarity_score,
    CuVector<BaseFloat> *deriv_v, CuVector<BaseFloat> *deriv_w,
    CuVector<BaseFloat> *deriv_S_and_b) {
  int32 d = is_same ? 1 : -1,
        S_dim = S.NumCols() * (S.NumCols() + 1) / 2;
  deriv_v->Resize(v.Dim(), kSetZero);
  deriv_w->Resize(v.Dim(), kSetZero);
  deriv_S_and_b->Resize(S_dim + 1, kSetZero);

  // This scalar is common to the different derivatives.
  BaseFloat deriv_coef = d * Exp(-1 * d * similarity_score)
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
  CuSubVector<BaseFloat> sub_deriv_S_and_b(*deriv_S_and_b, 0, S_dim);
  sub_deriv_S_and_b.AddVec(deriv_coef, deriv_S_vec);

  // Handle derivative with respect to b.
  (*deriv_S_and_b)(S_dim) = -deriv_coef;
}

BaseFloat SimilarityScore(const CuVector<BaseFloat> &v,
  const CuVector<BaseFloat> &w, const CuSpMatrix<BaseFloat> &S,
  BaseFloat b) {
  CuVector<BaseFloat> Sv(v.Dim());
  Sv.AddSpVec(1.0, S, v, 0);
  CuVector<BaseFloat> Sw(w.Dim());
  Sw.AddSpVec(1.0, S, w, 0);
  BaseFloat L = VecVec(v, w) - VecVec(v, Sv) - VecVec(w, Sw) + b;
  return L;
}

} // namespace kaldi
