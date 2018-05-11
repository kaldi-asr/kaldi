// xvector/xvector.cc

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

namespace kaldi {

void ComputeXvectorObjfAndDeriv(
    const CuMatrixBase<BaseFloat> &xvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S, BaseFloat *deriv_b,
    CuMatrixBase<BaseFloat> *scores_out,
    BaseFloat *tot_objf,
    BaseFloat *tot_weight) {

  int32 S_dim = S.NumCols() * (S.NumCols() + 1) / 2,
        N = xvector_pairs.NumRows(),
        xvector_dim = xvector_pairs.NumCols();
  (*tot_objf) = 0;

  if (deriv_xvector == NULL)
    KALDI_ASSERT(deriv_S == NULL && deriv_b == NULL);
  else {
    KALDI_ASSERT(deriv_xvector->NumCols() == xvector_dim);
    KALDI_ASSERT(deriv_xvector->NumRows() == N);
    KALDI_ASSERT(deriv_S->Dim() == S_dim);
    deriv_xvector->SetZero();
    deriv_S->SetZero();
  }


  CuMatrix<BaseFloat> S_tmp(S),
                      P(N, xvector_dim),
                      Q(N, N),
                      R(N, N),
                      scores(N, N),                 // The raw scores.
                      objf_terms(N, N, kUndefined),
                      scores_deriv(N, N,        // Derivative of the
                                   kUndefined); // objf w.r.t. the scores.
  CuVector<BaseFloat> r(N);

  P.AddMatMat(1.0, xvector_pairs, kNoTrans, S_tmp, kNoTrans, 0.0);
  r.AddDiagMatMat(1.0, xvector_pairs, kNoTrans, P, kTrans, 0.0);
  R.AddVecToRows(1.0, r);
  Q.SymAddMat2(1.0, xvector_pairs, kNoTrans, 0.0);
  Q.CopyLowerToUpper();
  scores.AddMat(1.0, Q, kNoTrans);
  scores.AddMat(-1.0, R, kTrans);
  scores.AddMat(-1.0, R, kNoTrans);
  scores.Add(b);
  if (scores_out != NULL) {
    KALDI_ASSERT(scores_out->NumCols() == scores.NumCols()
                 && scores_out->NumRows() == scores.NumRows());
    scores_out->CopyFromMat(scores);
  }

  cu::ComputeXvectorObjfFromScores(scores, &objf_terms, &scores_deriv);
  CuVector<BaseFloat> objf_terms_vec(N);
  objf_terms_vec.AddRowSumMat(1.0, objf_terms);
  (*tot_objf) = objf_terms_vec.Sum();

  if (deriv_xvector != NULL) {
    // compute the derivatives of tot_objf w.r.t the inputs.
    CuMatrix<BaseFloat> scores_deriv_plus_trans(scores_deriv, kTrans);
    scores_deriv_plus_trans.AddMat(1.0, scores_deriv, kNoTrans);
    CuVector<BaseFloat> r_deriv(N);
    r_deriv.AddRowSumMat(-1.0, scores_deriv_plus_trans, 0.0);

    // Compute derivative of the objf with respect to the xvectors.
    deriv_xvector->AddDiagVecMat(2.0, r_deriv, P, kNoTrans, 0.0);
    deriv_xvector->AddMatMat(1.0, scores_deriv_plus_trans, kNoTrans,
                             xvector_pairs, kNoTrans, 1.0);

    // Compute derivative of the objf with respect to the symmetric matrix S:
    // S_deriv += xvector_pairs' * diag(r_deriv) * xvector_pairs
    CuMatrix<BaseFloat> S_deriv_mat(xvector_dim, xvector_dim);
    // we don't need P any more so re-use it temporarily
    // rderiv_xvector_pairs is the product of diag(r_deriv) times xvector_pairs.
    CuMatrix<BaseFloat> &rderiv_xvector_pairs(P);
    rderiv_xvector_pairs.AddDiagVecMat(1.0, r_deriv, xvector_pairs, kNoTrans, 0.0);
    S_deriv_mat.AddMatMat(1.0, xvector_pairs, kTrans, rderiv_xvector_pairs, kNoTrans, 0.0);
    CuSpMatrix<BaseFloat> S_deriv_sp(xvector_dim);
    S_deriv_sp.CopyFromMat(S_deriv_mat, kTakeLower);

    // at this point S_deriv_sp represents the deriv w.r.t. S represented as a
    // symmetric matrix; but we need the deriv w.r.t. S represented as a packed
    // vector, which is a little different because each off-diagonal element is
    // only represented once in the packed vector.  This means we need
    // to scale the off-diag elements by 2.
    S_deriv_sp.Scale(2.0);
    S_deriv_sp.ScaleDiag(0.5);
    deriv_S->CopyFromVec(CuSubVector<BaseFloat>(S_deriv_sp.Data(),
                                                S_dim));

    // Compute derivative of objf with respect to the scalar offset b.
    (*deriv_b) = scores_deriv.Sum();
  }
  (*tot_weight) = N;
}

BaseFloat SimilarityScore(const Vector<BaseFloat> &v,
    const Vector<BaseFloat> &w, const SpMatrix<BaseFloat> &S,
    BaseFloat b) {
  KALDI_ASSERT(v.Dim() == w.Dim() && v.Dim() == S.NumRows());
  Vector<BaseFloat> Sv(v.Dim());
  Sv.AddSpVec(1.0, S, v, 0);
  Vector<BaseFloat> Sw(w.Dim());
  Sw.AddSpVec(1.0, S, w, 0);
  BaseFloat L = VecVec(v, w) - VecVec(v, Sv) - VecVec(w, Sw) + b;
  return L;
}

} // namespace kaldi
