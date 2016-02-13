// ivector/xvector.cc

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

#include "ivector/xvector.h"

namespace kaldi {

void ComputeXvectorObjfAndDeriv(
    const CuMatrixBase<BaseFloat> &xvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S_and_b, BaseFloat *tot_objf,
    BaseFloat *tot_weight) {

  int32 S_dim = S.NumCols() * (S.NumCols() + 1) / 2,
        N = xvector_pairs.NumRows(),
        xvector_dim = xvector_pairs.NumCols();
  BaseFloat K = 1.0 / (N - 2.0);
  (*tot_objf) = 0;

  if (deriv_xvector == NULL)
    KALDI_ASSERT(deriv_S_and_b == NULL);
  else {
    KALDI_ASSERT(deriv_xvector->NumCols() == xvector_dim);
    KALDI_ASSERT(deriv_xvector->NumRows() == N);
    KALDI_ASSERT(deriv_S_and_b->Dim() == S_dim + 1);
  }

  CuMatrix<BaseFloat> S_tmp(S);
  CuMatrix<BaseFloat> P(N, xvector_dim),
                      Q(N, N),
                      R(N, N),
                      T(N, N),
                      objf_terms(N, N),
                      objf_deriv_terms(N, N);

  CuVector<BaseFloat> r(N);
  P.AddMatMat(1.0, xvector_pairs, kNoTrans, S_tmp, kNoTrans, 0.0);
  r.AddDiagMatMat(1.0, xvector_pairs, kNoTrans, P, kTrans, 0.0);
  R.AddVecToRows(1.0, r);
  Q.SymAddMat2(1.0, xvector_pairs, kNoTrans, 0.0);
  Q.CopyLowerToUpper();
  T.AddMat(1.0, Q, kNoTrans);
  T.AddMat(-1.0, R, kTrans);
  T.AddMat(-1.0, R, kNoTrans);
  T.Add(b);

  cu::ComputeXvectorObjfFromScores<BaseFloat>(T, &objf_terms, &objf_deriv_terms);
  CuVector<BaseFloat> objf_terms_vec(N);
  objf_terms_vec.AddRowSumMat(1.0, objf_terms);
  (*tot_objf) = objf_terms_vec.Sum();

  if (deriv_xvector != NULL) {
    /* TODO: Call cu-math function that handles the derivatives of S
       and the xvectors.
    */
    (*deriv_S_and_b)(S_dim) = -objf_deriv_terms.Sum();
  }
  (*tot_weight) = N;
}

} // namespace kaldi
