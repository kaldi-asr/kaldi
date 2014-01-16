// nnet2/nnet-precondition-online-test.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet2/nnet-precondition-online.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet2 {


/* This simple version of the algorithm is intended to
   be a parallel computation used for testing the main one, that it's
   doing what we intend.  We don't do any sanity checks in this
   algorithm because there are so many in the original one.
 */
void PreconditionDirectionsOnlineSimple(BaseFloat eta,
                                        CuMatrixBase<BaseFloat> *N_in,
                                        CuMatrixBase<BaseFloat> *M_in) {
  Matrix<BaseFloat> N(*N_in), M(*M_in);
  int32 R = N_in->NumRows(), B = M_in->NumRows(), D = M_in->NumCols();
  
  Matrix<BaseFloat> Q(D, D);
  Q.SetUnit();
  Q.Range(0, R, 0, D).CopyFromMat(N);
  // now orthonormalize remaining rows of Q.
  for (int32 r = R; r < D; r++) {
    SubVector<BaseFloat> q_r(Q, r);
    for (int32 s = 0; s < r; s++) {
      SubVector<BaseFloat> q_s(Q, s);
      q_r.AddVec(-VecVec(q_s, q_r), q_s);
    }
    q_r.Scale(1.0 / q_r.Norm(2.0));
  }
  Matrix<BaseFloat> NMT(R, B);
  NMT.AddMatMat(1.0, N, kNoTrans, M, kTrans, 0.0);
  SpMatrix<BaseFloat> F_i(R);
  F_i.AddMat2(1.0 / B, NMT, kNoTrans, 0.0);
  SpMatrix<BaseFloat> F_rej(D - R);
  SubMatrix<BaseFloat> T(Q, R, D - R, 0, D);
  Matrix<BaseFloat> TMT(D - R, B);
  {
    SpMatrix<BaseFloat> T2(D - R);
    T2.AddMat2(1.0, T, kNoTrans, 0.0);
    KALDI_ASSERT(T2.IsUnit(0.001));
  }
  TMT.AddMatMat(1.0, T, kNoTrans, M, kTrans, 0.0);
  F_rej.AddMat2(1.0, TMT, kNoTrans, 0.0);
  BaseFloat beta_i = F_rej.Trace() / ((D - R) * B);
  KALDI_LOG << "beta_i = " << beta_i;
  Matrix<BaseFloat> Finv(D, D); // full-dimensional fisher.
  Finv.SetUnit();
  SubMatrix<BaseFloat> Finv_part(Finv, 0, R, 0, R);
  Finv_part.CopyFromSp(F_i);
  Finv_part.Invert();
  KALDI_LOG << "Finv trace is " << Finv_part.Trace();
  Finv_part.Scale(beta_i);
  Matrix<BaseFloat> Mproj(B, D);
  Mproj.AddMatMat(1.0, M, kNoTrans, Q, kTrans, 0.0);
  Matrix<BaseFloat> MprojScaled(B, D);
  MprojScaled.AddMatMat(1.0, Mproj, kNoTrans, Finv, kNoTrans, 0.0);
  Matrix<BaseFloat> Mfinal(B, D);
  Mfinal.AddMatMat(1.0, MprojScaled, kNoTrans, Q, kNoTrans, 0.0);
  M_in->CopyFromMat(Mfinal);
  // Now update N.
  Matrix<BaseFloat> O_i(R, D);
  O_i.AddMatMat(1.0, NMT, kNoTrans, M, kNoTrans, 0.0);
  BaseFloat eta_i = eta * sqrt(TraceMatMat(O_i, O_i, kTrans) / TraceMatMat(N, N, kTrans));
  KALDI_LOG << "eta_i = " << eta_i;
  Matrix<BaseFloat> P_i(O_i);
  P_i.AddMat(eta_i, N);
  SpMatrix<BaseFloat> Y_i(R);
  Y_i.AddMat2(1.0, P_i, kNoTrans, 0.0);
  TpMatrix<BaseFloat> Cinv(R);
  Cinv.Cholesky(Y_i);
  Cinv.Invert();
  Matrix<BaseFloat> Ni1(R, D);
  Ni1.AddTpMat(1.0, Cinv, kNoTrans, P_i, kNoTrans, 0.0);
  N_in->CopyFromMat(Ni1);
}


void UnitTestPreconditionDirectionsOnline() {
  MatrixIndexT R = 1 + rand() % 5,  // rank of correction
      B = (2 * R) + rand() % 30,  // batch size
      D = R + 1 + rand() % 20; // problem dimension.  Must be > R.
  
  BaseFloat eta = 0.1;
  if (rand() % 2 == 0) eta = 1.5;
  
  CuMatrix<BaseFloat> M(B, D);
  M.SetRandn();
  CuMatrix<BaseFloat> Mcopy(M);

  CuMatrix<BaseFloat> N(R, D); // The first call to PreconditionDirections will
                               // set N to a matrix with orthonormal rows.

  bool first_time = true;
  PreconditionDirectionsOnline(eta, first_time, &N, &M);

  BaseFloat orig_trace = TraceMatMat(Mcopy, Mcopy, kTrans),
      new_trace = TraceMatMat(Mcopy, M, kTrans),
      ratio = new_trace / orig_trace;
  KALDI_LOG << "ratio = " << ratio;
  KALDI_ASSERT(ratio > 0.0);
  // The rest of this function will do the computation the function is doing in
  // a different, less efficient way and compare with the function call.

  CuMatrix<BaseFloat> N1(N), M1(M), N2(N), M2(M);

  PreconditionDirectionsOnline(eta, false, &N1, &M1);
  PreconditionDirectionsOnlineSimple(eta, &N2, &M2);

  KALDI_LOG << "M1 frobenius norm is " << M1.FrobeniusNorm();
  KALDI_LOG << "M2 frobenius norm is " << M2.FrobeniusNorm();
  KALDI_LOG << "N1 frobenius norm is " << N1.FrobeniusNorm();
  KALDI_LOG << "N2 frobenius norm is " << N2.FrobeniusNorm();
  
  AssertEqual(M1, M2);
  AssertEqual(N1, N2);

  

  
  return;
}

/*
  CuSpMatrix<BaseFloat> G(D);
  G.SetUnit();
  G.ScaleDiag(lambda);
  // G += R^T R.
  G.AddMat2(1.0/(N-1), R, kTrans, 1.0);
  
  for (int32 n = 0; n < N; n++) {
    CuSubVector<BaseFloat> rn(R, n);
    CuSpMatrix<BaseFloat> Gn(G);
    Gn.AddVec2(-1.0/(N-1), rn); // subtract the
    // outer product of "this" vector.
    Gn.Invert();
    CuSubVector<BaseFloat> pn(P, n);
    CuVector<BaseFloat> pn_compare(D);
    pn_compare.AddSpVec(1.0, Gn, rn, 0.0);
    KALDI_ASSERT(pn.ApproxEqual(pn_compare, 0.1));
  }
}

*/
} // namespace nnet2
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;
  for (int32 i = 0; i < 10; i++)
    UnitTestPreconditionDirectionsOnline();
}
