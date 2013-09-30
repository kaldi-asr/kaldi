// nnet2/nnet-precondition-online.cc

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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

namespace kaldi {
namespace nnet2 {


static void CheckOrthogonal(CuMatrixBase<BaseFloat> *N,
                            bool quiet = false,
                            int32 recurse_count = 0) {
  if (recurse_count > 100)
    KALDI_ERR << "CheckOrthogonal recursed >100 times, something is wrong.";
  
  int32 R = N->NumRows();
  CuSpMatrix<BaseFloat> S(R);
  S.AddMat2(1.0, *N, kNoTrans, 0.0);
  if (!S.IsUnit(1.0e-04)) {
    {
      SpMatrix<BaseFloat> S_cpu(S);
      if (!quiet)
        KALDI_WARN << "Matrix N is not orthogonal, fixing it.  N N^T is "
                   << S_cpu;
      Vector<BaseFloat> s(R);
      S_cpu.Eig(&s);
      BaseFloat threshold = 0.001;
      if (s.Min() < threshold) {
        if (!quiet) {
          KALDI_WARN << "Minimum eigenvalue of N N^T is less than " << threshold
                     << ", may be hard to fix: re-initializing from random "
                     << "start. Eigs are" << s;
        }
        N->SetRandn();
        CheckOrthogonal(N, quiet, recurse_count + 1);
        return;
      }
    }
    CuTpMatrix<BaseFloat> Cinv(R);
    Cinv.Cholesky(S);
    Cinv.Invert();
    CuMatrix<BaseFloat> N_copy(*N);
    N->AddTpMat(1.0, Cinv, kNoTrans, N_copy, kNoTrans, 0.0);
    CheckOrthogonal(N, quiet, recurse_count + 1); // Check that it worked.
  }
}

void PreconditionDirectionsOnline(BaseFloat eta,
                                  bool first_time,
                                  CuMatrixBase<BaseFloat> *N,
                                  CuMatrixBase<BaseFloat> *M) {
  int32 R = N->NumRows(), B = M->NumRows(), D = M->NumCols();
  const BaseFloat epsilon = 1.0e-4, delta = 1.0e-10; // The algorithm should not be
  // sensitive to these values, it's just to avoid inverting singular matrices.
  
  KALDI_ASSERT(N->NumCols() == D && R > 0 && B > 0 && D > 0 && eta > 0);
  
  if (B < 2 * R) {
    KALDI_WARN << "Not preconditioning matrix since batch size " << B
               << " is too small relative to preconditioner rank " << R
               << " (partial minibatch?)";
    return;
  }
  if (R >= D) {
    KALDI_ERR << "Rank of preconditioner " << R << " must be less than "
              << "vector dimension " << D;
  }
  
  if (first_time) {
    CuMatrix<BaseFloat> M_tmp(*M);
    N->SetRandn();
    bool quiet = true;
    CheckOrthogonal(N, quiet);
    double first_time_eta = 0.001; // any small nonzero value will do.
    PreconditionDirectionsOnline(first_time_eta, false, N, &M_tmp);
    // Discard M_tmp; we only recursed in order to update N for one iteration
    // to get a not-quite-so-random value.
  }

  // The call to CheckOrthogonal below is really just out of an abundance of
  // caution; it shouldn't be necessary.
  if (rand() % 5 == 0) 
    CheckOrthogonal(N);

  // These are just for notation that's more consistent with the comment
  // in the header.
  const CuMatrixBase<BaseFloat> &N_i = *N, &M_i = *M;
  
  CuMatrix<BaseFloat> NMT_i(R, B);
  NMT_i.AddMatMat(1.0, N_i, kNoTrans, M_i, kTrans, 0.0);
  CuMatrix<BaseFloat> O_i(R, D);
  O_i.AddMatMat(1.0, NMT_i, kNoTrans, M_i, kNoTrans, 0.0);
  CuMatrix<BaseFloat> F_i(R, R);
  // Below, TODO: will change this to SymmetricAddMatMat when the
  // function has been written.
  F_i.AddMatMat(1.0 / B, O_i, kNoTrans, N_i, kTrans, 0.0);
  BaseFloat t_f = F_i.Trace(),
      t_m = TraceMatMat(M_i, M_i, kTrans), // Will have to implement this efficiently.
      beta_i = (t_m - B * t_f) / ((D - R) * B);
  if (beta_i <= 0.0) {
    // This really should not happen.
    KALDI_WARN << "Negative beta_i " << beta_i;
  }
  CuSpMatrix<BaseFloat> F_i_sp(F_i, kTakeLower); // we'll need the SpMatrix form
                                                 // of F_i later.
  CuSpMatrix<BaseFloat> F_i_inv(F_i_sp);
  F_i_inv.AddToDiag(epsilon * t_f / R + delta); // Ensure it will be invertible.  
  F_i_inv.Invert();
  CuSpMatrix<BaseFloat> &temp(F_i_inv);
  temp.Scale(beta_i);
  temp.AddToDiag(-1.0);
  // Now, temp contains (\beta_i F_i_inv - I).

  // we could choose to have temp2 the other way round (i.e. transposed)
  // which might affect the efficiency.
  CuMatrix<BaseFloat> temp2(B, R);
  temp2.AddMatSp(1.0, NMT_i, kTrans, temp, 0.0);
  // Now temp2 is NMT_i^T (\beta_i F_i_inv - I)
      
  CuMatrixBase<BaseFloat> &L_i(*M); // This is the output.  Its current value
                                    // equals M_i.
  // Next line does:  L_i = M_i  + NMT_i^T (\beta_i F_i_inv - I) N_i
  L_i.AddMatMat(1.0, temp2, kNoTrans, N_i, kNoTrans, 1.0);
  
  CuSpMatrix<BaseFloat> X_i(R);
  X_i.AddMat2(1.0, O_i, kNoTrans, 0.0); // X_i = O_i O_i^T
  
  BaseFloat eta_i = eta * sqrt(TraceMatMat(O_i, O_i, kTrans) / R);
  if (eta_i < delta)
    eta_i = delta;  
  
  CuSpMatrix<BaseFloat> &Y_i(X_i); // re-use that matrix for Y_i: Y_i = X_i.
  Y_i.AddToDiag(eta_i * eta_i); // Y_i += \eta_i^2 I.
  Y_i.AddSp(2.0 * eta_i * B, F_i_sp); // Y_i += (2 \eta_i B) F_i

  CuTpMatrix<BaseFloat> C_i_inv(R);
  CuMatrix<BaseFloat> &P_i(O_i); // re-use that matrix for P_i.
  CuMatrixBase<BaseFloat> &N_i1(*N); // N_{i+1}
  try {
    C_i_inv.Cholesky(Y_i);
    C_i_inv.Invert();
    P_i.AddMat(eta_i, N_i);
    N_i1.AddTpMat(1.0, C_i_inv, kNoTrans, P_i, kNoTrans, 0.0);
  } catch (...) {
    // A warning will already have been printed.
    // If Cholesky fails we simply won't update N this time.
    // This should not ever happen.
    CheckOrthogonal(N);
  }
}

}
}
