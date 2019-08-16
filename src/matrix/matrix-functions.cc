// matrix/matrix-functions.cc

// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.;  Jan Silovsky
//                      Yanmin Qian;  Saarland University;  Johns Hopkins University (Author: Daniel Povey)

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


#include "matrix/matrix-functions.h"
#include "matrix/sp-matrix.h"

namespace kaldi {


template<typename Real> void ComputeDctMatrix(Matrix<Real> *M) {
  //KALDI_ASSERT(M->NumRows() == M->NumCols());
  MatrixIndexT K = M->NumRows();
  MatrixIndexT N = M->NumCols();

  KALDI_ASSERT(K > 0);
  KALDI_ASSERT(N > 0);
  Real normalizer = std::sqrt(1.0 / static_cast<Real>(N));  // normalizer for
  // X_0.
  for (MatrixIndexT j = 0; j < N; j++) (*M)(0, j) = normalizer;
  normalizer = std::sqrt(2.0 / static_cast<Real>(N));  // normalizer for other
   // elements.
  for (MatrixIndexT k = 1; k < K; k++)
    for (MatrixIndexT n = 0; n < N; n++)
      (*M)(k, n) = normalizer
          * std::cos( static_cast<double>(M_PI)/N * (n + 0.5) * k );
}


template void ComputeDctMatrix(Matrix<float> *M);
template void ComputeDctMatrix(Matrix<double> *M);


template<typename Real>
void ComputePca(const MatrixBase<Real> &X,
                MatrixBase<Real> *U,
                MatrixBase<Real> *A,
                bool print_eigs,
                bool exact) {
  // Note that some of these matrices may be transposed w.r.t. the
  // way it's most natural to describe them in math... it's the rows
  // of X and U that correspond to the (data-points, basis elements).
  MatrixIndexT N = X.NumRows(), D = X.NumCols();
  // N = #points, D = feature dim.
  KALDI_ASSERT(U != NULL && U->NumCols() == D);
  MatrixIndexT G = U->NumRows();  // # of retained basis elements.
  KALDI_ASSERT(A == NULL || (A->NumRows() == N && A->NumCols() == G));
  KALDI_ASSERT(G <= N && G <= D);
  if (D < N) {  // Do conventional PCA.
    SpMatrix<Real> Msp(D);  // Matrix of outer products.
    Msp.AddMat2(1.0, X, kTrans, 0.0);  // M <-- X^T X
    Matrix<Real> Utmp;
    Vector<Real> l;
    if (exact) {
      Utmp.Resize(D, D);
      l.Resize(D);
      //Matrix<Real> M(Msp);
      //M.DestructiveSvd(&l, &Utmp, NULL);
      Msp.Eig(&l, &Utmp);
    } else {
      Utmp.Resize(D, G);
      l.Resize(G);
      Msp.TopEigs(&l, &Utmp);
    }
    SortSvd(&l, &Utmp);

    for (MatrixIndexT g = 0; g < G; g++)
      U->Row(g).CopyColFromMat(Utmp, g);
    if (print_eigs)
      KALDI_LOG << (exact ? "" : "Retained ")
                << "PCA eigenvalues are " << l;
    if (A != NULL)
      A->AddMatMat(1.0, X, kNoTrans, *U, kTrans, 0.0);
  } else {  // Do inner-product PCA.
    SpMatrix<Real> Nsp(N);  // Matrix of inner products.
    Nsp.AddMat2(1.0, X, kNoTrans, 0.0);  // M <-- X X^T

    Matrix<Real> Vtmp;
    Vector<Real> l;
    if (exact) {
      Vtmp.Resize(N, N);
      l.Resize(N);
      Matrix<Real> Nmat(Nsp);
      Nmat.DestructiveSvd(&l, &Vtmp, NULL);
    } else {
      Vtmp.Resize(N, G);
      l.Resize(G);
      Nsp.TopEigs(&l, &Vtmp);
    }

    MatrixIndexT num_zeroed = 0;
    for (MatrixIndexT g = 0; g < G; g++) {
      if (l(g) < 0.0) {
        KALDI_WARN << "In PCA, setting element " << l(g) << " to zero.";
        l(g) = 0.0;
        num_zeroed++;
      }
    }
    SortSvd(&l, &Vtmp); // Make sure zero elements are last, this
    // is necessary for Orthogonalize() to work properly later.

    Vtmp.Transpose();  // So eigenvalues are the rows.

    for (MatrixIndexT g = 0; g < G; g++) {
      Real sqrtlg = sqrt(l(g));
      if (l(g) != 0.0) {
        U->Row(g).AddMatVec(1.0 / sqrtlg, X, kTrans, Vtmp.Row(g), 0.0);
      } else {
        U->Row(g).SetZero();
        (*U)(g, g) = 1.0;  // arbitrary direction.  Will later orthogonalize.
      }
      if (A != NULL)
        for (MatrixIndexT n = 0; n < N; n++)
          (*A)(n, g) = sqrtlg * Vtmp(g, n);
    }
    // Now orthogonalize.  This is mainly useful in
    // case there were zero eigenvalues, but we do it
    // for all of them.
    U->OrthogonalizeRows();
    if (print_eigs)
      KALDI_LOG << "(inner-product) PCA eigenvalues are " << l;
  }
}


template
void ComputePca(const MatrixBase<float> &X,
                MatrixBase<float> *U,
                MatrixBase<float> *A,
                bool print_eigs,
                bool exact);

template
void ComputePca(const MatrixBase<double> &X,
                MatrixBase<double> *U,
                MatrixBase<double> *A,
                bool print_eigs,
                bool exact);


// Added by Dan, Feb. 13 2012.
// This function does: *plus += max(0, a b^T),
// *minus += max(0, -(a b^T)).
template<typename Real>
void AddOuterProductPlusMinus(Real alpha,
                              const VectorBase<Real> &a,
                              const VectorBase<Real> &b,
                              MatrixBase<Real> *plus,
                              MatrixBase<Real> *minus) {
  KALDI_ASSERT(a.Dim() == plus->NumRows() && b.Dim() == plus->NumCols()
               && a.Dim() == minus->NumRows() && b.Dim() == minus->NumCols());
  int32 nrows = a.Dim(), ncols = b.Dim(), pskip = plus->Stride() - ncols,
      mskip = minus->Stride() - ncols;
  const Real *adata = a.Data(), *bdata = b.Data();
  Real *plusdata = plus->Data(), *minusdata = minus->Data();

  for (int32 i = 0; i < nrows; i++) {
    const Real *btmp = bdata;
    Real multiple = alpha * *adata;
    if (multiple > 0.0) {
      for (int32 j = 0; j < ncols; j++, plusdata++, minusdata++, btmp++) {
        if (*btmp > 0.0) *plusdata += multiple * *btmp;
        else *minusdata -= multiple * *btmp;
      }
    } else {
      for (int32 j = 0; j < ncols; j++, plusdata++, minusdata++, btmp++) {
        if (*btmp < 0.0) *plusdata += multiple * *btmp;
        else *minusdata -= multiple * *btmp;
      }
    }
    plusdata += pskip;
    minusdata += mskip;
    adata++;
  }
}

// Instantiate template
template
void AddOuterProductPlusMinus<float>(float alpha,
                                     const VectorBase<float> &a,
                                     const VectorBase<float> &b,
                                     MatrixBase<float> *plus,
                                     MatrixBase<float> *minus);
template
void AddOuterProductPlusMinus<double>(double alpha,
                                      const VectorBase<double> &a,
                                      const VectorBase<double> &b,
                                      MatrixBase<double> *plus,
                                      MatrixBase<double> *minus);


} // end namespace kaldi
