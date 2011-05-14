// matrix/kaldi-matrix-inl.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_MATRIX_KALDI_MATRIX_INL_H_
#define KALDI_MATRIX_KALDI_MATRIX_INL_H_ 1

#include "matrix/kaldi-vector.h"

namespace kaldi {

/// Empty constructor
template<typename Real>
Matrix<Real>::Matrix(): MatrixBase<Real>(NULL, 0, 0, 0)
#ifdef KALDI_MEMALIGN_MANUAL
                      , free_data_(NULL)
#endif
{}


// state that we will implement separately for float and double.
template<>  void MatrixBase<float>::Invert(float *LogDet, float *DetSign, bool inverse_needed);

template<>  void MatrixBase<double>::Invert(double *LogDet, double *DetSign, bool inverse_needed);


template<>
void MatrixBase<float>::AddVecVec(const float alpha, const VectorBase<float>& ra, const VectorBase<float>& rb);

template<>
void MatrixBase<double>::AddVecVec(const double alpha, const VectorBase<double>& ra, const VectorBase<double>& rb);


template<>
void MatrixBase<float>::AddMatMat(const float alpha,
                                  const MatrixBase<float>& A, MatrixTransposeType transA,
                                  const MatrixBase<float>& B, MatrixTransposeType transB,
                                  const float beta);

template<>
void MatrixBase<double>::AddMatMat(const double alpha,
                                   const MatrixBase<double>& A, MatrixTransposeType transA,
                                   const MatrixBase<double>& B, MatrixTransposeType transB,
                                   const double beta);

template<>
void MatrixBase<float>::AddMat(const float alpha,
                               const MatrixBase<float>& A, MatrixTransposeType transA);

template<>
void MatrixBase<double>::AddMat(const double alpha,
                                const MatrixBase<double>& A, MatrixTransposeType transA);

template<>
void MatrixBase<float>::AddSpSp(const float alpha,
                                const SpMatrix<float>& A, const SpMatrix<float>& B,
                                const float beta);
template<>
void MatrixBase<double>::AddSpSp(const double alpha,
                                 const SpMatrix<double>& A,  const SpMatrix<double>& B,
                                 const double beta);

template<>
void MatrixBase<double>::Scale(double alpha);

template<>
void MatrixBase<float>::Scale(float alpha);


#ifndef HAVE_ATLAS
template<>
void
MatrixBase<double>::LapackGesvd(VectorBase<double> *s, MatrixBase<double> *U,
                                MatrixBase<double> *Vt);

template<>
void
MatrixBase<float>::LapackGesvd(VectorBase<float> *s, MatrixBase<float> *U,
                               MatrixBase<float> *Vt);
#endif


template<typename Real>
inline std::ostream & operator << (std::ostream & os, const MatrixBase<Real> & M) {
  M.Write(os, false);
  return os;
}

template<typename Real>
inline std::istream & operator >> (std::istream & is, Matrix<Real> & M) {
  M.Read(is, false);
  return is;
}


template<typename Real>
inline std::istream & operator >> (std::istream & is, MatrixBase<Real> & M) {
  M.Read(is, false);
  return is;
}

}// namespace kaldi


#endif  // KALDI_MATRIX_KALDI_MATRIX_INL_H_
