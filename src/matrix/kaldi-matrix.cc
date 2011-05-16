// matrix/kaldi-matrix.cc

// Copyright 2009-2011    Lukas Burget,  Ondrej Glembek,  Go Vivace Inc.,
//                   Microsoft Corporation,  Arnab Ghoshal,  Yanmin Qian,
//                   Petr Schwarz,  Jan Silovsky


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

#include "matrix/kaldi-matrix.h"
#include "matrix/sp-matrix.h"
#include "matrix/jama-svd.h"
#include "matrix/jama-eig.h"

namespace kaldi {
template<>
void MatrixBase<float>::Invert(float *LogDet, float *DetSign,
                               bool inverse_needed) {
  KALDI_ASSERT(num_rows_ == num_cols_);
#ifndef HAVE_ATLAS
    KaldiBlasInt* pivot = new KaldiBlasInt[num_rows_];
    KaldiBlasInt M = num_rows_;
    KaldiBlasInt N = num_cols_;
    KaldiBlasInt LDA = stride_;
    KaldiBlasInt result;
    KaldiBlasInt l_work = std::max<KaldiBlasInt>(1, N);
    float* p_work = new float[l_work];

    sgetrf_(&M, &N, data_, &LDA, pivot, &result);
    const int pivot_offset = 1;
#else
    int* pivot = new int[num_rows_];
    int result = clapack_sgetrf(CblasColMajor, num_rows_, num_cols_,
        data_, stride_, pivot);
    const int pivot_offset = 0;
#endif
    KALDI_ASSERT(result >= 0 && "Call to CLAPACK sgetrf_ or ATLAS clapack_sgetrf "
        "called with wrong arguments");
    if (result > 0) {
      if (inverse_needed) {
        KALDI_ERR << "Cannot invert: matrix is singular";
      } else {
        if (LogDet) *LogDet = -std::numeric_limits<float>::infinity();
        if (DetSign) *DetSign = 0;
        return;
      }
    }
    if (DetSign != NULL) {
      int sign = 1;
      for (MatrixIndexT i = 0; i < num_rows_; i++)
        if (pivot[i] != static_cast<int>(i) + pivot_offset) sign *= -1.0;
      *DetSign = sign;
    }
    if (LogDet != NULL || DetSign != NULL) {  // Compute log determinant.
      if (LogDet != NULL) *LogDet = 0.0;
      float prod = 1.0;
      for (MatrixIndexT i = 0; i < num_rows_; i++) {
        prod *= (*this)(i, i);
        if (i == num_rows_ - 1 || std::fabs(prod) < 1.0e-10 ||
            std::fabs(prod) > 1.0e+10) {
          if (LogDet != NULL) *LogDet += log(fabs(prod));
          if (DetSign != NULL) *DetSign *= (prod > 0 ? 1.0 : -1.0);
          prod = 1.0;
        }
      }
    }
#ifndef HAVE_ATLAS
    if (inverse_needed) sgetri_(&M, data_, &LDA, pivot, p_work, &l_work,
        &result);
    delete[] pivot;
    delete[] p_work;
#else
    if (inverse_needed)
      result = clapack_sgetri(CblasColMajor, num_rows_, data_, stride_, pivot);
    delete [] pivot;
#endif
    KALDI_ASSERT(result == 0 && "Call to CLAPACK sgetri_ or ATLAS clapack_sgetri "
        "called with wrong arguments");
}


//***************************************************************************
//***************************************************************************
template<>
void MatrixBase<double>::Invert(double *LogDet, double *DetSign,
                                bool inverse_needed) {
  KALDI_ASSERT(num_rows_ == num_cols_);
#ifndef HAVE_ATLAS
    KaldiBlasInt* pivot = new KaldiBlasInt[num_rows_];
    KaldiBlasInt M = num_rows_;
    KaldiBlasInt N = num_cols_;
    KaldiBlasInt LDA = stride_;
    KaldiBlasInt result;
    KaldiBlasInt l_work = std::max<KaldiBlasInt>(1, N);
    double* p_work = new double[l_work];

    dgetrf_(&M, &N, data_, &LDA, pivot, &result);
    const int pivot_offset = 1;
#else
    int* pivot = new int[num_rows_];
    int result = clapack_dgetrf(CblasColMajor, num_rows_, num_cols_, data_,
                                stride_, pivot);
    const int pivot_offset = 0;
#endif
    KALDI_ASSERT(result >= 0 && "Call to CLAPACK dgetrf_ or ATLAS clapack_dgetrf "
        "called with wrong arguments");
    if (result > 0) {
      if (inverse_needed)
        KALDI_ERR << "Cannot invert: matrix is singular";
      else {
        if (LogDet) *LogDet = -std::numeric_limits<float>::infinity();
        if (DetSign) *DetSign = 0;
        return;
      }
    }
    if (DetSign != NULL) {
      int sign = 1;
      for (MatrixIndexT i = 0; i < num_rows_; i++)
        if (pivot[i] != static_cast<int>(i) + pivot_offset) sign *= -1.0;
      *DetSign = sign;
    }
    if (LogDet != NULL || DetSign != NULL) {  // Compute log determinant...
      if (LogDet != NULL) *LogDet = 0.0;
      double prod = 1.0;
      for (MatrixIndexT i = 0; i < num_rows_; i++) {
        prod *= (*this)(i, i);
        if (i == num_rows_ - 1 || std::fabs(prod) < 1.0e-10 ||
            std::fabs(prod) > 1.0e+10) {
          if (LogDet != NULL) *LogDet += log(fabs(prod));
          if (DetSign != NULL) *DetSign *= (prod > 0 ? 1.0 : -1.0);
          prod = 1.0;
        }
      }
    }
#ifndef HAVE_ATLAS
    if (inverse_needed)
      dgetri_(&M, data_, &LDA, pivot, p_work, &l_work, &result);
    delete[] pivot;
    delete[] p_work;
#else
    if (inverse_needed)
      result = clapack_dgetri(CblasColMajor, num_rows_, data_, stride_, pivot);
    delete [] pivot;
#endif
    KALDI_ASSERT(result == 0 && "Call to CLAPACK dgetri_ or ATLAS clapack_dgetri "
        "called with wrong arguments");
}

template<>
void MatrixBase<float>::AddVecVec(const float alpha,
                                  const VectorBase<float>& ra,
                                  const VectorBase<float>& rb) {
  KALDI_ASSERT(ra.Dim() == num_rows_ && rb.Dim() == num_cols_);
  cblas_sger(CblasRowMajor, ra.Dim(), rb.Dim(), alpha, ra.Data(), 1, rb.Data(),
             1, data_, stride_);
}

template<class Real>
template<class OtherReal>
void MatrixBase<Real>::AddVecVec(const OtherReal alpha,
                                 const VectorBase<OtherReal>& a,
                                 const VectorBase<OtherReal>& b) {
  KALDI_ASSERT(a.Dim() == num_rows_ && b.Dim() == num_cols_);
  const OtherReal *a_data = a.Data(), *b_data = b.Data();
  Real *row_data = data_;
  for (MatrixIndexT i = 0; i < num_rows_; i++, row_data += stride_) {
    BaseFloat alpha_ai = alpha * a_data[i];
    for (MatrixIndexT j = 0; j < num_cols_; j++)
      row_data[j] += alpha_ai * b_data[j];
  }
}

// instantiate the template above.
template
void MatrixBase<float>::AddVecVec(const double alpha,
                                  const VectorBase<double>& a,
                                  const VectorBase<double>& b);
template
void MatrixBase<double>::AddVecVec(const float alpha,
                                   const VectorBase<float>& a,
                                   const VectorBase<float>& b);

template<>
void MatrixBase<double>::AddVecVec(const double alpha,
                                   const VectorBase<double>& ra,
                                   const VectorBase<double>& rb) {
  KALDI_ASSERT(ra.Dim() == num_rows_ && rb.Dim() == num_cols_);
  cblas_dger(CblasRowMajor, ra.Dim(), rb.Dim(), alpha, ra.Data(), 1, rb.Data(),
             1, data_, stride_);
}

template<>
void MatrixBase<float>::AddMatMat(const float alpha,
                                  const MatrixBase<float>& rA,
                                  MatrixTransposeType transA,
                                  const MatrixBase<float>& rB,
                                  MatrixTransposeType transB,
                                  const float beta) {
  KALDI_ASSERT((transA == kNoTrans && transB == kNoTrans && rA.num_cols_ == rB.num_rows_ && rA.num_rows_ == num_rows_ && rB.num_cols_ == num_cols_)
	     || (transA == kTrans && transB == kNoTrans && rA.num_rows_ == rB.num_rows_ && rA.num_cols_ == num_rows_ && rB.num_cols_ == num_cols_)
	     || (transA == kNoTrans && transB == kTrans && rA.num_cols_ == rB.num_cols_ && rA.num_rows_ == num_rows_ && rB.num_rows_ == num_cols_)
	     || (transA == kTrans && transB == kTrans && rA.num_rows_ == rB.num_cols_ && rA.num_cols_ == num_rows_ && rB.num_rows_ == num_cols_));
  KALDI_ASSERT(&rA !=  this && &rB != this);

  cblas_sgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA), static_cast<CBLAS_TRANSPOSE>(transB),
              num_rows_, num_cols_, transA == kNoTrans ? rA.num_cols_ : rA.num_rows_,
              alpha, rA.data_, rA.stride_, rB.data_, rB.stride_,
              beta, data_, stride_);
}

template<>
void MatrixBase<double>::AddMatMat(const double alpha,
                                   const MatrixBase<double>& rA,
                                   MatrixTransposeType transA,
                                   const MatrixBase<double>& rB,
                                   MatrixTransposeType transB,
                                   const double beta) {
  KALDI_ASSERT((transA == kNoTrans && transB == kNoTrans && rA.num_cols_ == rB.num_rows_ && rA.num_rows_ == num_rows_ && rB.num_cols_ == num_cols_)
	     || (transA == kTrans && transB == kNoTrans && rA.num_rows_ == rB.num_rows_ && rA.num_cols_ == num_rows_ && rB.num_cols_ == num_cols_)
	     || (transA == kNoTrans && transB == kTrans && rA.num_cols_ == rB.num_cols_ && rA.num_rows_ == num_rows_ && rB.num_rows_ == num_cols_)
	     || (transA == kTrans && transB == kTrans && rA.num_rows_ == rB.num_cols_ && rA.num_cols_ == num_rows_ && rB.num_rows_ == num_cols_));
  KALDI_ASSERT(&rA !=  this && &rB != this);

  cblas_dgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA), static_cast<CBLAS_TRANSPOSE>(transB),
              num_rows_, num_cols_, transA == kNoTrans ? rA.num_cols_ : rA.num_rows_,
              alpha, rA.data_, rA.stride_, rB.data_, rB.stride_,
              beta, data_, stride_);
}

template<>
void MatrixBase<float>::AddSpSp(const float alpha, const SpMatrix<float>& rA,
                                const SpMatrix<float>& rB, const float beta) {
  MatrixIndexT sz = num_rows_;
  KALDI_ASSERT(sz == num_cols_ && sz == rA.NumRows() && sz == rB.NumRows());

  Matrix<float> A(rA), B(rB);
  // CblasLower or CblasUpper would work below as symmetric matrix is copied
  // fully (to save work, we used the matrix constructor from SpMatrix).
  // CblasLeft means rA is on the left: C <-- alpha A B + beta C
  cblas_ssymm(CblasRowMajor, CblasLeft, CblasLower, sz, sz, alpha, A.data_,
              A.stride_, B.data_, B.stride_, beta, data_, stride_);
}

template<>
void MatrixBase<double>::AddSpSp(const double alpha,
                                 const SpMatrix<double>& rA,
                                 const SpMatrix<double>& rB,
                                 const double beta) {
  MatrixIndexT sz = num_rows_;
  KALDI_ASSERT(sz == num_cols_ && sz == rA.NumRows() && sz == rB.NumRows());

  Matrix<double> A(rA), B(rB);
  // CblasLower or CblasUpper would work below as symmetric matrix is copied
  // fully (to save work, we used the matrix constructor from SpMatrix).
  // CblasLeft means rA is on the left: C <-- alpha A B + beta C
  cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, sz, sz, alpha, A.data_,
              A.stride_, B.data_, B.stride_, beta, data_, stride_);
}


template<>
void MatrixBase<float>::AddMat(const float alpha, const MatrixBase<float>& rA,
                               MatrixTransposeType transA) {
  if (&rA == this) {  // Make it work in this case.
    if (!transA) {
      Scale(alpha + 1.0);
    } else {
      KALDI_ASSERT(num_rows_ == num_cols_ && "AddMat: adding to self(transposed): not symmetric.");
      float *data = data_;
      if (alpha == 1.0) {  // common case-- handle separately.
        for (MatrixIndexT row = 0; row < num_rows_; row++) {
          for (MatrixIndexT col = 0; col < row; col++) {
            float *lower = data + (row * stride_) + col, *upper = data + (col
                * stride_) + row;
            float sum = *lower + *upper;
            *lower = *upper = sum;
          }
          *(data + (row * stride_) + row) *= 2.0;  // diagonal.
        }
      } else {
        for (MatrixIndexT row = 0; row < num_rows_; row++) {
          for (MatrixIndexT col = 0; col < row; col++) {
            float *lower = data + (row * stride_) + col, *upper = data + (col
                * stride_) + row;
            float lower_tmp = *lower;
            *lower += alpha * *upper;
            *upper += alpha * lower_tmp;
          }
          *(data + (row * stride_) + row) *= (1.0 + alpha);  // diagonal.
        }
      }
    }
  } else {
    int aStride = (int) rA.stride_, stride = stride_;
    float *adata = rA.data_, *data = data_;
    if (transA == kNoTrans) {
      KALDI_ASSERT(rA.num_rows_ == num_rows_ && rA.num_cols_ == num_cols_);
      for (MatrixIndexT row = 0; row < num_rows_; row++, adata += aStride,
           data += stride) {
        cblas_saxpy(num_cols_, alpha, adata, 1, data, 1);
      }
    } else {
      KALDI_ASSERT(rA.num_cols_ == num_rows_ && rA.num_rows_ == num_cols_);
      for (MatrixIndexT row = 0; row < num_rows_; row++, adata++, data += stride)
        cblas_saxpy(num_cols_, alpha, adata, aStride, data, 1);
    }
  }
}

template<>
void MatrixBase<double>::AddMat(const double alpha,
                            const MatrixBase<double>& rA, MatrixTransposeType transA) {
  if (&rA == this) {  // Make it work in this case.
    if (!transA) {
      Scale(alpha+1.0);
    } else {
      KALDI_ASSERT(num_rows_ == num_cols_ && "AddMat: adding to self(transposed): not symmetric.");
      double *data = data_;
      if (alpha == 1.0) {  // common case-- handle separately.
        for (MatrixIndexT row = 0;row < num_rows_;row++) {
          for (MatrixIndexT col = 0;col < row;col++) {
            double *lower = data + (row*stride_) + col,
                   *upper = data + (col*stride_) + row;
            double sum= *lower+*upper;
            *lower = *upper = sum;
          }
          *(data + (row*stride_) + row) *= 2.0;  // diagonal.
        }
      } else {
        for (MatrixIndexT row = 0;row < num_rows_;row++) {
          for (MatrixIndexT col = 0;col < row;col++) {
            double *lower = data + (row*stride_) + col,
                   *upper = data + (col*stride_) + row;
            double lower_tmp = *lower;
            *lower += alpha * *upper;
            *upper += alpha * lower_tmp;
          }
          *(data + (row*stride_) + row) *= (1.0+alpha);  // diagonal.
        }
      }
    }
  } else {
    int aStride = (int)rA.stride_, stride = stride_;
    double *adata = rA.data_, *data = data_;
    if (transA == kNoTrans) {
      KALDI_ASSERT(rA.num_rows_ == num_rows_ && rA.num_cols_ == num_cols_);
      for (MatrixIndexT row = 0;row < num_rows_;row++, adata+=aStride, data+=stride)
        cblas_daxpy(num_cols_, alpha, adata, 1, data, 1);
    } else {
      KALDI_ASSERT(rA.num_cols_ == num_rows_ && rA.num_rows_ == num_cols_);
      for (MatrixIndexT row = 0;row < num_rows_;row++, adata++, data+=stride)
        cblas_daxpy(num_cols_, alpha, adata, aStride, data, 1);
    }
  }
}

template<class Real>
template<class OtherReal>
void MatrixBase<Real>::AddSp(const Real alpha, const SpMatrix<OtherReal> &S) {
  KALDI_ASSERT(S.NumRows() == NumRows() && S.NumRows() == NumCols());
  Real *data = data_; const OtherReal *sdata = S.Data();
  MatrixIndexT num_rows = NumRows(), stride = Stride();
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    for (MatrixIndexT j = 0; j < i; j++, sdata++) {
      data[i*stride + j] += alpha * *sdata;
      data[j*stride + i] += alpha * *sdata;
    }
    data[i*stride + i] += alpha * *sdata++;
  }
}

// instantiate the template above.
template
void MatrixBase<float>::AddSp(const float alpha, const SpMatrix<float> &S);
template
void MatrixBase<double>::AddSp(const double alpha, const SpMatrix<double> &S);
template
void MatrixBase<float>::AddSp(const float alpha, const SpMatrix<double> &S);
template
void MatrixBase<double>::AddSp(const double alpha, const SpMatrix<float> &S);


#ifndef HAVE_ATLAS
//****************************************************************************
//****************************************************************************
template<>
void MatrixBase<float>::LapackGesvd(VectorBase<float> *s, MatrixBase<float> *U_in, MatrixBase<float> *V_in) {
  KALDI_ASSERT(s != NULL && U_in != this && V_in != this);

  Matrix<float> tmpU, tmpV;
  if (U_in == NULL) tmpU.Resize(this->num_rows_, 1);  // work-space if U_in empty.
  if (V_in == NULL) tmpV.Resize(1, this->num_cols_);  // work-space if V_in empty.

  /// Impementation notes:
  /// Lapack works in column-order, therefore the dimensions of *this are
  /// swapped as well as the U and V matrices.

  KaldiBlasInt M   = num_cols_;
  KaldiBlasInt N   = num_rows_;
  KaldiBlasInt LDA = Stride();

  KALDI_ASSERT(N>=M);  // NumRows >= columns.

  if (U_in)
    KALDI_ASSERT((int)U_in->num_rows_ == N && (int)U_in->num_cols_ == M);
  if (V_in)
    KALDI_ASSERT((int)V_in->num_rows_ == M && (int)V_in->num_cols_ == M);

  KALDI_ASSERT((int)s->Dim() == std::min(M, N));

  MatrixBase<float> *U = (U_in ? U_in : &tmpU);
  MatrixBase<float> *V = (V_in ? V_in : &tmpV);

  KaldiBlasInt V_stride      = V->Stride();
  KaldiBlasInt U_stride      = U->Stride();

  // Original LAPACK recipe
  // KaldiBlasInt l_work = std::max(std::max<long int>
  //   (1, 3*std::min(M, N)+std::max(M, N)), 5*std::min(M, N))*2;
  KaldiBlasInt l_work = -1;
  float   work_query;
  KaldiBlasInt result;

  // query for work space
  char *u_job = const_cast<char*>(U_in ? "s" : "N");  // "s" == skinny, "N" == "none."
  char *v_job = const_cast<char*>(V_in ? "s" : "N");  // "s" == skinny, "N" == "none."
  sgesvd_(v_job, u_job,
          &M, &N, data_, &LDA,
          s->Data(),
          V->Data(), &V_stride,
          U->Data(), &U_stride,
          &work_query, &l_work,
		  &result);

  l_work = static_cast<KaldiBlasInt>(work_query);
  float* p_work = new float[l_work];

  // perform svd
  sgesvd_(v_job, u_job,
          &M, &N, data_, &LDA,
          s->Data(),
          V->Data(), &V_stride,
          U->Data(), &U_stride,
          p_work, &l_work,
          &result);

  KALDI_ASSERT(result >= 0 && "Call to CLAPACK sgesvd_ called with wrong arguments");

  if (result != 0) {
    KALDI_ERR << "CLAPACK sgesvd_ : some weird convergence not satisfied";
  }

  delete [] p_work;
}

template<>
void MatrixBase<double>::LapackGesvd(VectorBase<double> *s, MatrixBase<double> *U_in, MatrixBase<double> *V_in) {
  KALDI_ASSERT(s != NULL && U_in != this && V_in != this);

  Matrix<double> tmpU, tmpV;
  if (U_in == NULL) tmpU.Resize(this->num_rows_, 1);  // work-space if U_in empty.
  if (V_in == NULL) tmpV.Resize(1, this->num_cols_);  // work-space if V_in empty.

  /// Impementation notes:
  /// Lapack works in column-order, therefore the dimensions of *this are
  /// swapped as well as the U and V matrices.

  KaldiBlasInt M   = num_cols_;
  KaldiBlasInt N   = num_rows_;
  KaldiBlasInt LDA = Stride();

  KALDI_ASSERT(N>=M);  // NumRows >= columns.

  if (U_in) {
    KALDI_ASSERT((int)U_in->num_rows_ == N && (int)U_in->num_cols_ == M);
  }
  if (V_in) {
    KALDI_ASSERT((int)V_in->num_rows_ == M && (int)V_in->num_cols_ == M);
  }
  KALDI_ASSERT((int)s->Dim() == std::min(M, N));

  MatrixBase<double> *U = (U_in ? U_in : &tmpU);
  MatrixBase<double> *V = (V_in ? V_in : &tmpV);

  KaldiBlasInt V_stride      = V->Stride();
  KaldiBlasInt U_stride      = U->Stride();

  // Original LAPACK recipe
  // KaldiBlasInt l_work = std::max(std::max<long int>
  //   (1, 3*std::min(M, N)+std::max(M, N)), 5*std::min(M, N))*2;
  KaldiBlasInt l_work = -1;
  double   work_query;
  KaldiBlasInt result;

  // query for work space
  char *u_job = const_cast<char*>(U_in ? "s" : "N");  // "s" == skinny, "N" == "none."
  char *v_job = const_cast<char*>(V_in ? "s" : "N");  // "s" == skinny, "N" == "none."
  dgesvd_(v_job, u_job,
          &M, &N, data_, &LDA,
          s->Data(),
          V->Data(), &V_stride,
          U->Data(), &U_stride,
          &work_query, &l_work,
		  &result);

  l_work = static_cast<KaldiBlasInt>(work_query);
  double* p_work = new double[l_work];

  // perform svd
  dgesvd_(v_job, u_job,
          &M, &N, data_, &LDA,
          s->Data(),
          V->Data(), &V_stride,
          U->Data(), &U_stride,
          p_work, &l_work,
          &result);

  KALDI_ASSERT(result >= 0 && "Call to CLAPACK dgesvd_ called with wrong arguments");

  if (result != 0) {
    KALDI_ERR << "CLAPACK sgesvd_ : some weird convergence not satisfied";
  }

  delete [] p_work;
}

#endif

// Copy constructor.  Copies data to newly allocated memory.
template<typename Real>
Matrix<Real>::Matrix (const MatrixBase<Real> & M,
                      MatrixTransposeType trans/*=kNoTrans*/)
                      : MatrixBase<Real>() {
  if (trans == kNoTrans) {
    Resize(M.num_rows_, M.num_cols_);
    CopyFromMat(M);
  } else {
    Resize(M.num_cols_, M.num_rows_);
    CopyFromMat(M, kTrans);
  }
}

// Copy constructor.  Copies data to newly allocated memory.
template<typename Real>
Matrix<Real>::Matrix (const Matrix<Real> & M):
  MatrixBase<Real>() {
  Resize(M.num_rows_, M.num_cols_);
  CopyFromMat(M);
}

/// Copy constructor from another type.
template<typename Real>
template<typename OtherReal>
Matrix<Real>::Matrix(const MatrixBase<OtherReal> & M,
                     MatrixTransposeType trans) : MatrixBase<Real>() {
  if (trans == kNoTrans) {
    Resize(M.NumRows(), M.NumCols());
    CopyFromMat(M);
  } else {
    Resize(M.NumCols(), M.NumRows());
    CopyFromMat(M, kTrans);
  }
}

// Instantiate this constructor for float->double and double->float.
template
Matrix<float>::Matrix(const MatrixBase<double> & M,
                      MatrixTransposeType trans);
template
Matrix<double>::Matrix(const MatrixBase<float> & M,
                       MatrixTransposeType trans);

template<typename Real>
inline void Matrix<Real>::Init(const MatrixIndexT rows,
                        const MatrixIndexT cols) {
  if (rows*cols == 0) {
    KALDI_ASSERT(rows == 0 && cols == 0);
    this->num_rows_ = 0;
    this->num_cols_ = 0;
    this->stride_ = 0;
    this->data_ = NULL;
#ifdef KALDI_MEMALIGN_MANUAL
    free_data_=NULL;
#endif
    return;
  }
  // initialize some helping vars
  MatrixIndexT  skip;
  MatrixIndexT  real_cols;
  MatrixIndexT  size;
  void*   data;       // aligned memory block
  void*   free_data;  // memory block to be really freed

  // compute the size of skip and real cols
  skip      = ((16 / sizeof(Real)) - cols % (16 / sizeof(Real)))
      % (16 / sizeof(Real));
  real_cols = cols + skip;
  size      = rows * real_cols * sizeof(Real);

  // allocate the memory and set the right dimensions and parameters
  if (NULL != (data = KALDI_MEMALIGN(16, size, &free_data))) {
    MatrixBase<Real>::data_        = static_cast<Real *> (data);
#ifdef KALDI_MEMALIGN_MANUAL
    free_data_    = static_cast<Real *> (free_data);
#endif
    MatrixBase<Real>::num_rows_      = rows;
    MatrixBase<Real>::num_cols_      = cols;
    MatrixBase<Real>::stride_  = real_cols;
  } else {
    throw std::bad_alloc();
  }
}

template<typename Real>
void Matrix<Real>::Resize(const MatrixIndexT rows,
                          const MatrixIndexT cols,
                          MatrixResizeType resize_type) {
  // the next block uses recursion to handle what we have to do if
  // resize_type == kCopyData.
  if (resize_type == kCopyData) {
    if (this->data_ == NULL || rows == 0) resize_type = kSetZero;  // nothing to copy.
    else if (rows == this->num_rows_ && cols == this->num_cols_) { return; } // nothing to do.
    else {
      // set tmp to a matrix of the desired size; if new matrix
      // is bigger in some dimension, zero it.
      MatrixResizeType new_resize_type =
          (rows > this->num_rows_ || cols > this->num_cols_) ? kSetZero : kUndefined;
      Matrix<Real> tmp(rows, cols, new_resize_type);
      MatrixIndexT rows_min = std::min(rows, this->num_rows_),
          cols_min = std::min(cols, this->num_cols_);
      tmp.Range(0, rows_min, 0, cols_min).
          CopyFromMat(this->Range(0, rows_min, 0, cols_min));
      tmp.Swap(this);
      // and now let tmp go out of scope, deleting what was in *this.
      return;
    }
  }
  // At this point, resize_type == kSetZero or kUndefined.

  if (MatrixBase<Real>::data_ != NULL) {
    if (rows == MatrixBase<Real>::num_rows_
        && cols == MatrixBase<Real>::num_cols_) {
      if (resize_type == kSetZero)
        this->SetZero();
      return;
    }
    else
      Destroy();
  }
  Init(rows, cols);
  if (resize_type == kSetZero) MatrixBase<Real>::SetZero();
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromMat(const MatrixBase<OtherReal> & M,
                                   MatrixTransposeType Trans) {
  // CopyFromMat called from ourself.  Nothing to do.
  if (sizeof(Real) == sizeof(OtherReal) && (void*)(&M) == (void*)this)
    return;
  if (Trans == kNoTrans) {
    KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
    for (MatrixIndexT i = 0; i < num_rows_; i++)
      (*this).Row(i).CopyFromVec(M.Row(i));
  } else {
    KALDI_ASSERT(num_cols_ == M.NumRows() && num_rows_ == M.NumCols());
    for (MatrixIndexT i = 0; i < num_rows_; i++)
      for (MatrixIndexT j = 0; j < num_cols_; j++)
        (*this)(i, j) = M(j, i);
  }
}

// template instantiations.
template
void MatrixBase<float>::CopyFromMat(const MatrixBase<double> & M,
                                    MatrixTransposeType Trans);
template
void MatrixBase<double>::CopyFromMat(const MatrixBase<float> & M,
                                     MatrixTransposeType Trans);
template
void MatrixBase<float>::CopyFromMat(const MatrixBase<float> & M,
                                    MatrixTransposeType Trans);
template
void MatrixBase<double>::CopyFromMat(const MatrixBase<double> & M,
                                     MatrixTransposeType Trans);

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromSp(const SpMatrix<OtherReal> & M) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
  // MORE EFFICIENT IF LOWER TRIANGULAR!  Reverse code otherwise.
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    for (MatrixIndexT j = 0; j < i; j++) {
      (*this)(j, i)  = (*this)(i, j) = M(i, j);
    }
    (*this)(i, i) = M(i, i);
  }
}

// Instantiate this function
template
void MatrixBase<float>::CopyFromSp(const SpMatrix<float> & M);
template
void MatrixBase<float>::CopyFromSp(const SpMatrix<double> & M);
template
void MatrixBase<double>::CopyFromSp(const SpMatrix<float> & M);
template
void MatrixBase<double>::CopyFromSp(const SpMatrix<double> & M);


template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromTp(const TpMatrix<OtherReal> & M,
                                  MatrixTransposeType Trans) {
  if (Trans == kNoTrans) {
    KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
    SetZero();
    // MORE EFFICIENT IF LOWER TRIANGULAR!  Reverse code otherwise.
    for (MatrixIndexT i = 0;i < num_rows_; i++) {
      for (MatrixIndexT j = 0;j < i;j++) {
        (*this)(i, j) = M(i, j);
      }
      (*this)(i, i) = M(i, i);
    }
  } else {
    SetZero();
    KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
    for (MatrixIndexT i = 0; i < num_rows_; i++) {
      for (MatrixIndexT j = 0; j < i; j++) {
        (*this)(j, i) = M(i, j);
      }
      (*this)(i, i) = M(i, i);
    }
  }
}

template
void MatrixBase<float>::CopyFromTp(const TpMatrix<float> & M,
                                   MatrixTransposeType trans);
template
void MatrixBase<float>::CopyFromTp(const TpMatrix<double> & M,
                                   MatrixTransposeType trans);
template
void MatrixBase<double>::CopyFromTp(const TpMatrix<float> & M,
                                    MatrixTransposeType trans);
template
void MatrixBase<double>::CopyFromTp(const TpMatrix<double> & M,
                                    MatrixTransposeType trans);


template<typename Real>
void MatrixBase<Real>::CopyRowsFromVec(const VectorBase<Real> &rv) {
  KALDI_ASSERT(rv.Dim() == num_rows_*num_cols_);

  if (stride_ == num_cols_) {
    // one big copy operation.
    const Real *rv_data = rv.Data();
    std::memcpy(data_, rv_data, sizeof(Real)*num_rows_*num_cols_);
  } else {
    const Real *rv_data = rv.Data();
    for (MatrixIndexT r = 0; r < num_rows_; r++) {
      Real *row_data = RowData(r);
      for (MatrixIndexT c = 0; c < num_cols_; c++) {
        row_data[c] = rv_data[c];
      }
      rv_data += num_cols_;
    }
  }
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyRowsFromVec(const VectorBase<OtherReal> &rv) {
  KALDI_ASSERT(rv.Dim() == num_rows_*num_cols_);
  const OtherReal *rv_data = rv.Data();
  for (MatrixIndexT r = 0; r < num_rows_; r++) {
    Real *row_data = RowData(r);
    for (MatrixIndexT c = 0; c < num_cols_; c++) {
      row_data[c] = static_cast<Real>(rv_data[c]);
    }
    rv_data += num_cols_;
  }
}

template
void MatrixBase<float>::CopyRowsFromVec(const VectorBase<double> &rv);
template
void MatrixBase<double>::CopyRowsFromVec(const VectorBase<float> &rv);

template<typename Real>
void MatrixBase<Real>::CopyColsFromVec(const VectorBase<Real> &rv) {
  KALDI_ASSERT(rv.Dim() == num_rows_*num_cols_);

  const Real *v_inc_data = rv.Data();
  Real *m_inc_data = data_;

  for (MatrixIndexT c = 0; c < num_cols_; c++) {
    for (MatrixIndexT r = 0; r < num_rows_; r++) {
      m_inc_data[r * stride_] = v_inc_data[r];
    }
    v_inc_data += num_rows_;
    m_inc_data ++;
  }
}


template<typename Real>
void MatrixBase<Real>::CopyRowFromVec(const VectorBase<Real> &rv, const MatrixIndexT row) {
  KALDI_ASSERT(rv.Dim() == num_cols_ &&
               static_cast<UnsignedMatrixIndexT>(row) <
               static_cast<UnsignedMatrixIndexT>(num_rows_));

  const Real *rv_data = rv.Data();
  Real *row_data = RowData(row);

  std::memcpy(row_data, rv_data, num_cols_ * sizeof(Real));
}

template<typename Real>
void MatrixBase<Real>::CopyDiagFromVec(const VectorBase<Real> &rv) {
  KALDI_ASSERT(rv.Dim() == std::min(num_cols_, num_rows_));
  const Real *rv_data = rv.Data(), *rv_end = rv_data + rv.Dim();
  Real *my_data = this->Data();
  for (; rv_data != rv_end; rv_data++, my_data += (this->stride_+1))
    *my_data = *rv_data;
}

template<typename Real>
void MatrixBase<Real>::CopyColFromVec(const VectorBase<Real> &rv,
                                      const MatrixIndexT col) {
  KALDI_ASSERT(rv.Dim() == num_rows_ &&
               static_cast<UnsignedMatrixIndexT>(col) <
               static_cast<UnsignedMatrixIndexT>(num_cols_));

  const Real *rv_data = rv.Data();
  Real *col_data = data_ + col;

  for (MatrixIndexT r = 0; r < num_rows_; r++)
    col_data[r * stride_] = rv_data[r];
}



template<typename Real>
void Matrix<Real>::RemoveRow(MatrixIndexT i) {
  KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
               static_cast<UnsignedMatrixIndexT>(MatrixBase<Real>::num_rows_)
               && "Access out of matrix");
  for (MatrixIndexT j = i + 1; j <  MatrixBase<Real>::num_rows_; j++)
     MatrixBase<Real>::Row(j-1).CopyFromVec( MatrixBase<Real>::Row(j));
   MatrixBase<Real>::num_rows_--;
}

template<typename Real>
void Matrix<Real>::Destroy() {
  // we need to free the data block if it was defined
#ifndef KALDI_MEMALIGN_MANUAL
  if (NULL != MatrixBase<Real>::data_)
      KALDI_MEMALIGN_FREE( MatrixBase<Real>::data_);
#else
  if (NULL != MatrixBase<Real>::data_)
      KALDI_MEMALIGN_FREE(free_data_);
  free_data_ = NULL;
#endif
  MatrixBase<Real>::data_ = NULL;
  MatrixBase<Real>::num_rows_ = MatrixBase<Real>::num_cols_
                              = MatrixBase<Real>::stride_ = 0;
}



template<typename Real>
void MatrixBase<Real>::MulElements(const MatrixBase<Real>& a) {
  MatrixIndexT i;
  MatrixIndexT j;

  for (i = 0; i < num_rows_; ++i) {
    for (j = 0; j < num_cols_; ++j) {
      (*this)(i, j) *= a(i, j);
    }
  }
}

template<typename Real>
Real MatrixBase<Real>::Sum() const {
  double sum = 0.0;

  for (MatrixIndexT i = 0; i < num_rows_; ++i) {
    for (MatrixIndexT j = 0; j < num_cols_; ++j) {
      sum += (*this)(i, j);
    }
  }

  return (Real)sum;
}

template<>
void MatrixBase<float>::Scale(float alpha) {
  if(num_cols_ == stride_) {
    cblas_sscal(num_rows_ * num_cols_, alpha,
                data_, 1);
  } else {
    float *data = data_;
    for (MatrixIndexT i = 0; i < num_rows_; ++i, data += stride_) {
      cblas_sscal(num_cols_, alpha, data, 1);
    }
  }
}

template<>
void MatrixBase<double>::Scale(double alpha) {
  if(num_cols_ == stride_) {
    cblas_dscal(num_rows_ * num_cols_, alpha,
                data_, 1);
  } else {
    double *data = data_;
    for (MatrixIndexT i = 0; i < num_rows_; ++i, data += stride_) {
      cblas_dscal(num_cols_, alpha, data, 1);
    }
  }
}

template<typename Real>  // scales each row by scale[i].
void MatrixBase<Real>::MulRowsVec(const VectorBase<Real>& scale) {
  KALDI_ASSERT(scale.Dim() == num_rows_);
  MatrixIndexT M = num_rows_, N = num_cols_;

  for (MatrixIndexT i = 0; i < M; i++) {
    Real this_scale = scale(i);
    for (MatrixIndexT j = 0; j < N; j++) {
      (*this)(i, j) *= this_scale;
    }
  }
}

template<typename Real>  // scales each column by scale[i].
void MatrixBase<Real>::MulColsVec(const VectorBase<Real>& scale) {
  KALDI_ASSERT(scale.Dim() == num_cols_);
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    for (MatrixIndexT j = 0; j < num_cols_; j++) {
      Real this_scale = scale(j);
      (*this)(i, j) *= this_scale;
    }
  }
}

template<typename Real>
void MatrixBase<Real>::SetZero() {
  if (num_cols_ == stride_)
    memset(data_, 0, sizeof(Real)*num_rows_*num_cols_);
  else
    for (MatrixIndexT row = 0; row < num_rows_; row++)
      memset(data_ + row*stride_, 0, sizeof(Real)*num_cols_);
}

template<typename Real>
void MatrixBase<Real>::Set(Real value) {
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    for (MatrixIndexT col = 0; col < num_cols_; col++) {
      (*this)(row, col) = value;
    }
  }
}

template<typename Real>
void MatrixBase<Real>::SetUnit() {
  SetZero();
  for (MatrixIndexT row = 0; row < std::min(num_rows_, num_cols_); row++)
    (*this)(row, row) = 1.0;
}

template<typename Real>
void MatrixBase<Real>::SetRandn() {
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    for (MatrixIndexT col = 0; col < num_cols_; col++) {
      (*this)(row, col) = static_cast<Real>(kaldi::RandGauss());
    }
  }
}

template<typename Real>
void MatrixBase<Real>::Write(std::ostream &os, bool binary) const {
  if (!os.good()) {
    KALDI_ERR << "Failed to write matrix to stream: stream not good";
  }
  if (binary) {  // Use separate binary and text formats,
    // since in binary mode we need to know if it's float or double.
    std::string my_marker = (sizeof(Real) == 4 ? "FM" : "DM");

    WriteMarker(os, binary, my_marker);
    {
      int32 rows = this->num_rows_;  // make the size 32-bit on disk.
      int32 cols = this->num_cols_;
      KALDI_ASSERT(this->num_rows_ == (MatrixIndexT) rows);
      KALDI_ASSERT(this->num_cols_ == (MatrixIndexT) cols);
      WriteBasicType(os, binary, rows);
      WriteBasicType(os, binary, cols);
    }
    if (Stride() == NumCols())
      os.write(reinterpret_cast<const char*> (Data()), sizeof(Real)
               * num_rows_ * num_cols_);
    else
      for (MatrixIndexT i = 0; i < num_rows_; i++)
        os.write(reinterpret_cast<const char*> (RowData(i)), sizeof(Real)
                 * num_cols_);
    if (!os.good()) {
      KALDI_ERR << "Failed to write matrix to stream";
    }
  } else {  // text mode.
    if (num_cols_ == 0) {
      os << " [ ]\n";
    } else {
      os << " [";
      for (MatrixIndexT i = 0; i < num_rows_; i++) {
        os << "\n  ";
        for (MatrixIndexT j = 0; j < num_cols_; j++)
          os << (*this)(i, j) << " ";
      }
      os << "]\n";
    }
  }
}


template<typename Real>
void MatrixBase<Real>::Read(std::istream & is, bool binary, bool add) {
  if (add) {
    Matrix<Real> tmp(num_rows_, num_cols_);
    tmp.Read(is, binary, false);  // read without adding.
    if (tmp.num_rows_ != this->num_rows_ || tmp.num_cols_ != this->num_cols_)
      KALDI_ERR << "MatrixBase::Read, size mismatch "
                << this->num_rows_ << ", " << this->num_cols_
                << " vs. " << tmp.num_rows_ << ", " << tmp.num_cols_;
    this->AddMat(1.0, tmp);
    return;
  }
  // now assume add == false.

  //  In order to avoid rewriting this, we just declare a Matrix and
  // use it to read the data, then copy.
  Matrix<Real> tmp;
  tmp.Read(is, binary, false);
  if (tmp.NumRows() != NumRows() || tmp.NumCols() != NumCols()) {
    KALDI_ERR << "MatrixBase<Real>::Read, size mismatch "
              << NumRows() << " x " << NumCols() << " versus "
              << tmp.NumRows() << " x " << tmp.NumCols();
  }
  CopyFromMat(tmp);
}


template<typename Real>
void Matrix<Real>::Read(std::istream & is, bool binary, bool add) {
  if (add) {
    Matrix<Real> tmp;
    tmp.Read(is, binary, false);  // read without adding.
    if (this->num_rows_ == 0) this->Resize(tmp.num_rows_, tmp.num_cols_);
    else {
      if (this->num_rows_ != tmp.num_rows_ || this->num_cols_ != tmp.num_cols_) {
        if (tmp.num_rows_ == 0) return;  // do nothing in this case.
        else KALDI_ERR << "Matrix::Read, size mismatch "
                       << this->num_rows_ <<  ", " << this->num_cols_
                       << " vs. " << tmp.num_rows_ << ", " << tmp.num_cols_;
      }
    }
    this->AddMat(1.0, tmp);
    return;
  }

  // now assume add == false.
  MatrixIndexT pos_at_start = is.tellg();
  std::ostringstream specific_error;

  if (binary) {  // Read in binary mode.
    int peekval = PeekMarker(is, binary);
    const char *my_marker =  (sizeof(Real) == 4 ? "FM" : "DM");
    char other_marker_start = (sizeof(Real) == 4 ? 'D' : 'F');
    if (peekval == other_marker_start) {  // need to instantiate the other type to read it.
      typedef typename OtherReal<Real>::Real OtherType;  // if Real == float, OtherType == double, and vice versa.
      Matrix<OtherType> other(this->num_rows_, this->num_cols_);
      other.Read(is, binary, false);  // add is false at this point anyway.
      this->Resize(other.NumRows(), other.NumCols());
      this->CopyFromMat(other);
      return;
    }
    std::string marker;
    ReadMarker(is, binary, &marker);
    if (marker != my_marker) {
      specific_error << ": Expected marker " << my_marker << ", got " << marker;
      goto bad;
    }
    int32 rows, cols;
    ReadBasicType(is, binary, &rows);  // throws on error.
    ReadBasicType(is, binary, &cols);  // throws on error.
    if ((MatrixIndexT)rows != this->num_rows_ || (MatrixIndexT)cols != this->num_cols_) {
      this->Resize(rows, cols);
    }
    if (this->Stride() == this->NumCols() && rows*cols!=0) {
      is.read(reinterpret_cast<char*>(this->Data()),
              sizeof(Real)*rows*cols);
      if (is.fail()) goto bad;
    } else {
      for (MatrixIndexT i = 0; i < (MatrixIndexT)rows; i++) {
        is.read(reinterpret_cast<char*>(this->RowData(i)), sizeof(Real)*cols);
        if (is.fail()) goto bad;
      }
    }
    if (is.eof()) return;
    if (is.fail()) goto bad;
    return;
  } else {  // Text mode.
    std::string str;
    is >> str;  // get a token
    if (is.fail()) { specific_error << ": Expected \"[\", got EOF"; goto bad; }
    if (!str.compare("DM") || !str.compare("FM")) {  // Back compatibility.
      is >> str;  // get #rows
      is >> str;  // get #cols
      is >> str;  // get "["
    }
    if (str == "[]") { Resize(0, 0); return; } // Be tolerant of variants.
    else if (str != "[") {
      specific_error << ": Expected \"[\", got \"" << str << '"';
      goto bad;
    }
    // At this point, we have read "[".
    std::vector<std::vector<Real>* > data;
    std::vector<Real> *cur_row = new std::vector<Real>;
    while (1) {
      int i = is.peek();
      if (i == -1) { specific_error << "Got EOF while reading matrix data"; goto cleanup; }
      else if (static_cast<char>(i) == ']') {  // Finished reading matrix.
        is.get();  // eat the "]".
        i = is.peek();
        if (static_cast<char>(i) == '\r') {
          is.get();
          is.get();  // get \r\n (must eat what we wrote)
        } else if (static_cast<char>(i) == '\n') { is.get(); } // get \n (must eat what we wrote)
        if (is.fail()) {
          KALDI_WARN << "After end of matrix data, read error.";
          // we got the data we needed, so just warn for this error.
        }
        // Now process the data.
        if (!cur_row->empty()) data.push_back(cur_row);
        else delete(cur_row);
        if (data.empty()) { this->Resize(0, 0); return; }
        else {
          int32 num_rows = data.size(), num_cols = data[0]->size();
          this->Resize(num_rows, num_cols);
          for (int32 i = 0; i < num_rows; i++) {
            if (static_cast<int32>(data[i]->size()) != num_cols) {
              specific_error << "Matrix has inconsistent #cols: " << num_cols
                             << " vs." << data[i]->size() << " (processing row"
                             << i;
              goto cleanup;
            }
            for (int32 j = 0; j < num_cols; j++)
              (*this)(i, j) = (*(data[i]))[j];
            delete data[i];
          }
        }
        return;
      } else if (static_cast<char>(i) == '\n' || static_cast<char>(i) == ';') {
        // End of matrix row.
        is.get();
        if (cur_row->size() != 0) {
          data.push_back(cur_row);
          cur_row = new std::vector<Real>;
          cur_row->reserve(data.back()->size());
        }
      } else if ( (i >= '0' && i <= '9') || i == '-' ) {  // A number...
        Real r;
        is >> r;
        if (is.fail()) {
          specific_error << "Stream failure/EOF while reading matrix data.";
          goto cleanup;
        }
        cur_row->push_back(r);
      } else if (isspace(i)) {
        is.get();  // eat the space and do nothing.
      } else {  // NaN or inf or error.
        std::string str;
        is >> str;
        if (!KALDI_STRCASECMP(str.c_str(), "inf") ||
            !KALDI_STRCASECMP(str.c_str(), "infinity")) {
          cur_row->push_back(std::numeric_limits<Real>::infinity());
          KALDI_WARN << "Reading infinite value into matrix.";
        } else if (!KALDI_STRCASECMP(str.c_str(), "nan")) {
          cur_row->push_back(std::numeric_limits<Real>::quiet_NaN());
          KALDI_WARN << "Reading NaN value into matrix.";
        } else {
          specific_error << "Expecting numeric matrix data, got " << str;
          goto cleanup;
        }
      }
    }
    // Note, we never leave the while () loop before this
    // line (we return from it.)
 cleanup: // We only reach here in case of error in the while loop above.
    delete cur_row;
    for (size_t i = 0; i < data.size(); i++)
      delete data[i];
    // and then go on to "bad" below, where we print error.
  }
bad:
  KALDI_ERR << "Failed to read matrix from stream.  " << specific_error.str()
            << " File position at start is "
            << pos_at_start << ", currently " << is.tellg();
}


// Constructor... note that this is not const-safe as it would
// be quite complicated to implement a "const SubMatrix" class that
// would not allow its contents to be changed.
template<typename Real>
SubMatrix<Real>::SubMatrix(const MatrixBase<Real>& rT,
                             const MatrixIndexT    ro,
                             const MatrixIndexT    r,
                             const MatrixIndexT    co,
                             const MatrixIndexT    c) {
  KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(ro) <
               static_cast<UnsignedMatrixIndexT>(rT.num_rows_) &&
               static_cast<UnsignedMatrixIndexT>(co) <
               static_cast<UnsignedMatrixIndexT>(rT.num_cols_) &&
               static_cast<UnsignedMatrixIndexT>(r) <=
               static_cast<UnsignedMatrixIndexT>(rT.num_rows_ - ro) &&
               static_cast<UnsignedMatrixIndexT>(c) <=
               static_cast<UnsignedMatrixIndexT>(rT.num_cols_ - co));
  // point to the begining of window
  MatrixBase<Real>::num_rows_ = r;
  MatrixBase<Real>::num_cols_ = c;
  MatrixBase<Real>::stride_ = rT.Stride();
  MatrixBase<Real>::data_ = rT.Data_workaround() + co + ro * rT.Stride();
}


template<class Real>
void MatrixBase<Real>::Add(const Real alpha) {
  Real *data = data_;
  MatrixIndexT stride = stride_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      data[c + stride*r] += alpha;
}


template<class Real>
Real MatrixBase<Real>::Cond() const {
  KALDI_ASSERT(num_rows_ > 0&&num_cols_ > 0);
  Vector<Real> singular_values(std::min(num_rows_, num_cols_));
  Svd(&singular_values);  // Get singular values...
  Real min = singular_values(0), max = singular_values(0);  // both absolute values...
  for (MatrixIndexT i = 1;i < singular_values.Dim();i++) {
    min = std::min((Real)std::abs(singular_values(i)), min); max = std::max((Real)std::abs(singular_values(i)), max);
  }
  if (min > 0) return max/min;
  else return 1.0e+100;
}

template<class Real>
Real MatrixBase<Real>::Trace(bool check_square) const  {
  KALDI_ASSERT(!check_square || num_rows_ == num_cols_);
  Real ans = 0.0;
  for (MatrixIndexT r = 0;r < std::min(num_rows_, num_cols_);r++) ans += data_ [r + stride_*r];
  return ans;
}

template<class Real>
Real MatrixBase<Real>::Max() const {
  KALDI_ASSERT(num_rows_ * num_cols_ > 0);
  Real ans= *data_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      if (data_[c + stride_*r] > ans)
        ans = data_[c + stride_*r];
  return ans;
}

template<class Real>
Real MatrixBase<Real>::Min() const {
  KALDI_ASSERT(num_rows_ * num_cols_ > 0);
  Real ans= *data_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      if (data_[c + stride_*r] < ans)
        ans = data_[c + stride_*r];
  return ans;
}



template <class Real>
void MatrixBase<Real>::AddMatMatMat(Real alpha,
                                    const MatrixBase<Real> &A, MatrixTransposeType transA,
                                    const MatrixBase<Real> &B, MatrixTransposeType transB,
                                    const MatrixBase<Real> &C, MatrixTransposeType transC,
                                    Real beta) {
  // Note on time taken with different orders of computation.  Assume not transposed in this /
  // discussion. Firstly, normalize expressions using A.NumCols == B.NumRows and B.NumCols == C.NumRows, prefer
  // rows where there is a choice.
  // time taken for (AB) is:  A.NumRows*B.NumRows*C.Rows
  // time taken for (AB)C is A.NumRows*C.NumRows*C.Cols
  // so this order is A.NumRows*B.NumRows*C.NumRows + A.NumRows*C.NumRows*C.NumCols.

  // time taken for (BC) is: B.NumRows*C.NumRows*C.Cols
  // time taken for A(BC) is: A.NumRows*B.NumRows*C.Cols
  // so this order is B.NumRows*C.NumRows*C.NumCols + A.NumRows*B.NumRows*C.Cols

  MatrixIndexT ARows = A.num_rows_, ACols = A.num_cols_, BRows = B.num_rows_, BCols = B.num_cols_,
          CRows = C.num_rows_, CCols = C.num_cols_;
  if (transA == kTrans) std::swap(ARows, ACols);
  if (transB == kTrans) std::swap(BRows, BCols);
  if (transC == kTrans) std::swap(CRows, CCols);

  MatrixIndexT AB_C_time = ARows*BRows*CRows + ARows*CRows*CCols;
  MatrixIndexT A_BC_time = BRows*CRows*CCols + ARows*BRows*CCols;

  if (AB_C_time < A_BC_time) {
    Matrix<Real> AB(ARows, BCols);
    AB.AddMatMat(1.0, A, transA, B, transB, 0.0);  // AB = A * B.
    (*this).AddMatMat(alpha, AB, kNoTrans, C, transC, beta);
  } else {
    Matrix<Real> BC(BRows, CCols);
    BC.AddMatMat(1.0, B, transB, C, transC, 0.0);  // BC = B * C.
    (*this).AddMatMat(alpha, A, transA, BC, kNoTrans, beta);
  }
}




template<class Real>
void MatrixBase<Real>::DestructiveSvd(VectorBase<Real> *s, MatrixBase<Real> *U, MatrixBase<Real> *Vt) {
  // Svd, *this = U*diag(s)*Vt.
  // With (*this).num_rows_ == m, (*this).num_cols_ == n,
  // Support only skinny Svd with m>=n (NumRows>=NumCols), and zero sizes for U and Vt mean
  // we do not want that output.  We expect that s.Dim() == m,
  // U is either 0 by 0 or m by n, and rv is either 0 by 0 or n by n.
  // Throws exception on error.

  KALDI_ASSERT(num_rows_>=num_cols_ && "Svd requires that #rows by >= #cols.");  // For compatibility with JAMA code.
  KALDI_ASSERT(s->Dim() == num_cols_);  // s should be the smaller dim.
  KALDI_ASSERT(U == NULL || (U->num_rows_ == num_rows_&&U->num_cols_ == num_cols_));
  KALDI_ASSERT(Vt == NULL || (Vt->num_rows_ == num_cols_&&Vt->num_cols_ == num_cols_));

  Real prescale = 1.0;
  if ( std::abs((*this)(0, 0) ) < 1.0e-30) {  // Very tiny value... can cause problems in Svd.
    Real max_elem = LargestAbsElem();
    if (max_elem != 0) {
      prescale = 1.0 / max_elem;
      if (std::abs(prescale) == std::numeric_limits<Real>::infinity()) { prescale = 1.0e+40; }
      (*this).Scale(prescale);
    }
  }

#ifndef HAVE_ATLAS
  // "S" == skinny Svd (only one we support because of compatibility with Jama one which is only skinny),
  // "N"== no eigenvectors wanted.
  LapackGesvd(s, U, Vt);
#else
  /*  if(num_rows_ > 1 && num_cols_ > 1 && (*this)(0, 0) == (*this)(1, 1)
     && Max() == Min() && (*this)(0, 0) != 0.0) { // special case that JamaSvd sometimes crashes on.
    KALDI_WARN << "Jama SVD crashes on this type of matrix, perturbing it to prevent crash.";
    for(int32 i = 0; i < NumRows(); i++)
      (*this)(i, i)  *= 1.00001;
      }*/
  bool ans = JamaSvd(s, U, Vt);
  if (Vt != NULL) Vt->Transpose();  // possibly to do: change this and also the transpose inside the JamaSvd routine.  note, Vt is square.
  if (!ans) {
    KALDI_ERR << "Error doing Svd";  // This one will be caught.
  }
#endif
  if (prescale != 1.0) s->Scale(1.0/prescale);
}

template<class Real>
void MatrixBase<Real>::Svd(VectorBase<Real> *s, MatrixBase<Real> *U, MatrixBase<Real> *Vt) const {
  try {
    if (num_rows_ >= num_cols_) {
      Matrix<Real> tmp(*this);
      tmp.DestructiveSvd(s, U, Vt);
    } else {
      Matrix<Real> tmp(*this, kTrans);  // transpose of *this.
      // rVt will have different dim so cannot transpose in-place --> use a temp matrix.
      Matrix<Real> Vt_Trans(Vt ? Vt->num_cols_ : 0, Vt ? Vt->num_rows_ : 0);
      // U will be transpose
      tmp.DestructiveSvd(s, Vt ? &Vt_Trans : NULL, U);
      if (U) U->Transpose();
      if (Vt) Vt->CopyFromMat(Vt_Trans, kTrans);  // copy with transpose.
    }
  } catch (...) {
    KALDI_ERR << "Error doing Svd (did not converge), first part of matrix is\n"
              << SubMatrix<Real>(*this, 0, std::min((MatrixIndexT)10, num_rows_), 0, std::min((MatrixIndexT)10, num_cols_));
  }
}

template<class Real>
bool MatrixBase<Real>::IsSymmetric(Real cutoff) const {
  MatrixIndexT R = num_rows_, C = num_cols_;
  if (R != C) return false;
  Real bad_sum = 0.0, good_sum = 0.0;
  for (MatrixIndexT i = 0;i < R;i++) {
    for (MatrixIndexT j = 0;j < i;j++) {
      Real a = (*this)(i, j), b = (*this)(j, i), avg = 0.5*(a+b), diff = 0.5*(a-b);
      good_sum += std::abs(avg); bad_sum += std::abs(diff);
    }
    good_sum += std::abs((*this)(i, i));
  }
  if (bad_sum > cutoff*good_sum) return false;
  return true;
}

template<class Real>
bool MatrixBase<Real>::IsDiagonal(Real cutoff) const{
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real bad_sum = 0.0, good_sum = 0.0;
  for (MatrixIndexT i = 0;i < R;i++) {
    for (MatrixIndexT j = 0;j < C;j++) {
      if (i == j) good_sum += std::abs((*this)(i, j));
      else bad_sum += std::abs((*this)(i, j));
    }
  }
  return (!(bad_sum > good_sum * cutoff));
}

template<class Real>
bool MatrixBase<Real>::IsUnit(Real cutoff) const {
  MatrixIndexT R = num_rows_, C = num_cols_;
  // if (R != C) return false;
  Real bad_max = 0.0;
  for (MatrixIndexT i = 0;i < R;i++)
    for (MatrixIndexT j = 0;j < C;j++)
      bad_max = std::max(bad_max, static_cast<Real>(std::abs( (*this)(i, j) - (i == j?1.0:0.0))));
  return (bad_max <= cutoff);
}

template<class Real>
bool MatrixBase<Real>::IsZero(Real cutoff)const {
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real bad_max = 0.0;
  for (MatrixIndexT i = 0;i < R;i++)
    for (MatrixIndexT j = 0;j < C;j++)
      bad_max = std::max(bad_max, static_cast<Real>(std::abs( (*this)(i, j) )));
  return (bad_max <= cutoff);
}

template<class Real>
Real MatrixBase<Real>::FrobeniusNorm() const{
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real sum = 0.0;
  for (MatrixIndexT i = 0;i < R;i++)
    for (MatrixIndexT j = 0;j < C;j++) {
      Real tmp = (*this)(i, j);
      sum +=  tmp*tmp;
    }
  return sqrt(sum);
}

template<typename Real>
bool MatrixBase<Real>::ApproxEqual(const MatrixBase<Real> &other, float tol) const {
  if (num_rows_ != other.num_rows_ || num_cols_ != other.num_cols_)
      KALDI_ERR << "ApproxEqual: size mismatch.";
  Matrix<Real> tmp(*this);
  tmp.AddMat(-1.0, other);
  return (tmp.FrobeniusNorm() <= static_cast<Real>(tol) *
          this->FrobeniusNorm());
}

template<typename Real>
bool MatrixBase<Real>::Equal(const MatrixBase<Real> &other) const {
  if (num_rows_ != other.num_rows_ || num_cols_ != other.num_cols_)
    KALDI_ERR << "Equal: size mismatch.";
  for (MatrixIndexT i = 0; i < num_rows_; i++)
    for (MatrixIndexT j = 0; j < num_cols_; j++)
      if ( (*this)(i, j) != other(i, j))
        return false;
  return true;
}


template<class Real>
Real MatrixBase<Real>::LargestAbsElem() const{
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real largest = 0.0;
  for (MatrixIndexT i = 0;i < R;i++)
    for (MatrixIndexT j = 0;j < C;j++)
      largest = std::max(largest, (Real)std::abs((*this)(i, j)));
  return largest;
}



// Uses Svd to compute the eigenvalue decomposition of a symmetric positive semidefinite
//   matrix:
// (*this) = rU * diag(rs) * rU^T, with rU an orthogonal matrix so rU^{-1} = rU^T.
// Does this by computing svd (*this) = U diag(rs) V^T ... answer is just U diag(rs) U^T.
// Throws exception if this failed to within supplied precision (typically because *this was not
// symmetric positive definite).

template<class Real>
void MatrixBase<Real>::SymPosSemiDefEig(VectorBase<Real> *rs, MatrixBase<Real> *rU, Real check_thresh) // e.g. check_thresh = 0.001
{
  const MatrixIndexT D = num_rows_;

  KALDI_ASSERT(num_rows_ == num_cols_);
  KALDI_ASSERT(IsSymmetric() && "SymPosSemiDefEig: expecting input to be symmetrical.");
  KALDI_ASSERT(rU->num_rows_ == D && rU->num_cols_ == D && rs->Dim() == D);

  Matrix<Real>  Vt(D, D);
  Svd(rs, rU, &Vt);

  // First just zero any singular values if the column of U and V do not have +ve dot product--
  // this may mean we have small negative eigenvalues, and if we zero them the result will be closer to correct.
  for (MatrixIndexT i = 0;i < D;i++) {
    Real sum = 0.0;
    for (MatrixIndexT j = 0;j < D;j++) sum += (*rU)(j, i) * Vt(i, j);
    if (sum < 0.0) (*rs)(i) = 0.0;
  }

  {
    Matrix<Real> tmpU(*rU); Vector<Real> tmps(*rs); tmps.ApplyPow(0.5);
    tmpU.MulColsVec(tmps);
    SpMatrix<Real> tmpThis(D);
    tmpThis.AddMat2(1.0, tmpU, kNoTrans, 0.0);
    Matrix<Real> tmpThisFull(tmpThis);
    float new_norm = tmpThisFull.FrobeniusNorm();
    float old_norm = (*this).FrobeniusNorm();
    tmpThisFull.AddMat(-1.0, (*this));

    if (!(old_norm == 0 && new_norm == 0)) {
      float diff_norm = tmpThisFull.FrobeniusNorm();
      if (std::abs(new_norm-old_norm) > old_norm*check_thresh || diff_norm > old_norm*check_thresh) {
        KALDI_WARN << "SymPosSemiDefEig seems to have failed " << diff_norm << " !<< "
                   << check_thresh << "*" << old_norm << ", maybe matrix was not "
                   << "positive semi definite.  Continuing anyway.";
      }
    }
  }
}


template<class Real>
Real MatrixBase<Real>::LogDet(Real *det_sign) const {
  Real log_det;
  Matrix<Real> tmp(*this);
  tmp.Invert(&log_det, det_sign, false);  // false== output not needed (saves some computation).
  return log_det;
}

template<class Real>
void MatrixBase<Real>::InvertDouble(Real *LogDet, Real *DetSign,
                                    bool inverse_needed) {
  double LogDet_tmp, DetSign_tmp;
  Matrix<double> dmat(*this);
  dmat.Invert(&LogDet_tmp, &DetSign_tmp, inverse_needed);
  if (inverse_needed) (*this).CopyFromMat(dmat);
  if (LogDet) *LogDet = LogDet_tmp;
  if (DetSign) *DetSign = DetSign_tmp;
}

template<typename Real>
void MatrixBase<Real>::InvertElements() {
  for (MatrixIndexT r = 0; r < num_rows_; ++r) {
    for (MatrixIndexT c = 0; c < num_cols_; ++c) {
      (*this)(r, c) = static_cast<Real>(1.0 / (*this)(r, c));
    }
  }
}

template<class Real>
void MatrixBase<Real>::Transpose() {
  KALDI_ASSERT(num_rows_ == num_cols_);
  MatrixIndexT M = num_rows_;
  for (MatrixIndexT i = 0;i < M;i++)
    for (MatrixIndexT j = 0;j < i;j++) {
      Real &a = (*this)(i, j), &b = (*this)(j, i);
      std::swap(a, b);
    }
}


template<class Real>
void Matrix<Real>::Transpose() {
  if (this->num_rows_ != this->num_cols_) {
    Matrix<Real> tmp(*this, kTrans);
    Resize(this->num_cols_, this->num_rows_);
    CopyFromMat(tmp);
  } else {
    (static_cast<MatrixBase<Real>&>(*this)).Transpose();
  }
}

template<class Real>
void MatrixBase<Real>::ApplyFloor(Real floor_val) {
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    for (MatrixIndexT j = 0; j < num_cols_; j++) {
      if ((*this)(i, j) < floor_val) {
        (*this)(i, j) = floor_val;
  }}}
}

template<class Real>
void MatrixBase<Real>::ApplyLog() {
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    Row(i).ApplyLog();
  }
}

template<class Real>
void MatrixBase<Real>::ApplyPow(Real power) {
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    Row(i).ApplyPow(power);
  }
}


template<class Real>
bool MatrixBase<Real>::Power(Real power) {
  KALDI_ASSERT(num_rows_ > 0 && num_rows_ == num_cols_);
  MatrixIndexT n = num_rows_;
  Matrix<Real> P(n, n);
  Vector<Real> re(n), im(n);
  this->Eig(&P, &re, &im);
  // Now attempt to take the complex eigenvalues to this power.
  for (MatrixIndexT i = 0; i < n; i++)
    if (!AttemptComplexPower(&(re(i)), &(im(i)), power))
      return false;  // e.g. real and negative, or zero, eigenvalues.

  Matrix<Real> D(n, n);  // D to the power.
  CreateEigenvalueMatrix(re, im, &D);

  Matrix<Real> tmp(n, n);  // P times D
  tmp.AddMatMat(1.0, P, kNoTrans, D, kNoTrans, 0.0);  // tmp := P*D
  P.Invert();
  // next line is: *this = tmp * P^{-1} = P * D * P^{-1}
  (*this).AddMatMat(1.0, tmp, kNoTrans, P, kNoTrans, 0.0);
  return true;
}

template<class Real>
void Matrix<Real>::Swap(Matrix<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->num_cols_, other->num_cols_);
  std::swap(this->num_rows_, other->num_rows_);
  std::swap(this->stride_, other->stride_);
#ifdef KALDI_MEMALIGN_MANUAL
  std::swap(this->free_data_, other->free_data_);
#endif
}

// Repeating this comment that appeared in the header:
// Eigenvalue Decomposition of a square NxN matrix into the form (*this) = P D
// P^{-1}.  Be careful: the relationship of D to the eigenvalues we output is
// slightly complicated, due to the need for P to be real.  In the symmetric
// case D is diagonal and real, but in
// the non-symmetric case there may be complex-conjugate pairs of eigenvalues.
// In this case, for the equation (*this) = P D P^{-1} to hold, D must actually
// be block diagonal, with 2x2 blocks corresponding to any such pairs.  If a
// pair is lambda +- i*mu, D will have a corresponding 2x2 block
// [lambda, mu; -mu, lambda].
// Note that if the input matrix (*this) is non-invertible, P may not be invertible
// so in this case instead of the equation (*this) = P D P^{-1} holding, we have
// instead (*this) P = P D.
//
// By making the pointer arguments non-NULL or NULL, the user can choose to take
// not to take the eigenvalues directly, and/or the matrix D which is block-diagonal
// with 2x2 blocks.
template<class Real>
void MatrixBase<Real>::Eig(MatrixBase<Real> *P,
                            VectorBase<Real> *r,
                           VectorBase<Real> *i) const {
  EigenvalueDecomposition<Real>  eig(*this);
  if (P) eig.GetV(P);
  if (r) eig.GetRealEigenvalues(r);
  if (i) eig.GetImagEigenvalues(i);
}


template class Matrix<float>;
template class Matrix<double>;
template class MatrixBase<float>;
template class MatrixBase<double>;
template class SubMatrix<float>;
template class SubMatrix<double>;


// Begin non-member function definitions.

//  /**
//   * @brief Extension of the HTK header
//  */
// struct HtkHeaderExt
//  {
// INT_32 mHeaderSize;
// INT_32 mVersion;
// INT_32 mSampSize;
//};

template<class Real>
bool ReadHtk(std::istream &is, Matrix<Real> *M_ptr, HtkHeader *header_ptr)
{
  // check instantiated with double or float.
  KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  Matrix<Real> &M = *M_ptr;
  HtkHeader htk_hdr;

  is.read((char*)&htk_hdr, sizeof(htk_hdr));  // we're being really POSIX here!
  if (is.fail()) {
    KALDI_WARN << "Could not read header from HTK feature file ";
    return false;
  }

  KALDI_SWAP4(htk_hdr.mNSamples);
  KALDI_SWAP4(htk_hdr.mSamplePeriod);
  KALDI_SWAP2(htk_hdr.mSampleSize);
  KALDI_SWAP2(htk_hdr.mSampleKind);

  M.Resize(htk_hdr.mNSamples, htk_hdr.mSampleSize / sizeof(float));

  MatrixIndexT i;
  MatrixIndexT j;
  if (sizeof(Real) == sizeof(float)) {
    for (i = 0; i< M.NumRows(); ++i) {
      is.read((char*)M.RowData(i), sizeof(float)*M.NumCols());
      if (is.fail()) {
        KALDI_WARN << "Could not read data from HTK feature file ";
        return false;
      }
      if (MachineIsLittleEndian()) {
        MatrixIndexT C = M.NumCols();
        for (j = 0; j < C; j++) {
          KALDI_SWAP4((M(i, j)));  // The HTK standard is big-endian!
        }
      }
    }
  } else {
    float *pmem = new float[M.NumCols()];
    for (i = 0; i < M.NumRows(); i++) {
      is.read((char*)pmem, sizeof(float)*M.NumCols());
      if (is.fail()) {
        KALDI_WARN << "Could not read data from HTK feature file ";
        delete [] pmem;
        return false;
      }
      if (MachineIsLittleEndian()) {  // HTK standard is big-endian!
        MatrixIndexT C = M.NumCols();
        for (j = 0; j < C; ++j) {
          KALDI_SWAP4(pmem[j]);
          M(i, j) = static_cast<Real>(pmem[j]);
        }
      }
    }
    delete [] pmem;
  }
  if (header_ptr) *header_ptr = htk_hdr;
  return true;
}


template
bool ReadHtk(std::istream &is, Matrix<float> *M, HtkHeader *header_ptr);

template
bool ReadHtk(std::istream &is, Matrix<double> *M, HtkHeader *header_ptr);

    template<class Real>
bool WriteHtk(std::ostream &os, const MatrixBase<Real> &M, HtkHeader htk_hdr) // header may be derived from a previous call to ReadHtk.  Must be in binary mode.
{
  KALDI_ASSERT(M.NumRows() == static_cast<MatrixIndexT>(htk_hdr.mNSamples));
  KALDI_ASSERT(M.NumCols() == static_cast<MatrixIndexT>(htk_hdr.mSampleSize) /
               static_cast<MatrixIndexT>(sizeof(float)));

  KALDI_SWAP4(htk_hdr.mNSamples);
  KALDI_SWAP4(htk_hdr.mSamplePeriod);
  KALDI_SWAP2(htk_hdr.mSampleSize);
  KALDI_SWAP2(htk_hdr.mSampleKind);

  os.write((char*)&htk_hdr, sizeof(htk_hdr));
  if (os.fail())  goto bad;

  MatrixIndexT i;
  MatrixIndexT j;
  if (sizeof(Real) == sizeof(float) && !MachineIsLittleEndian()) {
    for (i = 0; i< M.NumRows(); ++i) {  // Unlikely to reach here ever!
      os.write((char*)M.RowData(i), sizeof(float)*M.NumCols());
      if (os.fail()) goto bad;
    }
  } else {
    float *pmem = new float[M.NumCols()];

    for (i = 0; i < M.NumRows(); i++) {
      const Real *rowData = M.RowData(i);
      for (j = 0;j < M.NumCols();j++)
        pmem[j] =  static_cast<float> ( rowData[j] );
      if (MachineIsLittleEndian())
        for (j = 0;j < M.NumCols();j++)
          KALDI_SWAP4(pmem[j]);
      os.write((char*)pmem, sizeof(float)*M.NumCols());
      if (os.fail()) {
        delete [] pmem;
        goto bad;
      }
    }
    delete [] pmem;
  }
  return true;
bad:
  KALDI_WARN << "Could not write to HTK feature file ";
  return false;
}


template
bool WriteHtk(std::ostream &os, const MatrixBase<float> &M, HtkHeader htk_hdr);

template
bool WriteHtk(std::ostream &os, const MatrixBase<double> &M, HtkHeader htk_hdr);


template <class Real>
Real TraceMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                      const MatrixBase<Real> &B, MatrixTransposeType transB,
                      const MatrixBase<Real> &C, MatrixTransposeType transC) {
  MatrixIndexT ARows = A.NumRows(), ACols = A.NumCols(), BRows = B.NumRows(), BCols = B.NumCols(),
          CRows = C.NumRows(), CCols = C.NumCols();
  if (transA == kTrans) std::swap(ARows, ACols);
  if (transB == kTrans) std::swap(BRows, BCols);
  if (transC == kTrans) std::swap(CRows, CCols);
  KALDI_ASSERT( CCols == ARows && ACols == BRows && BCols == CRows && "TraceMatMatMat: args have mismatched dimensions.");
  if (ARows*BCols < std::min(BRows*CCols, CRows*ACols)) {
    Matrix<Real> AB(ARows, BCols);
    AB.AddMatMat(1.0, A, transA, B, transB, 0.0);  // AB = A * B.
    return TraceMatMat(AB, C, transC);
  } else if ( BRows*CCols < CRows*ACols) {
    Matrix<Real> BC(BRows, CCols);
    BC.AddMatMat(1.0, B, transB, C, transC, 0.0);  // BC = B * C.
    return TraceMatMat(BC, A, transA);
  } else {
    Matrix<Real> CA(CRows, ACols);
    CA.AddMatMat(1.0, C, transC, A, transA, 0.0);  // CA = C * A
    return TraceMatMat(CA, B, transB);
  }
}

template
float TraceMatMatMat(const MatrixBase<float> &A, MatrixTransposeType transA,
                    const MatrixBase<float> &B, MatrixTransposeType transB,
                    const MatrixBase<float> &C, MatrixTransposeType transC);

template
double TraceMatMatMat(const MatrixBase<double> &A, MatrixTransposeType transA,
                    const MatrixBase<double> &B, MatrixTransposeType transB,
                    const MatrixBase<double> &C, MatrixTransposeType transC);


template <class Real>
Real TraceMatMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                       const MatrixBase<Real> &B, MatrixTransposeType transB,
                       const MatrixBase<Real> &C, MatrixTransposeType transC,
                       const MatrixBase<Real> &D, MatrixTransposeType transD) {
  MatrixIndexT ARows = A.NumRows(), ACols = A.NumCols(), BRows = B.NumRows(), BCols = B.NumCols(),
          CRows = C.NumRows(), CCols = C.NumCols(), DRows = D.NumRows(), DCols = D.NumCols();
  if (transA == kTrans) std::swap(ARows, ACols);
  if (transB == kTrans) std::swap(BRows, BCols);
  if (transC == kTrans) std::swap(CRows, CCols);
  if (transD == kTrans) std::swap(DRows, DCols);
  KALDI_ASSERT( DCols == ARows && ACols == BRows && BCols == CRows && CCols == DRows && "TraceMatMatMat: args have mismatched dimensions.");
  if (ARows*BCols < std::min(BRows*CCols, std::min(CRows*DCols, DRows*ACols))) {
    Matrix<Real> AB(ARows, BCols);
    AB.AddMatMat(1.0, A, transA, B, transB, 0.0);  // AB = A * B.
    return TraceMatMatMat(AB, kNoTrans, C, transC, D, transD);
  } else if ((BRows*CCols) < std::min(CRows*DCols, DRows*ACols)) {
    Matrix<Real> BC(BRows, CCols);
    BC.AddMatMat(1.0, B, transB, C, transC, 0.0);  // BC = B * C.
    return TraceMatMatMat(BC, kNoTrans, D, transD, A, transA);
  } else if (CRows*DCols < DRows*ACols) {
    Matrix<Real> CD(CRows, DCols);
    CD.AddMatMat(1.0, C, transC, D, transD, 0.0);  // CD = C * D
    return TraceMatMatMat(CD, kNoTrans, A, transA, B, transB);
  } else {
    Matrix<Real> DA(DRows, ACols);
    DA.AddMatMat(1.0, D, transD, A, transA, 0.0);  // DA = D * A
    return TraceMatMatMat(DA, kNoTrans, B, transB, C, transC);
  }
}

template
float TraceMatMatMatMat(const MatrixBase<float> &A, MatrixTransposeType transA,
                        const MatrixBase<float> &B, MatrixTransposeType transB,
                        const MatrixBase<float> &C, MatrixTransposeType transC,
                        const MatrixBase<float> &D, MatrixTransposeType transD);

template
double TraceMatMatMatMat(const MatrixBase<double> &A, MatrixTransposeType transA,
                         const MatrixBase<double> &B, MatrixTransposeType transB,
                         const MatrixBase<double> &C, MatrixTransposeType transC,
                         const MatrixBase<double> &D, MatrixTransposeType transD);

template<class Real> void  SortSvd(VectorBase<Real>* rs, MatrixBase<Real>* rU,
                                   MatrixBase<Real>* rVt) {
  // Makes sure the Svd is sorted (from greatest to least element).
  MatrixIndexT D = rs->Dim();
  KALDI_ASSERT(rU->NumCols() == D && (!rVt || rVt->NumRows() == D) && rs->Min() >= 0);
  std::vector<std::pair<Real, MatrixIndexT> > vec(D);
  for (MatrixIndexT d = 0;d<D;d++) vec[d] = std::pair<Real, MatrixIndexT>(-(*rs)(d), d);  // negative because we want reverse order.
  std::sort(vec.begin(), vec.end());
  for (MatrixIndexT d = 0;d < D;d++) (*rs)(d) = -vec[d].first;
  {
    Matrix<Real> rUtmp(*rU);
    MatrixIndexT R = rU->NumRows();
    for (MatrixIndexT d = 0;d < D;d++) {
      MatrixIndexT oldidx = vec[d].second;
      for (MatrixIndexT r = 0;r < R;r++) (*rU)(r, d) = rUtmp(r, oldidx);
    }
  }
  if (rVt) {
    Matrix<Real> rVttmp(*rVt);
    for (MatrixIndexT d = 0;d < D;d++) {
      MatrixIndexT oldidx = vec[d].second;
      (*rVt).Row(d).CopyFromVec(rVttmp.Row(oldidx));
    }
  }
}

template
void SortSvd(VectorBase<float>* rs, MatrixBase<float>* rU,
             MatrixBase<float>* rVt);

template
void SortSvd(VectorBase<double>* rs, MatrixBase<double>* rU,
             MatrixBase<double>* rVt);

template<class Real>
void CreateEigenvalueMatrix(const VectorBase<Real> &re, const VectorBase<Real> &im,
                            MatrixBase<Real> *D) {
  MatrixIndexT n = re.Dim();
  KALDI_ASSERT(im.Dim() == n && D->NumRows() == n && D->NumCols() == n);

  MatrixIndexT j = 0;
  D->SetZero();
  while (j < n) {
    if (im(j) == 0) {  // Real eigenvalue
      (*D)(j, j) = re(j);
      j++;
    } else {  // First of a complex pair
      KALDI_ASSERT(j+1 < n && ApproxEqual(im(j+1), -im(j))
                   && ApproxEqual(re(j+1), re(j)));
      /// if (im(j) < 0.0) KALDI_WARN << "Negative first im part of pair\n";  // TEMP
      Real lambda = re(j), mu = im(j);
      // create 2x2 block [lambda, mu; -mu, lambda]
      (*D)(j, j) = lambda;
      (*D)(j, j+1) = mu;
      (*D)(j+1, j) = -mu;
      (*D)(j+1, j+1) = lambda;
      j += 2;
    }
  }
}

template
void CreateEigenvalueMatrix(const VectorBase<float> &re, const VectorBase<float> &im,
                            MatrixBase<float> *D);
template
void CreateEigenvalueMatrix(const VectorBase<double> &re, const VectorBase<double> &im,
                            MatrixBase<double> *D);



template<class Real>
bool AttemptComplexPower(Real *x_re, Real *x_im, Real power) {
  // Used in Matrix<Real>::Power().
  // Attempts to take the complex value x to the power "power",
  // assuming that power is fractional (i.e. we don't treat integers as a
  // special case).  Returns false if this is not possible, either
  // because x is negative and real (hence there is no obvious answer
  // that is "closest to 1", and anyway this case does not make sense
  // in the Matrix<Real>::Power() routine);
  // or because power is negative, and x is zero.

  // First solve for r and theta in
  // x_re = r*cos(theta), x_im = r*sin(theta)
  if (*x_re < 0.0 && *x_im == 0.0) return false;  // can't do
  // it for negative real values.
  Real r = std::sqrt((*x_re * *x_re) + (*x_im * *x_im));  // r == radius.
  if (power < 0.0 && r == 0.0) return false;
  Real theta = std::atan2(*x_im, *x_re);
  // Take the power.
  r = std::pow(r, power);
  theta *= power;
  *x_re = r * std::cos(theta);
  *x_im = r * std::sin(theta);
  return true;
}

template
bool AttemptComplexPower(float *x_re, float *x_im, float power);
template
bool AttemptComplexPower(double *x_re, double *x_im, double power);


template <>  // non-member but friend!
double TraceMatMat(const MatrixBase<double> &A, const MatrixBase<double> &B, MatrixTransposeType trans) {  // tr(A B), equivalent to sum of each element of A times same element in B'
  MatrixIndexT aStride = A.stride_, bStride = B.stride_;
  if (trans == kNoTrans) {
    KALDI_ASSERT(A.NumRows() == B.NumCols() && A.NumCols() == B.NumRows());
    double ans = 0.0;
    double *adata = A.data_, *bdata = B.data_;
    MatrixIndexT arows = A.NumRows(), acols = A.NumCols();
    for (MatrixIndexT row = 0;row < arows;row++, adata+=aStride, bdata++)
      ans += cblas_ddot(acols, adata, 1, bdata, bStride);
    return ans;
  } else {
    KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
    double ans = 0.0;
    double *adata = A.data_, *bdata = B.data_;
    MatrixIndexT arows = A.NumRows(), acols = A.NumCols();
    for (MatrixIndexT row = 0;row < arows;row++, adata+=aStride, bdata+=bStride)
      ans += cblas_ddot(acols, adata, 1, bdata, 1);
    return ans;
  }
}

template <>  // non-member but friend!
float TraceMatMat(const MatrixBase<float> &A, const MatrixBase<float> &B, MatrixTransposeType trans) {  // tr(A B), equivalent to sum of each element of A times same element in B'
  MatrixIndexT aStride = A.stride_, bStride = B.stride_;
  if (trans == kNoTrans) {
    KALDI_ASSERT(A.NumRows() == B.NumCols() && A.NumCols() == B.NumRows());
    float ans = 0.0;
    float *adata = A.data_, *bdata = B.data_;
    MatrixIndexT arows = A.NumRows(), acols = A.NumCols();
    for (MatrixIndexT row = 0;row < arows;row++, adata+=aStride, bdata++)
      ans += cblas_sdot(acols, adata, 1, bdata, bStride);
    return ans;
  } else {
    KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
    float ans = 0.0;
    float *adata = A.data_, *bdata = B.data_;
    MatrixIndexT arows = A.NumRows(), acols = A.NumCols();
    for (MatrixIndexT row = 0;row < arows;row++, adata+=aStride, bdata+=bStride)
      ans += cblas_sdot(acols, adata, 1, bdata, 1);
    return ans;
  }
}

template<typename Real>
Real MatrixBase<Real>::LogSumExp(Real prune) const {
  Real sum;
  if (sizeof(sum) == 8) sum = kLogZeroDouble;
  else sum = kLogZeroFloat;
  Real max_elem = Max(), cutoff;
  if (sizeof(Real) == 4) cutoff = max_elem + kMinLogDiffFloat;
  else cutoff = max_elem + kMinLogDiffDouble;
  if (prune > 0.0 && max_elem - prune > cutoff) // explicit pruning...
    cutoff = max_elem - prune;

  double sum_relto_max_elem = 0.0;

  for (MatrixIndexT i = 0; i < num_rows_; ++i) {
    for (MatrixIndexT j = 0; j < num_cols_; ++j) {
      BaseFloat f = (*this)(i, j);
      if (f >= cutoff)
        sum_relto_max_elem += exp(f - max_elem);
    }
  }
  return max_elem + std::log(sum_relto_max_elem);
}
// instantiate.
template float MatrixBase<float>::LogSumExp(float) const;
template double MatrixBase<double>::LogSumExp(double) const;

template<typename Real>
Real MatrixBase<Real>::ApplySoftMax() {
  Real lse = LogSumExp();
  for (MatrixIndexT i = 0; i < num_rows_; ++i)
    for (MatrixIndexT j = 0; j < num_cols_; ++j)
      (*this)(i, j) = exp((*this)(i, j) - lse);
  return lse;
}

// instantiate.
template float MatrixBase<float>::ApplySoftMax();
template double MatrixBase<double>::ApplySoftMax();

} // namespace kaldi

