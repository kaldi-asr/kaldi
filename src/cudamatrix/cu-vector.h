// cudamatrix/cu-vector.h

// Copyright 2009-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)
//                      Lucas Ondel
//           2013       Xiaohui Zhang
//           2015       Guoguo Chen

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



#ifndef KALDI_CUDAMATRIX_CU_VECTOR_H_
#define KALDI_CUDAMATRIX_CU_VECTOR_H_

#include "matrix/kaldi-vector.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-value.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

template<typename Real> class CuMatrixBase;

template<typename Real>
Real VecVec(const CuVectorBase<Real> &v1, const CuVectorBase<Real> &v2);

template<typename Real, typename OtherReal>
Real VecVec(const CuVectorBase<Real> &v1, const CuVectorBase<OtherReal> &v2);

/**
 * Vector for CUDA computing
 */
template<typename Real>
class CuVectorBase {
 public:
  friend class CuVectorBase<float>;
  friend class CuVectorBase<double>;
  friend class CuMatrixBase<Real>;
  friend class MatrixBase<Real>;
  friend class CuPackedMatrix<Real>;
  friend class CuSpMatrix<Real>;
  friend class CuTpMatrix<Real>;

  template <typename OtherReal>
  friend OtherReal VecVec(const CuVectorBase<OtherReal> &v1,
                          const CuVectorBase<OtherReal> &v2);
  friend void cu::Splice<Real>(const CuMatrixBase<Real> &src,
                               const CuArray<int32> &frame_offsets,
                               CuMatrixBase<Real> *tgt);
  friend class CuRand<Real>;

  /// Dimensions
  MatrixIndexT Dim() const { return dim_;  }

  /// Returns a pointer to the start of the vector's data.
  inline Real* Data() { return data_; }
  /// Returns a pointer to the start of the vector's data (const).
  inline const Real* Data() const { return data_; }

  /// Copy functions; these will crash if the dimension
  /// do not match.  The operator = in class CuVector will
  /// also change the sizes for you.
  void CopyFromVec(const CuVectorBase<Real> &src);

  template<typename OtherReal>
  void CopyFromVec(const CuVectorBase<OtherReal> &M);

  template<typename OtherReal>
  void CopyFromVec(const VectorBase<OtherReal> &src);

  template<typename OtherReal>
  void CopyToVec(VectorBase<OtherReal> *dst) const;

  void CopyRowsFromMat(const CuMatrixBase<Real> &M);

  void CopyRowsFromMat(const MatrixBase<Real> &M);

  /// Math operations
  void SetZero();
  void Set(Real value);
  void Add(Real value);
  void Scale(Real value);

  void AddVec(Real alpha, const CuVectorBase<Real> &vec, Real beta = 1.0);

  template<typename OtherReal>
  void AddVec(Real alpha, const CuVectorBase<OtherReal> &vec, Real beta = 1.0);

  /// Sum the rows of the matrix, add to vector
  void AddRowSumMat(Real alpha, const CuMatrixBase<Real> &mat, Real beta = 1.0);
  /// Sum the columns of the matrix, add to vector
  void AddColSumMat(Real alpha, const CuMatrixBase<Real> &mat, Real beta = 1.0);

  /// Add triangular matrix times vector: this <-- beta*this + alpha*M*v.
  /// Works even if rv == *this.
  void AddTpVec(const Real alpha, const CuTpMatrix<Real>&M,
                const MatrixTransposeType trans, const CuVectorBase<Real> &v,
                const Real beta);  // **beta previously defaulted to 0.0**

  /// Multiplies this vector by lower-triangular marix:  *this <-- *this *M
  void MulTp(const CuTpMatrix<Real> &M, const MatrixTransposeType trans);

  bool ApproxEqual(const CuVectorBase<Real> &other, float tol = 0.01) const;

  void InvertElements();

  void ApplySoftMax();
  void ApplyExp();
  void ApplyLog();
  MatrixIndexT ApplyFloor(Real floor_val);
  MatrixIndexT ApplyCeiling(Real ceiling_val);
  void ApplyPow(Real power);
  Real Sum() const;

  void SetRandn();
  void SetRandUniform();

  CuSubVector<Real> Range(const MatrixIndexT o, const MatrixIndexT l) {
    return CuSubVector<Real>(*this, o, l);
  }

  const CuSubVector<Real> Range(const MatrixIndexT o,
                                const MatrixIndexT l) const {
    return CuSubVector<Real>(*this, o, l);
  }

  void CopyColFromMat(const CuMatrixBase<Real> &mat, MatrixIndexT col);

  template<typename OtherReal>
  void CopyColFromMat(const CuMatrixBase<OtherReal> &mat, MatrixIndexT col);

  void AddMatVec(const Real alpha, const CuMatrixBase<Real> &M,
                 MatrixTransposeType trans, const CuVectorBase<Real> &v,
                 const Real beta);
  void AddVecVec(Real alpha, const CuVectorBase<Real> &v,
                 const CuVectorBase<Real> &r, Real beta);

  void AddSpVec(const Real alpha, const CuSpMatrix<Real> &S,
                const CuVectorBase<Real> &v, const Real beta);

  /// Add the diagonal of a matrix times itself:
  /// *this = diag(M M^T) +  beta * *this (if trans == kNoTrans), or
  /// *this = diag(M^T M) +  beta * *this (if trans == kTrans).
  void AddDiagMat2(Real alpha, const CuMatrixBase<Real> &M,
                   MatrixTransposeType trans, Real beta);

  /// Add the diagonal of a matrix product: *this = diag(M N), assuming the
  /// "trans" arguments are both kNoTrans; for transpose arguments, it behaves
  /// as you would expect.
  void AddDiagMatMat(Real alpha, const CuMatrixBase<Real> &M, MatrixTransposeType transM,
                     const CuMatrixBase<Real> &N, MatrixTransposeType transN,
                     Real beta = 1.0);

  inline CuValue<Real> operator() (MatrixIndexT i) {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                          static_cast<UnsignedMatrixIndexT>(dim_));
    return CuValue<Real>(data_ + i);
  }

  Real Norm(Real p); // Only works for p = 1 and p = 2.

  inline Real operator() (MatrixIndexT i) const {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                          static_cast<UnsignedMatrixIndexT>(dim_));
    return CuValue<Real>(data_ + i); // will be casted to Real.
  }

  /// Extracts the diagonal of a packed matrix M; works for Sp or Tp.
  void CopyDiagFromPacked(const CuPackedMatrix<Real> &M);

  /// Extracts the diagonal of a matrix.
  void CopyDiagFromMat(const CuMatrix<Real> &M);

  /// Returns the maximum value of any element, or -infinity for the empty vector.
  Real Max() const;

  /// Returns the minimum value of any element, or +infinity for the empty vector.
  Real Min() const;

  // Set each element to y = (x == orig ? changed : x).
  void ReplaceValue(Real orig, Real changed);

  void MulElements(const CuVectorBase<Real> &v);
  // The following two functions should only be called if we did not compile
  // with CUDA or could not get a CUDA card; in that case the contents are
  // interpreted the same as a regular vector.
  // Do not use the following functions unless you know what you are doing!
  inline const VectorBase<Real> &Vec() const {
    return *(reinterpret_cast<const VectorBase<Real>* >(this));
  }
  inline VectorBase<Real> &Vec() {
    return *(reinterpret_cast<VectorBase<Real>* >(this));
  }

 protected:

  /// Default constructor: make it protected so the user cannot
  /// instantiate this class.
  CuVectorBase<Real>(): data_(NULL), dim_(0) { }

  Real *data_; ///< GPU data pointer (or regular data pointer
               ///< if CUDA is not compiled in or we have no GPU).
  MatrixIndexT dim_; ///< dimension of the vector

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CuVectorBase);
};

template<typename Real>
class CuVector: public CuVectorBase<Real> {
  friend class CuVectorBase<float>;
  friend class CuVectorBase<double>;
  friend class CuMatrixBase<Real>;
  friend class CuPackedMatrix<Real>;
  friend class CuSpMatrix<Real>;
  friend class CuTpMatrix<Real>;

 public:
  CuVector() { }
  CuVector(MatrixIndexT dim, MatrixResizeType t = kSetZero) { Resize(dim, t); }

  CuVector(const CuVectorBase<Real> &v);

  CuVector(const VectorBase<Real> &v);
  explicit CuVector(const CuVector<Real> &v) : CuVectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

  template<typename OtherReal>
  explicit CuVector(const CuVectorBase<OtherReal> &v) : CuVectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

  template<typename OtherReal>
  explicit CuVector(const VectorBase<OtherReal> &v) : CuVectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(Vector<Real>(v));
  }

  /// Allocate the memory
  void Resize(MatrixIndexT dim, MatrixResizeType t = kSetZero);

  ~CuVector() { Destroy(); }

  CuVector<Real> &operator = (const CuVectorBase<Real> &other) {
    Resize(other.Dim(), kUndefined);
    this->CopyFromVec(other);
    return *this;
  }

  CuVector<Real> &operator = (const CuVector<Real> &other) {
    Resize(other.Dim(), kUndefined);
    this->CopyFromVec(other);
    return *this;
  }
  CuVector<Real> &operator = (const VectorBase<Real> &other) {
    Resize(other.Dim());
    this->CopyFromVec(other);
    return *this;
  }


  /// I/O
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &is, bool binary) const;

  void Swap(Vector<Real> *vec);

 private:
  void Destroy();
};

// We'll fill out the following class if it's needed.
template<typename Real>
class CuSubVector: public CuVectorBase<Real> {
 public:
  CuSubVector(const CuVectorBase<Real> &t, const MatrixIndexT origin,
              const MatrixIndexT length) : CuVectorBase<Real>() {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(origin)+
                 static_cast<UnsignedMatrixIndexT>(length) <=
                 static_cast<UnsignedMatrixIndexT>(t.Dim()));
    CuVectorBase<Real>::data_ = const_cast<Real*>(t.Data()+origin);
    CuVectorBase<Real>::dim_ = length;
  }
  /// Copy constructor
  /// this constructor needed for Range() to work in base class.
  CuSubVector(const CuSubVector &other) : CuVectorBase<Real> () {
    CuVectorBase<Real>::data_ = other.data_;
    CuVectorBase<Real>::dim_ = other.dim_;
  }

  CuSubVector(const Real* data, MatrixIndexT length) : CuVectorBase<Real> () {
    // Yes, we're evading C's restrictions on const here, and yes, it can be used
    // to do wrong stuff; unfortunately the workaround would be very difficult.
    CuVectorBase<Real>::data_ = const_cast<Real*>(data);
    CuVectorBase<Real>::dim_ = length;
  }

  /// This operation does not preserve const-ness, so be careful.
  CuSubVector(const CuMatrixBase<Real> &matrix, MatrixIndexT row) {
    CuVectorBase<Real>::data_ = const_cast<Real*>(matrix.RowData(row));
    CuVectorBase<Real>::dim_ = matrix.NumCols();
  }


};

/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVectorBase<Real> &vec);


template<typename Real>
bool ApproxEqual(const CuVectorBase<Real> &a,
                 const CuVectorBase<Real> &b, Real tol = 0.01) {
  return a.ApproxEqual(b, tol);
}

template<typename Real>
inline void AssertEqual(const CuVectorBase<Real> &a,
                        const CuVectorBase<Real> &b, Real tol = 0.01) {
  KALDI_ASSERT(a.ApproxEqual(b, tol));
}

template<typename Real>
template<typename OtherReal>
void CuVectorBase<Real>::CopyFromVec(const CuVectorBase<OtherReal> &v) {
  v.CopyToVec(&this);
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyFromVec(const CuVectorBase<OtherReal> &cu) {
  cu.CopyToVec(this);
}

// declare template specializations.
template <>
template <>
void CuVectorBase<double>::CopyFromVec<float>(const CuVectorBase<float> &src);

template<>
template <>
void CuVectorBase<float>::CopyFromVec<double>(const CuVectorBase<double> &src);

template<typename Real>
template<typename OtherReal>
Vector<Real>::Vector(const CuVectorBase<OtherReal> &cu) {
  Init(cu.Dim());
  cu.CopyToVec(this);
}

} // namespace

#endif
