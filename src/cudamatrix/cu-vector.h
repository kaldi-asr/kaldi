// cudamatrix/cu-vector.h

// Copyright 2009-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CUVECTOR_H_
#define KALDI_CUDAMATRIX_CUVECTOR_H_

#include "matrix/kaldi-vector.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

template<typename Real> class CuMatrixBase;

/**
 * Vector for CUDA computing
 */
template<typename Real>
class CuVectorBase {
 public:
  friend class CuMatrixBase<Real>;
  friend void cu::Splice<Real>(const CuMatrix<Real> &src,
                               const CuStlVector<int32> &frame_offsets,
                               CuMatrix<Real> *tgt);
  
  
  /// Dimensions
  MatrixIndexT Dim() const { return dim_;  }   

  /// Copy functions; these will crash if the dimension
  /// do not match.  The operator = in class CuVector will
  /// also change the sizes for you.
  void CopyFromVec(const CuVectorBase<Real> &src);
  void CopyFromVec(const VectorBase<Real> &src);
  void CopyToVec(VectorBase<Real> *dst) const;

  /// Math operations
  void SetZero();
  void Set(Real value);
  void Add(Real value);
  void Scale(Real value);
  void AddVec(Real alpha, const CuVectorBase<Real> &vec, Real beta = 1.0);

  /// Sum the rows of the matrix, add to vector
  void AddRowSumMat(Real alpha, const CuMatrixBase<Real> &mat, Real beta = 1.0);
  /// Sum the columns of the matrix, add to vector
  void AddColSumMat(Real alpha, const CuMatrixBase<Real> &mat, Real beta = 1.0); 
  void InvertElements(); 


protected:
  // The following two functions should only be called if we did not compile
  // with CUDA or could not get a CUDA card; in that case the contents are
  // interpreted the same as a regular vector.
  inline const VectorBase<Real> &Vec() const {
    return *(reinterpret_cast<const VectorBase<Real>* >(this));
  }
  inline VectorBase<Real> &Vec() {
    return *(reinterpret_cast<VectorBase<Real>* >(this));
  }
  
  /// Default constructor: make it private so the user cannot
  /// instantiate this class.
  CuVectorBase<Real>(): data_(NULL), dim_(0) { }
  
  Real *data_; ///< GPU data pointer (or regular data pointer
               ///< if CUDA is not compiled in or we have no GPU).
  MatrixIndexT dim_; ///< dimension of the vector
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CuVectorBase);
};

template<class Real>
class CuVector: public CuVectorBase<Real> {
 public:
  CuVector() { }
  CuVector(MatrixIndexT dim, MatrixResizeType t = kSetZero) { Resize(dim, t); }
  CuVector(const CuVectorBase<Real> &v);
  CuVector(const VectorBase<Real> &v);  

  /// Allocate the memory
  void Resize(MatrixIndexT dim, MatrixResizeType t = kSetZero);
  
  ~CuVector() { Destroy(); }

  CuVector<Real> &operator = (const CuVectorBase<Real> &other) {
    Resize(other.Dim());
    CopyFromVec(other);
  }
  CuVector<Real> &operator = (const VectorBase<Real> &other) {
    Resize(other.Dim());
    CopyFromVec(other);
  }
      

  /// I/O 
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &is, bool binary) const;

  void Swap(Vector<Real> *vec);
 private:
  void Destroy();
};

// We'll fill out the following class if it's needed.
template<class Real>
class CuSubVector: public CuVectorBase<Real> {
 public:
 private:
};



/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVectorBase<Real> &vec);
 
  
} // namespace


#include "cu-vector-inl.h"

#endif
