// cudamatrix/cu-vector.h

// Copyright 2009-2012  Karel Vesely

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

namespace kaldi {

template<typename Real> class CuMatrix;


/**
 * Vector for CUDA computing
 */
template<typename Real>
class CuVector {
 typedef CuVector<Real> ThisType;

 public:

  /// Default Constructor
  CuVector<Real>()
   : dim_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuVector<Real>(MatrixIndexT dim)
   : dim_(0), data_(NULL) { 
    Resize(dim); 
  }

  /// Destructor
  ~CuVector() {
    Destroy(); 
  }

  /// Dimensions
  MatrixIndexT Dim() const { 
    return dim_; 
  }

  /// Get raw pointer
  const Real* Data() const;
  Real* Data();
 
  /// Allocate the memory
  ThisType& Resize(MatrixIndexT dim);

  /// Deallocate the memory
  void Destroy();

  /// Copy functions (lazy reallocation when needed)
  ThisType&        CopyFromVec(const CuVector<Real> &src);
  ThisType&        CopyFromVec(const Vector<Real> &src);
  void             CopyToVec(Vector<Real> *dst) const;
  
  /// I/O 
  void             Read(std::istream &is, bool binary);
  void             Write(std::ostream &is, bool binary) const;
  
  /// Math operations
  void SetZero();
  void Set(Real value);
  void AddVec(Real alpha, const CuVector<Real> &vec, Real beta=1.0); 
  /// Sum the rows of the matrix, add to vector
  void AddRowSumMat(Real alpha, const CuMatrix<Real> &mat, Real beta=1.0);
  /// Sum the columns of the matrix, add to vector
  void AddColSumMat(Real alpha, const CuMatrix<Real> &mat, Real beta=1.0); 
  void InvertElements(); 

  /// Accessor to non-GPU vector
  const VectorBase<Real>& Vec() const {
    return vec_;
  }
  VectorBase<Real>& Vec() {
    return vec_;
  }

private:
  MatrixIndexT dim_; ///< dimension of the vector
  Real *data_; ///< GPU data pointer
  Vector<Real> vec_; ///< non-GPU vector as back-up
};


/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVector<Real> &vec);
 
  
} // namespace


#include "cu-vector-inl.h"

#endif
