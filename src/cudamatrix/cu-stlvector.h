// cudamatrix/cu-stlvector.h

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



#ifndef KALDI_CUDAMATRIX_CUSTLVECTOR_H_
#define KALDI_CUDAMATRIX_CUSTLVECTOR_H_

#include "matrix/kaldi-vector.h"

namespace kaldi {

template<typename IntType> class CuMatrix;

/**
 * std::vector equivalent for CUDA computing
 */
template<typename IntType>
class CuStlVector {
 typedef CuStlVector<IntType> ThisType;

 public:

  /// Default Constructor
  CuStlVector<IntType>()
   : dim_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuStlVector<IntType>(MatrixIndexT dim)
   : dim_(0), data_(NULL) { 
    Resize(dim); 
  }

  /// Destructor
  ~CuStlVector() {
    Destroy(); 
  }

  /// Dimensions
  MatrixIndexT Dim() const { 
    return dim_; 
  }

  /// Get raw pointer
  const IntType* Data() const;
  IntType* Data();
 
  /// Allocate the memory
  ThisType& Resize(MatrixIndexT dim);

  /// Deallocate the memory
  void Destroy();

  /// Copy functions (reallocates when needed)
  ThisType&        CopyFromVec(const std::vector<IntType> &src);
  void             CopyToVec(std::vector<IntType> *dst) const;
  
  /// Math operations
  void SetZero();
  void Set(IntType value);

  /// Accessor to non-GPU vector
  const std::vector<IntType>& Vec() const {
    return vec_;
  }
  std::vector<IntType>& Vec() {
    return vec_;
  }

private:
  MatrixIndexT dim_;     ///< dimension of the vector
  IntType *data_;  ///< GPU data pointer
  std::vector<IntType> vec_; ///< non-GPU vector as back-up
};



/*
 * Signatures of general/specialized methods
 */
template<typename Real> void CuStlVector<Real>::Set(Real value) { KALDI_ERR << __func__ << " Not implemented"; }
template<> inline void CuStlVector<int32>::Set(int32 value);


/// I/O
template<typename IntType>
std::ostream &operator << (std::ostream &out, const CuStlVector<IntType> &vec);
 
} // namespace


#include "cu-stlvector-inl.h"

#endif

