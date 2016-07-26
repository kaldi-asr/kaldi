// cudamatrix/cu-array.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_ARRAY_H_
#define KALDI_CUDAMATRIX_CU_ARRAY_H_

#include "matrix/kaldi-vector.h"

namespace kaldi {


/**
 * std::vector equivalent for CUDA computing.  This class is mostly intended as
 * a CUDA-based mirror of a std::vector object that lives on the CPU.  We don't
 * call constructors, initializers, etc., on the GPU.
 */
template<typename T>
class CuArray {
  typedef CuArray<T> ThisType;
 public:

  /// Default Constructor
  CuArray<T>() : dim_(0), data_(NULL) {  }

  /// Constructor with memory initialisation.  resize_type may be kSetZero or
  /// kUndefined.
  explicit CuArray<T>(MatrixIndexT dim, MatrixResizeType resize_type = kSetZero):
    dim_(0), data_(NULL) { Resize(dim, resize_type); }

  /// Constructor from CPU-based int vector
  explicit CuArray<T>(const std::vector<T> &src):
    dim_(0), data_(NULL) { CopyFromVec(src); }

  /// Copy constructor.  We don't make this explicit because we want to be able
  /// to create a std::vector<CuArray>.
  CuArray<T>(const CuArray<T> &src):
   dim_(0), data_(NULL) { CopyFromArray(src); }

  /// Destructor
  ~CuArray() { Destroy(); }

  /// Return the vector dimension
  MatrixIndexT Dim() const { return dim_;  }

  /// Get raw pointer
  const T* Data() const { return data_; }

  T* Data() { return data_; }
 
  /// Allocate the memory.  resize_type may be kSetZero or kUndefined.
  /// kCopyData not yet supported (can be implemented if needed).
  void Resize(MatrixIndexT dim, MatrixResizeType resize_type = kSetZero);
  
  /// Deallocate the memory and set dim_ and data_ to zero.  Does not call any
  /// destructors of the objects stored.
  void Destroy();
  
  /// This function resizes if needed.  Note: copying to GPU is done via memcpy,
  /// and any constructors or assignment operators are not called.
  void CopyFromVec(const std::vector<T> &src);

  /// This function resizes if needed.
  void CopyFromArray(const CuArray<T> &src);

  /// This function resizes *dst if needed.  On resize of "dst", the STL vector
  /// may call copy-constructors, initializers, and assignment operators for
  /// existing objects (which will be overwritten), but the copy from GPU to CPU
  /// is done via memcpy.  So be very careful calling this function if your
  /// objects are more than plain structs.
  void CopyToVec(std::vector<T> *dst) const;

  /// Version of the above function that copies contents to a host array.
  /// This function requires *dst to be allocated before calling. The allocated
  /// size should be dim_ * sizeof(T)
  void CopyToHost(T *dst) const;

  /// Sets the memory for the object to zero, via memset.  You should verify
  /// that this makes sense for type T.
  void SetZero();
  
  /// Set to a constant value.  Note: any copying is done as if using memcpy, and
  /// assignment operators or destructors are not called.  This is NOT IMPLEMENTED
  /// YET except for T == int32 (the current implementation will just crash).
  void Set(const T &value);
  
  /// Add a constant value. This is NOT IMPLEMENTED YET except for T == int32 
  /// (the current implementation will just crash).
  void Add(const T &value);

  /// Get minimum value (for now implemented on CPU, reimplement if slow).
  /// Asserts the vector is non-empty, otherwise crashes.
  T Min() const;

  /// Get minimum value (for now implemented on CPU, reimplement if slow).
  /// Asserts the vector is non-empty, otherwise crashes.
  T Max() const;

  CuArray<T> &operator= (const CuArray<T> &in) {
    this->CopyFromArray(in); return *this;
  }

  CuArray<T> &operator= (const std::vector<T> &in) {
    this->CopyFromVec(in); return *this;
  }

  /// I/O
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &is, bool binary) const;
  
 private:
  MatrixIndexT dim_;     ///< dimension of the vector
  T *data_;  ///< GPU data pointer (if GPU not available,
             ///< will point to CPU memory).
};


/// I/O
template<typename T>
std::ostream &operator << (std::ostream &out, const CuArray<T> &vec);

} // namespace

#include "cudamatrix/cu-array-inl.h"

#endif

