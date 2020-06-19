// cudamatrix/cu-array.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2017  Shiyin Kang

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

template <typename T> class CuArray;
template <typename T> class CuSubArray;


/**
   Class CuArrayBase, CuSubArray and CuArray are analogues of classes
   CuVectorBase, CuSubVector and CuVector, except that they are intended to
   store things other than float/double: they are intended to store integers or
   small structs.  Their CPU-based equivalents are std::vector, and we provide
   ways to copy to/from a std::vector of the same type.
*/
template<typename T>
class CuArrayBase {
  friend class CuArray<T>;
  friend class CuSubArray<T>;
 public:
  /// Return the vector dimension
  MatrixIndexT Dim() const { return dim_;  }

  /// Get raw pointer
  const T* Data() const { return data_; }

  T* Data() { return data_; }

  /// Sets the memory for the object to zero, via memset.  You should verify
  /// that this makes sense for type T.
  void SetZero();

  /// The caller is responsible to ensure dim is equal between *this and src.
  /// Note: copying to GPU is done via memcpy,
  /// and any constructors or assignment operators are not called.
  void CopyFromArray(const CuArrayBase<T> &src);

  /// The caller is responsible to ensure dim is equal between *this and src.
  /// Note: copying to GPU is done via memcpy,
  /// and any constructors or assignment operators are not called.
  void CopyFromVec(const std::vector<T> &src);

  /// This function resizes *dst if needed.  On resize of "dst", the STL vector
  /// may call copy-constructors, initializers, and assignment operators for
  /// existing objects (which will be overwritten), but the copy from GPU to CPU
  /// is done via memcpy.  So be very careful calling this function if your
  /// objects are more than plain structs.
  void CopyToVec(std::vector<T> *dst) const;

  /// Version of the above function that copies contents to a host array
  /// (i.e. to regular memory, not GPU memory, assuming we're using a GPU).
  /// This function requires *dst to be allocated before calling. The allocated
  /// size should be dim_ * sizeof(T)
  void CopyToHost(T *dst) const;


  /// Set to a constant value.  Note: any copying is done as if using memcpy, and
  /// assignment operators or destructors are not called.  This is NOT IMPLEMENTED
  /// YET except for T == int32 (the current implementation will just crash).
  void Set(const T &value);

  /// Fill with the sequence [base ... base + Dim())
  /// This is not implemented except for T=int32
  void Sequence(const T base);

  /// Add a constant value. This is NOT IMPLEMENTED YET except for T == int32
  /// (the current implementation will just crash).
  void Add(const T &value);

  /// Get minimum value (for now implemented on CPU, reimplement if slow).
  /// Asserts the vector is non-empty, otherwise crashes.
  T Min() const;

  /// Get minimum value (for now implemented on CPU, reimplement if slow).
  /// Asserts the vector is non-empty, otherwise crashes.
  T Max() const;

 protected:
  /// Default constructor: make it protected so the user cannot
  /// instantiate this class.
  CuArrayBase<T>(): data_(NULL), dim_(0) { }


  T *data_;  ///< GPU data pointer (if GPU not available,
             ///< will point to CPU memory).
  MatrixIndexT dim_;     ///< dimension of the vector

};

/**
   Class CuArray represents a vector of an integer or struct of type T.  If we
   are using a GPU then the memory is on the GPU, otherwise it's on the CPU.
   This class owns the data that it contains from a memory allocation
   perspective; see also CuSubArrary which does not own the data it contains.
 */
template<typename T>
class CuArray: public CuArrayBase<T> {
 public:

  /// Default constructor, initialized data_ to NULL and dim_ to 0 via
  /// constructor of CuArrayBase.
  CuArray<T>() { }

  /// Constructor with memory initialisation.  resize_type may be kSetZero or
  /// kUndefined.
  explicit CuArray<T>(MatrixIndexT dim, MatrixResizeType resize_type = kSetZero)
     { Resize(dim, resize_type); }

  /// Constructor from CPU-based int vector
  explicit CuArray<T>(const std::vector<T> &src) { CopyFromVec(src); }

  /// Copy constructor.  We don't make this explicit because we want to be able
  /// to create a std::vector<CuArray>.
  CuArray<T>(const CuArray<T> &src) { CopyFromArray(src); }

  /// Destructor
  ~CuArray() { Destroy(); }

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
  void CopyFromArray(const CuArrayBase<T> &src);

  CuArray<T> &operator= (const CuArray<T> &in) {
    this->CopyFromArray(in); return *this;
  }

  CuArray<T> &operator= (const std::vector<T> &in) {
    this->CopyFromVec(in); return *this;
  }

  /// Shallow swap with another CuArray<T>.
  void Swap(CuArray<T> *other);

  /// I/O
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &is, bool binary) const;

};


template<typename T>
class CuSubArray: public CuArrayBase<T> {
 public:
  /// Constructor as a range of an existing CuArray or CuSubArray.  Note: like
  /// similar constructors in class CuVector and others, it can be used to evade
  /// 'const' constraints; don't do that.
  explicit CuSubArray<T>(const CuArrayBase<T> &src,
                         MatrixIndexT offset, MatrixIndexT dim);

  /// Construct from raw pointers
  CuSubArray(const T* data, MatrixIndexT length) {
    // Yes, we're evading C's restrictions on const here, and yes, it can be used
    // to do wrong stuff; unfortunately the workaround would be very difficult.
    this->data_ = const_cast<T*>(data);
    this->dim_ = length;
  }
};



/// I/O
template<typename T>
std::ostream &operator << (std::ostream &out, const CuArray<T> &vec);

} // namespace

#include "cudamatrix/cu-array-inl.h"

#endif
