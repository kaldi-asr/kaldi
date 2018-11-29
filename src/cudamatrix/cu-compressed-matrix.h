// cudamatrix/cu-compressed-matrix.h

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_COMPRESSED_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_COMPRESSED_MATRIX_H_

#include "cudamatrix/cu-matrix.h"

namespace kaldi {

/**
   Class CuCompressedMatrixBase is an abstract base class that allows you to
   compress a matrix of type CuMatrix<BaseFloat>.  When you instantiate it you
   would choose the child-class type (by allocating the appropriate child-class
   type via 'new').
 */
class CuCompressedMatrixBase {
 public:

  /// Sets *this to an appropriately compressed copy of 'mat', which
  /// includes resizing *this.  The details of how this is done will be
  /// different in different child classes.
  virtual void CopyFromMat(const CuMatrixBase<BaseFloat> &mat) = 0;

  /// Copies the contents of *this to 'mat', which should be
  /// correctly sized beforehand.
  virtual void CopyToMat(CuMatrixBase<BaseFloat> *mat) const = 0;


  // The number of rows in *this.
  virtual int32 NumRows() const = 0;

  // The number of columns in *this.
  virtual int32 NumCols() const = 0;

  virtual ~CuCompressedMatrixBase() { }
};



/**
   Class CuCompressedMatrix, templated on an integer type (expected to be one
   of: int8, uint8, int16, uint16), this provides a way to approximate a
   CuMatrix in a more memory-efficient format.  It's used in nnet3 to
   reduce memory use for large networks.

   It is *not* a CUDA equivalent for class CompressedMatrix (of
   ../matrix/compressed-matrix.h).  Note: this class is only to be used when you
   are using a GPU.  If you didn't compile for CUDA or you are not using a GPU,
   you are not supposed to create an instance of this class, and doing so will
   cause a runtime error.
 */
template <typename I>
class CuCompressedMatrix: public CuCompressedMatrixBase {
 public:

  /// Constructor which sets 'scale_' according to
  /// scale_ = range / std::numeric_limits<I>::max().
  ///
  /// range = 0 (only supported for I == int8) is a special case in which only
  /// the sign of the input is retained; and when we reconstruct, the output
  /// will be -1, 0 or 1.
  ///
  /// truncate (only relevant if range != 0) should be true if it's possible
  /// that the input could exceed the allowed input range, i.e. [0, range] if I
  /// is unsigned, and [-range, range] if I is signed; and it may be false if
  /// you know that the input (the matrix given to CopyFromMat) will have
  /// elements only in the allowed range.  Setting 'truncate' to false
  /// allows the compression code to avoid the bounds check.
  CuCompressedMatrix(BaseFloat range, bool truncate = true);

  virtual void CopyFromMat(const CuMatrixBase<BaseFloat> &mat);

  virtual void CopyToMat(CuMatrixBase<BaseFloat> *mat) const;

  virtual MatrixIndexT NumRows() const { return num_rows_; }

  virtual MatrixIndexT NumCols() const { return num_cols_; }


  virtual ~CuCompressedMatrix() { Destroy(); }

 private:
  // If there was data in 'data_', frees it, and sets it to NULL.
  void Destroy();

  // The raw data.
  I *data_;

  // scale_ affects how the raw data is interpreted as a floating point value.
  // When uncompressing to a CuMatrix, we'll do:
  //  f  = scale_ * i
  // where f is the floating point value we're writing to, and i is the integer
  // value.
  //
  // scale_ = 0 is treated specially; in this case we just take notice of the
  // sign of the input, and when uncompressing we do it with a scale such
  // that the output becomes -1, 0 and 1.
  BaseFloat scale_;

  // 'truncate_' affects the code that compresses data to integer values.
  // If the data we're compressing might possibly be outside of the representable
  // range, then you should set truncate to true (this is the default in the
  // constructor).  This way, values larger than the minimum or maximum will
  // be set to the minimum or maximum value.  If truncate_ is false, it will
  // just wrap around, but the compression code will be slightly faster as
  // it doesn't need to check.
  bool truncate_;

  MatrixIndexT num_rows_;
  MatrixIndexT num_cols_;
  // stride_ is currently always equal to num_cols_; it was added mainly to
  // point the way to possible future extension.
  MatrixIndexT stride_;
};



// This enum value is used to encode the type you want to instantiate
// a CuCompressedMatrix with.  It's used in class NnetComputation
// (cast to int32) as one of the arguments of kCompressMatrix.
enum  CuCompressedMatrixType {
  kCompressedMatrixInt8 = 1,
  kCompressedMatrixUint8 = 2,
  kCompressedMatrixInt16 = 3,
  kCompressedMatrixUint16 = 4
};

/**
   This function allocates a new CuCompressedMatrix with type determined
   by t, and with the 'range' and 'truncate' parameters provided to the
   constructor of class CuCompressedMatrix.

   It will crash at runtime if called when CUDA is not compiled in, or not
   enabled.
 */
CuCompressedMatrixBase *NewCuCompressedMatrix(CuCompressedMatrixType t,
                                              BaseFloat range,
                                              bool truncate = true);


} // namespace kaldi

#endif
