// matrix/matrix-common.h

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_MATRIX_MATRIX_COMMON_H_
#define KALDI_MATRIX_MATRIX_COMMON_H_

// This file contains some #includes, forward declarations
// and typedefs that are needed by all the main header
// files in this directory.

#include "base/kaldi-common.h"
#include "cblasext/kaldi-blas.h"
#include "cblasext/cblas-wrappers.h"

namespace kaldi {


// Define Kaldi's MatrixTransposeType (which is basically equivalent to enum
// CBLAS_TRANSPOSE) in case we're including this in a context where it was not
// already defined.  This is part of a kludge to be able to use this enum while
// not including the cblas headers in our headers; cblas headers can cause
// problems because they can bring in a lot of junk (types in the global
// namespace; preprocessor macros), and there are different flavors of cblas
// which might put different *kinds* of junk there.
typedef enum {
  kTrans    = 112, // = CblasTrans
  kNoTrans  = 111  // = CblasNoTrans
} MatrixTransposeType;

typedef enum {
  kSetZero,
  kUndefined,
  kCopyData
} MatrixResizeType;


typedef enum {
  kDefaultStride,
  kStrideEqualNumCols,
} MatrixStrideType;

typedef enum {
  kTakeLower,
  kTakeUpper,
  kTakeMean,
  kTakeMeanAndCheck
} SpCopyType;

template<typename Real> class VectorBase;
template<typename Real> class Vector;
template<typename Real> class SubVector;
template<typename Real> class MatrixBase;
template<typename Real> class SubMatrix;
template<typename Real> class Matrix;
template<typename Real> class SpMatrix;
template<typename Real> class TpMatrix;
template<typename Real> class PackedMatrix;
template<typename Real> class SparseMatrix;

// these are classes that won't be defined in this
// directory; they're mostly needed for friend declarations.
template<typename Real> class CuMatrixBase;
template<typename Real> class CuSubMatrix;
template<typename Real> class CuMatrix;
template<typename Real> class CuVectorBase;
template<typename Real> class CuSubVector;
template<typename Real> class CuVector;
template<typename Real> class CuPackedMatrix;
template<typename Real> class CuSpMatrix;
template<typename Real> class CuTpMatrix;
template<typename Real> class CuSparseMatrix;

class CompressedMatrix;
class GeneralMatrix;

/// This class provides a way for switching between double and float types.
template<typename T> class OtherReal { };  // useful in reading+writing routines
                                           // to switch double and float.
/// A specialized class for switching from float to double.
template<> class OtherReal<float> {
 public:
  typedef double Real;
};
/// A specialized class for switching from double to float.
template<> class OtherReal<double> {
 public:
  typedef float Real;
};


// BLAS's interface has 'int' which on even many 64 bit systems is
// 32 bits, so using 64 bits for the matrix index would be like making
// a promise we can't keep.
typedef int32 MatrixIndexT;
typedef uint32 UnsignedMatrixIndexT;


}



#endif  // KALDI_MATRIX_MATRIX_COMMON_H_
