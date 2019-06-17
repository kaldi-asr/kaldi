// matrix/cblas-extensions.h

// Copyright 2012-2019  Johns Hopkins University (author: Daniel Povey);
//                      Haihua Xu; Wei Shi

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
#ifndef KALDI_MATRIX_CBLAS_EXTENSIONS_H_
#define KALDI_MATRIX_CBLAS_EXTENSIONS_H_ 1


#include "cblasext/kaldi-blas.h"
#include "cblasext/cblas-wrappers.h"

// In directories other than this directory, this file is intended to mostly be
// included from .cc files, not from headers, since it includes cblas headers
// (via kaldi-blas.h) and those can be quite polluting.

// This file contains templated wrappers for CBLAS functions, which enable C++
// code calling these functions to be templated.
namespace kaldi {



// This has the same interface as cblas_Xgemv, i.e. it does y = alpha M x + beta y;
// it is just specialized for the case where the vector 'x' has a lot of zeros.
template<typename Real>
void cblasext_Xgemv_sparsevec(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                              KaldiBlasInt num_cols, Real alpha, const Real *Mdata,
                              KaldiBlasInt stride, const Real *xdata,
                              KaldiBlasInt incX, Real beta, Real *ydata,
                              KaldiBlasInt incY);



/// This is not really a wrapper for CBLAS as CBLAS does not have this; in future we could
/// extend this somehow.
inline void mul_elements(
    const KaldiBlasInt dim,
    const double *a,
    double *b) { // does b *= a, elementwise.
  double c1, c2, c3, c4;
  KaldiBlasInt i;
  for (i = 0; i + 4 <= dim; i += 4) {
    c1 = a[i] * b[i];
    c2 = a[i+1] * b[i+1];
    c3 = a[i+2] * b[i+2];
    c4 = a[i+3] * b[i+3];
    b[i] = c1;
    b[i+1] = c2;
    b[i+2] = c3;
    b[i+3] = c4;
  }
  for (; i < dim; i++)
    b[i] *= a[i];
}

inline void mul_elements(
    const KaldiBlasInt dim,
    const float *a,
    float *b) { // does b *= a, elementwise.
  float c1, c2, c3, c4;
  KaldiBlasInt i;
  for (i = 0; i + 4 <= dim; i += 4) {
    c1 = a[i] * b[i];
    c2 = a[i+1] * b[i+1];
    c3 = a[i+2] * b[i+2];
    c4 = a[i+3] * b[i+3];
    b[i] = c1;
    b[i+1] = c2;
    b[i+2] = c3;
    b[i+3] = c4;
  }
  for (; i < dim; i++)
    b[i] *= a[i];
}



}
// namespace kaldi

#endif
