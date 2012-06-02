// cudamatrix/cu-randkernels.h

// Copyright 2012  Karel Vesely

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




#ifndef KALDI_CUDAMATRIX_CU_RANDKERNELS_H_
#define KALDI_CUDAMATRIX_CU_RANDKERNELS_H_


#include "cudamatrix/cu-kernels.h"

#if HAVE_CUDA==1

extern "C" {
  // **************
  // float
  //
  void cudaF_rand(dim3 Gr, dim3 Bl, float *mat, unsigned *z1, unsigned *z2, unsigned *z3, unsigned *z4, MatrixDim d);
  void cudaF_gauss_rand(dim3 Gr, dim3 Bl, float *mat, unsigned *z1, unsigned *z2, unsigned *z3, unsigned *z4, MatrixDim d);
  void cudaF_binarize_probs(dim3 Gr, dim3 Bl, float *states, const float *probs, float *rand, MatrixDim d);

}

#endif // HAVE_CUDA

#endif
