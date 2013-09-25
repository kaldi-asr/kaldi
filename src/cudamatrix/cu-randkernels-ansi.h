// cudamatrix/cu-randkernels-ansi.h

// Copyright 2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CU_RANDKERNELS_ANSI_H_
#define KALDI_CUDAMATRIX_CU_RANDKERNELS_ANSI_H_

#include "cudamatrix/cu-matrixdim.h"
#include "cudamatrix/cu-kernels-ansi.h"

#if HAVE_CUDA == 1

extern "C" {

/*********************************************************
 * float CUDA kernel calls
 */
void cudaF_rand(dim3 Gr, dim3 Bl, float *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d);
void cudaF_gauss_rand(dim3 Gr, dim3 Bl, float *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d);
void cudaF_vec_gauss_rand(int Gr, int Bl, float *v, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, int dim);
void cudaF_binarize_probs(dim3 Gr, dim3 Bl, float *states, const float *probs, float *rand, MatrixDim d);

/*********************************************************
 * double CUDA kernel calls
 */
void cudaD_rand(dim3 Gr, dim3 Bl, double *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d);
void cudaD_gauss_rand(dim3 Gr, dim3 Bl, double *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d);
void cudaD_vec_gauss_rand(int Gr, int Bl, double *v, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, int dim);
void cudaD_binarize_probs(dim3 Gr, dim3 Bl, double *states, const double *probs, double *rand, MatrixDim d);

}



#endif // HAVE_CUDA

#endif
