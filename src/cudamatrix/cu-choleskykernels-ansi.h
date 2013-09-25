// cudamatrix/cu-choleskykernel-ansi.h

// Copyright 2010-2013Dr. Stephan Kramer
//  Institut für Numerische und Angewandte Mathematik

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

#ifndef KALDI_CUDAMATRIX_CU_CHOLESKYKERNELS_ANSI_H_
#define KALDI_CUDAMATRIX_CU_CHOLESKYKERNELS_ANSI_H_

#include <stdlib.h>
#include <stdio.h>

#include "cudamatrix/cu-matrixdim.h"

#if HAVE_CUDA == 1

extern "C" {

/*********************************************************
 * float CUDA kernel calls
 */
void cudaF_factorize_diagonal_block(float* A, int block_offset, MatrixDim d);
void cudaF_strip_update(float* A, int block_offset, int n_remaining_blocks, MatrixDim d);
void cudaF_diag_update(float* A, int block_offset, int n_remaining_blocks, MatrixDim d);
void cudaF_lo_update(float* A, int block_offset, int n_blocks, int n_remaining_blocks, MatrixDim d);


/*********************************************************
 * double CUDA kernel calls
 */
void cudaD_factorize_diagonal_block(double* A, int block_offset, MatrixDim d);
void cudaD_strip_update(double* A, int block_offset, int n_remaining_blocks, MatrixDim d);
void cudaD_diag_update(double* A, int block_offset, int n_remaining_blocks, MatrixDim d);
void cudaD_lo_update(double* A, int block_offset, int n_blocks, int n_remaining_blocks, MatrixDim d);
}

#endif // HAVE_CUDA

#endif
