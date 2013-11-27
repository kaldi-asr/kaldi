// cudamatrix/cu-choleskykernel.h

// Copyright 2010-2013  Dr. Stephan Kramer
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

#ifndef KALDI_CUDAMATRIX_CU_CHOLESKYKERNELS_H_
#define KALDI_CUDAMATRIX_CU_CHOLESKYKERNELS_H_

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "cudamatrix/cu-choleskykernels-ansi.h"

/*
 * In this file are C++ templated wrappers
 * of the ANSI-C CUDA kernels
 */

namespace kaldi {

/*********************************************************
* base templates
*/
template<typename Real> inline void cuda_factorize_diagonal_block(Real* A, int block_offset, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_strip_update(Real* A, int block_offset, int n_remaining_blocks, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_diag_update(Real* A, int block_offset, int n_remaining_blocks, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_lo_update(Real* A, int block_offset, int n_blocks, int n_remaining_blocks, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
/*********************************************************
* float specialization
*/
template<> inline void cuda_factorize_diagonal_block<float>(float* A, int block_offset, MatrixDim d) { cudaF_factorize_diagonal_block(A,block_offset,d); }
template<> inline void cuda_strip_update<float>(float* A, int block_offset, int n_remaining_blocks, MatrixDim d) { cudaF_strip_update(A,block_offset,n_remaining_blocks,d); }
template<> inline void cuda_diag_update<float>(float* A, int block_offset, int n_remaining_blocks, MatrixDim d) { cudaF_diag_update(A,block_offset,n_remaining_blocks,d); }
template<> inline void cuda_lo_update<float>(float* A, int block_offset, int n_blocks, int n_remaining_blocks, MatrixDim d) { cudaF_lo_update(A,block_offset,n_blocks,n_remaining_blocks,d); }
/*********************************************************
* double specialization
*/
template<> inline void cuda_factorize_diagonal_block<double>(double* A, int block_offset, MatrixDim d) { cudaD_factorize_diagonal_block(A,block_offset,d); }
template<> inline void cuda_strip_update<double>(double* A, int block_offset, int n_remaining_blocks, MatrixDim d) { cudaD_strip_update(A,block_offset,n_remaining_blocks,d); }
template<> inline void cuda_diag_update<double>(double* A, int block_offset, int n_remaining_blocks, MatrixDim d) { cudaD_diag_update(A,block_offset,n_remaining_blocks,d); }
template<> inline void cuda_lo_update<double>(double* A, int block_offset, int n_blocks, int n_remaining_blocks, MatrixDim d) { cudaD_lo_update(A,block_offset,n_blocks,n_remaining_blocks,d); }

} // namespace

#endif // HAVE_CUDA

#endif
