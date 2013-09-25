// cudamatrix/cu-randkernels.h

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



#ifndef KALDI_CUDAMATRIX_CU_RANDKERNELS_H_
#define KALDI_CUDAMATRIX_CU_RANDKERNELS_H_

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "cudamatrix/cu-randkernels-ansi.h"

/*
 * In this file are C++ templated wrappers 
 * of the ANSI-C CUDA kernels
 */
namespace kaldi {

/*********************************************************
 * base templates
 */
template<typename Real> inline void cuda_rand(dim3 Gr, dim3 Bl, Real *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_gauss_rand(dim3 Gr, dim3 Bl, Real *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_vec_gauss_rand(int Gr, int Bl, Real *v, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_binarize_probs(dim3 Gr, dim3 Bl, Real *states, const Real *probs, Real *rand, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }

/*********************************************************
 * float specializations
 */
template<> inline void cuda_rand<float>(dim3 Gr, dim3 Bl, float *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d) { cudaF_rand(Gr,Bl,mat,z1,z2,z3,z4,d); }
template<> inline void cuda_gauss_rand<float>(dim3 Gr, dim3 Bl, float *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d) { cudaF_gauss_rand(Gr,Bl,mat,z1,z2,z3,z4,d); } 
template<> inline void cuda_vec_gauss_rand<float>(int Gr, int Bl, float *v, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, int dim) { cudaF_vec_gauss_rand(Gr,Bl,v,z1,z2,z3,z4,dim); } 
template<> inline void cuda_binarize_probs<float>(dim3 Gr, dim3 Bl, float *states, const float *probs, float *rand, MatrixDim d) { cudaF_binarize_probs(Gr,Bl,states,probs,rand,d); } 

/*********************************************************
 * double specializations
 */
template<> inline void cuda_rand<double>(dim3 Gr, dim3 Bl, double *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d) { cudaD_rand(Gr,Bl,mat,z1,z2,z3,z4,d); }
template<> inline void cuda_gauss_rand<double>(dim3 Gr, dim3 Bl, double *mat, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, MatrixDim d) { cudaD_gauss_rand(Gr,Bl,mat,z1,z2,z3,z4,d); } 
template<> inline void cuda_vec_gauss_rand<double>(int Gr, int Bl, double *v, uint32_cuda *z1, uint32_cuda *z2, uint32_cuda *z3, uint32_cuda *z4, int dim) { cudaD_vec_gauss_rand(Gr,Bl,v,z1,z2,z3,z4,dim); } 
template<> inline void cuda_binarize_probs<double>(dim3 Gr, dim3 Bl, double *states, const double *probs, double *rand, MatrixDim d) { cudaD_binarize_probs(Gr,Bl,states,probs,rand,d); } 

} // namespace

#endif // HAVE_CUDA

#endif


