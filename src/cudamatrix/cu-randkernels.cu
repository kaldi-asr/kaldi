// cudamatrix/cu-randkernels.cu

// Copyright 2012  Karel Vesely
//           2013 Johns Hopkins University (author: Daniel Povey)

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



/*
 * Hybrid Tauss/LCG random number generator
 *
 * Based on : http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
 * 
 * z1,z2,z3,z4 are matrices of IntType
 * (inner state of grid-like random number generator)
 *
 * ie. each matrix elemnent has its own random number sequence, 
 * which is given by z1,z2,z3,z4
 *
 * In this file is the CUDA code of the CUDA kernels, plus the ANSI-C wrappers
 *
 */

#include "cudamatrix/cu-randkernels-ansi.h"



/***********************************************************************
 * CUDA kernels
 * some functions are templated to have the float/double operations
 */

// S1, S2, S3, and M are all constants, z is the inner state  
__device__
static uint32_cuda TausStep(uint32_cuda &z, int32_cuda S1, int32_cuda S2, int32_cuda S3, uint32_cuda M) {  
  uint32_cuda b=(((z << S1) ^ z) >> S2);  
  return z = (((z & M) << S3) ^ b);  
}  

// A and C are constants  
__device__
static uint32_cuda LCGStep(uint32_cuda &z, uint32_cuda A, uint32_cuda C) {  
  return z=(A*z+C);  
} 

template<typename Real>
__device__
static Real HybridTaus(uint32_cuda& z1, uint32_cuda& z2, uint32_cuda& z3, uint32_cuda& z4) {  
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  Real randval;
  do { 
   randval = 2.3283064365387e-10 * (          // Periods  
    TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1  
    TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1  
    TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1  
    LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32  
   );
  } while (!(randval > 0.0 && randval < 1.0));
  return randval;
}  

template<typename Real>
__global__
static void _rand(Real* mat, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) {
    mat[index] = HybridTaus<Real>(z1[index],z2[index],z3[index],z4[index]);
  }
}


 
template<typename Real>
__device__
static Real BoxMuller(uint32_cuda& z1, uint32_cuda& z2, uint32_cuda& z3, uint32_cuda& z4) {
  const Real M_2PI = 6.283185307179586476925286766558;
  Real u0 = HybridTaus<Real>(z1,z2,z3,z4), u1 = HybridTaus<Real>(z1,z2,z3,z4);
  Real r = sqrt(-2.0 * log(u0));
  Real theta = M_2PI * u1;
  return r*sin(theta);
}  



template<typename Real>
__global__
static void _gauss_rand(Real* mat, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) {
    mat[index] = BoxMuller<Real>(z1[index],z2[index],z3[index],z4[index]);
  }
}



template<typename Real>
__global__
static void _vec_gauss_rand(Real* v, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.y > 0)
     return;

  if ( i < dim ) {
    v[i] = BoxMuller<Real>(z1[i],z2[i],z3[i],z4[i]);
  }
}



template<typename Real>
__global__
static void _binarize_probs(Real* states, const Real* probs, const Real* rand, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) {
    states[index] = ((probs[index] > rand[index])? 1.0 : 0.0);
  }
}



/***********************************************************************
 * ANSI-C wrappers of CUDA kernels
 */

/*
 * float 
 */
void cudaF_rand(dim3 Gr, dim3 Bl, float* mat, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, MatrixDim d) { 
  _rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); 
}

void cudaF_gauss_rand(dim3 Gr, dim3 Bl, float* mat, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, MatrixDim d) { 
  _gauss_rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); 
}

void cudaF_vec_gauss_rand(int Gr, int Bl, float* v, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, int dim) {
  _vec_gauss_rand<<<Gr,Bl>>>(v,z1,z2,z3,z4,dim);
}

void cudaF_binarize_probs(dim3 Gr, dim3 Bl, float* states, const float* probs, float* rand, MatrixDim d) { 
  _binarize_probs<<<Gr,Bl>>>(states,probs,rand,d); 
}



/*
 * double 
 */
void cudaD_rand(dim3 Gr, dim3 Bl, double* mat, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, MatrixDim d) { 
  _rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); 
}

void cudaD_gauss_rand(dim3 Gr, dim3 Bl, double* mat, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, MatrixDim d) { 
  _gauss_rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); 
}

void cudaD_vec_gauss_rand(int Gr, int Bl, double* v, uint32_cuda* z1, uint32_cuda* z2, uint32_cuda* z3, uint32_cuda* z4, int dim) {
  _vec_gauss_rand<<<Gr,Bl>>>(v,z1,z2,z3,z4,dim);
}

void cudaD_binarize_probs(dim3 Gr, dim3 Bl, double* states, const double* probs, double* rand, MatrixDim d) { 
  _binarize_probs<<<Gr,Bl>>>(states,probs,rand,d); 
}



