// cudamatrix/cu-randkernels.cc

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



#include "cudamatrix/cu-randkernels.h"



//
//Hybrid Tauss/LCG random number generator
//
//http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html


// S1, S2, S3, and M are all constants, and z is part of the  
// private per-thread generator state.
__device__
static unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)  
{  
  unsigned b=(((z << S1) ^ z) >> S2);  
  return z = (((z & M) << S3) ^ b);  
}  

// A and C are constants  
__device__
static unsigned LCGStep(unsigned &z, unsigned A, unsigned C)  
{  
  return z=(A*z+C);  
} 

template<typename T>
__device__
static T HybridTaus(unsigned& z1, unsigned& z2, unsigned& z3, unsigned& z4)  
{  
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  T randval;
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




template<typename T>
__global__
static void _rand(T* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) {
    mat[index] = HybridTaus<T>(z1[index],z2[index],z3[index],z4[index]);
  }
}

/*
float2 BoxMuller()  
{  
  float u0=HybridTaus (), u1=HybridTaus ();  
  float r=sqrt(-2 log(u0));  
  float theta=2*PI*u1;  
  return make_float2(r*sin(theta),r*cos(theta));  
} 
*/
 
template<typename T>
__device__
static T BoxMuller(unsigned& z1, unsigned& z2, unsigned& z3, unsigned& z4)  
{
  const T M_2PI = 6.283185307179586476925286766558;

  T u0 = HybridTaus<T>(z1,z2,z3,z4), u1 = HybridTaus<T>(z1,z2,z3,z4);
  T r = sqrt(-2.0 * log(u0));
  T theta = M_2PI * u1;
  return r*sin(theta);
  
}  


template<typename T>
__global__
static void _gauss_rand(T* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) {
    mat[index] = BoxMuller<T>(z1[index],z2[index],z3[index],z4[index]);
  }
}


template<typename T>
__global__
static void _binarize_probs(T* states, const T* probs, const T* rand, MatrixDim d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) {
    states[index] = ((probs[index] > rand[index])? 1.0 : 0.0);
  }
}



/************
 * :FLOAT:
 */
void cudaF_rand(dim3 Gr, dim3 Bl, float* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d)
{ _rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); }

void cudaF_gauss_rand(dim3 Gr, dim3 Bl, float* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d)
{ _gauss_rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); }

void cudaF_binarize_probs(dim3 Gr, dim3 Bl, float* states, const float* probs, float* rand, MatrixDim d) 
{ _binarize_probs<<<Gr,Bl>>>(states,probs,rand,d); }


/************
 * :DOUBLE:
 */
void cudaD_rand(dim3 Gr, dim3 Bl, double* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d)
{ _rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); }

void cudaD_gauss_rand(dim3 Gr, dim3 Bl, double* mat, unsigned* z1, unsigned* z2, unsigned* z3, unsigned* z4, MatrixDim d)
{ _gauss_rand<<<Gr,Bl>>>(mat,z1,z2,z3,z4,d); }

void cudaD_binarize_probs(dim3 Gr, dim3 Bl, double* states, const double* probs, double* rand, MatrixDim d) 
{ _binarize_probs<<<Gr,Bl>>>(states,probs,rand,d); }

