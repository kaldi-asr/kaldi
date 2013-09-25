// cudamatrix/cu-choleskykernel.cu

// Copyright 2010-2013  Dr. Stephan Kramer
//  Institut fur Numerische und Angewandte Mathematik

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

#include "cudamatrix/cu-choleskykernels-ansi.h"
#include <stdio.h>


#define TILE_SIZE 16

/***********************************************************************
 * CUDA kernels
 * some functions are templated to have the float/double operations
 */
__device__ int lex_index_2D (int r, int c, int row_length) {
  return c +  r*row_length;
}


__device__ int global_pos(int t_pos, int block_offset) {
  return t_pos + TILE_SIZE*block_offset;
}


__device__ float inv_sqrt(float x) {
  return rsqrtf(x);
}


__device__ double inv_sqrt(double x) {
  return rsqrt(x);
}


template<typename T>
__global__
void __factorize_diagonal_block(T* A, int block_offset, MatrixDim d) {
  int global_row_length = d.stride;

  int col = threadIdx.x;
  int row = threadIdx.y;

  int global_row = global_pos(row,block_offset);
  int global_col = global_pos(col,block_offset);

  if ((global_row >= d.cols) || (global_col >= d.cols))
    return;

  int k_max = TILE_SIZE;
  if (d.cols - global_pos(0,block_offset) < TILE_SIZE)
    k_max = d.cols % TILE_SIZE;


  int idx = lex_index_2D(global_row, global_col, global_row_length);
  
  __shared__ T L[TILE_SIZE][TILE_SIZE+1];

  L[row][col] = 0;
  L[row][col] = A[idx];
  __syncthreads();

  if ((row >= k_max) || (col >= k_max))
    return;


  T fac;

  for (int k = 0; k < k_max; k++) {
    __syncthreads();
    fac = inv_sqrt(L[k][k]);
    __syncthreads();

    if ((row==k)&&(col>=k))
      L[col][row] = (L[col][row])*fac;

    __syncthreads();

    if ((row>=col)&&(col>k))
      L[row][col] = L[row][col] - L[col][k]*L[row][k];
  }
  __syncthreads();

    
  if (row >= col) {
    A[idx] = L[row][col];
    if (A[idx] > 100000)
      A[idx] = 1;
  }
}


template<typename T>
__global__
void __strip_update(T* A, int block_offset, MatrixDim d) {
  int global_row_length = d.stride;

  int boffy = block_offset;
  int boffx = blockIdx.x + boffy + 1;
  
  int col = threadIdx.x;
  int row = threadIdx.y;

  __shared__ T topleft[TILE_SIZE][TILE_SIZE+1];
  __shared__ T workingmat[TILE_SIZE][TILE_SIZE+1];

  int global_row = global_pos(row,block_offset);
  int global_col = global_pos(col,block_offset);

  if ((global_row >= d.cols) || (global_col >= d.cols))
    return;

  int idx = lex_index_2D(global_row, global_col, global_row_length);

  topleft[row][col] = 0;  
  topleft[row][col] = A[idx];
  //__syncthreads();
  
  global_row = global_pos(row,boffx);
  
  if (global_row >= d.cols)
    return;

  int idx_w = lex_index_2D(global_row, global_col, global_row_length);
  //int row2 = row + block_offset * TILE_SIZE;
  //int idx_w = row2 + col*global_row_length;
  workingmat[col][row]=0;
  workingmat[col][row]=A[idx_w];

  __syncthreads();
  
  if (row==0) {
    for (int k = 0; k < TILE_SIZE; k++) {
      T sum=0.0;
      for (int m = 0; m < k; m++) 
        sum = sum + topleft[k][m]*workingmat[m][col];
	
      workingmat[k][col] = (workingmat[k][col] - sum) / topleft[k][k];
    }
  }

  __syncthreads();

  A[idx_w] = workingmat[col][row];
  if (A[idx_w] > 100000)
    A[idx_w] = 1;
  //A[idx_w] = 1;
}


template<typename T>
__global__
void __diag_update(T* A, int block_offset, MatrixDim d) {
  int global_row_length = d.stride;
  int boffx = blockIdx.x + block_offset + 1;

  int col = threadIdx.x;
  int row = threadIdx.y;

  int global_row = global_pos(row,boffx);
  int global_col = global_pos(col,block_offset);

  if ((global_row >= d.cols) || (global_col >= d.cols))
    return;

  int idx = lex_index_2D(global_row, global_col, global_row_length);

  __shared__ T left[TILE_SIZE][TILE_SIZE+1];
  
  left[row][col] = 0;
  left[row][col] = A[idx];
  
  __syncthreads();

  T sum = 0.0;


  if (row >= col) {
    for (int kk = 0; kk < TILE_SIZE; kk++)
      sum = sum + left[row][kk]*left[col][kk];
    
    //__syncthreads();
  
    global_col = global_pos(col, boffx);
 
    if (global_col >= d.cols)
      return;

    idx = lex_index_2D(global_row, global_col, global_row_length);

    A[idx] = A[idx] - sum;
 
  }
}


template<typename T>
__global__
void __lo_update(T* A, int block_offset, int n_blocks, MatrixDim d) {
  int global_row_length = d.stride;
  int col = threadIdx.x;
  int row = threadIdx.y;
  
  int boffy = blockIdx.y + block_offset + 1;
  //int boffx = boffy + 1;
  int boffx = boffy + 1;

  __shared__ T left[TILE_SIZE][TILE_SIZE];

  __shared__ T upt[TILE_SIZE][TILE_SIZE + 1];
  
  int global_row = global_pos(row,boffy);
  int global_col_src = global_pos(col,block_offset);

  if ((global_row >= d.cols) || (global_col_src >= d.cols))
    return;

  int idx = lex_index_2D(global_row, global_col_src, global_row_length);
  
  upt[row][col] = 0;
  upt[row][col] = A[idx];
  __syncthreads();

  for (; boffx < n_blocks; boffx++) {
    global_row = global_pos(row,boffx);

    if (global_row >= d.cols) 
      return;

    idx = lex_index_2D(global_row, global_col_src, global_row_length);
    
    left[row][col] = 0;    
    left[row][col] = A[idx];
    
    __syncthreads();

    if (global_row >= d.cols)
      return;

    T matrixprod = 0.0;
    
    for (int kk = 0; kk < TILE_SIZE; kk++)
      matrixprod += left[row][kk]*upt[col][kk];

    __syncthreads();

    int global_col = global_pos(col,boffy);
    if (global_col >= d.cols)
      return;
        
    idx = lex_index_2D(global_row, global_col, global_row_length);
    A[idx] = A[idx] - matrixprod;
  }
}

/***********************************************************************
 * ANSI-C wrappers of CUDA kernels
 */

/*
 * float
 */

void cudaF_factorize_diagonal_block(float* A, int block_offset, MatrixDim d) {
  dim3 threads(TILE_SIZE,TILE_SIZE);
  __factorize_diagonal_block<<<1,threads>>>(A,block_offset,d);
  cudaThreadSynchronize();
}

void cudaF_strip_update(float* A, int block_offset, int n_remaining_blocks, MatrixDim d) {
  dim3 threads(TILE_SIZE,TILE_SIZE);
  if (n_remaining_blocks >= 2) {
    dim3 stripgrid(n_remaining_blocks-1);
    __strip_update<<<stripgrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();    
  } else {
    int stripgrid = 1;
    __strip_update<<<stripgrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();    
  }
}

void cudaF_diag_update(float* A, int block_offset, int n_remaining_blocks, MatrixDim d) {
  dim3 threads(TILE_SIZE,TILE_SIZE);
  if (n_remaining_blocks >= 2) {
    dim3 diaggrid(n_remaining_blocks-1);
    __diag_update<<<diaggrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();
  } else {
    int diaggrid = 1;
    __diag_update<<<diaggrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();
  }
}

void cudaF_lo_update(float* A, int block_offset, int n_blocks, int n_remaining_blocks, MatrixDim d) {
  dim3 logrid;
  logrid.x = 1;
  logrid.y = n_remaining_blocks-2;
  dim3 threads(TILE_SIZE,TILE_SIZE);
  __lo_update<<<logrid,threads>>>(A,block_offset,n_blocks,d);
  cudaThreadSynchronize();
}
/*
 * double
 */
void cudaD_factorize_diagonal_block(double* A, int block_offset, MatrixDim d) {
  dim3 threads(TILE_SIZE,TILE_SIZE);
  __factorize_diagonal_block<<<1,threads>>>(A,block_offset,d);
  cudaThreadSynchronize();
}

void cudaD_strip_update(double* A, int block_offset, int n_remaining_blocks, MatrixDim d) {
  dim3 threads(TILE_SIZE,TILE_SIZE);
  if (n_remaining_blocks >= 2) {
    dim3 stripgrid(n_remaining_blocks-1);
    __strip_update<<<stripgrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();    
  } else {
    int stripgrid = 1;
    __strip_update<<<stripgrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();    
  }
}

void cudaD_diag_update(double* A, int block_offset, int n_remaining_blocks, MatrixDim d) {
  dim3 threads(TILE_SIZE,TILE_SIZE);
  if (n_remaining_blocks >= 2) {
    dim3 diaggrid(n_remaining_blocks-1);
    __diag_update<<<diaggrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();
  } else {
    int diaggrid = 1;
    __diag_update<<<diaggrid,threads>>>(A,block_offset,d);
    cudaThreadSynchronize();
  }
}

void cudaD_lo_update(double* A, int block_offset, int n_blocks, int n_remaining_blocks, MatrixDim d) {
  dim3 logrid;
  logrid.x = 1;
  logrid.y = n_remaining_blocks-2;
  dim3 threads(TILE_SIZE,TILE_SIZE);
  __lo_update<<<logrid,threads>>>(A,block_offset,n_blocks,d);
  cudaThreadSynchronize();
}
