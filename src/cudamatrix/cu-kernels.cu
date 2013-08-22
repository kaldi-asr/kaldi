// cudamatrix/cu-kernels.cu

// Copyright 2009-2012  Karel Vesely

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



// In this file is the CUDA code of the CUDA kernels, plus the ANSI-C wrappers

#include <cfloat>
#include "cu-kernels-ansi.h"



/***********************************************************************
 * Generic __device__ functions
 */
template<typename Real>
__device__
static Real _sum_reduce(Real buffer[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;	
  __syncthreads();
  // perform tree-based reduction (sum)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      Real temp = 0.0;
      if(threadIdx.x+halfPoint < nTotalThreads) {
        temp = buffer[threadIdx.x + halfPoint];
      }
      buffer[threadIdx.x] += temp;
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return buffer[0];
}


template<typename Real>
__device__
static Real _min_reduce(Real buffer[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;
  __syncthreads();
  // perform tree-based reduction (min)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1); // divide by two
    // only the first half of the threads will be active
    if (threadIdx.x < halfPoint) {
      // Get the shared value stored by another thread
      Real temp = -1e20;
      if (threadIdx.x + halfPoint < nTotalThreads) {
        temp = buffer[threadIdx.x + halfPoint];
      }
      if (temp < buffer[threadIdx.x]) buffer[threadIdx.x] = temp;
    }

    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1); // divide by two
  }
  // the result
  return buffer[0];
}


template<typename Real>
__device__
static Real _max_reduce(Real buffer[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;	
  __syncthreads();
  // perform tree-based reduction (max)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      Real temp = -1e20;
      if(threadIdx.x+halfPoint < nTotalThreads) {
        temp = buffer[threadIdx.x + halfPoint];
      }
      if (temp > buffer[threadIdx.x]) buffer[threadIdx.x] = temp;
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return buffer[0];
}



template<typename Real>
__device__
static int32_cuda _max_id_reduce(Real val[], int32_cuda idx[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;	
  __syncthreads();
  // perform tree-based reduction (get index of maximum)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      Real temp = -1e20;
      if(threadIdx.x+halfPoint < nTotalThreads) {
        temp = val[idx[threadIdx.x + halfPoint]];
      }
      if (temp > val[idx[threadIdx.x]]) idx[threadIdx.x]=idx[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return idx[0];
}




/***********************************************************************
 * CUDA kernels
 * the functions are templated to have the float/double operations
 */

/*
 * CuMatrix
 */
template<typename Real>
__global__
static void _ammdm_elements(Real alpha, Real* mat, const Real* A, const Real* B,
       	    	            const Real* C, Real beta, MatrixDim d) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + j * d.stride;
  if ( i < d.cols && j < d.rows) {
    if ( C[index] != 0) {
      mat[index] = alpha * A[index] * B[index] / C[index] + beta * mat[index];
    } else {
      mat[index] = alpha * A[index] * B[index] + beta * mat[index];
    }
  }
}


template<typename Real>
__global__
static void _copy_from_tp_trans(Real* A, const Real* B, MatrixDim dmat) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i * (i+1) / 2;
  if ( i < dmat.rows ) {
    for (int32_cuda j = 0; j <= i; j++) {
      int32_cuda index_A = i + j * dmat.stride;
      A[index_A] = B[index + j];
    }
  }
}


template<typename Real>
__global__
static void _copy_from_tp(Real* A, const Real* B, MatrixDim dmat) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i * (i+1) / 2;
  
  if ( i < dmat.rows ) {
     for (int32_cuda j = 0; j <= i; j++) {
       int32_cuda index_A = j + i * dmat.stride;
       A[index_A] = B[index + j];
     }     
  }
}


template<typename Real>
__global__
static void _trace_sp_sp_fd(const float* A, const Real* B, float* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i * (i+1) / 2;
  if ( i < dim ) {
    float sum = 0.0;
    for (int32_cuda j = 0; j < i; j++) {
      sum = sum + 2 * (A[index + j] * B[index + j]);
    }
    sum = sum + A[index + i] * B[index + i];

    __shared__ double row_data[256];
    for (int32_cuda m = 0; m < 255; m++)
      row_data[m] = 0;

    row_data[i] = sum;
    __syncthreads();
    
    *value = _sum_reduce(row_data);
  }   
}


template<typename Real>
__global__
static void _trace_sp_sp_df(const double* A, const Real* B, double* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i * (i+1) / 2;
  if ( i < dim ) {
    double sum = 0.0;
    for (int32_cuda j = 0; j < i; j++) {
      sum = sum + 2 * (A[index + j] * B[index + j]);
    }
    sum = sum + A[index + i] * B[index + i];

    __shared__ double row_data[256];
    for (int32_cuda m = 0; m < 255; m++)
      row_data[m] = 0;

    row_data[i] = sum;
    __syncthreads();
    
    *value = _sum_reduce(row_data);
  }   
}


template<typename Real>
__global__
static void _copy_from_mat_df(double* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index_in = i + j * d_in.stride;
  int32_cuda index_out = i + j * d_out.stride;
  if ( i < d_in.cols && j < d_in.rows) {
    mat_out[index_out] = static_cast<double>(mat_in[index_in]);
  }
}


template<typename Real>
__global__
static void _copy_from_mat_df_trans(double* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index_in = i + j * d_in.stride;
  int32_cuda index_out = j + i * d_out.stride;
  if ( i < d_in.cols && j < d_in.rows) {
    mat_out[index_out] = static_cast<double>(mat_in[index_in]);
  }
}


template<typename Real>
__global__
static void _copy_from_mat_fd(float* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index_in = i + j * d_in.stride;
  int32_cuda index_out = i + j * d_out.stride;
  if ( i < d_in.cols && j < d_in.rows) {
    mat_out[index_out] = (float) mat_in[index_in];
  }
}


template<typename Real>
__global__
static void _copy_from_mat_fd_trans(float* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index_in = i + j * d_in.stride;
  int32_cuda index_out = j + i * d_out.stride;
  if ( i < d_in.cols && j < d_in.rows) {
    mat_out[index_out] = (float) mat_in[index_in];
  }
}


template<typename Real>
__global__
static void _copy_col_from_vec(Real* mat, const Real* v, int col, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < d.rows ) {
    int32_cuda index = col + i * d.stride;
    mat[index] = v[i];
  }
}


template<typename Real>
__global__
static void _apply_exp(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + j * d.stride;
  if ( i < d.cols && j < d.rows ) {
    mat[index] = exp(mat[index]);
  }
}


template<typename Real>
__global__
static void _scale_diag(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i+1)*(i+2)/2) - 1;
  if ( i < dim ) {
     mat[index] = value * mat[index];
  }
}

template<typename Real>
__global__
static void _set_diag(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + i*d.stride;
  if ( i < d.rows ) {
    mat[index] = 1;
  }
}

template<typename Real>
__global__
static void _set_diag_packed(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i+1)*(i+2)/2) - 1;
  if ( i < dim ) {
     mat[index] = value;
  }
}

template<typename Real>
__global__
static void _add_diag_packed(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i+1)*(i+2)/2) - 1;
  if ( i < dim ) {
    mat[index] = mat[index] + value;
  }
}


template<typename Real>
__global__
static void _set_const(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] = value;
}


template<typename Real>
__global__
static void _set_zero_above_diag(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index  = i + j * d.stride;
  if ( i < d.cols && j < i)
    mat[index] = 0;    
}


template<typename Real>
__global__
static void _add(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] = mat[index] + value;
}


template<typename Real>
__global__
static void _scale(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] = mat[index] * value;
}


template<typename Real>
__global__
static void _apply_log(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] = log(mat[index]);
}


template<typename Real>
__global__
static void _mul_elements(Real* mat, const Real* A, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] = mat[index] * A[index];
}

template<typename Real>
__global__
static void _vec_mul_elements(Real* v, const Real* a, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim)
    v[i] = v[i] * a[i];
}


template<typename Real>
__global__
static void _mul_cols_vec(Real* mat, const Real* scale, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] *= scale[i];
}


template<typename Real>
__global__
static void _mul_rows_vec(Real* mat, const Real* scale, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    mat[index] *= scale[j];
}


template<typename Real>
__global__
static void _div_rows_vec(Real* mat, const Real* vec_div, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;

  if( j >= d.rows ) return;

  //invert divider in shared memory
  __shared__ Real inv[16];
  if(threadIdx.x==0) {
    inv[threadIdx.y] = 1.0/vec_div[j];
  }
  __syncthreads();
 
  //multiply elements
  if ( i < d.cols  &&  j < d.rows )
    mat[index] *= inv[threadIdx.y];
}


template<typename Real>
__global__
static void _add_mat(Real alpha, const Real* A, Real beta, Real* dst, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    dst[index] = alpha*A[index] + beta*dst[index];
}



template<typename Real>
__global__
static void _add_vec_to_cols(Real alpha, const Real* col, Real beta, Real* dst, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    dst[index] = alpha*col[j] + beta*dst[index];
}



template<typename Real>
__global__
static void _add_vec_to_rows(Real alpha, const Real* row, Real beta, Real* dst, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    dst[index] = alpha*row[i] + beta*dst[index];
}


template<typename Real>
__global__
static void _apply_mask(Real* mat, const char* mask, MatrixDim dmat, MatrixDim dmask) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*dmat.stride;
  int32_cuda index2 = i + j*dmask.stride;
  if ( i < dmat.cols  &&  j < dmat.rows ) 
    if(mask[index2] == 0) mat[index] = 0;
}



/*
 * CuVector
 */
// very limited application!
template<typename Real>
__global__
static void _set_bias_params(Real* v, const Real* a, Real param_1, Real param_2, Real param_3, int* flag, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < dim ) {
    Real ratio = a[i] / param_3;
    if ( ( ratio < 0.0 ) || ( ratio >= 1.01 )) {
      *flag = 1;
      return;
    }
    if ( ratio < param_1 ) {
      Real factor = ((param_1/ratio) > param_2) ? param_2 : (param_1/ratio);
      v[i] = v[i] / factor;
    } else if ( ratio > param_1 ) {
      Real factor = ((ratio/param_1) > param_2) ? param_2 : (ratio/param_1);
      v[i] = v[i] * factor; 
    }
  }
}


template<typename Real>
__global__
static void _copy_from_vec_df(double* v_out, const Real* v_in, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0) return;
  
  if ( i < dim) {
    v_out[i] = (double) v_in[i];
  }
}


template<typename Real>
__global__
static void _copy_from_vec_fd(float* v_out, const Real* v_in, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0) return;
  
  if ( i < dim) {
    v_out[i] = (float) v_in[i];
  }
}


template<typename Real>
__global__
static void _min(const Real* v, Real* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if(blockIdx.x > 0) return;

  if ( i < dim) {
    __shared__ Real row_data[256];

   //copy the input to row_data
   row_data[i] = v[i];
   __syncthreads();

   //get the sum
   *value = _min_reduce(row_data);
  }  
}


template<typename Real>
__global__
static void _trace_mat_mat(const Real* A, const Real* B, MatrixDim dA, MatrixDim dB, Real* value) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ( i < dA.rows ) {
    Real sum = 0;
    for (int32_cuda j = 0; j < dA.cols; j++) {
      int32_cuda index_A = j + i * dA.stride;
      int32_cuda index_B = i + j * dB.stride;
      sum = sum + A[index_A] * B[index_B];
    }
  
    __shared__ Real row_data[256];

    for (int32_cuda m = 0; m < 255; m++)
      row_data[m] = 0;

    row_data[i] = sum;
    __syncthreads();

    *value = _sum_reduce(row_data);
  }
}


template<typename Real>
__global__
static void _trace_mat_mat_trans(const Real* A, const Real* B, MatrixDim dA, MatrixDim dB, Real* value) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < dA.rows ) {
    Real sum = 0;
    for (int32_cuda j = 0; j < dA.cols; j++) {
      int32_cuda index = j + i * dA.stride;

      sum = sum + A[index] * B[index];
    }
  
    __shared__ Real row_data[256];

    for (int32_cuda m = 0; m < 255; m++)
      row_data[m] = 0;

    row_data[i] = sum;
    __syncthreads();

    *value = _sum_reduce(row_data);
  }
}


template<typename Real>
__global__
static void _add_diag_mat(Real alpha, Real* v, const Real* mat, Real beta, MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > dmat.rows)
    return;

  if (i < dim) {
    Real sum = 0.0;
    for (int32_cuda j = 0; j < dmat.cols; j++) {
      int32_cuda index = j + i * dmat.stride;
      sum += mat[index] * mat[index];
    }
    v[i] = beta * v[i] + alpha * sum;
  } 

}


template<typename Real>
__global__
static void _add_diag_mat_trans(Real alpha, Real* v, const Real* mat, Real beta, MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0)
    return;

  if (i > dmat.cols)
    return;

  if (i < dim) {
    Real sum = 0.0;
    for (int32_cuda j = 0; j < dmat.rows; j++) {
      int32_cuda index = i + j * dmat.stride;
      sum += mat[index] * mat[index];
    }
    v[i] = beta * v[i] + alpha * sum;
  } 

}


template<typename Real>
__global__
static void _add_vec_vec(Real alpha, Real* v, const Real* x, const Real* y, Real beta, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0)
    return;

  if (i < dim)
    v[i] = alpha * x[i] * y[i] + beta * v[i];
}


template<typename Real>
__global__
static void _copy_col_from_mat(Real* v, int col, const Real* mat, MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = col + i * dmat.stride;
  if (blockIdx.x > 0)
    return;

  if (i < dim)
    v[i] = mat[index];
}


template<typename Real>
__global__
static void _copy_col_from_mat_df(double* v, int col, const Real* mat, MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = col + i * dmat.stride;
  if (blockIdx.x > 0)
    return;

  if (i < dim)
    v[i] = (double) mat[index];
}


template<typename Real>
__global__
static void _copy_col_from_mat_fd(float* v, int col, const Real* mat, MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = col + i * dmat.stride;
  if (blockIdx.x > 0)
    return;

  if (i < dim)
    v[i] = (float) mat[index];
}


template<typename Real>
__global__
static void _vec_apply_exp(Real* v, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0) return;
  
  if (i < dim) {
    v[i] = exp(v[i]);
  } 
}


template<typename Real>
__global__
static void _vec_apply_log(Real* v, Real* flag, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0) return;
  
  if (i < dim) {
    if (v[i] < 0) {
      *flag = 1;
      return;
    }
    v[i] = log(v[i]);
  }
}

template<typename Real>
__global__
static void _sum(Real* mat, Real* value, MatrixDim d) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + j * d.stride;

  if (i < d.cols && j < d.rows) {
    __shared__ Real row_data[256];
    row_data[index] = mat[index];
    __syncthreads();
    
    *value = _sum_reduce(row_data);
  }
}


template<typename Real>
__global__
static void _vec_sum(Real* v, Real* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0) return;
  
  if (i < dim) {
    __shared__ Real row_data[256];
    row_data[i] = v[i];
    __syncthreads();
    
    *value = _sum_reduce(row_data);
  }
}


template<typename Real>
__global__
static void _trace(const Real* mat, Real* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < dim) {
    int32_cuda index = ((i+1) * (i+2) / 2) - 1;
    __shared__ Real row_data[256];

   //copy the input to row_data
   row_data[i] = mat[index];
   __syncthreads();

   //get the sum
   *value = _sum_reduce(row_data);
}
}


template<typename Real>
__global__
static void _vec_apply_floor(Real* v, Real floor_val, int* num, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if(blockIdx.x > 0) return;

  if ( i < dim) {
    __shared__ int row_data[256];

    //copy the input to row_data
    if ( v[i] < floor_val) {
      row_data[i] = 1;
      v[i] = floor_val;
    } else {
      row_data[i] = 0;
    }

    __syncthreads();

    //get the sum
    *num = _sum_reduce(row_data);
}
}


template<typename Real>
__global__
static void _apply_pow(Real* mat, Real power, MatrixDim d) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + j * d.stride;

  if (i < d.cols && j < d.rows) {
    if (power == 1.0)
      return;
    if (power == 2.0) {
      mat[index] = mat[index] * mat[index];
    } else if (power == 0.5) {
      if (!(mat[index] >= 0.0))
        return;
      mat[index] = sqrt(mat[index]);
    } else {
      mat[index] = pow(mat[index], power);
    }
  }
}


template<typename Real>
__global__
static void _apply_floor(Real* mat, Real floor_val, MatrixDim d) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + j * d.stride;

  if (i < d.cols && j < d.rows ) {
    if (mat[index] < floor_val)
      mat[index] = floor_val;
  }
}


template<typename Real>
__global__
static void _add_row_sum_mat(const Real* mat, Real* vec_sum, MatrixDim d) {
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y; //col
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x; //row

  if(blockIdx.x > 0) return;
  if(blockDim.y > 1) return;

  __shared__ Real row_data[256];

  //copy the input to row_data
  row_data[j] = mat[i+j*d.stride];
  __syncthreads();

  //get the sum
  Real sum = _sum_reduce(row_data);
  __syncthreads();
  
  //add to previously accumulated sum
  if(threadIdx.x == 0)
    vec_sum[i] += sum;
}


template<typename Real>
__global__
static void _add_col_sum_mat(const Real* mat, Real* vec_sum, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; //row
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; //col

  if(blockIdx.x > 0) return;
  if(blockDim.y > 1) return;

  __shared__ Real row_data[256];

  //copy the input to row_data
  row_data[i] = mat[i+j*d.stride];
  __syncthreads();

  //get the sum
  Real sum = _sum_reduce(row_data);
  __syncthreads();
  
  //add to previously accumulated sum
  if(threadIdx.x == 0) 
    vec_sum[j] += sum;
}


template<typename Real>
__global__
static void _invert_elements(Real* data, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows )
    data[index] = 1.0/data[index];
}



/*
 * cu::
 */
template<typename Real>
__global__
static void _sigmoid(Real*y, const Real*x, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if( i < d.cols  &&  j < d.rows ) {
    Real res = 1.0 / (1.0 + exp(-x[index]));
    y[index] = res;
  }
}


template<typename Real>
__global__
static void _diff_sigmoid(Real*eout, const Real*e, const Real*y, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) 
    eout[index] = y[index]*(1.0-y[index]) * e[index];
}


template<typename Real>
__global__
static void _tanh(Real*y, const Real*x, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if( i < d.cols  &&  j < d.rows ) {
    Real exp_2x = exp(2.0*x[index]);
    Real res;
    if(isinf(exp_2x)) {
      res = 1.0;
    } else {
      res = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
    y[index] = res;
  }
}


template<typename Real>
__global__
static void _diff_tanh(Real*eout, const Real*e, const Real*y, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if( i < d.cols  && j < d.rows ) 
    eout[index] = (1.0 - y[index]*y[index]) * e[index];
}

template<typename Real>
__global__
static void _vec_soft_max(Real* x, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  double max = -1e20;
  double sum = 0.0;
  if ( i < dim ) {
    for (int32_cuda j = 0; j < dim; j++) {
      if ( max < x[i] )
        max = x[i];
    }
    for (int32_cuda j = 0; j < dim; j++) {
      x[i] = exp(x[i] - max);
      sum += x[i];
    }
    for (int32_cuda j = 0; j < dim; j++) {
      x[i] /= sum;
    }  
  }
}


template<typename Real>
__global__
static void _softmax(Real*y, const Real*x, MatrixDim d) {
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j >= d.rows) return;

  //copy to output and find max...
  double max = -1e20;
  double sum = 0.0;
  for(int32_cuda i=0; i<d.cols; i++) {
    if(max < x[i+j*d.stride]) max = x[i+j*d.stride];
    y[i+j*d.stride] = x[i+j*d.stride];
  }
  //subtract max, apply exp, sum up...
  for(int32_cuda i=0; i<d.cols; i++) {
    y[i+j*d.stride] = exp(y[i+j*d.stride] - max);
    sum += y[i+j*d.stride];
  }
  //normalize by sum...
  for(int32_cuda i=0; i<d.cols; i++) {
    y[i+j*d.stride] /= sum;
  }
}



template<typename Real>
__global__
static void _splice(Real* y, const Real* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if( i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_col = i % d_in.cols;
    int32_cuda src_row = j + off[i / d_in.cols];
    if(src_row < 0) src_row = 0;
    if(src_row >= d_in.rows) src_row = d_in.rows-1;
    y[index] = x[src_col + src_row*d_in.stride];
  }
}

template<typename Real>
__global__
static void _take_mean(const Real* x, Real* y, MatrixDim d_in, int d_out) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index1 = i + j * d_in.stride;
  int32_cuda index2 = j + i * d_in.stride;
  if (i <= j && j < d_in.rows) {
    int32_cuda index_sp = (j * (j+1) / 2) + i;
    y[index_sp] = 0.5 * (x[index1] + x[index2]);
  }
}

template<typename Real>
__global__
static void _take_lower(const Real* x, Real* y, MatrixDim d_in, int d_out) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d_in.stride;
  int32_cuda index_sp = (j * (j+1) / 2) + i;
  int32_cuda nr = d_out * (d_out + 1) / 2;
  if (i <= j  &&  j < d_in.rows && index_sp < nr) {
    y[index_sp] = x[index];
  }
}

template<typename Real>
__global__
static void _take_upper(const Real* x, Real* y, MatrixDim d_in, int d_out) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d_in.stride;
  if ( i >= j  &&  i < d_in.rows) {
    int32_cuda index_sp = (i * (i+1) / 2) + j;
    y[index_sp] = x[index];
  }
}

template<typename Real>
__global__
static void _copy_diag(Real* y, const Real* x, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i+1) * (i+2) / 2) - 1;
  if (i < dim) {
     y[i] = *(x+index);
  }
}

template<typename Real>
__global__
static void _copy_from_sp(const Real* x, Real* y, int d_in, MatrixDim d_out) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < d_in) {
    for (int32_cuda j = 0; j <= i; j++) {
      int32_cuda index1 = j + i * d_out.stride;
      int32_cuda index2 = i + j * d_out.stride;
      y[index1] = x[(i * (i+1) / 2) + j];
      y[index2] = x[(i * (i+1) / 2) + j];
    }
  }
}

template<typename Real>
__global__
static void _copy(Real* y, const Real* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if( i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_col = copy_from[i];
    if(src_col >= 0 && src_col < d_in.cols) {
      y[index] = x[src_col + j*d_in.stride];
    } else {
      y[index] = 1.0/0.0;
    }
  }
}

template<typename Real>
__global__
static void _one(Real* x, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < dim ) {
     x[i] = 1.0;
  }
}

template<typename Real>
__global__
static void _randomize(Real* y, const Real* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if( i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_row = copy_from[j];
    y[index] = x[i + src_row*d_in.stride];
  }
}


template<typename Real>
__global__
static void _regularize_l1(Real* wei, Real* grad, Real l1, Real lr, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows ) {

    if(wei[index]==0.0) return; //skip L1 if zero weight!
    
    Real l1_signed = l1;
    if(wei[index] < 0.0) //flip sign
      l1_signed = -l1;

    Real before = wei[index];
    Real after = wei[index] -lr*grad[index] -l1_signed;//simulate update
    if((after > 0.0) ^ (before > 0.0)) { //sign changed?
      wei[index] = 0.0;
      grad[index] = 0.0;
    } else {
      wei[index] -= l1_signed;
    }
  }
}



template<typename Real>
__global__
static void _find_row_max_id(const Real* mat, Real* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;

  if(blockIdx.x > 0) return;
  if(blockDim.y != 1) return;

  __shared__ Real value[256];
  __shared__ int32_cuda index[256];

  //copy to shared memory
  value[threadIdx.x] = mat[i+j*d.stride];
  index[threadIdx.x] = threadIdx.x;
  __syncthreads();
  
  //get the id of the max value
  int32_cuda out_max = _max_id_reduce(value,index);
  __syncthreads();

  //see if it's bigger value
  if(threadIdx.x == 0) {
    if(vec_val[j] <= mat[out_max+j*d.stride]) {
      vec_val[j] = mat[out_max+j*d.stride];
      vec_id[j]  = voff+out_max;
    }
  }
}



template<typename Real>
__global__
static void _diff_xent(const int32_cuda* vec_tgt, Real* mat_net_out, Real* vec_log_post, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i>0) return;
  if(j<d.rows) {
    int32_cuda index = vec_tgt[j] + j*d.stride;
    Real value = mat_net_out[index];
    if(value < 1e-20) value = 1e-20;
    vec_log_post[j] = log(value);
    mat_net_out[index] -= 1.0;
  }
}



template<typename Real>
__global__
static void _softmax_part(const Real* X, const int32_cuda* vec_ids, Real* Y, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if ( i < d.cols  &&  j < d.rows ) {
    Real tmp = X[index] - X[vec_ids[j] + j*d.stride];
    Y[index] = exp(tmp);
  }
}



/***********************************************************************
 * ANSI-C wrappers of CUDA kernels
 */

/*
 * "int32" 
 */
void cudaI32_set_const(dim3 Gr, dim3 Bl, int32_cuda* mat, int32_cuda value, MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d); 
}



/*
 * "float"
 */

/*
 * CuMatrix
 */
void cudaF_ammdm_elements(dim3 Gr, dim3 Bl, float alpha, float* mat, const float* A, const float* B, const float* C, float beta, MatrixDim d) {
  _ammdm_elements<<<Gr,Bl>>>(alpha,mat,A,B,C,beta,d);
}

void cudaF_copy_from_tp_trans(int Gr, int Bl, float* A, const float* B, MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}

void cudaF_copy_from_tp(int Gr, int Bl, float* A, const float* B, MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}

void cudaF_trace_sp_sp_fd(int Gr, int Bl, const float* A, const float* B, float* value, int dim) {
  _trace_sp_sp_fd<<<Gr,Bl>>>(A,B,value,dim);
}

void cudaF_trace_sp_sp_df(int Gr, int Bl, const double* A, const float* B, double* value, int dim) {
  _trace_sp_sp_df<<<Gr,Bl>>>(A,B,value,dim);
}

void cudaF_copy_from_mat_fd(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_fd<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cudaF_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_fd_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cudaF_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_df<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cudaF_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_df_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cudaF_copy_col_from_vec(int Gr, int Bl, float* mat, const float* v, int col, MatrixDim d) {
  _copy_col_from_vec<<<Gr,Bl>>>(mat,v,col,d);
}

void cudaF_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}

void cudaF_sum(dim3 Gr, dim3 Bl, float* mat, float* value, MatrixDim d) {
  _sum<<<Gr,Bl>>>(mat, value, d);
}

void cudaF_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}

void cudaF_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}

void cudaF_set_diag(int Gr, int Bl, float* mat, float value, MatrixDim d) {
  _set_diag<<<Gr,Bl>>>(mat,value,d);
}

void cudaF_set_diag_packed(int Gr, int Bl, float* mat, float value, int dim) {
  _set_diag_packed<<<Gr,Bl>>>(mat,value,dim);
}

void cudaF_add_diag_packed(int Gr, int Bl, float* mat, float value, int dim) {
  _add_diag_packed<<<Gr,Bl>>>(mat,value,dim);
}

void cudaF_set_const(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d); 
}

void cudaF_set_zero_above_diag(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _set_zero_above_diag<<<Gr,Bl>>>(mat, d);
}

void cudaF_add(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _add<<<Gr,Bl>>>(mat,value,d); 
}

void cudaF_scale_diag(int Gr, int Bl, float* mat, float value, int dim) {
  _scale_diag<<<Gr,Bl>>>(mat,value,dim);
}

void cudaF_scale(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _scale<<<Gr,Bl>>>(mat,value,d); 
}

void cudaF_apply_log(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d); 
}

void cudaF_mul_elements(dim3 Gr, dim3 Bl, float* mat, const float* A, MatrixDim d) {
  _mul_elements<<<Gr,Bl>>>(mat,A,d); 
}

void cudaF_mul_cols_vec(dim3 Gr, dim3 Bl, float* mat, const float* scale, MatrixDim d) {
  _mul_cols_vec<<<Gr,Bl>>>(mat,scale,d); 
}

void cudaF_mul_rows_vec(dim3 Gr, dim3 Bl, float* mat, const float* scale, MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaF_div_rows_vec(dim3 Gr, dim3 Bl, float* mat, const float* vec_div, MatrixDim d) {
  _div_rows_vec<<<Gr,Bl>>>(mat, vec_div, d);
}

void cudaF_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* A, float beta, float* dst, MatrixDim d) {
  _add_mat<<<Gr,Bl>>>(alpha,A,beta,dst,d); 
}

void cudaF_add_vec_to_cols(dim3 Gr, dim3 Bl, float alpha, const float* col, float beta, float* dst, MatrixDim d) {
  _add_vec_to_cols<<<Gr,Bl>>>(alpha,col,beta,dst,d); 
}


void cudaF_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float* row, float beta, float* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d); 
}

// CURRENTLY UNUSED...
void cudaF_apply_mask(dim3 Gr, dim3 Bl, float* mat, const char* mask, MatrixDim dmat, MatrixDim dmask) {
  _apply_mask<<<Gr,Bl>>>(mat,mask,dmat,dmask); 
}


/*
 * CuVector
 */
void cudaF_set_bias_params(int Gr, int Bl, float* v, const float* a, float param_1, float param_2, float param_3, int* flag, int dim) {
  _set_bias_params<<<Gr,Bl>>>(v,a,param_1,param_2,param_3,flag,dim);
}

void cudaF_copy_from_vec_df(int Gr, int Bl, double* v_out, const float* v_in, int dim) {
  _copy_from_vec_df<<<Gr,Bl>>>(v_out,v_in,dim);
}

void cudaF_copy_from_vec_fd(int Gr, int Bl, float* v_out, const float* v_in, int dim) {
  _copy_from_vec_fd<<<Gr,Bl>>>(v_out,v_in,dim);
}

void cudaF_vec_mul_elements(int Gr, int Bl, float* v, const float* a, int dim) {
  _vec_mul_elements<<<Gr,Bl>>>(v, a, dim);
}

void cudaF_vec_soft_max(int Gr, int Bl, float* v, int dim) {
  _vec_soft_max<<<Gr,Bl>>>(v, dim);
}

void cudaF_min(int Gr, int Bl, const float* v, float* value, int dim) {
  _min<<<Gr,Bl>>>(v, value, dim);
}

void cudaF_trace_mat_mat_trans(int Gr, int Bl, const float* A, const float* B, MatrixDim dA, MatrixDim dB, float* value) {
  _trace_mat_mat_trans<<<Gr,Bl>>>(A, B, dA, dB, value);
}

void cudaF_trace_mat_mat(int Gr, int Bl, const float* A, const float* B, MatrixDim dA, MatrixDim dB, float* value) {
  _trace_mat_mat<<<Gr,Bl>>>(A, B, dA, dB, value);
}

void cudaF_add_diag_mat_trans(int Gr, int Bl, float alpha, float* v, const float* mat, float beta,  MatrixDim dmat, int dim) {
  _add_diag_mat_trans<<<Gr,Bl>>>(alpha,v,mat,beta,dmat,dim);
}

void cudaF_add_diag_mat(int Gr, int Bl, float alpha, float* v, const float* mat, float beta,  MatrixDim dmat, int dim) {
  _add_diag_mat<<<Gr,Bl>>>(alpha,v,mat,beta,dmat,dim);
}

void cudaF_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}

void cudaF_copy_col_from_mat(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaF_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const float* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_df<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaF_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_fd<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaF_vec_sum(int Gr, int Bl, float* v, float* value, int dim) {
  _vec_sum<<<Gr,Bl>>>(v,value,dim);
}

void cudaF_vec_apply_floor(int Gr, int Bl, float* v, float floor_val, int* num, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,num,dim);
}

void cudaF_vec_apply_exp(int Gr, int Bl, float* v, int dim) {
  _vec_apply_exp<<<Gr,Bl>>>(v,dim);
}

void cudaF_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim) {
  _vec_apply_log<<<Gr,Bl>>>(v,flag,dim);
}

void cudaF_trace(int Gr, int Bl, float* mat, float* value, int dim) {

  _trace<<<Gr,Bl>>>(mat,value,dim);
}

void cudaF_add_row_sum_mat(dim3 Gr, dim3 Bl, const float* mat, float* vec_sum, MatrixDim d) {
  _add_row_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}

void cudaF_add_col_sum_mat(dim3 Gr, dim3 Bl, const float* mat, float* vec_sum, MatrixDim d) {
  _add_col_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}

void cudaF_invert_elements(dim3 Gr, dim3 Bl, float* data, MatrixDim d) {
  _invert_elements<<<Gr,Bl>>>(data, d);
}

/*
 * cu::
 */
void cudaF_sigmoid (dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d) {
  _sigmoid<<<Gr,Bl>>>(y, x, d); 
}

void cudaF_diff_sigmoid (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d);
}

void cudaF_tanh (dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d) {
  _tanh<<<Gr,Bl>>>(y, x, d); 
}

void cudaF_diff_tanh (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d);
}

void cudaF_softmax (size_t Gr, size_t Bl, float* y, const float* x, MatrixDim d) { 
  _softmax<<<Gr,Bl>>>(y, x, d); 
}

void cudaF_softmax_part(dim3 Gr, dim3 Bl, const float* X, const int32_cuda* vec_ids, float* Y, MatrixDim d) {
  _softmax_part<<<Gr,Bl>>>(X,vec_ids,Y,d);
}

void cudaF_splice(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in); 
}

void cudaF_one(int Gr, int Bl, float* x, int dim) {
  _one<<<Gr,Bl>>>(x,dim);
}

void cudaF_take_mean(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out) {
  _take_mean<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaF_take_lower(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out) {
  _take_lower<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaF_take_upper(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out) {
  _take_upper<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaF_copy_diag(int Gr, int Bl, float* y, const float* x, int dim) {
  _copy_diag<<<Gr,Bl>>>(y,x,dim);
}

void cudaF_copy_from_sp(int Gr, int Bl, const float* x, float* y, int d_in, MatrixDim d_out) {
  _copy_from_sp<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaF_copy(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}
  
void cudaF_randomize(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) { 
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}


void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float* wei, float* grad, float l1, float lr, MatrixDim d) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d); 
}

void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float* mat, float* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, voff, d);
}

void cudaF_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda* vec_tgt, float* mat_net_out, float* vec_log_post, MatrixDim d) {
  _diff_xent<<<Gr,Bl>>>(vec_tgt,mat_net_out,vec_log_post,d);
}




/*
 * "double" 
 */

/*
 * CuMatrix
 */
void cudaD_ammdm_elements(dim3 Gr, dim3 Bl, double alpha, double* mat, const double* A, const double* B, const double* C, double beta, MatrixDim d) {
  _ammdm_elements<<<Gr,Bl>>>(alpha,mat,A,B,C,beta,d);
}

void cudaD_copy_from_tp_trans(int Gr, int Bl, double* A, const double* B, MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}

void cudaD_copy_from_tp(int Gr, int Bl, double* A, const double* B, MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}

void cudaD_trace_sp_sp_fd(int Gr, int Bl, const float* A, const double* B, float* value, int dim) {
  _trace_sp_sp_fd<<<Gr,Bl>>>(A,B,value,dim);
}

void cudaD_trace_sp_sp_df(int Gr, int Bl, const double* A, const double* B, double* value, int dim) {
  _trace_sp_sp_df<<<Gr,Bl>>>(A,B,value,dim);
}

void cudaD_copy_from_mat_fd(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_fd<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cudaD_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_fd_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cudaD_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_df<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}


void cudaD_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_df_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cudaD_copy_col_from_vec(int Gr, int Bl, double* mat, const double* v, int col, MatrixDim d) {
  _copy_col_from_vec<<<Gr,Bl>>>(mat,v,col,d);
}

void cudaD_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}

void cudaD_sum(dim3 Gr, dim3 Bl, double* mat, double* value, MatrixDim d) {
  _sum<<<Gr,Bl>>>(mat, value, d);
}

void cudaD_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}

void cudaD_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}

void cudaD_set_diag(int Gr, int Bl, double* mat, double value, MatrixDim d) {
  _set_diag<<<Gr,Bl>>>(mat,value,d);
}

void cudaD_set_diag_packed(int Gr, int Bl, double* mat, double value, int dim) {
  _set_diag_packed<<<Gr,Bl>>>(mat,value,dim);
}

void cudaD_add_diag_packed(int Gr, int Bl, double* mat, double value, int dim) {
  _add_diag_packed<<<Gr,Bl>>>(mat,value,dim);
}

void cudaD_set_const(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d); 
}

void cudaD_set_zero_above_diag(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _set_zero_above_diag<<<Gr,Bl>>>(mat, d);
}

void cudaD_add(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _add<<<Gr,Bl>>>(mat,value,d); 
}

void cudaD_scale_diag(int Gr, int Bl, double* mat, double value, int dim) {
  _scale_diag<<<Gr,Bl>>>(mat,value,dim);
}

void cudaD_scale(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _scale<<<Gr,Bl>>>(mat,value,d); 
}

void cudaD_apply_log(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d); 
}

void cudaD_mul_elements(dim3 Gr, dim3 Bl, double* mat, const double* A, MatrixDim d) {
  _mul_elements<<<Gr,Bl>>>(mat,A,d); 
}

void cudaD_mul_cols_vec(dim3 Gr, dim3 Bl, double* mat, const double* scale, MatrixDim d) {
  _mul_cols_vec<<<Gr,Bl>>>(mat,scale,d); 
}

void cudaD_mul_rows_vec(dim3 Gr, dim3 Bl, double* mat, const double* scale, MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaD_div_rows_vec(dim3 Gr, dim3 Bl, double* mat, const double* vec_div, MatrixDim d) {
  _div_rows_vec<<<Gr,Bl>>>(mat, vec_div, d);
}

void cudaD_add_mat(dim3 Gr, dim3 Bl, double alpha, const double* A, double beta, double* dst, MatrixDim d) {
  _add_mat<<<Gr,Bl>>>(alpha,A,beta,dst,d); 
}

void cudaD_add_vec_to_cols(dim3 Gr, dim3 Bl, double alpha, const double* col, double beta, double* dst, MatrixDim d) {
  _add_vec_to_cols<<<Gr,Bl>>>(alpha,col,beta,dst,d); 
}

void cudaD_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double* row, double beta, double* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d); 
}

// CURRENTLY UNUSED...
void cudaD_apply_mask(dim3 Gr, dim3 Bl, double* mat, const char* mask, MatrixDim dmat, MatrixDim dmask) {
  _apply_mask<<<Gr,Bl>>>(mat,mask,dmat,dmask); 
}



/*
 * CuVector
 */
void cudaD_set_bias_params(int Gr, int Bl, double* v, const double* a, double param_1, double param_2, double param_3, int* flag, int dim) {
  _set_bias_params<<<Gr,Bl>>>(v,a,param_1,param_2,param_3,flag,dim);
}

void cudaD_copy_from_vec_df(int Gr, int Bl, double* v_out, const double* v_in, int dim) {
  _copy_from_vec_df<<<Gr,Bl>>>(v_out,v_in,dim);
}

void cudaD_copy_from_vec_fd(int Gr, int Bl, float* v_out, const double* v_in, int dim) {
  _copy_from_vec_fd<<<Gr,Bl>>>(v_out,v_in,dim);
}

void cudaD_vec_mul_elements(int Gr, int Bl, double* v, const double* a, int dim) {
  _vec_mul_elements<<<Gr,Bl>>>(v, a, dim);
}

void cudaD_vec_soft_max(int Gr, int Bl, double* v, int dim) {
  _vec_soft_max<<<Gr,Bl>>>(v, dim);
}

void cudaD_min(int Gr, int Bl, const double* v, double* value, int dim) {
  _min<<<Gr,Bl>>>(v, value, dim);
}

void cudaD_trace_mat_mat_trans(int Gr, int Bl, const double* A, const double* B, MatrixDim dA, MatrixDim dB, double* value) {
  _trace_mat_mat_trans<<<Gr,Bl>>>(A,B,dA,dB,value);
}

void cudaD_trace_mat_mat(int Gr, int Bl, const double* A, const double* B, MatrixDim dA, MatrixDim dB, double* value) {
  _trace_mat_mat<<<Gr,Bl>>>(A,B,dA,dB,value);
}

void cudaD_add_diag_mat_trans(int Gr, int Bl, double alpha, double* v, const double* mat, double beta,  MatrixDim dmat, int dim) {
  _add_diag_mat_trans<<<Gr,Bl>>>(alpha,v,mat,beta,dmat,dim);
}

void cudaD_add_diag_mat(int Gr, int Bl, double alpha, double* v, const double* mat, double beta,  MatrixDim dmat, int dim) {
  _add_diag_mat<<<Gr,Bl>>>(alpha,v,mat,beta,dmat,dim);
}

void cudaD_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}

void cudaD_copy_col_from_mat(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaD_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_df<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaD_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const double* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_fd<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaD_vec_sum(int Gr, int Bl, double* v, double* value, int dim) {
  _vec_sum<<<Gr,Bl>>>(v,value,dim);
}

void cudaD_vec_apply_floor(int Gr, int Bl, double* v, double floor_val, int* num, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,num,dim);
}

void cudaD_vec_apply_exp(int Gr, int Bl, double* v, int dim) {
  _vec_apply_exp<<<Gr,Bl>>>(v,dim);
}

void cudaD_vec_apply_log(int Gr, int Bl, double* v, double* flag, int dim) {
  _vec_apply_log<<<Gr,Bl>>>(v,flag,dim);
}

void cudaD_trace(int Gr, int Bl, double* mat, double* value, int dim) {
  _trace<<<Gr,Bl>>>(mat,value,dim);
}

void cudaD_add_row_sum_mat(dim3 Gr, dim3 Bl, const double* mat, double* vec_sum, MatrixDim d) {
  _add_row_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}

void cudaD_add_col_sum_mat(dim3 Gr, dim3 Bl, const double* mat, double* vec_sum, MatrixDim d) {
  _add_col_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}

void cudaD_invert_elements(dim3 Gr, dim3 Bl, double* data, MatrixDim d) {
  _invert_elements<<<Gr,Bl>>>(data, d);
}

/*
 * cu::
 */
void cudaD_sigmoid (dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d) {
  _sigmoid<<<Gr,Bl>>>(y, x, d); 
}

void cudaD_diff_sigmoid (dim3 Gr, dim3 Bl, double* eout, const double* e, const double* y, MatrixDim d) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d);
}

void cudaD_tanh (dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d) {
  _tanh<<<Gr,Bl>>>(y, x, d); 
}

void cudaD_diff_tanh (dim3 Gr, dim3 Bl, double* eout, const double* e, const double* y, MatrixDim d) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d);
}


void cudaD_softmax (size_t Gr, size_t Bl, double* y, const double* x, MatrixDim d) { 
  _softmax<<<Gr,Bl>>>(y, x, d); 
}

void cudaD_softmax_part(dim3 Gr, dim3 Bl, const double* X, const int32_cuda* vec_ids, double* Y, MatrixDim d) {
  _softmax_part<<<Gr,Bl>>>(X,vec_ids,Y,d);
}

void cudaD_splice(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in); 
}

void cudaD_one(int Gr, int Bl, double* x, int dim) {
  _one<<<Gr,Bl>>>(x,dim);
}

void cudaD_take_mean(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out) {
  _take_mean<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaD_take_lower(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out) {
  _take_lower<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaD_take_upper(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out) {
  _take_upper<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaD_copy_diag(int Gr, int Bl, double* y, const double* x, int dim) {
  _copy_diag<<<Gr,Bl>>>(y,x,dim);
}

void cudaD_copy_from_sp(int Gr, int Bl, const double* x, double* y, int d_in, MatrixDim d_out) {
  _copy_from_sp<<<Gr,Bl>>>(x,y,d_in,d_out);
}

void cudaD_copy(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}
  
void cudaD_randomize(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) { 
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}

void cudaD_regularize_l1(dim3 Gr, dim3 Bl, double* wei, double* grad, double l1, double lr, MatrixDim d) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d); 
}

void cudaD_find_row_max_id(dim3 Gr, dim3 Bl, const double* mat, double* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, voff, d);
}

void cudaD_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda* vec_tgt, double* mat_net_out, double* vec_log_post, MatrixDim d) {
  _diff_xent<<<Gr,Bl>>>(vec_tgt,mat_net_out,vec_log_post,d);
}







