// cudamatrix/cu-kernels.cu

// Copyright 2009-2012  Karel Vesely
//                2013  Ehsan Variani
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//           2013-2015  Guoguo Chen

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
#include "cudamatrix/cu-kernels-ansi.h"


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
    if (threadIdx.x >= halfPoint)  { // was <
      // Get the shared value stored by another thread
      Real temp = 0.0;
      if(threadIdx.x < nTotalThreads) { // was +halfPoint
        temp = buffer[threadIdx.x]; // was +halfPoint
      }
      buffer[threadIdx.x - halfPoint] += temp;
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
      if (threadIdx.x + halfPoint < nTotalThreads) {
        Real temp = buffer[threadIdx.x + halfPoint];
        if (temp < buffer[threadIdx.x])
           buffer[threadIdx.x] = temp;
      }
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
      if(threadIdx.x+halfPoint < nTotalThreads) {
        Real temp = buffer[threadIdx.x + halfPoint];
        if (temp > buffer[threadIdx.x])
          buffer[threadIdx.x] = temp;
      }
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
static void _copy_low_upp(Real* A, MatrixDim dimA) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i <= j || i >= dimA.rows) return;
  int index_1 = i * dimA.stride + j;
  int index_2 = j * dimA.stride + i;
  A[index_2] = A[index_1];
}


template<typename Real>
__global__
static void _copy_upp_low(Real* A, MatrixDim dimA) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j <= i || j >= dimA.rows) return;
  int index_1 = i * dimA.stride + j;
  int index_2 = j * dimA.stride + i;
  A[index_2] = A[index_1];
}

// mat += diag(vec) * mat2.
template<typename Real>
__global__
static void _add_diag_vec_mat(Real alpha, Real *mat, MatrixDim mat_dim,
                              const Real *vec, const Real *mat2, int mat2_row_stride,
                              int mat2_col_stride, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // column index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // row index

  int index = j * mat_dim.stride + i,
      index2 = j * mat2_row_stride + i * mat2_col_stride;

  if (i < mat_dim.cols && j < mat_dim.rows) {
    mat[index] = alpha * vec[j] * mat2[index2] + beta * mat[index];
  }
}


template<typename Real, typename OtherReal>
__global__
static void _copy_from_tp(Real* A, const OtherReal* B, MatrixDim dmat) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dmat.cols && j < dmat.rows) {
    int32_cuda index_B = (j * (j+1) / 2) + i;
    int32_cuda index_A = j * dmat.stride + i;
    if (i <= j) {
      A[index_A] = B[index_B];
    } else {
      A[index_A] = 0.0;
    }
  }
}


template<typename Real, typename OtherReal>
__global__
static void _copy_from_tp_trans(Real* A, const OtherReal* B, MatrixDim dmat) {
  // we interpret these indexes oppositely from normal, but it doesn't
  // matter as it's invoked in a symmetric way.
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  // transpose the indices used to index the source TpMatrix.
  if (i < dmat.rows && j < dmat.cols) {
    int32_cuda index_B = (j * (j+1) / 2) + i;
    int32_cuda index_A = i * dmat.stride + j;
    if (i <= j) {
      A[index_A] = B[index_B];
    } else {
      A[index_A] = 0.0;
    }
  }
}


template<typename Real, typename OtherReal>
__global__
static void _copy_from_mat(Real* mat_out, const OtherReal* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // col-index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row-index.
  int32_cuda index_out = i + j * d_out.stride;
  int32_cuda index_in = i + j * d_in.stride;
  if (i < d_out.cols && j < d_out.rows)
    mat_out[index_out] = static_cast<Real>(mat_in[index_in]);
}


template<typename Real, typename OtherReal>
__global__
static void _copy_from_mat_trans(Real* mat_out, const OtherReal* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // col-index out
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // row-index out
  int32_cuda index_out = i + j * d_out.stride;
  int32_cuda index_in = j + i * d_in.stride;
  if (j < d_out.rows && i < d_out.cols)
    mat_out[index_out] = static_cast<Real>(mat_in[index_in]);
}

template<typename Real, typename OtherReal>
__global__
static void _copy_from_smat(Real* mat_out, const MatrixElement<OtherReal>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= d_in) return;
  int data_index = smat_in[smat_index].row * d_out.stride + smat_in[smat_index].column;
  mat_out[data_index] = smat_in[smat_index].weight;
}

template<typename Real, typename OtherReal>
__global__
static void _copy_from_smat_trans(Real* mat_out, const MatrixElement<OtherReal>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= d_in) return;
  int data_index = smat_in[smat_index].column * d_out.stride + smat_in[smat_index].row;
  mat_out[data_index] = smat_in[smat_index].weight;
}

template<typename Real>
__global__
static void _trace_mat_smat_trans(const Real* mat_in, const MatrixElement<Real>* smat_in, MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in, Real* trace_vec_out) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= smat_d_in) return;
  int mat_index = smat_in[smat_index].row * mat_d_in.stride + smat_in[smat_index].column;
  trace_vec_out[smat_index] = mat_in[mat_index] * smat_in[smat_index].weight;
}

template<typename Real>
__global__
static void _trace_mat_smat(const Real* mat_in, const MatrixElement<Real>* smat_in, MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in, Real* trace_vec_out) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= smat_d_in) return;
  int mat_index = smat_in[smat_index].column * mat_d_in.stride + smat_in[smat_index].row;
  trace_vec_out[smat_index] = mat_in[mat_index] * smat_in[smat_index].weight;
}

template<typename Real>
__global__
static void _transpose_matrix(Real* mat, MatrixDim d) {
  // Transposes a square matrix in-place.
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // col-index
  if (j >= i || i >= d.rows) { return; } // Only half the threads act.
  int32_cuda index_a = j + i * d.stride,
      index_b = i + j * d.stride;
  Real a = mat[index_a], b = mat[index_b];
  mat[index_a] = b;
  mat[index_b] = a;
}

template<typename Real>
__global__
static void _apply_exp(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows) {
    mat[index] = exp(mat[index]);
  }
}


template<typename Real>
__global__
static void _scale_diag_packed(Real* mat, Real value, int dim) {
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
  if ( i < d.rows && i < d.cols) {
    mat[index] = value;
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
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // column
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = value;
}


template<typename Real>
__global__
static void _set_zero_above_diag(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index  = i + j * d.stride;
  if (i < d.cols && j < i)
    mat[index] = 0.0;
}


template<typename Real>
__global__
static void _add(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = mat[index] + value;
}


template<typename Real>
__global__
static void _scale(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = mat[index] * value;
}


template<typename Real>
__global__
static void _apply_log(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = log(mat[index]);
}

template<typename Real>
__global__
static void _mul_elements(Real* mat, const Real* A, MatrixDim dst_d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda dst_index = i + j*dst_d.stride, src_index = i + j*src_stride;
  if (i < dst_d.cols  &&  j < dst_d.rows)
    mat[dst_index] = mat[dst_index] * A[src_index];
}

template<typename Real>
__global__
static void _div_elements(Real* mat, const Real* A, MatrixDim dst_d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda dst_index = i + j*dst_d.stride, src_index = i + j*src_stride;
  if (i < dst_d.cols  &&  j < dst_d.rows)
    mat[dst_index] = mat[dst_index] / A[src_index];
}

template<typename Real>
__global__
static void _max(Real* mat, const Real* A, MatrixDim dst_d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda dst_index = i + j*dst_d.stride, src_index = i + j*src_stride;
  if ( i < dst_d.cols  &&  j < dst_d.rows ) {
    Real a = mat[dst_index], b = A[src_index];
    mat[dst_index] = (a > b ? a : b);
  }
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
  if (i < d.cols && j < d.rows)
    mat[index] *= scale[i];
}


template<typename Real>
__global__
static void _mul_rows_vec(Real* mat, const Real* scale, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] *= scale[j];
}

template<typename Real>
__global__
static void _mul_rows_group_mat(Real *y, const Real *x, MatrixDim d,
                                int src_stride, int group_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < d.rows && i < d.cols ) {
    int dst_index = i + j * d.stride;
    int src_index = i / group_size + j * src_stride;
    y[dst_index] *= x[src_index];
  }
}

/// y is the derivative we will output; vec is the input we're computing
/// the group p-norm on, "norm" is the previously computed group p-norm.
template<typename Real>
__global__
static void _calc_pnorm_deriv(Real *deriv, const Real *vec, const Real *norm,
        MatrixDim d, int src_stride, int group_size, Real power) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < d.rows  && i < d.cols ) {
    int dst_index = i + j * d.stride,
        src_index = i / group_size + j * src_stride;
    Real vec_element = vec[dst_index], // this is the element of the original vector.
         norm_element = norm[src_index]; // this is the pnorm
    Real vec_element_sign = (vec_element > 0 ? 1 : -1);
    Real ans;
    if (norm_element <= 0.0) ans = 0.0; // The derivative is either zero or undefined at the origin.
    else ans = vec_element_sign * pow(std::abs(vec_element), power - 1) *
             pow(norm_element, 1 - power);
    deriv[dst_index] = ans;
  }
}

/// deriv is the derivative we will output; vec is the input we're computing
/// the group max on, "maxv" is the previously computed group max.
template<typename Real>
__global__
static void _calc_group_max_deriv(Real *deriv, const Real *vec, const Real *maxv,
        MatrixDim d, int src_stride, int group_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < d.rows  && i < d.cols ) {
    int dst_index = i + j * d.stride,
        src_index = i / group_size + j * src_stride;
    Real vec_element = vec[dst_index], // this is the element of the original vector.
         max_element = maxv[src_index]; // this is the max value
    Real ans = (max_element == vec_element ? 1.0 : 0.0);
    deriv[dst_index] = ans;
  }
}

/// Set each element to y = (x == orig ? changed : x).
template<typename Real>
__global__
static void _replace_value(Real *vec, int dim, Real orig, Real changed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim)
    if (vec[i] == orig) vec[i] = changed;
}


template<typename Real>
__global__
static void _div_rows_vec(Real* mat, const Real* vec_div, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;

  if (j >= d.rows ) return;

  //invert divider in shared memory
  __shared__ Real inv[16];
  if(threadIdx.x==0) {
    inv[threadIdx.y] = 1.0/vec_div[j];
  }
  __syncthreads();

  //multiply elements
  if (i < d.cols && j < d.rows)
    mat[index] *= inv[threadIdx.y];
}


template<typename Real>
__global__
static void _add_mat(Real alpha, const Real* src, Real* dst, MatrixDim d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = i + j * src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha * src[index_src] + dst[index];
}

template<typename Real>
__global__
static void _add_mat_trans(Real alpha, const Real* src, Real* dst, MatrixDim d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j *d.stride;
  int32_cuda index_src = j + i*src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha*src[index_src] + dst[index];
}

template<typename Real>
__global__
static void _add_mat_blocks(Real alpha, const Real* src, int32_cuda num_row_blocks, int32_cuda num_col_blocks, Real* dst, MatrixDim d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = i + j * src_stride;
  if (i < d.cols && j < d.rows)
    for (int32_cuda p = 0; p < num_row_blocks; p++) {
      for (int32_cuda q = 0; q < num_col_blocks; q++) {
        dst[index] = alpha * src[index_src + p * src_stride * d.rows + q * d.cols] + dst[index];
      }
    }
}

template<typename Real>
__global__
static void _add_mat_blocks_trans(Real alpha, const Real* src, int32_cuda num_row_blocks, int32_cuda num_col_blocks, Real* dst, MatrixDim d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = j + i * src_stride;
  if (i < d.cols && j < d.rows)
    for (int32_cuda p = 0; p < num_row_blocks; p++) {
      for (int32_cuda q = 0; q < num_col_blocks; q++) {
        dst[index] = alpha * src[index_src + p * src_stride * d.cols + q * d.rows] + dst[index];
      }
    }
}

template<typename Real>
__global__
static void _add_mat_mat_div_mat(const Real* A, const Real* B, const Real* C, Real* dst, MatrixDim d, int stride_a,
                                 int stride_b, int stride_c) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride,
             a_index = i + j*stride_a,
             b_index = i + j*stride_b,
             c_index = i + j*stride_c;
  if (i < d.cols && j < d.rows)
    if (C[c_index] == 0)
      dst[index] = A[a_index];
    else
      dst[index] = A[a_index] * B[b_index] / C[c_index];
}

// Given a matrix input S (not packed!) and a lower-triangular matrix L,
// this function does S = beta S + alpha * L^T L.  This is used in PSD matrix inversion.
// The i index is the row of the destination S and the j the column (although of
// course the output is symmetric so it doesn't matter in a sense).  The main point
// of this is to make use of various symmetries and zero-ness.
template<typename Real>
__global__
static void _sy_add_tr2(Real alpha, Real beta, const Real *T, MatrixDim tdim, Real *S,
                        MatrixDim sdim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= sdim.rows || j > i) return;

  // this thread computes the dot-product of the i'th column of
  // L with the j'th column of L.  The values we're multiplying
  // are only nonzero for row-index k greater or equal to
  // max(i, j), which equals i.

  Real sum = 0.0;
  for (int k = i; k < sdim.rows; k++) {
    int i_index = i + tdim.stride * k,
         j_index = j + tdim.stride * k;
     sum += T[i_index] * T[j_index];
  }
  int output_index1 = i * sdim.stride + j,
      output_index2 = j * sdim.stride + i;
  S[output_index1] = alpha * sum + beta * S[output_index1];
  S[output_index2] = alpha * sum + beta * S[output_index2];
}





template<typename Real>
__global__
static void _add_vec_to_cols(Real alpha, const Real* col, Real beta, Real* dst, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha*col[j] + beta*dst[index];
}



template<typename Real>
__global__
static void _add_vec_to_rows(Real alpha, const Real* row, Real beta, Real* dst, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
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

template<typename Real>
__global__
static void _add_mat_diag_vec(Real alpha, Real *mat, MatrixDim mat_dim,
                              const Real *mat2, int mat2_row_stride, int mat2_col_stride,
                              const Real *vec, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // column index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // row index

  int index = i + j * mat_dim.stride,
      index2 = i * mat2_col_stride + j * mat2_row_stride;
  if (j < mat_dim.rows && i < mat_dim.cols)
    mat[index] = alpha * mat2[index2] * vec[i] + beta * mat[index];
}

template<typename Real>
__global__
static void _add_mat_mat_elements(Real *data, const Real *srcA_data, const Real *srcB_data, MatrixDim dim, int srcA_stride, int srcB_stride, Real alpha, Real beta) {
    int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
    int32_cuda tgt_index = i + j*dim.stride;
    int32_cuda srcA_index = i + j*srcA_stride;
    int32_cuda srcB_index = i + j*srcB_stride;
    if (i < dim.cols && j < dim.rows) {
        data[tgt_index] = alpha * srcA_data[srcA_index] * srcB_data[srcB_index] + beta * data[tgt_index] ;
    }
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
  //  if (blockIdx.y > 0) return;

  if (i < dim) {
    v_out[i] = (double) v_in[i];
  }
}


// This kernel writes a copy of the vector "v_in" to each row of the matrix
// "m_out".  the dimension of v_in should be equal to the #columns of m_out.
template<typename Real>
__global__
static void _copy_rows_from_vec(Real* m_out, MatrixDim d, const Real* v_in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // column index.
  int j = blockIdx.y * blockDim.y + threadIdx.y; // row index.
  if (i < d.cols && j < d.rows) {
    int index = i + j * d.stride;
    m_out[index] = v_in[i];
  }
}

template<typename Real>
__global__
static void _copy_from_vec_fd(float* v_out, const Real* v_in, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  //  if (blockIdx.y > 0) return;

  if (i < dim) {
    v_out[i] = (float) v_in[i];
  }
}


template<typename Real>
__global__
static void _vec_min(const Real* v, Real* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= CU1DBLOCK) return;

  __shared__ Real row_data[CU1DBLOCK];

  int block_size = (dim + CU1DBLOCK - 1) / CU1DBLOCK;

  Real min = 1.0 / 0.0; // infinity.

  for (int j = i * block_size; j < (i+1) * block_size && j < dim; j++) {
     Real v_j = v[j];
     if (v_j < min) min = v_j;
  }

  row_data[i] = min;

  __syncthreads();

  //get the sum
  *value = _min_reduce(row_data);
}


template<typename Real>
__global__
static void _vec_max(const Real* v, Real* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if(blockIdx.y > 0) return;

  __shared__ Real row_data[CU1DBLOCK];

  if(i >= CU1DBLOCK) return;

  int block_size = (dim + CU1DBLOCK - 1) / CU1DBLOCK;

  Real max = -1.0 / 0.0; // -infinity.

  for (int j = i * block_size; j < (i+1) * block_size && j < dim; j++) {
     Real v_j = v[j];
     if (v_j > max) max = v_j;
  }

  row_data[i] = max;

  __syncthreads();

  //get the sum
  *value = _max_reduce(row_data);
}


// _trace_mat_mat expects to be called with 1 blocks, each of dimension
// CU1DBLOCK.  Each block outputs a partial sum to value[blockIdx.x],
// i.e. value[0 through 0].
template<typename Real, int num_blocks>
__global__
static void _trace_mat_mat(const Real* A, const Real* B, MatrixDim dA, int B_stride, Real* value) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if(blockIdx.x > num_blocks || threadIdx.x > CU1DBLOCK) return;

  int num_elements = dA.rows * dA.cols,
      num_threads = CU1DBLOCK * num_blocks;
  int block_size = (num_elements + num_threads - 1) / num_threads;
  int loop_start = i * block_size, loop_end = (i + 1) * block_size;
  if (loop_end > num_elements)
    loop_end = num_elements;

  Real sum = 0.0;
  for (int j = loop_start; j < loop_end; j++) {
    // for (int j = i; j < num_elements; j += num_threads) {
    int row = j / dA.cols, col = j % dA.cols; // "row" is row-index in A, "col" is
                                              // col-index in A; in B, it's reversed.
    int index_A = col + row * dA.stride,
        index_B = row + col * B_stride;
    sum += A[index_A] * B[index_B];
  }
  __shared__ Real row_data[CU1DBLOCK];

  row_data[threadIdx.x] = sum;

  __syncthreads();

  Real ans = _sum_reduce(row_data);
  if (threadIdx.x == 0)
    value[blockIdx.x] = ans;
}

// _trace_mat_mat_trans expects to be called with 4 blocks, each of dimension
// CU1DBLOCK.  Each block outputs a partial sum to value[blockIdx.x],
// i.e. value[0 through 3].
template<typename Real, int num_blocks>
__global__
static void _trace_mat_mat_trans(const Real* A, const Real* B, MatrixDim dA, int B_stride, Real* value) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if(blockIdx.x > num_blocks || threadIdx.x > CU1DBLOCK) return;

  int num_elements = dA.rows * dA.cols,
      num_threads = CU1DBLOCK * num_blocks;
  // int block_size = (num_elements + num_threads - 1) / num_threads;
  // int loop_start = i * block_size, loop_end = (i + 1) * block_size;
  // if (loop_end > num_elements)
  //  loop_end = num_elements;

  Real sum = 0.0;
  // for (int j = loop_start; j < loop_end; j++) {
  for (int j = i; j < num_elements; j += num_threads) {
    int row = j / dA.cols, col = j % dA.cols; // "row" is row-index in A, "col" is
                                              // col-index in A; in B, it's reversed.
    int index_A = col + row * dA.stride,
        index_B = col + row * B_stride;
    sum += A[index_A] * B[index_B];
  }
  __shared__ Real row_data[CU1DBLOCK];

  row_data[threadIdx.x] = sum;

  __syncthreads();

  Real ans = _sum_reduce(row_data);
  if (threadIdx.x == 0)
    value[blockIdx.x] = ans;
}


// Adds diag(M N) to v, where M and N are matrices.  We supply row_stride and
// col_stride arguments for M and N, and swapping them allows us to transpose
// those matrices.  Note: we imagine row-major indexing here, just like Kaldi
// and CBLAS (but unlike CUBLAS).
// This kernel expects the blockDim to be (CU1DBLOCK, 1) and the
// gridDim times CU1DBLOCK to be at least num-rows-of-v * threads_per_element.
// threads_per_element should be a power of 2.
template<typename Real>
__global__
static void _add_diag_mat_mat(
       Real alpha, Real* v, int v_dim, const Real* M, int M_cols, int M_row_stride,
       int M_col_stride, const Real *N, int N_row_stride, int N_col_stride,
       int threads_per_element, Real beta) {

  // we actually assume blockDim.x == CU1DBLOCK here.
  // Each diagonal element of v is processed by "threads_per_element" threads.
  __shared__ Real temp_data[CU1DBLOCK];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v_idx = i / threads_per_element,   // v_idx is the index into v that we are supposed to
      sub_idx = i % threads_per_element; // add to; 0 <= sub_idx < threads_per_element tells
                                         // us which block of elements we sum up.
  if (v_idx < v_dim) {
    Real sum = 0.0;
    for (int j = sub_idx; j < M_cols; j += threads_per_element) {
      int M_index = v_idx * M_row_stride + j * M_col_stride,
          N_index = j * N_row_stride + v_idx * N_col_stride;
      sum += M[M_index] * N[N_index];
    }
    temp_data[threadIdx.x] = sum;
  }

  // start_idx = threadIdx.x - sub_idx; // start of the position in temp_data
                                     // that we want to sum up.
  // The following is a tree-based reduction of the elements of temp_data from
  // start_idx to start_idx + threads_per_element - 1; our own index is "sub_idx".
  __syncthreads();
  int num_total_threads = threads_per_element;
  while (num_total_threads > 1) {
    int half_point = ((1 + num_total_threads) >> 1);
    if (sub_idx < half_point) {
      Real temp = 0.0;
      if (sub_idx + half_point < num_total_threads) {
        temp = temp_data[threadIdx.x + half_point];
      }
      temp_data[threadIdx.x] += temp;
    }
    __syncthreads();
    num_total_threads = half_point;
  }
  if (sub_idx == 0 && v_idx < v_dim) {
    v[v_idx] = beta * v[v_idx] + alpha * temp_data[threadIdx.x];
  }
}


template<typename Real>
__global__
static void _add_vec_vec(Real alpha, Real* v, const Real* x, const Real* y, Real beta, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (blockIdx.y > 0) return;

  if (i < dim)
    v[i] = alpha * x[i] * y[i] + beta * v[i];
}


template<typename Real>
__global__
static void _copy_col_from_mat_df(double* v, int col, const Real* mat, MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = col + i * dmat.stride;
  // if (blockIdx.y > 0)  return;

  if (i < dim)
    v[i] = (double) mat[index];
}


template<typename Real>
__global__
static void _copy_col_from_mat_fd(float* v, int col, const Real* mat, MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = col + i * dmat.stride;
  // if (blockIdx.y > 0)  return;

  if (i < dim)
    v[i] = (float) mat[index];
}


template<typename Real>
__global__
static void _vec_apply_exp(Real* v, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (blockIdx.y > 0) return;

  if (i < dim) {
    v[i] = exp(v[i]);
  }
}


template<typename Real>
__global__
static void _vec_apply_log(Real* v, Real* flag, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  //  if (blockIdx.y > 0) return;

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
static void _cuda_comp_obj_deriv(MatrixElement<Real> *x, int s, const Real* z, MatrixDim d, Real* z2, MatrixDim d2, Real* t) {
  int i = threadIdx.x;
  __shared__ Real tot_objf[CU1DBLOCK];
  __shared__ Real tot_weight[CU1DBLOCK];


  Real tmp_weight_sum = 0;
  Real tmp_tot_objf = 0;
  int size = s / CU1DBLOCK; //the least size in a loop (later part)
  int threshold = s - size * CU1DBLOCK; //any loop below this number would + 1

  int loop_start;
  int loop_end;
  if(i < threshold) {
    loop_start = i * (size + 1);
    loop_end = (i+1) * (size + 1);
  }
  else {
    loop_start = threshold + i*size;
    loop_end = threshold + (i+1)*size;
  }
  for(int j = loop_start; j< loop_end; j++) {
    int m = (x + j)->row;   //* ((int*) ((size_t)x + j * (2 * sizeof(int) + sizeof(Real) )) );
    int label = (x + j)->column; //*(int*) ((size_t)x + j * (2 * sizeof(int) + sizeof(Real) )+ sizeof(int));
    Real weight = (x + j)->weight; //*(Real*) ((size_t)x + j * (2 * sizeof(int) + sizeof(Real) ) + 2 * sizeof(int));
    tmp_weight_sum += weight;
    Real this_prob =  *(z + m * d.stride + label);
    tmp_tot_objf += weight * log(this_prob);

    *(z2 + m * d2.stride + label ) += weight / this_prob;// there might be problems here....
  }
  tot_objf[i] = tmp_tot_objf;
  tot_weight[i] = tmp_weight_sum;
  __syncthreads();
  *t = _sum_reduce(tot_objf);
  __syncthreads();
  *(t+1) = _sum_reduce(tot_weight);
  return;
}

template<typename Real>
__global__
static void _cuda_matrix_add_elements(Real *data, MatrixDim dim, Real alpha, MatrixElement<Real>* x, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
        return;
    data[x[i].row * dim.stride + x[i].column] += alpha * x[i].weight;
}

template<typename Real>
__global__
static void _cuda_matrix_add_indexed_values(MatrixDim dim, Real alpha,
                                            const Int32Pair* indices, const Real* x,
                                            int s, Real* data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s)
    return;
  int data_i = indices[i].first * dim.stride + indices[i].second;
  data[data_i] += alpha * x[i];
}


template<typename Real>
__global__
static void _matrix_lookup(const Real *data, MatrixDim dim,
                           const Int32Pair *indices,
                           int indices_size, Real *output) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= indices_size) return;
  int data_ind = indices[ind].first * dim.stride + indices[ind].second;
  output[ind] = data[data_ind];

}

template<typename Real>
__global__
static void _equal_element_mask(const Real *mat1, const Real *mat2, Real *mask, MatrixDim mat1_dim, int mat2_stride, int mask_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // col
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // row
  int32_cuda index_mat1 = i + j*mat1_dim.stride;
  int32_cuda index_mat2 = i + j*mat2_stride;
  int32_cuda index_mask = i + j*mask_stride;
  if (i < mat1_dim.cols && j < mat1_dim.rows)
    mask[index_mask] = (mat1[index_mat1] == mat2[index_mat2] ? 1.0 : 0.0);
}

template<typename Real>
__global__
static void _vec_sum(Real *v, Real *sum, int dim, int inc) {
  int i = threadIdx.x;
  __shared__ Real row_data[CU1DBLOCK];

  if (i >= CU1DBLOCK) return;

  Real tmp_sum = 0;
  int size = dim / CU1DBLOCK; //the least size in a loop (later part)
  int threshold = dim - size * CU1DBLOCK; //any loop below this number would + 1

  int loop_start;
  int loop_end;
  if(i < threshold) {
    loop_start = i * (size + 1);
    loop_end = (i+1) * (size + 1);
  }
  else {
    loop_start = threshold + i * size;
    loop_end = threshold + (i+1) * size;
  }
  for(int j = loop_start; j< loop_end; j++) {
    tmp_sum += v[j * inc];
  }

  row_data[threadIdx.x] = tmp_sum;
  __syncthreads();
  *sum = _sum_reduce(row_data);
}


template<typename Real>
__global__
static void _pvec_sum(Real* v, Real* g, int dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int start = size * i;
  int end = start + size;
  if (end > dim) end = dim;
  __shared__ Real row_data[CU1DBLOCK];
  Real sum = 0;
  for (int j = start; j < end; j++)
    sum += v[j];
  row_data[threadIdx.x] = sum;
  __syncthreads();
  g[blockIdx.x] = _sum_reduce(row_data);
}



template<typename Real>
__global__
static void _vec_apply_floor(Real *v, Real floor_val, float *count, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < dim) {
    if ( v[i] < floor_val) {
      v[i] = floor_val;
      count[i] = 1;
    } else {
      count[i] = 0;
    }
  }
}

template<typename Real>
__global__
static void _vec_apply_ceiling(Real *v, Real ceiling_val, float *count, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < dim) {
    if ( v[i] > ceiling_val) {
      v[i] = ceiling_val;
      count[i] = 1;
    } else {
      count[i] = 0;
    }
  }
}

template<typename Real>
__global__
static void _apply_pow(Real* mat, Real power, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  int index = i + j * d.stride;
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
static void _apply_pow_abs(Real* mat, Real power, bool include_sign, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  int index = i + j * d.stride;
  if (i < d.cols && j < d.rows) {
    if (include_sign == true && mat[index] < 0) {
      if (power == 1.0)
        mat[index] = -std::abs(mat[index]);
      if (power == 2.0) {
        mat[index] = -mat[index] * mat[index];
      } else if (power == 0.5) {
        mat[index] = -sqrt(std::abs(mat[index]));
      } else {
        mat[index] = -pow(std::abs(mat[index]), power);
      }
    } else {
      if (power == 1.0)
        mat[index] = std::abs(mat[index]);
      if (power == 2.0) {
        mat[index] = mat[index] * mat[index];
      } else if (power == 0.5) {
        mat[index] = sqrt(std::abs(mat[index]));
      } else if (power < 0.0 && mat[index] == 0.0) {
        mat[index] = 0.0;
      } else {
        mat[index] = pow(std::abs(mat[index]), power);
      }
    }
  }
}

template<typename Real>
__global__
static void _apply_heaviside(Real* mat, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  int index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = (mat[index] > 0.0 ? 1.0 : 0.0);
}


template<typename Real>
__global__
static void _apply_floor(Real* mat, Real floor_val, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  int index = i + j * d.stride;

  if (i < d.cols && j < d.rows) {
    if (mat[index] < floor_val)
      mat[index] = floor_val;
  }
}


template<typename Real>
__global__
static void _copy_cols(Real* dst, const Real *src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int index = reorder[i],
        dst_index = j * dst_dim.stride + i;
    if (index >= 0) {
      int src_index = j * src_stride + reorder[i];
      Real val = src[src_index];
      dst[dst_index] = val;
    } else {
      dst[dst_index] = 0.0;
    }
  }
}

template<typename Real>
__global__
static void _add_cols(Real* dst, const Real *src, const MatrixIndexT_cuda* reorder,
                      MatrixDim dst_dim, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int index = reorder[i],
        dst_index = j * dst_dim.stride + i;
    if (index >= 0) {
      int src_index = j * src_stride + index;
      Real val = src[src_index];
      dst[dst_index] += val;
    }
  }
}

template<typename Real>
__global__
static void _copy_rows(Real* dst, const Real *src, const MatrixIndexT_cuda* reorder,
                       MatrixDim dst_dim, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int index = reorder[j],
        dst_index = j * dst_dim.stride + i;
    if (index >= 0) {
      int src_index = reorder[j] * src_stride + i;
      Real val = src[src_index];
      dst[dst_index] = val;
    } else {
      dst[dst_index] = 0;
    }
  }
}

template<typename Real>
__global__
static void _copy_rows(Real* dst, const Real *const *src, MatrixDim dst_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int dst_index = j * dst_dim.stride + i;
    const Real *pointer = src[j];
    if (pointer != NULL) {
      dst[dst_index] = pointer[i];
    } else {
      dst[dst_index] = 0;
    }
  }
}

template<typename Real>
__global__
static void _copy_to_rows(Real* const* dst,
                          const Real *src, MatrixDim src_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < src_dim.cols && j < src_dim.rows) {
    Real *pointer = dst[j];
    if (pointer != NULL) {
      pointer[i] = src[j * src_dim.stride + i];
    }
  }
}

template<typename Real>
__global__
static void _add_rows(Real alpha, Real* dst, const Real *src,
                     const MatrixIndexT_cuda* reorder,
                     MatrixDim dst_dim, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int dst_index = j * dst_dim.stride + i;
    if (reorder[j] >= 0) {
      int src_index = reorder[j] * src_stride + i;
      dst[dst_index] += alpha * src[src_index];
    }
  }
}

template<typename Real>
__global__
static void _add_rows(Real alpha,
                      Real* dst, const Real *const *src, MatrixDim dst_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int dst_index = j * dst_dim.stride + i;
    if (src[j] != NULL) {
      dst[dst_index] += alpha * src[j][i];
    }
  }
}

template<typename Real>
__global__
static void _add_to_rows(Real alpha,
                         Real* const* dst, const Real *src, MatrixDim src_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < src_dim.cols && j < src_dim.rows) {
    if (dst[j] != NULL) {
      dst[j][i] += alpha * src[j * src_dim.stride + i];
    }
  }
}

template<typename Real>
__global__
static void _apply_ceiling(Real* mat, Real ceiling_val, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j * d.stride;

  if (i < d.cols && j < d.rows ) {
    if (mat[index] > ceiling_val)
      mat[index] = ceiling_val;
  }
}



template<typename Real>
__global__
static void _invert_elements(Real* data, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    data[index] = 1.0/data[index];
}


// matrix-wise, do data = alpha * data + beta * A * B^T,
// where B is a block matrix.
template<typename Real>
__global__
static void _add_mat_blockmat_trans(Real *data, MatrixDim dim, const Real *A_data, int A_num_rows, int A_num_cols,
                                    int A_row_stride, int A_col_stride, const CuBlockMatrixData *B_cu_data,
                                    int B_num_blocks, Real alpha, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index into "data"
  int j = blockIdx.y * blockDim.y + threadIdx.y; // block-index into B.
  if (i >= A_num_rows || j >= B_num_blocks) return;

  const CuBlockMatrixData &cu_data = B_cu_data[j];

  // BT means B transposed.
  int BT_row_start = cu_data.col_offset,
      BT_col_start = cu_data.row_offset,
      BT_num_rows = cu_data.matrix_dim.cols,
      BT_num_cols = cu_data.matrix_dim.rows,
      BT_col_stride = cu_data.matrix_dim.stride;
  const Real *B_data = static_cast<Real*>(cu_data.matrix_data); // Cast from void;
  // we avoided a bunch of hassle by doing this (relates to Ansi-C requirement).

  for (int k = 0; k < BT_num_cols; k++) {
    const Real *this_BT_col = B_data + k * BT_col_stride;
    const Real *this_A_row = A_data + i * A_row_stride + BT_row_start * A_col_stride;
    // this_A_row points to the element A[i][BT_row_start], it's really just
    // part of this row of A.
    Real sum = 0.0;
    for (int l = 0; l < BT_num_rows; l++) // l indexes rows of B.
      sum += this_BT_col[l] * this_A_row[l * A_col_stride];

    int index = i * dim.stride + (k + BT_col_start);
    data[index] = alpha * sum + beta * data[index];
  }
}

template<typename Real>
__global__
static void _add_mat_blockmat(Real *data, MatrixDim dim, const Real *A_data, int A_num_rows, int A_num_cols,
                              int A_row_stride, int A_col_stride, const CuBlockMatrixData *B_cu_data,
                              int B_num_blocks, Real alpha, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index into "data"
  int j = blockIdx.y * blockDim.y + threadIdx.y; // block-index into B.
  if (i >= A_num_rows || j >= B_num_blocks) return;

  const CuBlockMatrixData &block_data = B_cu_data[j];

  int B_row_start = block_data.row_offset,
      B_col_start = block_data.col_offset,
      B_num_rows = block_data.matrix_dim.rows,
      B_num_cols = block_data.matrix_dim.cols,
      B_row_stride = block_data.matrix_dim.stride;
  const Real *B_data = static_cast<Real*>(block_data.matrix_data); // Cast from void;
  // we avoided a bunch of hassle by doing this (relates to Ansi-C requirement).

  for (int k = 0; k < B_num_cols; k++) {
    const Real *this_B_col = B_data + k;
    const Real *this_A_row = A_data + i * A_row_stride + B_row_start * A_col_stride;
    // this_A_row points to the element A[i][B_row_start], it's really just
    // part of this row of A.
    Real sum = 0.0;
    for (int l = 0; l < B_num_rows; l++) // l indexes rows of B.
      sum += this_B_col[l * B_row_stride] * this_A_row[l * A_col_stride];

    int index = i * dim.stride + (k + B_col_start);
    data[index] = alpha * sum + beta * data[index];
  }
}



// For a block matrix B, does B = alpha * C * D + beta * B.
// the (x,y,z) indices are the block index, then the row
// and column indices within the block.  Note: transposition of C and D
// is handled by swapping the (num_rows,num_cols) and (row_stride,col_stride),
// so it's invisible to this code.  The num-cols and num-rows of C and D
// are only provided to the extent that they are not already determined
// by other quantities.
template<typename Real>
__global__
static void _block_add_mat_mat(CuBlockMatrixData *B_cu_data, int num_blocks,
                               const Real *C_data, int C_num_cols,
                               int C_row_stride, int C_col_stride,
                               const Real *D_data,
                               int D_row_stride, int D_col_stride,
                               Real alpha, Real beta) {
  int b = blockIdx.x * blockDim.x + threadIdx.x; // block-index into B.
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row-index into b'th block
  int j = blockIdx.z * blockDim.z + threadIdx.z; // col-index into b'th block
  if (b >= num_blocks) return;

  const CuBlockMatrixData &block_data = B_cu_data[b];

  if (i >= block_data.matrix_dim.rows || j >= block_data.matrix_dim.cols)
    return; // we're outside the dimensions of the b'th block.

  // B_elem is the element of B we're writing to.
  Real *B_elem = reinterpret_cast<Real*>(block_data.matrix_data) +
      i * block_data.matrix_dim.stride + j;

  Real B_val = *B_elem;

  // B_row and B_col are the (row, col) index into the full matrix B.
  int B_row = block_data.row_offset + i, B_col = block_data.col_offset + j;

  const Real *C_row_data = C_data + C_row_stride * B_row,
      *D_col_data = D_data + D_col_stride * B_col;

  Real sum = 0.0;
  for (int k = 0; k < C_num_cols; k++) {
    sum += C_row_data[k * C_col_stride] * D_col_data[k * D_row_stride];
  }
  *B_elem = alpha * sum + beta * B_val;
}


template<typename Real>
__global__
static void _blockadd_mat_blockmat_trans(Real *data, MatrixDim dim, const Real *A_data, int A_num_rows, int A_num_cols,
                                    int A_row_stride, int A_col_stride, const CuBlockMatrixData *B_cu_data,
                                    int B_num_blocks, Real alpha, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index into "data"
  int j = blockIdx.y * blockDim.y + threadIdx.y; // block-index into B.
  if (i >= A_num_rows || j >= B_num_blocks) return;

  const CuBlockMatrixData &cu_data = B_cu_data[j];

  // BT means B transposed.
  int BT_row_start = cu_data.col_offset,
      BT_col_start = cu_data.row_offset,
      BT_num_rows = cu_data.matrix_dim.cols,
      BT_num_cols = cu_data.matrix_dim.rows,
      BT_col_stride = cu_data.matrix_dim.stride;
  const Real *B_data = static_cast<Real*>(cu_data.matrix_data); // Cast from void;
  // we avoided a bunch of hassle by doing this (relates to Ansi-C requirement).

  for (int k = 0; k < BT_num_cols; k++) {
    const Real *this_BT_col = B_data + k * BT_col_stride;
    const Real *this_A_row = A_data + i * A_row_stride + BT_row_start * A_col_stride;
    // this_A_row points to the element A[i][BT_row_start], it's really just
    // part of this row of A.
    Real sum = 0.0;
    for (int l = 0; l < BT_num_rows; l++) // l indexes rows of B.
      sum += this_BT_col[l] * this_A_row[l * A_col_stride];

    int index = i * dim.stride + (k + BT_col_start);
    data[index] = alpha * sum + beta * data[index];
  }
}

template<typename Real>
__global__
static void _sum_column_ranges(Real *data, MatrixDim dim,
                               const Real *src_data,
                               MatrixDim src_dim,
                               const Int32Pair *indices) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= dim.rows || col >= dim.cols)
    return;
  int dst_index = row * dim.stride + col,
    src_start_index = row * src_dim.stride + indices[col].first,
      src_end_index = row * src_dim.stride + indices[col].second;
  Real sum = 0.0;
  for (int index = src_start_index; index < src_end_index; index++)
    sum += src_data[index];
  data[dst_index] = sum;
}

template<typename Real>
__global__
static void _add_row_ranges(Real *data, MatrixDim dim, const Real *src_data,
                            MatrixDim src_dim, const Int32Pair *indexes) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= dim.rows || col >= dim.cols)
    return;
  int dst_index = row * dim.stride + col;
  int src_index_start = indexes[row].first,
      src_index_end = indexes[row].second;
  for (int row_index = src_index_start; row_index < src_index_end;
       row_index++)
    data[dst_index] += src_data[row_index * src_dim.stride + col];
}

template<typename Real>
__global__
static void _soft_hinge(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride, src_index = i + j*src_stride;
  // compute the function y[index] = log(1 + exp(x[index]))
  if(i < d.cols && j < d.rows) {
    Real val = x[src_index], result;
    if (val >= 10.0) result = val; // function approaches y=x as x gets large
    else result = log1p(exp(val));
    y[dst_index] = result;
  }
}

template<typename Real>
__global__
static void _group_pnorm(Real *y, const Real *x, MatrixDim d, int src_stride,
			 int group_size, Real power) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < d.rows  && i < d.cols) {
    int dst_index = i + j * d.stride;
    Real tmp = 0;
    int src_begin_index = i * group_size + j * src_stride;
    int src_end_index = src_begin_index + group_size;
    for (int src_index = src_begin_index; src_index < src_end_index;
         src_index ++) {
      tmp += pow(std::abs(x[src_index]), power);
    }
    tmp = pow(tmp, Real(1.0 / power));
    if (!isnan(tmp)) {
      y[dst_index] = tmp;
    } else {
      Real max_value = x[src_begin_index], min_value = max_value;
      for (int src_index = src_begin_index + 1;
      	   src_index < src_end_index; src_index ++) {
        if (x[src_index] > max_value)
          max_value = x[src_index];
        if (x[src_index] < min_value)
          min_value = x[src_index];
      }
      tmp = 0.0;
      Real max_abs_value = (max_value > -min_value ?
                            max_value : -min_value); // let max_value be the
                                                     // largest abs(value)
      if (max_abs_value == 0) {
        y[dst_index] = 0.0;
      } else {
        for (int src_index = src_begin_index;
             src_index < src_end_index; src_index ++) {
          Real x_scaled = x[src_index] / max_abs_value;
          tmp += pow(std::abs(x_scaled), Real(power));
        }
        y[dst_index] = pow(tmp, Real(1.0 / power)) * max_abs_value;
      }
    }
  }
}

template<typename Real>
__global__
static void _group_max(Real *y, const Real *x, MatrixDim d, int src_stride,
                       int group_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < d.rows  && i < d.cols) {
    int dst_index = i + j * d.stride;
    int src_begin_index = i * group_size + j * src_stride;
    Real max_value = -1e20;
    int src_end_index = src_begin_index + group_size;
    for (int src_index = src_begin_index; src_index < src_end_index;
         src_index ++) {
      if (!isnan(x[src_index]) && x[src_index] > max_value)
        max_value = x[src_index];
    }
    y[dst_index] = max_value;
  }
}

/*
 * cu::
 */
template<typename Real>
__global__
static void _sigmoid(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride, src_index = i + j*src_stride;
  if(i < d.cols && j < d.rows) {
    Real res = 1.0 / (1.0 + exp(-x[src_index]));
    y[dst_index] = res;
  }
}

template<typename Real>
__global__
static void _diff_sigmoid(Real*eout, const Real*e, const Real*y, MatrixDim d, int e_stride, int y_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride;
  int e_index = i + j*e_stride;
  int y_index = i + j*y_stride;
  if (i < d.cols  && j < d.rows )
    eout[dst_index] = y[y_index]*(1.0-y[y_index]) * e[e_index];
}


template<typename Real>
__global__
static void _tanh(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride, src_index = i + j * src_stride;
  if(i < d.cols && j < d.rows) {
    Real exp_2x = exp(2.0*x[src_index]);
    Real res;
    if(isinf(exp_2x)) {
      res = 1.0;
    } else {
      res = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
    y[dst_index] = res;
  }
}


template<typename Real>
__global__
static void _diff_tanh(Real*eout, const Real*e, const Real*y, MatrixDim d, int e_stride, int y_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride;
  int e_index   = i + j*e_stride;
  int y_index   = i + j*y_stride;
  if (i < d.cols  && j < d.rows )
    eout[dst_index] = (1.0 - y[y_index]*y[y_index]) * e[e_index];
}

template<typename Real>
__global__
static void _softmax_reduce(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int j = blockIdx.x;
  int THREADS = blockDim.x;
  if (j >= d.rows) return;

  __shared__ Real aux[CU1DBLOCK];
  int steps = (d.cols - 1) / THREADS + 1;

  //copy input to aux
  aux[threadIdx.x] = x[threadIdx.x+j*d.stride];
  for(int i=1; i<steps; ++i) {
    if(threadIdx.x+i*THREADS < d.cols && aux[threadIdx.x] < x[threadIdx.x+i*THREADS+j*d.stride])
	aux[threadIdx.x] = x[threadIdx.x+i*THREADS+j*d.stride];
  }

  //get the maximum value
  int nTotalThreads = THREADS;
  __syncthreads();
  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);   // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      if(threadIdx.x+halfPoint < nTotalThreads && aux[threadIdx.x] < aux[threadIdx.x+halfPoint])
        aux[threadIdx.x] = aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);   // divide by two.
  }
  Real max = aux[0];
  __syncthreads();

   // subtract max, apply exp, sum up...
  y[threadIdx.x+j*d.stride] = exp(x[threadIdx.x+j*d.stride] - max);
  aux[threadIdx.x] = y[threadIdx.x+j*d.stride];
  for(int i=1; i<steps; i++) {
    if(threadIdx.x+i*THREADS < d.cols) {
      y[threadIdx.x+i*THREADS+j*d.stride] = exp(x[threadIdx.x+i*THREADS+j*d.stride] - max);
      aux[threadIdx.x] += y[threadIdx.x+i*THREADS+j*d.stride];
    }
  }
  nTotalThreads = THREADS;
  __syncthreads();
  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);   // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      if(threadIdx.x+halfPoint < nTotalThreads)
        aux[threadIdx.x] += aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);   // divide by two.
  }
  Real sum = aux[0];
  __syncthreads();

  //normalize by sum...
  for(int i=0; i<steps; i++) {
    if(threadIdx.x+i*THREADS < d.cols) {
      y[threadIdx.x+i*THREADS+j*d.stride] = y[threadIdx.x+i*THREADS+j*d.stride] / sum;
    }
  }

}

template<typename Real>
__global__
static void _log_softmax_reduce(Real *y, const Real *x,
                                MatrixDim d, int src_stride) {
  int j = blockIdx.x;
  int THREADS = blockDim.x;
  if (j >= d.rows) return;

  __shared__ Real aux[CU1DBLOCK];
  int steps = (d.cols - 1) / THREADS + 1;

  // Maximum step 1: loads input data to <aux>. If <d.cols> is larger than
  //                 <blockDim.x>, then we do a first pass filtering and only
  //                 keep a <blockDim.x> size array.
  aux[threadIdx.x] = x[threadIdx.x + j * d.stride];
  for (int i = 1; i < steps; ++i) {
    if (threadIdx.x + i * THREADS < d.cols
        && aux[threadIdx.x] < x[threadIdx.x + i * THREADS + j * d.stride])
      aux[threadIdx.x] = x[threadIdx.x + i * THREADS + j * d.stride];
  }

  // Maximum step 2: the standard max reduce.
  int nTotalThreads = THREADS;
  __syncthreads();
  while (nTotalThreads > 1) {
    int halfPoint = ((1 + nTotalThreads) >> 1);
    if (threadIdx.x < halfPoint) {
      if (threadIdx.x + halfPoint < nTotalThreads
          && aux[threadIdx.x] < aux[threadIdx.x + halfPoint])
        aux[threadIdx.x] = aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1 + nTotalThreads) >> 1);
  }
  Real max = aux[0];
  __syncthreads();

  // Log sum step 1: substracts max, and takes exponentials.
  y[threadIdx.x + j * d.stride] = x[threadIdx.x + j * d.stride] - max;
  aux[threadIdx.x] = exp(y[threadIdx.x + j * d.stride]);
  for (int i = 1; i < steps; ++i) {
    if (threadIdx.x + i * THREADS < d.cols) {
      y[threadIdx.x + i * THREADS + j * d.stride] =
        x[threadIdx.x + i * THREADS + j * d.stride] - max;
      aux[threadIdx.x] += exp(y[threadIdx.x + i * THREADS + j * d.stride]);
    }
  }

  // Log sum step 2: comptes summation and then takes logarithm.
  nTotalThreads = THREADS;
  __syncthreads();
  while (nTotalThreads > 1) {
    int halfPoint = ((1 + nTotalThreads) >> 1);
    if (threadIdx.x < halfPoint)  {
      if (threadIdx.x + halfPoint < nTotalThreads)
        aux[threadIdx.x] += aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1 + nTotalThreads) >> 1);
  }
  Real log_sum = log(aux[0]);
  __syncthreads();

  // Computes log softmax.
  for (int i = 0; i < steps; ++i) {
    if (threadIdx.x + i * THREADS < d.cols) {
      y[threadIdx.x + i * THREADS + j * d.stride] -= log_sum;
    }
  }
}


template<typename Real>
__global__
static void _splice(Real* y, const Real* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if (i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_col = i % d_in.cols;
    int32_cuda src_row = j + off[i / d_in.cols];
    if(src_row < 0) src_row = 0;
    if(src_row >= d_in.rows) src_row = d_in.rows-1;
    y[index] = x[src_col + src_row*d_in.stride];
  }
}

template<typename Real>
__global__
static void _take_mean(const Real* x, Real* y, MatrixDim d_in) {
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
static void _take_lower(const Real* x, Real* y, MatrixDim d_in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index
  if (j > i || i >= d_in.rows) return;
  int index = i * d_in.stride + j;
  Real val = x[index];
  int index_sp = (i * (i+1) / 2) + j;
  y[index_sp] = val;
}

template<typename Real>
__global__
static void _take_upper(const Real* x, Real* y, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // col-index
  if (j < i  || j >= d_in.rows) return;
  int32_cuda index = i * d_in.stride + j;
  int32_cuda index_sp = (j * (j+1) / 2) + i;
  y[index_sp] = x[index];
}

template<typename Real>
__global__
static void _vec_copy_diag_from_packed(Real* y, const Real* x, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i+1) * (i+2) / 2) - 1;
  if (i < dim) {
     y[i] = x[index];
  }
}

template<typename Real>
__global__
static void _copy_from_sp(const Real* x, Real* y, MatrixDim dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  //
  if (i < dim.cols && j < dim.rows) {
    int dst_index = i + j * dim.stride, src_index;
    if (j <= i) {  // no transpose
      src_index = (i * (i+1) / 2) + j;
    } else { // transpose.
      src_index = (j * (j+1) / 2) + i;
    }
    y[dst_index] = x[src_index];
  }
}

template<typename Real>
__global__
static void _copy(Real* y, const Real* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if (i < d_out.cols  && j < d_out.rows ) {
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
  if (i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_row = copy_from[j];
    y[index] = x[i + src_row*d_in.stride];
  }
}


template<typename Real>
__global__
static void _regularize_l1(Real* wei, Real* grad, Real l1, Real lr, MatrixDim d, int stride_grad) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride,
             grad_index = i + j*stride_grad;
  if (i < d.cols && j < d.rows) {

    if(wei[index]==0.0) return; //skip L1 if zero weight!

    Real l1_signed = l1;
    if(wei[index] < 0.0) //flip sign
      l1_signed = -l1;

    Real before = wei[index];
    Real after = wei[index] -lr*grad[grad_index] -l1_signed;//simulate update
    if((after > 0.0) ^ (before > 0.0)) { //sign changed?
      wei[index] = 0.0;
      grad[grad_index] = 0.0;
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

  __shared__ Real value[CU1DBLOCK];
  __shared__ int32_cuda index[CU1DBLOCK];

  //copy to shared memory
  value[threadIdx.x] = mat[i+j*d.stride];
  index[threadIdx.x] = threadIdx.x;
  __syncthreads();

  //get the id of the max value
  int32_cuda out_max = _max_id_reduce(value, index);
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
void cudaF_copy_upp_low(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) { _copy_upp_low<<<Gr,Bl>>>(A,dimA); }
void cudaF_copy_low_upp(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) { _copy_low_upp<<<Gr,Bl>>>(A,dimA); }
void cudaF_add_diag_vec_mat(dim3 Gr, dim3 Bl, float alpha, float *mat, MatrixDim mat_dim,
                            const float *vec, const float *mat2, int mat2_row_stride,
                            int mat2_col_stride, float beta) {
  _add_diag_vec_mat<<<Gr,Bl>>>(alpha, mat, mat_dim, vec, mat2, mat2_row_stride,
                               mat2_col_stride, beta);
}

void cudaF_copy_from_tp_trans(dim3 Gr, dim3 Bl, float* A, const float* B, MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}
void cudaFD_copy_from_tp_trans(dim3 Gr, dim3 Bl, float* A, const double* B, MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}

void cudaF_copy_from_tp(dim3 Gr, dim3 Bl, float* A, const float* B, MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}
void cudaFD_copy_from_tp(dim3 Gr, dim3 Bl, float* A, const double* B, MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}

void cudaF_transpose_matrix(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _transpose_matrix<<<Gr,Bl>>>(mat, d);
}

void cudaF_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}

void cudaF_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}

void cudaF_apply_pow_abs(dim3 Gr, dim3 Bl, float* mat, float power, bool include_sign, MatrixDim d) {
  _apply_pow_abs<<<Gr,Bl>>>(mat, power, include_sign, d);
}

void cudaF_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_heaviside<<<Gr,Bl>>>(mat, d);
}

void cudaF_copy_cols(dim3 Gr, dim3 Bl, float* dst, const float* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _copy_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaF_add_cols(dim3 Gr, dim3 Bl, float* dst, const float* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _add_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaF_copy_rows(dim3 Gr, dim3 Bl, float* dst, const float* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _copy_rows<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaF_copy_rows_direct(dim3 Gr, dim3 Bl, float* dst, const float* const* src, MatrixDim dst_dim) {
  _copy_rows<<<Gr,Bl>>>(dst, src, dst_dim);
}

void cudaF_copy_to_rows_direct(dim3 Gr, dim3 Bl, float* const* dst, const float* src, MatrixDim src_dim) {
  _copy_to_rows<<<Gr,Bl>>>(dst, src, src_dim);
}

void cudaF_add_rows(dim3 Gr, dim3 Bl, float alpha, float* dst, const float* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, reorder, dst_dim, src_stride);
}

void cudaF_add_rows_direct(dim3 Gr, dim3 Bl, float alpha, float* dst, const float* const* src, MatrixDim dst_dim) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, dst_dim);
}

void cudaF_add_to_rows_direct(dim3 Gr, dim3 Bl, float alpha, float* const* dst, const float* src, MatrixDim src_dim) {
  _add_to_rows<<<Gr,Bl>>>(alpha, dst, src, src_dim);
}

void cudaF_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}

void cudaF_apply_ceiling(dim3 Gr, dim3 Bl, float* mat, float ceiling_val, MatrixDim d) {
  _apply_ceiling<<<Gr,Bl>>>(mat, ceiling_val, d);
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

void cudaF_scale_diag_packed(int Gr, int Bl, float* mat, float value, int dim) {
  _scale_diag_packed<<<Gr,Bl>>>(mat,value,dim);
}

void cudaF_scale(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _scale<<<Gr,Bl>>>(mat,value,d);
}

void cudaF_apply_log(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d);
}

void cudaF_mul_elements(dim3 Gr, dim3 Bl, float* mat, const float* A, MatrixDim dst_d, int src_stride) {
  _mul_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaF_div_elements(dim3 Gr, dim3 Bl, float* mat, const float* A, MatrixDim dst_d, int src_stride) {
  _div_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaF_max(dim3 Gr, dim3 Bl, float* mat, const float* A, MatrixDim dst_d, int src_stride) {
  _max<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaF_mul_cols_vec(dim3 Gr, dim3 Bl, float* mat, const float* scale, MatrixDim d) {
  _mul_cols_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaF_mul_rows_vec(dim3 Gr, dim3 Bl, float* mat, const float* scale, MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaF_mul_rows_group_mat(dim3 Gr, dim3 Bl, float *y, const float *x,
			      MatrixDim d, int src_stride, int group_size) {
  _mul_rows_group_mat<<<Gr,Bl>>>(y, x, d, src_stride, group_size);
}

void cudaF_calc_pnorm_deriv(dim3 Gr, dim3 Bl, float *y, const float *x1,
			    const float *x2, MatrixDim d, int src_stride,
			    int group_size, float power) {
  _calc_pnorm_deriv<<<Gr,Bl>>>(y, x1, x2, d, src_stride, group_size, power);
}

void cudaF_calc_group_max_deriv(dim3 Gr, dim3 Bl, float *y, const float *x1,
			        const float *x2, MatrixDim d, int src_stride,
			        int group_size) {
  _calc_group_max_deriv<<<Gr,Bl>>>(y, x1, x2, d, src_stride, group_size);
}

void cudaF_div_rows_vec(dim3 Gr, dim3 Bl, float* mat, const float* vec_div, MatrixDim d) {
  _div_rows_vec<<<Gr,Bl>>>(mat, vec_div, d);
}

void cudaF_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* src, float* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_trans<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  } else {
    _add_mat<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  }
}

void cudaF_add_mat_blocks(dim3 Gr, dim3 Bl, float alpha, const float* src, int32_cuda num_row_blocks, int32_cuda num_col_blocks, float* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_blocks_trans<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks, dst, d, src_stride);
  } else {
    _add_mat_blocks<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks, dst, d, src_stride);
  }
}

void cudaF_add_mat_mat_div_mat(dim3 Gr, dim3 Bl, const float *A, const float *B, const float *C, float *dst, MatrixDim d, int stride_a, int stride_b, int stride_c) {
  _add_mat_mat_div_mat<<<Gr,Bl>>>(A,B,C,dst,d, stride_a, stride_b, stride_c);
}

void cudaF_sy_add_tr2(dim3 Gr, dim3 Bl, float alpha, float beta, const float* T, MatrixDim tdim,
                      float *S, MatrixDim sdim) {
  _sy_add_tr2<<<Gr,Bl>>>(alpha, beta, T, tdim, S, sdim);
}

void cudaF_add_vec_to_cols(dim3 Gr, dim3 Bl, float alpha, const float* col, float beta, float* dst, MatrixDim d) {
  _add_vec_to_cols<<<Gr,Bl>>>(alpha,col,beta,dst,d);
}


void cudaF_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float* row, float beta, float* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d);
}

void cudaF_add_mat_diag_vec(dim3 Gr, dim3 Bl, float alpha, float *mat, MatrixDim mat_dim, const float *mat2, int mat2_row_stride, int mat2_col_stride, const float *vec,  float beta) {
  _add_mat_diag_vec<<<Gr,Bl>>>(alpha, mat, mat_dim, mat2, mat2_row_stride, mat2_col_stride, vec, beta);
}

void cudaF_add_mat_mat_elements(dim3 Gr, dim3 Bl, float *data, const float *srcA_data, const float *srcB_data, MatrixDim dim, int srcA_stride, int srcB_stride, float alpha, float beta) {
    _add_mat_mat_elements<<<Gr, Bl>>>(data, srcA_data, srcB_data, dim, srcA_stride, srcB_stride, alpha, beta);
}


// CURRENTLY UNUSED...
void cudaF_apply_mask(dim3 Gr, dim3 Bl, float* mat, const char* mask, MatrixDim dmat, MatrixDim dmask) {
  _apply_mask<<<Gr,Bl>>>(mat,mask,dmat,dmask);
}


/*
 * CuVector
 */

void cudaF_replace_value(int Gr, int Bl, float *v, int dim, float orig, float changed) {
  _replace_value<<<Gr,Bl>>>(v, dim, orig, changed);
}

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

void cudaF_vec_min(const float* v, float* value, int dim) {
  _vec_min<<<1,CU1DBLOCK>>>(v, value, dim);
}

void cudaF_vec_max(const float* v, float* value, int dim) {
  _vec_max<<<1,CU1DBLOCK>>>(v, value, dim);
}

void cudaF_trace_mat_mat_trans(const float* A, const float* B, MatrixDim dA, int B_stride, float* value) {
  _trace_mat_mat_trans<float,4> <<<4,CU1DBLOCK>>>(A,B,dA,B_stride,value);
}

void cudaF_trace_mat_mat(const float* A, const float* B, MatrixDim dA, int B_stride, float* value) {
  _trace_mat_mat<float,2> <<<2,CU1DBLOCK>>>(A,B,dA,B_stride,value);
}


void cudaF_add_diag_mat_mat(int Gr, int Bl, float alpha, float* v, int v_dim, const float* M,
     int M_cols, int M_row_stride, int M_col_stride, const float *N, int N_row_stride,
                            int N_col_stride, int threads_per_element, float beta) {
   _add_diag_mat_mat<<<Gr,Bl>>>(alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride,
                                N, N_row_stride, N_col_stride, threads_per_element, beta);
}

void cudaF_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}

void cudaF_vec_sum(int Gr, int Bl, float* v, float* value, int dim, int inc) {
  _vec_sum<<<Gr,Bl>>>(v, value, dim, inc);
}

void cudaF_pvec_sum(int Gr, int Bl, float* v, float* pvec_sum, int dim, int size) {
  _pvec_sum<<<Gr,Bl>>>(v, pvec_sum, dim, size);
}

void cudaF_matrix_add_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim dim, float alpha, MatrixElement<float>* x, int num_elements) {
  _cuda_matrix_add_elements<<<Gr, Bl>>>(data, dim, alpha, x, num_elements);
}

void cudaF_matrix_add_indexed_values(dim3 Gr, dim3 Bl, MatrixDim dim, float alpha, const Int32Pair* indices, const float* x, int s, float* data) {
  _cuda_matrix_add_indexed_values<<<Gr, Bl>>>(dim, alpha, indices, x, s, data);
}

void cudaF_comp_obj_deriv(dim3 Gr, dim3 Bl, MatrixElement<float>* x, int s, const float* z, MatrixDim d, float* z2, MatrixDim d2, float* t) {
  _cuda_comp_obj_deriv<<<Gr,Bl>>>(x,s,z,d,z2,d2,t);
}

void cudaD_comp_obj_deriv(dim3 Gr,dim3 Bl, MatrixElement<double>* x, int s, const double* z, MatrixDim d, double* z2, MatrixDim d2, double* t) {
  _cuda_comp_obj_deriv<<<Gr,Bl>>>(x,s,z,d,z2,d2,t);
}

void cudaF_vec_copy_diag_from_packed(int Gr, int Bl, float *dst, const float *src, int dim) {
  _vec_copy_diag_from_packed<<<Gr,Bl>>>(dst,src,dim);
}

void cudaF_vec_apply_floor(int Gr, int Bl, float* v, float floor_val, float *count, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,count,dim);
}

void cudaF_vec_apply_ceiling(int Gr, int Bl, float* v, float ceiling_val, float *count, int dim) {
  _vec_apply_ceiling<<<Gr,Bl>>>(v, ceiling_val,count,dim);
}

void cudaF_vec_apply_exp(int Gr, int Bl, float* v, int dim) {
  _vec_apply_exp<<<Gr,Bl>>>(v,dim);
}

void cudaF_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim) {
  _vec_apply_log<<<Gr,Bl>>>(v,flag,dim);
}

void cudaF_invert_elements(dim3 Gr, dim3 Bl, float* data, MatrixDim d) {
  _invert_elements<<<Gr,Bl>>>(data, d);
}


void cudaF_add_mat_blockmat(dim3 Gr, dim3 Bl, float *data, MatrixDim d, const float *Adata,
                            int A_num_rows, int A_num_cols, int A_row_stride, int A_col_stride,
                            const CuBlockMatrixData *B_cu_data, int B_num_blocks,
                            float alpha, float beta, int B_trans) {
  if (B_trans) {
    _add_mat_blockmat_trans<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
                                       A_row_stride, A_col_stride, B_cu_data,
                                       B_num_blocks, alpha, beta);
  } else {
    _add_mat_blockmat<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
                                 A_row_stride, A_col_stride, B_cu_data,
                                 B_num_blocks, alpha, beta);

  }
}

void cudaF_block_add_mat_mat(dim3 Gr, dim3 Bl, CuBlockMatrixData *B_cu_data, int num_blocks,
                             const float *C_data, int C_num_cols, int C_row_stride, int C_col_stride,
                             const float *D_data, int D_row_stride, int D_col_stride,
                             float alpha, float beta) {
  _block_add_mat_mat<<<Gr,Bl>>>(B_cu_data, num_blocks, C_data, C_num_cols,
                                C_row_stride, C_col_stride, D_data, D_row_stride,
                                D_col_stride, alpha, beta);
}

/*
 * cu::
 */
void cudaF_soft_hinge (dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _soft_hinge<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_group_pnorm(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride, int group_size, float power) {
  _group_pnorm<<<Gr,Bl>>>(y, x, d, src_stride, group_size, power);
}

void cudaF_group_max(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride, int group_size) {
  _group_max<<<Gr,Bl>>>(y, x, d, src_stride, group_size);
}

void cudaF_sigmoid (dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _sigmoid<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_diff_sigmoid (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaF_tanh (dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _tanh<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_diff_tanh (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaF_softmax_reduce (size_t Gr, size_t Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_log_softmax_reduce (size_t Gr, size_t Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _log_softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_splice(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in);
}

void cudaF_one(int Gr, int Bl, float* x, int dim) {
  _one<<<Gr,Bl>>>(x,dim);
}

void cudaF_take_mean(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in) {
  _take_mean<<<Gr,Bl>>>(x,y,d_in);
}

void cudaF_take_lower(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in) {
  _take_lower<<<Gr,Bl>>>(x,y,d_in);
}

void cudaF_take_upper(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in) {
  _take_upper<<<Gr,Bl>>>(x,y,d_in);
}

void cudaF_copy_from_sp(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim dim) {
  _copy_from_sp<<<Gr,Bl>>>(x, y, dim);
}

void cudaF_copy(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaF_randomize(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}


void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float* wei, float* grad, float l1, float lr, MatrixDim d, int stride_grad) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d,stride_grad);
}

void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float* mat, float* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, voff, d);
}

void cudaF_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda* vec_tgt, float* mat_net_out, float* vec_log_post, MatrixDim d) {
  _diff_xent<<<Gr,Bl>>>(vec_tgt,mat_net_out,vec_log_post,d);
}

void cudaF_copy_rows_from_vec(dim3 Gr, dim3 Bl, float *mat_out, MatrixDim d_out, const float *v_in) {
  _copy_rows_from_vec<<<Gr,Bl>>>(mat_out, d_out, v_in);
}

void cudaF_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const float* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_df<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaF_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_fd<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaF_sum_column_ranges(dim3 Gr, dim3 Bl, float *data, MatrixDim dim,
                             const float *src_data, MatrixDim src_dim,
                             const Int32Pair *indices) {
  _sum_column_ranges<<<Gr,Bl>>>(data, dim, src_data, src_dim, indices);
}

void cudaF_add_row_ranges(dim3 Gr, dim3 Bl, float *data, MatrixDim dim,
                          const float *src_data, MatrixDim src_dim,
                          const Int32Pair *indexes) {
  _add_row_ranges<<<Gr,Bl>>>(data, dim, src_data, src_dim, indexes);
}

void cudaF_matrix_lookup(dim3 Gr, dim3 Bl, const float *data, MatrixDim dim,
                         const Int32Pair *indices, int indices_size,
                         float *output) {
  _matrix_lookup<<<Gr,Bl>>>(data, dim, indices, indices_size, output);
}

void cudaF_equal_element_mask(dim3 Gr, dim3 Bl, const float *mat1,
                              const float *mat2, float *mask, MatrixDim mat1_dim,
                              int mat2_stride, int mask_stride) {
  _equal_element_mask<<<Gr,Bl>>>(mat1, mat2, mask, mat1_dim, mat2_stride, mask_stride);
}

/*
 * "double"
 */

/*
 * CuMatrix
 */
void cudaD_copy_upp_low(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) { _copy_upp_low<<<Gr,Bl>>>(A,dimA); }
void cudaD_copy_low_upp(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) { _copy_low_upp<<<Gr,Bl>>>(A,dimA); }
void cudaD_add_diag_vec_mat(dim3 Gr, dim3 Bl, double alpha, double *mat, MatrixDim mat_dim,
                            const double *vec, const double *mat2, int mat2_row_stride,
                            int mat2_col_stride, double beta) {
  _add_diag_vec_mat<<<Gr,Bl>>>(alpha, mat, mat_dim, vec, mat2, mat2_row_stride,
                               mat2_col_stride, beta);
}

void cudaD_copy_from_tp_trans(dim3 Gr, dim3 Bl, double* A, const double* B, MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}
void cudaDF_copy_from_tp_trans(dim3 Gr, dim3 Bl, double* A, const float* B, MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}

void cudaD_copy_from_tp(dim3 Gr, dim3 Bl, double* A, const double* B, MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}
void cudaDF_copy_from_tp(dim3 Gr, dim3 Bl, double* A, const float* B, MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}

void cudaD_transpose_matrix(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _transpose_matrix<<<Gr,Bl>>>(mat, d);
}

void cudaD_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}

void cudaD_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}

void cudaD_apply_pow_abs(dim3 Gr, dim3 Bl, double* mat, double power, bool include_sign, MatrixDim d) {
  _apply_pow_abs<<<Gr,Bl>>>(mat, power, include_sign, d);
}

void cudaD_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_heaviside<<<Gr,Bl>>>(mat, d);
}

void cudaD_copy_cols(dim3 Gr, dim3 Bl, double* dst, const double* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _copy_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaD_add_cols(dim3 Gr, dim3 Bl, double* dst, const double* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _add_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaD_copy_rows(dim3 Gr, dim3 Bl, double* dst, const double* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _copy_rows<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaD_copy_rows_direct(dim3 Gr, dim3 Bl, double* dst, const double* const* src, MatrixDim dst_dim) {
  _copy_rows<<<Gr,Bl>>>(dst, src, dst_dim);
}

void cudaD_copy_to_rows_direct(dim3 Gr, dim3 Bl, double* const* dst, const double* src, MatrixDim src_dim) {
  _copy_to_rows<<<Gr,Bl>>>(dst, src, src_dim);
}

void cudaD_add_rows(dim3 Gr, dim3 Bl, double alpha, double* dst, const double* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, reorder, dst_dim, src_stride);
}

void cudaD_add_rows_direct(dim3 Gr, dim3 Bl, double alpha, double* dst, const double* const* src, MatrixDim dst_dim) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, dst_dim);
}

void cudaD_add_to_rows_direct(dim3 Gr, dim3 Bl, double alpha, double* const* dst, const double* src, MatrixDim src_dim) {
  _add_to_rows<<<Gr,Bl>>>(alpha, dst, src, src_dim);
}

void cudaD_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}

void cudaD_apply_ceiling(dim3 Gr, dim3 Bl, double* mat, double ceiling_val, MatrixDim d) {
  _apply_ceiling<<<Gr,Bl>>>(mat, ceiling_val, d);
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

void cudaD_scale_diag_packed(int Gr, int Bl, double* mat, double value, int dim) {
  _scale_diag_packed<<<Gr,Bl>>>(mat,value,dim);
}

void cudaD_scale(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _scale<<<Gr,Bl>>>(mat,value,d);
}

void cudaD_apply_log(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d);
}

void cudaD_mul_elements(dim3 Gr, dim3 Bl, double* mat, const double* A, MatrixDim dst_d, int src_stride) {
  _mul_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaD_div_elements(dim3 Gr, dim3 Bl, double* mat, const double* A, MatrixDim dst_d, int src_stride) {
  _div_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaD_max(dim3 Gr, dim3 Bl, double* mat, const double* A, MatrixDim dst_d, int src_stride) {
  _max<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaD_mul_cols_vec(dim3 Gr, dim3 Bl, double* mat, const double* scale, MatrixDim d) {
  _mul_cols_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaD_mul_rows_vec(dim3 Gr, dim3 Bl, double* mat, const double* scale, MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaD_mul_rows_group_mat(dim3 Gr, dim3 Bl, double* y, const double* x,
			      MatrixDim d, int src_stride, int group_size) {
  _mul_rows_group_mat<<<Gr,Bl>>>(y, x, d, src_stride, group_size);
}

void cudaD_calc_pnorm_deriv(dim3 Gr, dim3 Bl, double*y, const double* x1,
			    const double* x2, MatrixDim d, int src_stride,
			    int group_size, double power) {
  _calc_pnorm_deriv<<<Gr,Bl>>>(y, x1, x2, d, src_stride, group_size, power);
}

void cudaD_calc_group_max_deriv(dim3 Gr, dim3 Bl, double*y, const double* x1,
			        const double* x2, MatrixDim d, int src_stride,
			        int group_size) {
  _calc_group_max_deriv<<<Gr,Bl>>>(y, x1, x2, d, src_stride, group_size);
}

void cudaD_div_rows_vec(dim3 Gr, dim3 Bl, double* mat, const double* vec_div, MatrixDim d) {
  _div_rows_vec<<<Gr,Bl>>>(mat, vec_div, d);
}

void cudaD_add_mat(dim3 Gr, dim3 Bl, double alpha, const double* src, double* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_trans<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  } else {
    _add_mat<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  }
}

void cudaD_add_mat_blocks(dim3 Gr, dim3 Bl, double alpha, const double* src, int32_cuda num_row_blocks, int32_cuda num_col_blocks, double* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_blocks_trans<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks, dst, d, src_stride);
  } else {
    _add_mat_blocks<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks, dst, d, src_stride);
  }
}

void cudaD_add_mat_mat_div_mat(dim3 Gr, dim3 Bl, const double *A, const double *B, const double *C, double *dst, MatrixDim d, int stride_a, int stride_b, int stride_c) {
  _add_mat_mat_div_mat<<<Gr,Bl>>>(A,B,C,dst,d,stride_a,stride_b,stride_c);
}

void cudaD_sy_add_tr2(dim3 Gr, dim3 Bl, double alpha, double beta, const double* T, MatrixDim tdim,
                      double *S, MatrixDim sdim) {
  _sy_add_tr2<<<Gr,Bl>>>(alpha, beta, T, tdim, S, sdim);
}

void cudaD_add_vec_to_cols(dim3 Gr, dim3 Bl, double alpha, const double* col, double beta, double* dst, MatrixDim d) {
  _add_vec_to_cols<<<Gr,Bl>>>(alpha,col,beta,dst,d);
}

void cudaD_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double* row, double beta, double* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d);
}

void cudaD_add_mat_diag_vec(dim3 Gr, dim3 Bl, double alpha, double *mat, MatrixDim mat_dim, const double *mat2, int mat2_row_stride, int mat2_col_stride, const double *vec,  double beta) {
  _add_mat_diag_vec<<<Gr,Bl>>>(alpha, mat, mat_dim, mat2, mat2_row_stride, mat2_col_stride, vec, beta);
}

void cudaD_add_mat_mat_elements(dim3 Gr, dim3 Bl, double *data, const double *srcA_data, const double *srcB_data, MatrixDim dim, int srcA_stride, int srcB_stride, double alpha, double beta) {
    _add_mat_mat_elements<<<Gr, Bl>>>(data, srcA_data, srcB_data, dim, srcA_stride, srcB_stride, alpha, beta);
}

// CURRENTLY UNUSED...
void cudaD_apply_mask(dim3 Gr, dim3 Bl, double* mat, const char* mask, MatrixDim dmat, MatrixDim dmask) {
  _apply_mask<<<Gr,Bl>>>(mat,mask,dmat,dmask);
}



/*
 * CuVector
 */
void cudaD_replace_value(int Gr, int Bl, double *v, int dim, double orig, double changed) {
  _replace_value<<<Gr,Bl>>>(v, dim, orig, changed);
}

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

void cudaD_vec_min(const double* v, double* value, int dim) {
  _vec_min<<<1,CU1DBLOCK>>>(v, value, dim);
}

void cudaD_vec_max(const double* v, double* value, int dim) {
  _vec_max<<<1,CU1DBLOCK>>>(v, value, dim);
}

void cudaD_trace_mat_mat_trans(const double* A, const double* B, MatrixDim dA, int B_stride, double* value) {
  _trace_mat_mat_trans<double,4> <<<4,CU1DBLOCK>>>(A,B,dA,B_stride,value);
}

void cudaD_trace_mat_mat(const double* A, const double* B, MatrixDim dA, int B_stride, double* value) {
  _trace_mat_mat<double,2> <<<2,CU1DBLOCK>>>(A,B,dA,B_stride,value);
}

void cudaD_add_diag_mat_mat(int Gr, int Bl, double alpha, double* v, int v_dim, const double* M,
     int M_cols, int M_row_stride, int M_col_stride, const double *N, int N_row_stride,
     int N_col_stride, int threads_per_element, double beta) {
   _add_diag_mat_mat<<<Gr,Bl>>>(alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride,
                                N, N_row_stride, N_col_stride, threads_per_element, beta);
}

void cudaD_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}

void cudaD_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_df<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaD_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const double* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_fd<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaD_vec_sum(int Gr, int Bl, double* v, double* value, int dim, int inc) {
  _vec_sum<<<Gr,Bl>>>(v,value,dim,inc);
}

void cudaD_pvec_sum(int Gr, int Bl, double* v, double* pvec_sum, int dim, int size) {
  _pvec_sum<<<Gr,Bl>>>(v,pvec_sum,dim,size);
}

void cudaD_matrix_add_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim dim, double alpha, MatrixElement<double>* x, int num_elements) {
  _cuda_matrix_add_elements<<<Gr, Bl>>>(data, dim, alpha, x, num_elements);
}

void cudaD_matrix_add_indexed_values(dim3 Gr, dim3 Bl, MatrixDim dim, double alpha, const Int32Pair* indices, const double* x, int s, double* data) {
  _cuda_matrix_add_indexed_values<<<Gr, Bl>>>(dim, alpha, indices, x, s, data);
}

void cudaD_vec_copy_diag_from_packed(int Gr, int Bl, double *dst, const double *src, int dim) {
  _vec_copy_diag_from_packed<<<Gr,Bl>>>(dst,src,dim);
}

void cudaD_vec_apply_floor(int Gr, int Bl, double* v, double floor_val, float *count, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,count,dim);
}

void cudaD_vec_apply_ceiling(int Gr, int Bl, double* v, double ceiling_val, float *count, int dim) {
  _vec_apply_ceiling<<<Gr,Bl>>>(v,ceiling_val,count,dim);
}

void cudaD_vec_apply_exp(int Gr, int Bl, double* v, int dim) {
  _vec_apply_exp<<<Gr,Bl>>>(v,dim);
}

void cudaD_vec_apply_log(int Gr, int Bl, double* v, double* flag, int dim) {
  _vec_apply_log<<<Gr,Bl>>>(v,flag,dim);
}

void cudaD_invert_elements(dim3 Gr, dim3 Bl, double* data, MatrixDim d) {
  _invert_elements<<<Gr,Bl>>>(data, d);
}

void cudaD_add_mat_blockmat(dim3 Gr, dim3 Bl, double *data, MatrixDim d, const double *Adata,
                            int A_num_rows, int A_num_cols, int A_row_stride, int A_col_stride,
                            const CuBlockMatrixData *B_cu_data, int B_num_blocks,
                            double alpha, double beta, int B_trans) {
  if (B_trans) {
    _add_mat_blockmat_trans<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
                                       A_row_stride, A_col_stride, B_cu_data,
                                       B_num_blocks, alpha, beta);
  } else {
    _add_mat_blockmat<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
                                 A_row_stride, A_col_stride, B_cu_data,
                                 B_num_blocks, alpha, beta);
  }
}

void cudaD_block_add_mat_mat(dim3 Gr, dim3 Bl, CuBlockMatrixData *B_cu_data, int num_blocks,
                             const double *C_data, int C_num_cols, int C_row_stride, int C_col_stride,
                             const double *D_data, int D_row_stride, int D_col_stride,
                             double alpha, double beta) {
  _block_add_mat_mat<<<Gr,Bl>>>(B_cu_data, num_blocks, C_data, C_num_cols,
                                C_row_stride, C_col_stride, D_data, D_row_stride,
                                D_col_stride, alpha, beta);
}

/*
 * cu::
 */
void cudaD_soft_hinge (dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _soft_hinge<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_group_pnorm(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d,
		       int src_stride, int group_size, double power) {
  _group_pnorm<<<Gr,Bl>>>(y, x, d, src_stride, group_size, power);
}

void cudaD_group_max(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d,
		     int src_stride, int group_size) {
  _group_max<<<Gr,Bl>>>(y, x, d, src_stride, group_size);
}

void cudaD_sigmoid (dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _sigmoid<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_diff_sigmoid (dim3 Gr, dim3 Bl, double* eout, const double* e, const double* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaD_tanh (dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _tanh<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_diff_tanh (dim3 Gr, dim3 Bl, double* eout, const double* e, const double* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaD_softmax_reduce (size_t Gr, size_t Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_log_softmax_reduce (size_t Gr, size_t Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _log_softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_splice(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in);
}

void cudaD_one(int Gr, int Bl, double* x, int dim) {
  _one<<<Gr,Bl>>>(x,dim);
}

void cudaD_take_mean(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in) {
  _take_mean<<<Gr,Bl>>>(x,y,d_in);
}

void cudaD_take_lower(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in) {
  _take_lower<<<Gr,Bl>>>(x,y,d_in);
}

void cudaD_take_upper(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in) {
  _take_upper<<<Gr,Bl>>>(x,y,d_in);
}

void cudaD_copy_from_sp(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_out) {
  _copy_from_sp<<<Gr,Bl>>>(x,y,d_out);
}

void cudaD_copy(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaD_randomize(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaD_regularize_l1(dim3 Gr, dim3 Bl, double* wei, double* grad, double l1, double lr, MatrixDim d,int stride_grad) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d,stride_grad);
}

void cudaD_find_row_max_id(dim3 Gr, dim3 Bl, const double* mat, double* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, voff, d);
}

void cudaD_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda* vec_tgt, double* mat_net_out, double* vec_log_post, MatrixDim d) {
  _diff_xent<<<Gr,Bl>>>(vec_tgt,mat_net_out,vec_log_post,d);
}

void cudaD_copy_rows_from_vec(dim3 Gr, dim3 Bl, double *mat_out, MatrixDim d_out, const double *v_in) {
  _copy_rows_from_vec<<<Gr,Bl>>>(mat_out, d_out, v_in);
}

void cudaD_sum_column_ranges(dim3 Gr, dim3 Bl, double *data, MatrixDim dim,
                             const double *src_data, MatrixDim src_dim,
                             const Int32Pair *indices) {
  _sum_column_ranges<<<Gr,Bl>>>(data, dim, src_data, src_dim, indices);
}

void cudaD_add_row_ranges(dim3 Gr, dim3 Bl, double *data, MatrixDim dim,
                          const double *src_data, MatrixDim src_dim,
                          const Int32Pair *indexes) {
  _add_row_ranges<<<Gr,Bl>>>(data, dim, src_data, src_dim, indexes);
}

void cudaD_matrix_lookup(dim3 Gr, dim3 Bl, const double *data, MatrixDim dim,
                         const Int32Pair *indices, int indices_size,
                         double *output) {
  _matrix_lookup<<<Gr,Bl>>>(data, dim, indices, indices_size, output);
}

void cudaD_equal_element_mask(dim3 Gr, dim3 Bl, const double *mat1,
                              const double *mat2, double *mask, MatrixDim mat1_dim,
                              int mat2_stride, int mask_stride) {
  _equal_element_mask<<<Gr,Bl>>>(mat1, mat2, mask, mat1_dim, mat2_stride, mask_stride);
}



/* Some conversion kernels for which it's more convenient to not name them F or D. */

void cuda_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_ff(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_fd(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_dd(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_ff_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_dd_trans(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_smat_ff(dim3 Gr, dim3 Bl, float* mat_out, const MatrixElement<float>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_fd(dim3 Gr, dim3 Bl, float* mat_out, const MatrixElement<double>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_df(dim3 Gr, dim3 Bl, double* mat_out, const MatrixElement<float>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_dd(dim3 Gr, dim3 Bl, double* mat_out, const MatrixElement<double>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_ff_trans(dim3 Gr, dim3 Bl, float* mat_out, const MatrixElement<float>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_fd_trans(dim3 Gr, dim3 Bl, float* mat_out, const MatrixElement<double>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const MatrixElement<float>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_dd_trans(dim3 Gr, dim3 Bl, double* mat_out, const MatrixElement<double>* smat_in, MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}

void cudaF_trace_mat_smat(dim3 Gr, dim3 Bl, const float* mat_in, const MatrixElement<float>* smat_in, MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in, float* trace_vec_out) {
  _trace_mat_smat<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in, trace_vec_out);
}
void cudaF_trace_mat_smat_trans(dim3 Gr, dim3 Bl, const float* mat_in, const MatrixElement<float>* smat_in, MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in, float* trace_vec_out) {
  _trace_mat_smat_trans<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in, trace_vec_out);
}
void cudaD_trace_mat_smat(dim3 Gr, dim3 Bl, const double* mat_in, const MatrixElement<double>* smat_in, MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in, double* trace_vec_out) {
  _trace_mat_smat<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in, trace_vec_out);
}
void cudaD_trace_mat_smat_trans(dim3 Gr, dim3 Bl, const double* mat_in, const MatrixElement<double>* smat_in, MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in, double* trace_vec_out) {
  _trace_mat_smat_trans<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in, trace_vec_out);
}

