// cudamatrix/cu-kernels.cu

// Copyright 2009-2012  Karel Vesely
//                2013  Ehsan Variani
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//           2013-2015  Guoguo Chen
//                2016  Shiyin Kang

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
#include <limits>
#include <math_constants.h>
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
  while (nTotalThreads > 1) {
    int32_cuda halfPoint = ((1 + nTotalThreads) >> 1); // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x >= halfPoint) { // was <
      // Get the shared value stored by another thread
      Real temp = 0.0;
      if (threadIdx.x < nTotalThreads) { // was +halfPoint
        temp = buffer[threadIdx.x]; // was +halfPoint
      }
      buffer[threadIdx.x - halfPoint] += temp;
    }
    __syncthreads();
    nTotalThreads = ((1 + nTotalThreads) >> 1); // divide by two.
  }
  // the result
  return buffer[0];
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
  if (i <= j || i >= dimA.rows)
    return;
  int index_1 = i * dimA.stride + j;
  int index_2 = j * dimA.stride + i;
  A[index_2] = A[index_1];
}

template<typename Real>
__global__
static void _copy_upp_low(Real* A, MatrixDim dimA) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j <= i || j >= dimA.rows)
    return;
  int index_1 = i * dimA.stride + j;
  int index_2 = j * dimA.stride + i;
  A[index_2] = A[index_1];
}

// mat += diag(vec) * mat2.
template<typename Real>
__global__
static void _add_diag_vec_mat(Real alpha, Real *mat, MatrixDim mat_dim,
                              const Real *vec, const Real *mat2,
                              int mat2_row_stride, int mat2_col_stride,
                              Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // column index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // row index

  int index = j * mat_dim.stride + i, index2 = j * mat2_row_stride
      + i * mat2_col_stride;

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
    int32_cuda index_B = (j * (j + 1) / 2) + i;
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
    int32_cuda index_B = (j * (j + 1) / 2) + i;
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
static void _copy_from_mat(Real* mat_out, const OtherReal* mat_in,
                           MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // col-index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row-index.
  int32_cuda index_out = i + j * d_out.stride;
  int32_cuda index_in = i + j * d_in.stride;
  if (i < d_out.cols && j < d_out.rows)
    mat_out[index_out] = static_cast<Real>(mat_in[index_in]);
}

template<int TileDim, typename Real, typename OtherReal>
__global__
static void _copy_from_mat_trans(Real* mat_out, const OtherReal* mat_in,
                                 MatrixDim d_out, MatrixDim d_in) {
  // Use shared meme to achieve both coalesced memory reading and writing
  // '+1' to avoid bank conflict when reading sbuf
  __shared__ Real sbuf[TileDim][TileDim + 1];

  const int32_cuda i_in = blockIdx.y * TileDim + threadIdx.y; // row-index
  const int32_cuda j_in = blockIdx.x * TileDim + threadIdx.x; // col-index
  const int32_cuda tile_stride_in = CU1DBLOCK / TileDim * d_in.stride;
  int32_cuda index_in = i_in * d_in.stride + j_in;

# pragma unroll
  for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
    if (i_in + i < d_in.rows && j_in < d_in.cols) {
      sbuf[threadIdx.y + i][threadIdx.x] = static_cast<Real>(mat_in[index_in]);
    }
    index_in += tile_stride_in;
  }
  __syncthreads();

  // Grid is transposed, but block is not yet.
  // Warp (blockDim.x) is always along the row-dim.
  const int32_cuda i_out = blockIdx.x * TileDim + threadIdx.y;
  const int32_cuda j_out = blockIdx.y * TileDim + threadIdx.x;
  const int32_cuda tile_stride_out = CU1DBLOCK / TileDim * d_out.stride;
  int32_cuda index_out = i_out * d_out.stride + j_out;

# pragma unroll
  for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
    if (i_out + i < d_out.rows && j_out < d_out.cols) {
      // block is tranposed when reading sbuf
      mat_out[index_out] = sbuf[threadIdx.x][threadIdx.y + i];
    }
    index_out += tile_stride_out;
  }
}

template<typename Real, typename OtherReal>
__global__
static void _copy_from_smat(Real* mat_out,
                            const MatrixElement<OtherReal>* smat_in,
                            MatrixDim d_out, MatrixIndexT_cuda d_in) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= d_in)
    return;
  int data_index = smat_in[smat_index].row * d_out.stride
      + smat_in[smat_index].column;
  mat_out[data_index] = smat_in[smat_index].weight;
}

template<typename Real, typename OtherReal>
__global__
static void _copy_from_smat_trans(Real* mat_out,
                                  const MatrixElement<OtherReal>* smat_in,
                                  MatrixDim d_out, MatrixIndexT_cuda d_in) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= d_in)
    return;
  int data_index = smat_in[smat_index].column * d_out.stride
      + smat_in[smat_index].row;
  mat_out[data_index] = smat_in[smat_index].weight;
}

template<typename Real>
__global__
static void _trace_mat_smat_trans(const Real* mat_in,
                                  const MatrixElement<Real>* smat_in,
                                  MatrixDim mat_d_in,
                                  MatrixIndexT_cuda smat_d_in,
                                  Real* trace_vec_out) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= smat_d_in)
    return;
  int mat_index = smat_in[smat_index].row * mat_d_in.stride
      + smat_in[smat_index].column;
  trace_vec_out[smat_index] = mat_in[mat_index] * smat_in[smat_index].weight;
}

template<typename Real>
__global__
static void _trace_mat_smat(const Real* mat_in,
                            const MatrixElement<Real>* smat_in,
                            MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in,
                            Real* trace_vec_out) {
  int smat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (smat_index >= smat_d_in)
    return;
  int mat_index = smat_in[smat_index].column * mat_d_in.stride
      + smat_in[smat_index].row;
  trace_vec_out[smat_index] = mat_in[mat_index] * smat_in[smat_index].weight;
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
  int32_cuda index = ((i + 1) * (i + 2) / 2) - 1;
  if (i < dim) {
    mat[index] = value * mat[index];
  }
}

template<typename Real>
__global__
static void _set_diag(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + i * d.stride;
  if (i < d.rows && i < d.cols) {
    mat[index] = value;
  }
}

template<typename Real>
__global__
static void _set_diag_packed(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i + 1) * (i + 2) / 2) - 1;
  if (i < dim) {
    mat[index] = value;
  }
}

template<typename Real>
__global__
static void _add_diag_packed(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i + 1) * (i + 2) / 2) - 1;
  if (i < dim) {
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
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < i)
    mat[index] = 0.0;
}

template<typename Real>
__global__
static void _add(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = mat[index] + value;
}

template<typename Real>
__global__
static void _scale(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = mat[index] * value;
}

template<typename Real>
__global__
static void _apply_log(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = log(mat[index]);
}

template<typename Real>
__global__
static void _mul_elements(Real* mat, const Real* A, MatrixDim dst_d,
                          int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda dst_index = i + j * dst_d.stride, src_index = i + j * src_stride;
  if (i < dst_d.cols && j < dst_d.rows)
    mat[dst_index] = mat[dst_index] * A[src_index];
}

template<typename Real>
__global__
static void _div_elements(Real* mat, const Real* A, MatrixDim dst_d,
                          int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda dst_index = i + j * dst_d.stride, src_index = i + j * src_stride;
  if (i < dst_d.cols && j < dst_d.rows)
    mat[dst_index] = mat[dst_index] / A[src_index];
}

template<typename Real>
__global__
static void _max(Real* mat, const Real* A, MatrixDim dst_d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda dst_index = i + j * dst_d.stride, src_index = i + j * src_stride;
  if (i < dst_d.cols && j < dst_d.rows) {
    Real a = mat[dst_index], b = A[src_index];
    mat[dst_index] = fmax(a, b);
  }
}

template<typename Real>
__global__
static void _min(Real* mat, const Real* other, MatrixDim mat_d,
                 int other_stride) {
  int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda mat_index = i * mat_d.stride + j;
  int32_cuda other_index = i * other_stride + j;
  if (j < mat_d.cols && i < mat_d.rows) {
    Real a = mat[mat_index], b = other[other_index];
    mat[mat_index] = fmin(a, b);
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
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] *= scale[i];
}

template<typename Real>
__global__
static void _mul_rows_vec(Real* mat, const Real* scale, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] *= scale[j];
}

template<typename Real>
__global__
static void _mul_rows_group_mat(Real *y, const Real *x, MatrixDim d,
                                int src_stride, int group_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < d.rows && i < d.cols) {
    int dst_index = i + j * d.stride;
    int src_index = i / group_size + j * src_stride;
    y[dst_index] *= x[src_index];
  }
}


template<typename Real>
__global__
void _diff_group_pnorm(Real *id, const Real *iv, const Real *ov, const Real* od,
                       MatrixDim id_dim, int iv_stride, int ov_stride,
                       int od_stride, int group_size, Real power) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < id_dim.cols) {
    const int grid_stride = gridDim.y * blockDim.y;
    const int src_j = j / group_size;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    for (; i < id_dim.rows; i += grid_stride) {
      const int iv_index = j + i * iv_stride;
      Real iv_ij = iv[iv_index];
      Real ans;
      if (power == Real(2)) {
        const int ov_index = src_j + i * ov_stride;
        Real ov_ij = ov[ov_index];
        ans = ov_ij <= 0.0 ? 0.0 : iv_ij / ov_ij;
      } else if (power == Real(1)) {
        Real iv_ij_sign = (iv_ij >= 0 ? 1 : -1);
        ans = (iv_ij == Real(0) ? 0.0 : iv_ij_sign);
      } else if (power
          == (sizeof(Real) == sizeof(float) ? CUDART_INF_F : CUDART_INF)) {
        const int ov_index = src_j + i * ov_stride;
        Real ov_ij = ov[ov_index];
        Real iv_ij_sign = (iv_ij >= 0 ? 1 : -1);
        ans =
            ov_ij <= 0.0 ?
                0.0 : (iv_ij_sign * (abs(iv_ij) == ov_ij ? 1.0 : 0.0));
      } else {
        const int ov_index = src_j + i * ov_stride;
        Real ov_ij = ov[ov_index];
        Real iv_ij_sign = (iv_ij >= 0 ? 1 : -1);
        if (ov_ij <= 0.0) {
          ans = 0.0; // The derivative is either 0 or undefined at the origin.
        } else {
          ans = iv_ij_sign * pow(std::abs(iv_ij), power - 1)
              * pow(ov_ij, 1 - power);
        }
      }
      const int od_index = src_j + i * od_stride;
      const int id_index = j + i * id_dim.stride;
      id[id_index] = ans * od[od_index];
    }
  }
}

/// deriv is the derivative we will output; vec is the input we're computing
/// the group max on, "maxv" is the previously computed group max.
template<typename Real>
__global__
static void _calc_group_max_deriv(Real *deriv, const Real *vec,
                                  const Real *maxv, MatrixDim deriv_dim,
                                  int vec_stride, int maxv_stride,
                                  int group_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < deriv_dim.rows && i < deriv_dim.cols) {
    int deriv_index = i + j * deriv_dim.stride;
    int vec_index = i + j * vec_stride;
    int maxv_index = i / group_size + j * maxv_stride;
    Real vec_element = vec[vec_index], // The element of the original vector.
        max_element = maxv[maxv_index]; // this is the max value
    Real ans = (max_element == vec_element ? 1.0 : 0.0);
    deriv[deriv_index] = ans;
  }
}

/// Set each element to y = (x == orig ? changed : x).
template<typename Real>
__global__
static void _replace_value(Real *vec, int dim, Real orig, Real changed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim)
    if (vec[i] == orig)
      vec[i] = changed;
}

template<typename Real>
__global__
static void _div_rows_vec(Real* mat, const Real* vec_div, MatrixDim d) {
  const int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < d.rows) {
    const int32_cuda start = i * d.stride;
    const Real scale = Real(1) / vec_div[i];
    const int32_cuda grid_stride = blockDim.x * gridDim.x;
    for (int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x; j < d.cols; j +=
        grid_stride) {
      mat[start + j] *= scale;
    }
  }
}

template<typename Real>
__global__
static void _add_mat(Real alpha, const Real* src, Real* dst, MatrixDim d,
                     int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = i + j * src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha * src[index_src] + dst[index];
}

template<typename Real>
__global__
static void _add_mat_trans(Real alpha, const Real* src, Real* dst, MatrixDim d,
                           int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = j + i * src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha * src[index_src] + dst[index];
}

template<typename Real>
__global__
static void _add_mat_blocks(Real alpha, const Real* src,
                            int32_cuda num_row_blocks,
                            int32_cuda num_col_blocks, Real* dst, MatrixDim d,
                            int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = i + j * src_stride;
  if (i < d.cols && j < d.rows)
    for (int32_cuda p = 0; p < num_row_blocks; p++) {
      for (int32_cuda q = 0; q < num_col_blocks; q++) {
        dst[index] = alpha
            * src[index_src + p * src_stride * d.rows + q * d.cols]
            + dst[index];
      }
    }
}

template<typename Real>
__global__
static void _add_mat_repeated(Real alpha, const Real* src,
                              MatrixDim src_dim, Real* dst,
                              MatrixDim dst_dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda src_i = i % src_dim.cols,
      src_j = j % src_dim.rows,
      dst_index = i + j * dst_dim.stride,
      src_index = src_i + src_j * src_dim.stride;
  if (i < dst_dim.cols && j < dst_dim.rows)
    dst[dst_index] += alpha * src[src_index];
}


template<typename Real>
__global__
static void _add_mat_blocks_trans(Real alpha, const Real* src,
                                  int32_cuda num_row_blocks,
                                  int32_cuda num_col_blocks, Real* dst,
                                  MatrixDim d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = j + i * src_stride;
  if (i < d.cols && j < d.rows)
    for (int32_cuda p = 0; p < num_row_blocks; p++) {
      for (int32_cuda q = 0; q < num_col_blocks; q++) {
        dst[index] = alpha
            * src[index_src + p * src_stride * d.cols + q * d.rows]
            + dst[index];
      }
    }
}

template<typename Real>
__global__
static void _set_mat_mat_div_mat(const Real* A, const Real* B, const Real* C,
                                 Real* dst, MatrixDim d, int stride_a,
                                 int stride_b, int stride_c) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride, a_index = i + j * stride_a, b_index = i
      + j * stride_b, c_index = i + j * stride_c;
  if (i < d.cols && j < d.rows)
    if (C[c_index] == 0)
      dst[index] = A[a_index];
    else
      dst[index] = A[a_index] * B[b_index] / C[c_index];
}

// Given a matrix input S (not packed!) and a lower-triangular matrix L, this
// function does S = beta S + alpha * L^T L.  This is used in PSD matrix
// inversion. The i index is the row of the destination S and the j the column
// (although of course the output is symmetric so it doesn't matter in a sense).
// The main point of this is to make use of various symmetries and zero-ness.
template<typename Real>
__global__
static void _sy_add_tr2(Real alpha, Real beta, const Real *T, MatrixDim tdim,
                        Real *S, MatrixDim sdim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= sdim.rows || j > i)
    return;

  // this thread computes the dot-product of the i'th column of
  // L with the j'th column of L.  The values we're multiplying
  // are only nonzero for row-index k greater or equal to
  // max(i, j), which equals i.

  Real sum = 0.0;
  for (int k = i; k < sdim.rows; k++) {
    int i_index = i + tdim.stride * k, j_index = j + tdim.stride * k;
    sum += T[i_index] * T[j_index];
  }
  int output_index1 = i * sdim.stride + j, output_index2 = j * sdim.stride + i;
  S[output_index1] = alpha * sum + beta * S[output_index1];
  S[output_index2] = alpha * sum + beta * S[output_index2];
}

template<typename Real>
__global__
static void _add_vec_to_cols(Real alpha, const Real* col, Real beta, Real* dst,
                             MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha * col[j] + beta * dst[index];
}

template<typename Real>
__global__
static void _add_vec_to_rows(Real alpha, const Real* row, Real beta, Real* dst,
                             MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha * row[i] + beta * dst[index];
}

template<typename Real>
__global__
static void _apply_mask(Real* mat, const char* mask, MatrixDim dmat,
                        MatrixDim dmask) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * dmat.stride;
  int32_cuda index2 = i + j * dmask.stride;
  if (i < dmat.cols && j < dmat.rows)
    if (mask[index2] == 0)
      mat[index] = 0;
}

template<typename Real>
__global__
static void _add_mat_diag_vec(Real alpha, Real *mat, MatrixDim mat_dim,
                              const Real *mat2, int mat2_row_stride,
                              int mat2_col_stride, const Real *vec, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // column index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // row index

  int index = i + j * mat_dim.stride, index2 = i * mat2_col_stride
      + j * mat2_row_stride;
  if (j < mat_dim.rows && i < mat_dim.cols)
    mat[index] = alpha * mat2[index2] * vec[i] + beta * mat[index];
}

template<typename Real>
__global__
static void _add_mat_mat_elements(Real *data, const Real *srcA_data,
                                  const Real *srcB_data, MatrixDim dim,
                                  int srcA_stride, int srcB_stride, Real alpha,
                                  Real beta) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda tgt_index = i + j * dim.stride;
  int32_cuda srcA_index = i + j * srcA_stride;
  int32_cuda srcB_index = i + j * srcB_stride;
  if (i < dim.cols && j < dim.rows) {
    data[tgt_index] = alpha * srcA_data[srcA_index] * srcB_data[srcB_index]
        + beta * data[tgt_index];
  }
}

/*
 * CuVector
 */
// very limited application!
template<typename Real>
__global__
static void _set_bias_params(Real* v, const Real* a, Real param_1, Real param_2,
                             Real param_3, int* flag, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    Real ratio = a[i] / param_3;
    if ((ratio < 0.0) || (ratio >= 1.01)) {
      *flag = 1;
      return;
    }
    if (ratio < param_1) {
      Real factor = ((param_1 / ratio) > param_2) ? param_2 : (param_1 / ratio);
      v[i] = v[i] / factor;
    } else if (ratio > param_1) {
      Real factor = ((ratio / param_1) > param_2) ? param_2 : (ratio / param_1);
      v[i] = v[i] * factor;
    }
  }
}

template<typename Real, typename OtherReal>
__global__
static void _cublas_copy_kaldi(int n, const Real* x, int incx, OtherReal* y,
                               int incy) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    y[i * incy] = static_cast<OtherReal>(x[i * incx]);
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

// This kernel writes a copy of the vector "v_in" to each col of the matrix
// "m_out".  the dimension of v_in should be equal to the #row of m_out.
template<typename Real>
__global__
static void _copy_cols_from_vec(Real* m_out, MatrixDim d, const Real* v_in) {
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row id
  int j = blockIdx.x * blockDim.x + threadIdx.x; // col id
  if (i < d.rows && j < d.cols) {
    m_out[i * d.stride + j] = v_in[i];
  }
}

// _trace_mat_mat reduce the partial sum to
// value[blockIdx.y * gridDim.x + blockIdx.x]
// It use shared mem to transpose matrix B to ensure coalesced memory access
template<int TileDim, typename Real>
__global__
static void _trace_mat_mat(const Real* A, const Real* B, MatrixDim dA,
                           int B_stride, Real* value) {
  // Reuse shared mem and make indexing easier. "+1" to avoid bank conflict
  __shared__ union {
    Real trans[TileDim][TileDim + 1];
    Real sum[CU1DBLOCK];
  } smem;
  // linear thread id;
  const int32_cuda tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_cuda grid_height = gridDim.y * TileDim;

  const int32_cuda ja = blockIdx.x * TileDim + threadIdx.x;
  const int32_cuda ib = blockIdx.x * TileDim + threadIdx.y;
  int32_cuda ia = blockIdx.y * TileDim + threadIdx.y;
  int32_cuda jb = blockIdx.y * TileDim + threadIdx.x;

  // Grid reduce
  Real tsum = Real(0);
  for (int32_cuda i0 = 0; i0 < dA.rows; i0 += grid_height) {
    // Load from B, transpose the block and store in shared mem
    if (jb < dA.rows) {
#     pragma unroll
      for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
        if (ib + i < dA.cols) {
          smem.trans[threadIdx.x][threadIdx.y + i] =
              B[(ib + i) * B_stride + jb];
        }
      }
    }
    __syncthreads();

    // Load from A, sum up the product.
    if (ja < dA.cols) {
#     pragma unroll
      for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
        if (ia + i < dA.rows) {
          tsum += A[(ia + i) * dA.stride + ja]
              * smem.trans[threadIdx.y + i][threadIdx.x];
        }
      }
    }
    __syncthreads();

    ia += grid_height;
    jb += grid_height;
  }

  smem.sum[tid] = tsum;
  __syncthreads();

  // Block reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      smem.sum[tid] += smem.sum[tid + shift];
    __syncthreads();
  }

  // Warp reduce. Implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      smem.sum[tid] += smem.sum[tid + shift];
    }
  }

  // output 1 sum per thread block
  if (tid == 0) {
    value[blockIdx.y * gridDim.x + blockIdx.x] = smem.sum[0];
  }
}

// _trace_mat_mat_trans reduce the partial sum to
// value[blockIdx.y * gridDim.x + blockIdx.x]
template<typename Real>
__global__
static void _trace_mat_mat_trans(const Real* A, const Real* B, MatrixDim dA,
                                 int B_stride, Real* value) {
  __shared__ Real ssum[CU1DBLOCK];
  // linear thread id;
  const int32_cuda tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_cuda j = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_cuda grid_height = gridDim.y * blockDim.y;
  int32_cuda i = blockIdx.y * blockDim.y + threadIdx.y;

  // Grid reduce
  Real tsum = Real(0);
  if (j < dA.cols) {
    while (i < dA.rows) {
      tsum += A[i * dA.stride + j] * B[i * B_stride + j];
      i += grid_height;
    }
  }
  ssum[tid] = tsum;
  __syncthreads();

  // Block reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      ssum[tid] += ssum[tid + shift];
    __syncthreads();
  }

  // Warp reduce. Implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      ssum[tid] += ssum[tid + shift];
    }
  }

  // output 1 sum per thread block
  if (tid == 0) {
    value[blockIdx.y * gridDim.x + blockIdx.x] = ssum[0];
  }
}

// v = alpha * diag(M * N^T) + beta * v
template<typename Real>
__global__
static void _add_diag_mat_mat_MNT(const Real alpha, const Real* M,
                                  const MatrixDim dim_M, const Real* N,
                                  const int stride_N, const Real beta,
                                  Real* v) {
  __shared__ Real ssum[CU1DBLOCK];
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const int m_start = i * dim_M.stride;
  const int n_start = i * stride_N;

  // Loop along the matrix row. Reduce to CU1DBLOCK elements per row.
  Real tsum = Real(0);
  for (int j = tid; j < dim_M.cols; j += CU1DBLOCK) {
    tsum += M[m_start + j] * N[n_start + j];
  }
  ssum[tid] = tsum;
  __syncthreads();

  // Tree reduce to 2x warpSize elements.
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      ssum[tid] += ssum[tid + shift];
    __syncthreads();
  }

  // Warp reduce to 1 element. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      ssum[tid] += ssum[tid + shift];
    }
  }

  // output 1 sum per thread block
  if (tid == 0) {
    v[i] = alpha * ssum[0] + beta * v[i];
  }
}

// v = alpha * diag(M^T * N) + beta * v
template<int TileDim, typename Real>
__global__
static void _add_diag_mat_mat_MTN(const Real alpha, const Real* M,
                                  const int stride_M, const Real* N,
                                  const MatrixDim dim_N, const Real beta,
                                  Real* v) {
  __shared__ Real ssum[CU1DBLOCK];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j >= dim_N.cols)
    return;

  // Loop along the matrix column.
  // Reduce to CU1DBLOCK / TileDim elements per column.
  Real tsum = Real(0);
  for (int i = threadIdx.y; i < dim_N.rows; i += blockDim.y) {
    tsum += M[i * stride_M + j] * N[i * dim_N.stride + j];
  }
  ssum[tid] = tsum;
  __syncthreads();

  // Tree reduce to 2x warpSize / TileDim elements per column.
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize && shift >= TileDim;
      shift >>= 1) {
    if (tid < shift) {
      ssum[tid] += ssum[tid + shift];
    }
    __syncthreads();
  }

  // Warp reduce to 1 element per column.
  // Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift >= TileDim; shift >>= 1) {
      ssum[tid] += ssum[tid + shift];
    }
  }

  // output TileDim sums per thread block
  if (tid < TileDim) {
    v[j] = alpha * ssum[tid] + beta * v[j];
  }
}

// v = alpha * diag(M * N) + beta * v
template<int TileDim, typename Real>
__global__
static void _add_diag_mat_mat_MN(const Real alpha, const Real* M,
                                 const int stride_M, const Real* N,
                                 const MatrixDim dim_N, const Real beta,
                                 Real* v) {
  // Reuse shared mem and make indexing easier. "+1" to avoid bank conflict
  __shared__ union {
    Real trans[TileDim][TileDim + 1];
    Real sum[CU1DBLOCK];
  } smem;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i_m = blockIdx.x * TileDim + threadIdx.y;
  const int j_n = blockIdx.x * TileDim + threadIdx.x;
  int i_n = threadIdx.y;
  int j_m = threadIdx.x;

  // Loop along the matrix column.
  // Reduce to CU1DBLOCK / TileDim elements per column.
  Real tsum = Real(0);
  for (int block_i_n = 0; block_i_n < dim_N.rows; block_i_n += TileDim) {

    // Load, transpose and store M to shared mem.
    if (j_m < dim_N.rows) {
#     pragma unroll
      for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
        if (i_m + i < dim_N.cols) {
          smem.trans[threadIdx.x][threadIdx.y + i] = M[(i_m + i) * stride_M
              + j_m];
        }
      }
    }
    __syncthreads();

    // Load N, sum up the product.
    if (j_n < dim_N.cols) {
#     pragma unroll
      for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
        if (i_n + i < dim_N.rows) {
          tsum += N[(i_n + i) * dim_N.stride + j_n]
              * smem.trans[threadIdx.y + i][threadIdx.x];
        }
      }
    }
    __syncthreads();

    i_n += TileDim;
    j_m += TileDim;
  }
  smem.sum[tid] = tsum;
  __syncthreads();

  // Tree reduce to 2x warpSize / TileDim elements per column.
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize && shift >= TileDim;
      shift >>= 1) {
    if (tid < shift) {
      smem.sum[tid] += smem.sum[tid + shift];
    }
    __syncthreads();
  }

  // Warp reduce to 1 element per column.
  // Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift >= TileDim; shift >>= 1) {
      smem.sum[tid] += smem.sum[tid + shift];
    }
  }

  // output TileDim sums per thread block
  if (tid < TileDim && j_n < dim_N.cols) {
    v[j_n] = alpha * smem.sum[tid] + beta * v[j_n];
  }
}

template<typename Real>
__global__
static void _add_vec_vec(Real alpha, Real* v, const Real* x, const Real* y,
                         Real beta, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (blockIdx.y > 0) return;

  if (i < dim)
    v[i] = alpha * x[i] * y[i] + beta * v[i];
}

template<typename Real>
__global__
static void _copy_col_from_mat_df(double* v, int col, const Real* mat,
                                  MatrixDim dmat, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = col + i * dmat.stride;
  // if (blockIdx.y > 0)  return;

  if (i < dim)
    v[i] = (double) mat[index];
}

template<typename Real>
__global__
static void _copy_col_from_mat_fd(float* v, int col, const Real* mat,
                                  MatrixDim dmat, int dim) {
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
static void _cuda_comp_obj_deriv(MatrixElement<Real> *x, int s, const Real* z,
                                 MatrixDim d, Real* z2, MatrixDim d2, Real* t) {
  int i = threadIdx.x;
  __shared__ Real tot_objf[CU1DBLOCK];
  __shared__ Real tot_weight[CU1DBLOCK];

  Real tmp_weight_sum = 0;
  Real tmp_tot_objf = 0;
  int size = s / CU1DBLOCK; //the least size in a loop (later part)
  int threshold = s - size * CU1DBLOCK; //any loop below this number would + 1

  int loop_start;
  int loop_end;
  if (i < threshold) {
    loop_start = i * (size + 1);
    loop_end = (i + 1) * (size + 1);
  } else {
    loop_start = threshold + i * size;
    loop_end = threshold + (i + 1) * size;
  }
  for (int j = loop_start; j < loop_end; j++) {
    //* ((int*) ((size_t)x + j * (2 * sizeof(int) + sizeof(Real) )) );
    int m = (x + j)->row;
    //*(int*) ((size_t)x + j * (2 * sizeof(int) + sizeof(Real) )+ sizeof(int));
    int label = (x + j)->column;
    // *(Real*) ((size_t)x + j*(2*sizeof(int) + sizeof(Real)) + 2*sizeof(int));
    Real weight = (x + j)->weight;
    tmp_weight_sum += weight;
    Real this_prob = *(z + m * d.stride + label);
    tmp_tot_objf += weight * log(this_prob);

    // there might be problems here....
    *(z2 + m * d2.stride + label) += weight / this_prob;
  }
  tot_objf[i] = tmp_tot_objf;
  tot_weight[i] = tmp_weight_sum;
  __syncthreads();
  *t = _sum_reduce(tot_objf);
  __syncthreads();
  *(t + 1) = _sum_reduce(tot_weight);
  return;
}

template<typename Real>
__global__
static void _cuda_matrix_add_elements(Real *data, MatrixDim dim, Real alpha,
                                      MatrixElement<Real>* x,
                                      int num_elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_elements)
    return;
  data[x[i].row * dim.stride + x[i].column] += alpha * x[i].weight;
}

template<typename Real>
__global__
static void _cuda_matrix_add_indexed_values(MatrixDim dim, Real alpha,
                                            const Int32Pair* indices,
                                            const Real* x, int s, Real* data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s)
    return;
  int data_i = indices[i].first * dim.stride + indices[i].second;
  data[data_i] += alpha * x[i];
}

template<typename Real>
__global__
static void _matrix_lookup(const Real *data, MatrixDim dim,
                           const Int32Pair *indices, int indices_size,
                           Real *output) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= indices_size)
    return;
  int data_ind = indices[ind].first * dim.stride + indices[ind].second;
  output[ind] = data[data_ind];

}

template<typename Real>
__global__
static void _equal_element_mask(const Real *mat1, const Real *mat2, Real *mask,
                                MatrixDim mat1_dim, int mat2_stride,
                                int mask_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // col
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // row
  int32_cuda index_mat1 = i + j * mat1_dim.stride;
  int32_cuda index_mat2 = i + j * mat2_stride;
  int32_cuda index_mask = i + j * mask_stride;
  if (i < mat1_dim.cols && j < mat1_dim.rows)
    mask[index_mask] = (mat1[index_mat1] == mat2[index_mat2] ? 1.0 : 0.0);
}

enum EnumTransformReduce {
  SUMAB, SUM, MAX, MIN, LINFNORM, L2NORM, L1NORM, L0NORM, LPNORM
};

template<EnumTransformReduce TransReduceType, typename Real>
struct TransReduceOp {
  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return Real(0);
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return Real(0);
  }
};

template<typename Real>
struct TransReduceOp<SUMAB, Real> {
  const Real alpha_;
  const Real beta_;
  TransReduceOp(const Real& a, const Real& b) :
      alpha_(a), beta_(b) {
  }
  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return x;
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return a + b;
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    if (beta_ == Real(0)) {
      return alpha_ * x;
    } else {
      return alpha_ * x + beta_ * output;
    }
  }
};

template<typename Real>
struct TransReduceOp<SUM, Real> {
  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return x;
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return a + b;
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return x;
  }
};

template<typename Real>
struct TransReduceOp<MAX, Real> {
  __forceinline__
  __device__ Real InitValue() const {
    return sizeof(Real) == sizeof(float) ? -CUDART_INF_F : -CUDART_INF;
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return x;
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return fmax(a, b);
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return x;
  }
};

template<typename Real>
struct TransReduceOp<MIN, Real> {
  __forceinline__
  __device__ Real InitValue() const {
    return sizeof(Real) == sizeof(float) ? CUDART_INF_F : CUDART_INF;
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return x;
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return min(a, b);
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return x;
  }
};

template<typename Real>
struct TransReduceOp<LINFNORM, Real> {
  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return abs(x);
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return fmax(a, b);
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return x;
  }
};

template<typename Real>
struct TransReduceOp<L2NORM, Real> {
  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return x * x;
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return a + b;
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return sqrt(x);
  }
};

template<typename Real>
struct TransReduceOp<L1NORM, Real> {
  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return abs(x);
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return a + b;
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return x;
  }
};

template<typename Real>
struct TransReduceOp<L0NORM, Real> {
  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return Real(x == Real(0) ? 0 : 1);
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return a + b;
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return x;
  }
};

template<typename Real>
struct TransReduceOp<LPNORM, Real> {

  const Real power_;
  TransReduceOp(const Real& p) :
      power_(p) {
  }

  __forceinline__
  __device__ Real InitValue() const {
    return Real(0);
  }
  __forceinline__
  __device__ Real Transform(const Real& x) const {
    return pow(abs(x), power_);
  }
  __forceinline__
  __device__ Real Reduce(const Real& a, const Real& b) const {
    return a + b;
  }
  __forceinline__
  __device__ Real PostReduce(const Real& x, const Real& output) const {
    return pow(x, Real(1) / power_);
  }
};

// Vector reduce.
template<EnumTransformReduce TransReduceType, typename Real>
__global__
static void _vec_transform_reduce(
    const Real* v, Real* result, const int dim, const int inc,
    const TransReduceOp<TransReduceType, Real> op) {

  __shared__ Real sdata[CU1DBLOCK];
  Real tdata = op.InitValue();

  const int tid = threadIdx.x;
  const int vec_len = dim * inc;
  const int grid_stride = gridDim.x * blockDim.x * inc;
  int i = (blockIdx.x * blockDim.x + tid) * inc;

  // Grid reduce. Loop over the whole vector v.
  for (; i < vec_len; i += grid_stride) {
    tdata = op.Reduce(tdata, op.Transform(v[i]));
  }
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
    __syncthreads();
  }

  // Reduce last warp. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
  }

  // Output to vector result.
  if (tid == 0)
    result[blockIdx.x] = op.PostReduce(sdata[0], result[blockIdx.x]);
}

// Reduce a matrix 'mat' to a column vector 'result'
template<EnumTransformReduce TransReduceType, typename Real>
__global__
static void _transform_reduce_mat_cols(
    Real *result, const Real *mat, const MatrixDim d,
    const TransReduceOp<TransReduceType, Real> op) {

  __shared__ Real sdata[CU1DBLOCK];
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const int row_start = i * d.stride;

  Real tdata = op.InitValue();
  for (int j = tid; j < d.cols; j += CU1DBLOCK) {
    tdata = op.Reduce(tdata, op.Transform(mat[row_start + j]));
  }
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    __syncthreads();
  }

  // Reduce last warp. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1)
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
  }

  // Output to vector result.
  if (tid == 0) {
    result[i] = op.PostReduce(sdata[0], result[i]);
  }
}

template<EnumTransformReduce TransReduceType, typename Real>
__global__
static void _group_transform_reduce(
    Real *y, const Real *x, const MatrixDim d, const int src_stride,
    const int group_size, const TransReduceOp<TransReduceType, Real> op) {

  __shared__ Real sreduction[CU1DBLOCK];
  const int i = blockIdx.x;
  const int x_start = i * src_stride;
  const int y_start = i * d.stride;
  const int threads_per_group = blockDim.x;

  // Reduce n groups per thread block
  const int n = blockDim.y;
  const int len = group_size * n;
  // linear thread id
  const int tid = threadIdx.y * threads_per_group + threadIdx.x;
  int j = threadIdx.y * group_size + threadIdx.x; // col-id of *x
  int group_id = threadIdx.y;                     // col-id of *y
  int group_end = x_start + (group_id + 1) * group_size;

  while (group_id < d.cols) {
    // reduce to threads_per_group elements per group
    int x_idx = x_start + j;
    Real treduction = op.Transform(x[x_idx]);
    x_idx += threads_per_group;
    while (x_idx < group_end) {
      treduction = op.Reduce(treduction, op.Transform(x[x_idx]));
      x_idx += threads_per_group;
    }
    sreduction[tid] = treduction;
    if (threads_per_group > warpSize) {
      __syncthreads();
    }

    // tree-reduce to 2x warpSize elements per group
#   pragma unroll
    for (int shift = threads_per_group / 2; shift > warpSize; shift >>= 1) {
      if (threadIdx.x < shift) {
        sreduction[tid] = op.Reduce(sreduction[tid], sreduction[tid + shift]);
      }
      __syncthreads();
    }

    // Warp-reduce to 1 element per group.
    // Threads implicitly synchronized within the warp.
    const int warp_reduce_size =
        threads_per_group / 2 < warpSize ? threads_per_group / 2 : warpSize;
    if (threadIdx.x < warp_reduce_size) {
#     pragma unroll
      for (int shift = warp_reduce_size; shift > 0; shift >>= 1) {
        sreduction[tid] = op.Reduce(sreduction[tid], sreduction[tid + shift]);
      }
    }

    // Store the result.
    if (threadIdx.x == 0) {
      y[y_start + group_id] = op.PostReduce(sreduction[tid],
                                            y[y_start + group_id]);
    }

    j += len;
    group_end += len;
    group_id += n;
  }
}

template<typename Real>
__global__
static void _vec_apply_floor(Real *v, Real floor_val, float *count, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < dim) {
    if (v[i] < floor_val) {
      v[i] = floor_val;
      count[i] = 1;
    } else {
      count[i] = 0;
    }
  }
}

template<typename Real>
__global__
static void _vec_apply_ceiling(Real *v, Real ceiling_val, float *count,
                               int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < dim) {
    if (v[i] > ceiling_val) {
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
static void _apply_pow_abs(Real* mat, Real power, bool include_sign,
                           MatrixDim d) {
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
static void _copy_cols(Real* dst, const Real *src,
                       const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                       int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int index = reorder[i], dst_index = j * dst_dim.stride + i;
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
static void _add_cols(Real* dst, const Real *src,
                      const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                      int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int index = reorder[i], dst_index = j * dst_dim.stride + i;
    if (index >= 0) {
      int src_index = j * src_stride + index;
      Real val = src[src_index];
      dst[dst_index] += val;
    }
  }
}

template<typename Real>
__global__
static void _copy_rows(Real* dst, const Real *src,
                       const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                       int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dst_dim.cols && j < dst_dim.rows) {
    int index = reorder[j], dst_index = j * dst_dim.stride + i;
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
static void _copy_rows(Real* dst, const Real * const *src, MatrixDim dst_dim) {
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
static void _copy_to_rows(Real* const * dst, const Real *src,
                          MatrixDim src_dim) {
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
                      const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                      int src_stride) {
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
static void _add_rows(Real alpha, Real* dst, const Real * const *src,
                      MatrixDim dst_dim) {
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
static void _add_to_rows(Real alpha, Real* const * dst, const Real *src,
                         MatrixDim src_dim) {
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

  if (i < d.cols && j < d.rows) {
    if (mat[index] > ceiling_val)
      mat[index] = ceiling_val;
  }
}

template<typename Real>
__global__
static void _invert_elements(Real* data, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    data[index] = 1.0 / data[index];
}

// matrix-wise, do data = alpha * data + beta * A * B^T,
// where B is a block matrix.
template<typename Real>
__global__
static void _add_mat_blockmat_trans(Real *data, MatrixDim dim,
                                    const Real *A_data, int A_num_rows,
                                    int A_num_cols, int A_row_stride,
                                    int A_col_stride,
                                    const CuBlockMatrixData *B_cu_data,
                                    int B_num_blocks, Real alpha, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index into "data"
  int j = blockIdx.y * blockDim.y + threadIdx.y; // block-index into B.
  if (i >= A_num_rows || j >= B_num_blocks)
    return;

  const CuBlockMatrixData &cu_data = B_cu_data[j];

  // BT means B transposed.
  int BT_row_start = cu_data.col_offset, BT_col_start = cu_data.row_offset,
      BT_num_rows = cu_data.matrix_dim.cols, BT_num_cols =
          cu_data.matrix_dim.rows, BT_col_stride = cu_data.matrix_dim.stride;
  // Cast from void;
  const Real *B_data = static_cast<Real*>(cu_data.matrix_data);
  // we avoided a bunch of hassle by doing this (relates to Ansi-C requirement).

  for (int k = 0; k < BT_num_cols; k++) {
    const Real *this_BT_col = B_data + k * BT_col_stride;
    const Real *this_A_row = A_data + i * A_row_stride
        + BT_row_start * A_col_stride;
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
static void _add_mat_blockmat(Real *data, MatrixDim dim, const Real *A_data,
                              int A_num_rows, int A_num_cols, int A_row_stride,
                              int A_col_stride,
                              const CuBlockMatrixData *B_cu_data,
                              int B_num_blocks, Real alpha, Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index into "data"
  int j = blockIdx.y * blockDim.y + threadIdx.y; // block-index into B.
  if (i >= A_num_rows || j >= B_num_blocks)
    return;

  const CuBlockMatrixData &block_data = B_cu_data[j];

  int B_row_start = block_data.row_offset, B_col_start = block_data.col_offset,
      B_num_rows = block_data.matrix_dim.rows, B_num_cols =
          block_data.matrix_dim.cols, B_row_stride =
          block_data.matrix_dim.stride;
  // Cast from void;
  const Real *B_data = static_cast<Real*>(block_data.matrix_data);
  // we avoided a bunch of hassle by doing this (relates to Ansi-C requirement).

  for (int k = 0; k < B_num_cols; k++) {
    const Real *this_B_col = B_data + k;
    const Real *this_A_row = A_data + i * A_row_stride
        + B_row_start * A_col_stride;
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
                               const Real *D_data, int D_row_stride,
                               int D_col_stride, Real alpha, Real beta) {
  int b = blockIdx.x * blockDim.x + threadIdx.x; // block-index into B.
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row-index into b'th block
  int j = blockIdx.z * blockDim.z + threadIdx.z; // col-index into b'th block
  if (b >= num_blocks)
    return;

  const CuBlockMatrixData &block_data = B_cu_data[b];

  if (i >= block_data.matrix_dim.rows || j >= block_data.matrix_dim.cols)
    return; // we're outside the dimensions of the b'th block.

  // B_elem is the element of B we're writing to.
  Real *B_elem = reinterpret_cast<Real*>(block_data.matrix_data)
      + i * block_data.matrix_dim.stride + j;

  Real B_val = *B_elem;

  // B_row and B_col are the (row, col) index into the full matrix B.
  int B_row = block_data.row_offset + i, B_col = block_data.col_offset + j;

  const Real *C_row_data = C_data + C_row_stride * B_row, *D_col_data = D_data
      + D_col_stride * B_col;

  Real sum = 0.0;
  for (int k = 0; k < C_num_cols; k++) {
    sum += C_row_data[k * C_col_stride] * D_col_data[k * D_row_stride];
  }
  *B_elem = alpha * sum + beta * B_val;
}

template<typename Real>
__global__
static void _blockadd_mat_blockmat_trans(Real *data, MatrixDim dim,
                                         const Real *A_data, int A_num_rows,
                                         int A_num_cols, int A_row_stride,
                                         int A_col_stride,
                                         const CuBlockMatrixData *B_cu_data,
                                         int B_num_blocks, Real alpha,
                                         Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index into "data"
  int j = blockIdx.y * blockDim.y + threadIdx.y; // block-index into B.
  if (i >= A_num_rows || j >= B_num_blocks)
    return;

  const CuBlockMatrixData &cu_data = B_cu_data[j];

  // BT means B transposed.
  int BT_row_start = cu_data.col_offset, BT_col_start = cu_data.row_offset,
      BT_num_rows = cu_data.matrix_dim.cols, BT_num_cols =
          cu_data.matrix_dim.rows, BT_col_stride = cu_data.matrix_dim.stride;
  // Cast from void;
  const Real *B_data = static_cast<Real*>(cu_data.matrix_data);
  // we avoided a bunch of hassle by doing this (relates to Ansi-C requirement).

  for (int k = 0; k < BT_num_cols; k++) {
    const Real *this_BT_col = B_data + k * BT_col_stride;
    const Real *this_A_row = A_data + i * A_row_stride
        + BT_row_start * A_col_stride;
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
static void _sum_column_ranges(Real *data, MatrixDim dim, const Real *src_data,
                               MatrixDim src_dim, const Int32Pair *indices) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= dim.rows || col >= dim.cols)
    return;
  int dst_index = row * dim.stride + col, src_start_index = row * src_dim.stride
      + indices[col].first, src_end_index = row * src_dim.stride
      + indices[col].second;
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
  int src_index_start = indexes[row].first, src_index_end = indexes[row].second;
  for (int row_index = src_index_start; row_index < src_index_end; row_index++)
    data[dst_index] += src_data[row_index * src_dim.stride + col];
}

template<typename Real>
__global__
static void _soft_hinge(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j * d.stride, src_index = i + j * src_stride;
  // compute the function y[index] = log(1 + exp(x[index]))
  if (i < d.cols && j < d.rows) {
    Real val = x[src_index], result;
    if (val >= 10.0)
      result = val; // function approaches y=x as x gets large
    else
      result = log1p(exp(val));
    y[dst_index] = result;
  }
}

template<typename Real>
__global__
static void _group_pnorm(Real *y, const Real *x, MatrixDim d, int src_stride,
                         int group_size, Real power) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < d.rows && i < d.cols) {
    int dst_index = i + j * d.stride;
    Real tmp = 0;
    int src_begin_index = i * group_size + j * src_stride;
    int src_end_index = src_begin_index + group_size;
    for (int src_index = src_begin_index; src_index < src_end_index;
        src_index++) {
      tmp += pow(std::abs(x[src_index]), power);
    }
    tmp = pow(tmp, Real(1.0 / power));
    if (!isnan(tmp)) {
      y[dst_index] = tmp;
    } else {
      Real max_value = x[src_begin_index], min_value = max_value;
      for (int src_index = src_begin_index + 1; src_index < src_end_index;
          src_index++) {
        if (x[src_index] > max_value)
          max_value = x[src_index];
        if (x[src_index] < min_value)
          min_value = x[src_index];
      }
      tmp = 0.0;
      // let max_value be the largest abs(value)
      Real max_abs_value = (max_value > -min_value ? max_value : -min_value);
      if (max_abs_value == 0) {
        y[dst_index] = 0.0;
      } else {
        for (int src_index = src_begin_index; src_index < src_end_index;
            src_index++) {
          Real x_scaled = x[src_index] / max_abs_value;
          tmp += pow(std::abs(x_scaled), Real(power));
        }
        y[dst_index] = pow(tmp, Real(1.0 / power)) * max_abs_value;
      }
    }
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
  int dst_index = i + j * d.stride, src_index = i + j * src_stride;
  if (i < d.cols && j < d.rows) {
    Real res = 1.0 / (1.0 + exp(-x[src_index]));
    y[dst_index] = res;
  }
}

template<typename Real>
__global__
static void _diff_sigmoid(Real*eout, const Real*e, const Real*y, MatrixDim d,
                          int e_stride, int y_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j * d.stride;
  int e_index = i + j * e_stride;
  int y_index = i + j * y_stride;
  if (i < d.cols && j < d.rows)
    eout[dst_index] = y[y_index] * (1.0 - y[y_index]) * e[e_index];
}

template<typename Real>
__global__
static void _tanh(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j * d.stride, src_index = i + j * src_stride;
  if (i < d.cols && j < d.rows) {
    Real exp_2x = exp(2.0 * x[src_index]);
    Real res;
    if (isinf(exp_2x)) {
      res = 1.0;
    } else {
      res = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
    y[dst_index] = res;
  }
}

template<typename Real>
__global__
static void _diff_tanh(Real*eout, const Real*e, const Real*y, MatrixDim d,
                       int e_stride, int y_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j * d.stride;
  int e_index = i + j * e_stride;
  int y_index = i + j * y_stride;
  if (i < d.cols && j < d.rows)
    eout[dst_index] = (1.0 - y[y_index] * y[y_index]) * e[e_index];
}

template<typename Real>
__global__
static void _parametric_relu(Real* y, const Real* x, MatrixDim d, int src_stride,
                             const Real* a, const Real* b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j * d.stride,
      src_index = i + j * src_stride;
  if (i < d.cols && j < d.rows) {
    Real res = (x[src_index] > 0.0) ? a[i] * x[src_index] : b[i] * x[src_index];
    y[dst_index] = res;
  }
}

template<typename Real>
__global__
static void _diff_parametric_relu(Real* eout, const Real* e, const Real* y,
                                  MatrixDim d, int e_stride, int y_stride,
                                  const Real* a, const Real* b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j * d.stride;
  int e_index   = i + j * e_stride;
  int y_index   = i + j * y_stride;
  if (i < d.cols  && j < d.rows )
    eout[dst_index] = (y[y_index] > 0.0 ? a[i] * e[e_index] : b[i] * e[e_index]);
}

template<typename Real>
__global__
static void _heaviside(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j * d.stride, src_index = i + j * src_stride;
  if (i < d.cols && j < d.rows) {
    Real res = (x[src_index] > 0.0 ? 1.0 : 0.0);
    y[dst_index] = res;
  }
}

template<typename Real>
__global__
static void _softmax_reduce(Real*y, const Real*x, MatrixDim d, int src_stride) {
  __shared__ Real smem[CU1DBLOCK];
  const int i = blockIdx.x;
  const int x_start = i * src_stride;
  const int y_start = i * d.stride;
  const int tid = threadIdx.x;

  // find max element of the row
  // reduce to CU1DBLOCK elements per row.
  Real tmax = sizeof(Real) == sizeof(float) ? -CUDART_INF_F : -CUDART_INF;
  for (int j = tid; j < d.cols; j += CU1DBLOCK) {
    tmax = fmax(tmax, x[x_start + j]);
  }
  smem[tid] = tmax;
  __syncthreads();

  // reduce to 2x warpSize elements per row
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      smem[tid] = fmax(smem[tid], smem[tid + shift]);
    }
    __syncthreads();
  }

  // reduce to 1 element per row
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      smem[tid] = fmax(smem[tid], smem[tid + shift]);
    }
  }

  // broadcast max to all threads
  __syncthreads();
  Real max = smem[0];

  // sum_j(exp(x(i,j)-max))
  // reduce to CU1DBLOCK elements per row.
  Real tsum = Real(0);
  for (int j = tid; j < d.cols; j += CU1DBLOCK) {
    tsum += exp(x[x_start + j] - max);
  }
  smem[tid] = tsum;
  __syncthreads();

  // reduce to 2x warpSize elements per row
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      smem[tid] += smem[tid + shift];
    }
    __syncthreads();
  }

  // reduce to 1 element per row
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      smem[tid] += smem[tid + shift];
    }
  }

  // broadcast sum to all threads
  __syncthreads();
  Real inv_sum = Real(1) / smem[0];

  // normalize the row
  for (int j = tid; j < d.cols; j += CU1DBLOCK) {
    y[y_start + j] = exp(x[x_start + j] - max) * inv_sum;
  }
}

// The output y_i = scale * x_i,
// and we want to RMS value of the y_i to equal target_rms,
// so y^t y = D * target_rms^2 (if y is one row of the input).
// we need to have scale = 1.0 / sqrt(x^t x / (D * target_rms^2)).
// there is also flooring involved, to avoid division-by-zero
// problems.  It's important for the backprop, that the floor's
// square root is exactly representable as float.
// If add_log_stddev is true, log(max(epsi, sqrt(x^t x / D)))
// is an extra dimension of the output.
//
// 1D grid is used. Each 256-thread block works on 1 row of the data matrix.
// The block is also of 1D. Strided memory access is used if the length of the
// row is longer than 256.
template<typename Real>
__global__
static void _normalize_per_row(Real *y, int y_stride, const Real *x,
                               MatrixDim x_d, Real target_rms,
                               bool add_log_stddev) {
  const int i = blockIdx.x;
  const int tid = threadIdx.x;
  const Real* x_row = x + i * x_d.stride;
  __shared__ Real ssum[CU1DBLOCK];

  // Reduce x_j^2 to CU1DBLOCK elements per row
  Real tsum = Real(0);
  for (int j = tid; j < x_d.cols; j += CU1DBLOCK) {
    tsum += x_row[j] * x_row[j];
  }
  ssum[tid] = tsum;
  __syncthreads();

  // Tree reduce to 2x warpSize elements per row
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      ssum[tid] += ssum[tid + shift];
    __syncthreads();
  }

  // Reduce last warp to 1 element per row.
  // Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      ssum[tid] += ssum[tid + shift];
    }
  }

  const Real kSquaredNormFloor = 1.3552527156068805425e-20; // 2^-66
  if (tid == 0) {
    ssum[0] = sqrt(
        fmax(ssum[0] / (target_rms * target_rms * x_d.cols), kSquaredNormFloor));
  }

  // Broadcast floored stddev to all threads.
  __syncthreads();
  const Real stddev_div_target_rms = ssum[0];
  const Real scale = Real(1) / stddev_div_target_rms;

  // Store normalized input to output
  Real* y_row = y + i * y_stride;
  for (int j = tid; j < x_d.cols; j += CU1DBLOCK) {
    y_row[j] = x_row[j] * scale;
  }

  if (tid == 0 && add_log_stddev) {
    y_row[x_d.cols] = log(stddev_div_target_rms * target_rms);
  }
}


template<typename Real>
__global__
static void _diff_normalize_per_row(Real *id, int id_stride, const Real *iv,
                                    MatrixDim iv_dim, const Real* od,
                                    int od_stride, Real target_rms,
                                    bool add_log_stddev) {

  const Real kSquaredNormFloor = 1.3552527156068805425e-20; // 2^-66
  const Real kInvNormFloor = 8589934592.0;

  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const Real* iv_row = iv + i * iv_dim.stride;
  const Real* od_row = od + i * od_stride;

  // reduce to CU1DBLOCK elements per row
  Real dot_products = Real(0);
  Real in_norm = Real(0);
  for (int j = tid; j < iv_dim.cols; j += CU1DBLOCK) {
    const Real iv_ij = iv_row[j];
    dot_products += iv_ij * od_row[j];
    in_norm += iv_ij * iv_ij;
  }
  __shared__ Real sprod[CU1DBLOCK];
  __shared__ Real snorm[CU1DBLOCK];
  sprod[tid] = dot_products;
  snorm[tid] = in_norm;
  __syncthreads();

  // reduce to 2x warpSize elements per row
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      sprod[tid] += sprod[tid + shift];
      snorm[tid] += snorm[tid + shift];
    }
    __syncthreads();
  }

  // reduce to 1 element per row
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      sprod[tid] += sprod[tid + shift];
      snorm[tid] += snorm[tid + shift];
    }
  }

  // broadcast the sum results
  __syncthreads();
  dot_products = sprod[0];
  in_norm = snorm[0];

  Real log_stddev_deriv;
  if (add_log_stddev) {
    log_stddev_deriv = Real(1) / max(in_norm, iv_dim.cols * kSquaredNormFloor)
        * od_row[iv_dim.cols];
  }

  const Real inv_d_scaled = Real(1) / (iv_dim.cols * target_rms * target_rms);
  in_norm = Real(1) / sqrt(max(in_norm * inv_d_scaled, kSquaredNormFloor));

  const Real f = in_norm == kInvNormFloor ? Real(0) : in_norm;
  dot_products *= f * f * f * inv_d_scaled;

  for (int j = tid; j < iv_dim.cols; j += CU1DBLOCK) {
    const Real iv_ij = iv_row[j];
    Real id_ij = id[i * id_stride + j];
    if (add_log_stddev) {
      id_ij += log_stddev_deriv * iv_ij;
    }
    if (id != od) {
      id_ij += in_norm * od_row[j];
    } else {
      id_ij *= in_norm;
    }
    id_ij -= dot_products * iv_ij;
    id[i * id_stride + j] = id_ij;
  }
}

// Per-row log-softmax operation on 'x', with writing to 'y'.
// note, x and y may point to the same memory.  This is equivalent to setting
// matrix y to matrix x and then, for each row of y, subtracting the offset that
// will make exp(y.row[j]) sum to 1 for each row j.
//
// It expects to be called with CU1DBLOCK threads.
// The number of blocks [i.e. the gridDim] equals to y_dim.rows,
// so one block of threads processes each row.  x and y are
// expected to have the same dimension, but possibly different row strides.
template<typename Real>
__global__
static void _log_softmax_reduce(Real* y, const Real* x, MatrixDim y_dim,
                                int x_stride) {
  __shared__ Real smem[CU1DBLOCK];
  const int i = blockIdx.x;
  const int x_start = i * x_stride;
  const int y_start = i * y_dim.stride;
  const int tid = threadIdx.x;

  // find max element of the row
  // reduce to CU1DBLOCK elements per row.
  Real tmax = -1e20;
  for (int j = tid; j < y_dim.cols; j += CU1DBLOCK) {
    tmax = fmax(tmax, x[x_start + j]);
  }
  smem[tid] = tmax;
  __syncthreads();

  // reduce to 2x warpSize elements per row
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      smem[tid] = fmax(smem[tid], smem[tid + shift]);
    }
    __syncthreads();
  }

  // reduce to 1 element per row
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      smem[tid] = fmax(smem[tid], smem[tid + shift]);
    }
  }

  // broadcast max to all threads
  __syncthreads();
  Real max = smem[0];

  // sum_j(exp(x(i,j)-max))
  // reduce to CU1DBLOCK elements per row.
  Real tsum = Real(0);
  for (int j = tid; j < y_dim.cols; j += CU1DBLOCK) {
    tsum += exp(x[x_start + j] - max);
  }
  smem[tid] = tsum;
  __syncthreads();

  // reduce to 2x warpSize elements per row
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      smem[tid] += smem[tid + shift];
    }
    __syncthreads();
  }

  // reduce to 1 element per row
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      smem[tid] += smem[tid + shift];
    }
  }

  // broadcast sum to all threads
  __syncthreads();
  Real log_sum = log(smem[0]);

  // normalize the row
  for (int j = tid; j < y_dim.cols; j += CU1DBLOCK) {
    y[y_start + j] = x[x_start + j] - max - log_sum;
  }
}

template<typename Real>
__global__
static void _splice(Real* y, const Real* x, const int32_cuda* off,
                    MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d_out.stride;
  if (i < d_out.cols && j < d_out.rows) {
    int32_cuda src_col = i % d_in.cols;
    int32_cuda src_row = j + off[i / d_in.cols];
    if (src_row < 0)
      src_row = 0;
    if (src_row >= d_in.rows)
      src_row = d_in.rows - 1;
    y[index] = x[src_col + src_row * d_in.stride];
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
    int32_cuda index_sp = (j * (j + 1) / 2) + i;
    y[index_sp] = 0.5 * (x[index1] + x[index2]);
  }
}

template<typename Real>
__global__
static void _take_lower(const Real* x, Real* y, MatrixDim d_in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col-index
  if (j > i || i >= d_in.rows)
    return;
  int index = i * d_in.stride + j;
  Real val = x[index];
  int index_sp = (i * (i + 1) / 2) + j;
  y[index_sp] = val;
}

template<typename Real>
__global__
static void _take_upper(const Real* x, Real* y, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // col-index
  if (j < i || j >= d_in.rows)
    return;
  int32_cuda index = i * d_in.stride + j;
  int32_cuda index_sp = (j * (j + 1) / 2) + i;
  y[index_sp] = x[index];
}

template<typename Real>
__global__
static void _vec_copy_diag_from_packed(Real* y, const Real* x, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i + 1) * (i + 2) / 2) - 1;
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
      src_index = (i * (i + 1) / 2) + j;
    } else { // transpose.
      src_index = (j * (j + 1) / 2) + i;
    }
    y[dst_index] = x[src_index];
  }
}

template<typename Real>
__global__
static void _copy(Real* y, const Real* x, const int32_cuda* copy_from,
                  MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d_out.stride;
  if (i < d_out.cols && j < d_out.rows) {
    int32_cuda src_col = copy_from[i];
    if (src_col >= 0 && src_col < d_in.cols) {
      y[index] = x[src_col + j * d_in.stride];
    } else {
      y[index] = 1.0 / 0.0;
    }
  }
}

template<typename Real>
__global__
static void _one(Real* x, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    x[i] = 1.0;
  }
}

template<typename Real>
__global__
static void _randomize(Real* y, const Real* x, const int32_cuda* copy_from,
                       MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d_out.stride;
  if (i < d_out.cols && j < d_out.rows) {
    int32_cuda src_row = copy_from[j];
    y[index] = x[i + src_row * d_in.stride];
  }
}

template<typename Real>
__global__
static void _regularize_l1(Real* wei, Real* grad, Real l1, Real lr, MatrixDim d,
                           int stride_grad) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride, grad_index = i + j * stride_grad;
  if (i < d.cols && j < d.rows) {

    if (wei[index] == 0.0)
      return; //skip L1 if zero weight!

    Real l1_signed = l1;
    if (wei[index] < 0.0) //flip sign
      l1_signed = -l1;

    Real before = wei[index];
    //simulate update
    Real after = wei[index] - lr * grad[grad_index] - l1_signed;
    if ((after > 0.0) ^ (before > 0.0)) { //sign changed?
      wei[index] = 0.0;
      grad[grad_index] = 0.0;
    } else {
      wei[index] -= l1_signed;
    }
  }
}

template<typename Real>
__global__
static void _find_row_max_id(const Real* mat, Real* vec_val, int32_cuda* vec_id,
                             MatrixDim d) {
  const int32_cuda i = blockIdx.x;
  const int32_cuda base = i * d.stride;
  const int32_cuda tid = threadIdx.x;

  __shared__ Real smax[CU1DBLOCK];
  __shared__ int32_cuda sidx[CU1DBLOCK];

  Real tmax = -1e20;
  int32_cuda tidx = -1;

  // Loop over blocks for coalesced memory access.
  for (int32_cuda j = tid; j < d.cols; j += CU1DBLOCK) {
    const Real val = mat[base + j];
    if (val > tmax) {
      tmax = val;
      tidx = j;
    }
  }

  smax[tid] = tmax;
  sidx[tid] = tidx;

  // Parallel reduce
#pragma unroll
  for (int32_cuda num_working_threads = CU1DBLOCK / 2;
      num_working_threads >= warpSize; num_working_threads >>= 1) {
    __syncthreads();
    if (tid < num_working_threads) {
      if (smax[tid + num_working_threads] > smax[tid]) {
        smax[tid] = smax[tid + num_working_threads];
        sidx[tid] = sidx[tid + num_working_threads];
      }
    }
  }
  // Warp reduce without __syncthreads()
  // (note.: synchronizes implicitly within a warp at the multiprocessor)
  if (tid < warpSize / 2) {
#pragma unroll
    for (int32_cuda num_working_threads = warpSize / 2; num_working_threads > 0;
        num_working_threads >>= 1) {
      if (smax[tid + num_working_threads] > smax[tid]) {
        smax[tid] = smax[tid + num_working_threads];
        sidx[tid] = sidx[tid + num_working_threads];
      }
    }
  }

  if (tid == 0) {
    if (vec_val) {
      vec_val[i] = smax[0];
    }
    vec_id[i] = sidx[0];
  }
}

template<typename Real>
__global__
static void _diff_xent(const int32_cuda* vec_tgt, Real* mat_net_out,
                       Real* vec_log_post, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0)
    return;
  if (j < d.rows) {
    int32_cuda index = vec_tgt[j] + j * d.stride;
    Real value = mat_net_out[index];
    if (value < 1e-20)
      value = 1e-20;
    vec_log_post[j] = log(value);
    mat_net_out[index] -= 1.0;
  }
}

template<typename Real>
__global__
static void _diff_softmax(Real* x, const MatrixDim dim, const Real* value,
                          const int value_stride, const Real* diff,
                          const int diff_stride) {
  __shared__ Real ssum[CU1DBLOCK];
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const int value_start = i * value_stride;
  const int diff_start = i * diff_stride;
  const int x_start = i * dim.stride;

  // Loop along the matrix row. Reduce to CU1DBLOCK elements per row.
  Real tsum = Real(0);
  for (int j = tid; j < dim.cols; j += CU1DBLOCK) {
    tsum += value[value_start + j] * diff[diff_start + j];
  }
  ssum[tid] = tsum;
  __syncthreads();

  // Tree reduce to 2x warpSize elements.
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      ssum[tid] += ssum[tid + shift];
    }
    __syncthreads();
  }

  // Warp reduce to 1 element. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      ssum[tid] += ssum[tid + shift];
    }
  }

  // Broadcast result to all threads
  __syncthreads();
  const Real pe = ssum[0];

  // Apply element-wise x = value * (diff - pe)
  for (int j = tid; j < dim.cols; j += CU1DBLOCK) {
    x[x_start + j] = value[value_start + j] * (diff[diff_start + j] - pe);
  }
}

// Differentiate backward through the log softmax function.
// "out_value" is the log softmax output. Does, for each row i,
// in_deriv(i) =  out_deriv(i) - sum(out_deriv(i)) .* exp(out_value(i))
// ???(i) is row-vector.
// CUDA thread layout: 1 thread block (CU1DBLOCK == 256 threads) per matrix-row.
template<typename Real>
__global__
static void _diff_log_softmax(const MatrixDim in_deriv_dim,
                              const Real* out_value, const int out_value_stride,
                              const Real* out_deriv, const int out_deriv_stride,
                              Real* in_deriv) {

  __shared__ Real ssum[CU1DBLOCK];
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const int out_value_start = i * out_value_stride;
  const int out_deriv_start = i * out_deriv_stride;
  const int in_deriv_start = i * in_deriv_dim.stride;

  // Loop along the matrix row. Reduce to CU1DBLOCK elements per row.
  Real tsum = Real(0);
  for (int j = tid; j < in_deriv_dim.cols; j += CU1DBLOCK) {
    tsum += out_deriv[out_deriv_start + j];
  }
  ssum[tid] = tsum;
  __syncthreads();

  // Tree reduce to 2x warpSize elements.
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      ssum[tid] += ssum[tid + shift];
    }
    __syncthreads();
  }

  // Warp reduce to 1 element. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
#   pragma unroll
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      ssum[tid] += ssum[tid + shift];
    }
  }

  // Broadcast result to all threads
  __syncthreads();
  const Real sum_e = ssum[0];

  // Apply element-wise x = out_deriv - exp(value) * sum_e
  for (int j = tid; j < in_deriv_dim.cols; j += CU1DBLOCK) {
    in_deriv[in_deriv_start + j] = out_deriv[out_deriv_start + j]
        - exp(out_value[out_value_start + j]) * sum_e;
  }
}


/**
 this function computes the core part of the LSTM nonlinearity.
 @param [in] in      A matrix, of dimension num_rows by 5*cell_dim
                     (i.e. its num-cols must be a multiple of 5).
                     The column-space is interpreted as 5
                     consecutive blocks, each of dimension cell_dim,
                     which we name:
                     (i_part, f_part, c_part, o_part, c_{t-1}).
                     If 'have_dropout_mask' is nonzero, each row of
                     'in' will have 3 extra elements, interpreted
                     as dropout masks/scales for i_t, f_t and o_t.
 @param [in] params  A matrix, of dimension 3 by cell_dim,
                     with rows containing the 3 diagonal parameter matrices
                     used in LSTMs, namely
                     w_{ic}, w_{fc} and w_{oc}.
 @param [out] out    A matrix, of dimension num_rows by 2*cell_dim.
                     The quantities c_t and m_t respectively are put there
                     (in two blocks of column-dimension cell_dim),
                     according to the following equations:

                     i_t = Sigmoid(i_part + w_{ic}*c_{t-1})
                     f_t = Sigmoid(f_part + w_{fc}*c_{t-1})
                     c_t = f_t*c_{t-1} + i_t * Tanh(c_part)
                     o_t = Sigmoid(o_part + w_{oc}*c_t)
                     m_t = o_t * Tanh(c_t)

We use 1D thread block with CU1DBLOCK threads.
It works best when cell_dim is a multiple of CU1DBLOCK.
We use 1d Grid. Each block is working on one row of the in and out matrices.
*/
template<typename Real>
__global__
static void _lstm_nonlinearity(const Real* in, const int in_stride,
                               const Real* params, const int params_stride,
                               const int out_stride, const int cell_dim,
                               const int have_dropout_mask, const int num_rows,
                               Real* out) {
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const Real* i_part = in + i * in_stride;
  const Real* f_part = in + i * in_stride + cell_dim;
  const Real* c_part = in + i * in_stride + cell_dim * 2;
  const Real* o_part = in + i * in_stride + cell_dim * 3;
  const Real* c_tm1 = in + i * in_stride + cell_dim * 4;
  const Real* w_ic = params;
  const Real* w_fc = params + params_stride;
  const Real* w_oc = params + params_stride * 2;
  Real* c_t = out + i * out_stride;
  Real* m_t = out + i * out_stride + cell_dim;
  Real i_scale = (have_dropout_mask ? in[i * in_stride + cell_dim * 5] : 1),
       f_scale = (have_dropout_mask ? in[i * in_stride + cell_dim * 5 + 1] : 1),
       o_scale = (have_dropout_mask ? in[i * in_stride + cell_dim * 5 + 2] : 1);

  for (int j = tid; j < cell_dim; j += CU1DBLOCK) {
    Real c_tm1_j = c_tm1[j];
    Real i_t_j = Real(1) / (Real(1) + exp(-i_part[j] - w_ic[j] * c_tm1_j));
    Real f_t_j = Real(1) / (Real(1) + exp(-f_part[j] - w_fc[j] * c_tm1_j));
    Real c_t_j = f_t_j * f_scale * c_tm1_j + i_t_j * i_scale * tanh(c_part[j]);
    Real o_t_j = Real(1) / (Real(1) + exp(-o_part[j] - w_oc[j] * c_t_j));
    c_t[j] = c_t_j;
    m_t[j] = o_t_j * o_scale * tanh(c_t_j);
  }
}


/**
   This function does the 'backward' pass corresponding to the function
   ComputeLstmNonlinearity.  It's a little more complicated than you might
   expect because of the 'self-repair' mechanism that we use to prevent the
   sigmoid and tanh nonlinearities oversaturating,  and because of the
   average-activation and average-derivative stats that we store for these
   nonlinearites (these stats are used both to control the self-repair
   mechanism, and for diagnostic purposes).

   Because the forward pass computes various intermediate values that are not
   output, this function actually has to do the same computations as the
   forward pass before it actually does the backprop.

   In the following description, `C` is for `cell_dim`, `N` is for `num_rows`.

 @param [in]  input  The same as in ComputeLstmNonlinearity().
                     A matrix, of dimension N by 5C (i.e. its num-cols must be
                     a multiple of 5).  The column-space is interpreted as 5
                     consecutive blocks, each of dimension C, which we name:
                     (i_part, f_part, c_part, o_part, c_{t-1}).
                     If 'have_dropout_mask' is nonzero, each row of
                     'in' will have 3 extra elements, interpreted
                     as dropout masks/scales for i_t, f_t and o_t.
 @param [in] params  The same as in ComputeLstmNonlinearity().
                     A matrix, of dimension 3 by C, with rows containing the
                     three diagonal parameter matrices used in LSTMs, namely
                     w_{ic}, w_{fc} and w_{oc}.
 @param [in] output_deriv
                     A matrix, of dimension N by 2C, containing the derivative
                     of the objective function we're backpropagating,
                     w.r.t. the quantities c_t and m_t (in two blocks of
                     column-dimension C).
 @param [in] deriv_sum_in
                     This is used in the self-repair code to identify
                     oversaturated nonlinearities.
                     It is a matrix, of dimension 5 by C, corresponding to
                     the totals of the derivatives of the 5 sigmoid and tanh
                     nonlinearities, in they order they appear in the equations
                     in the documentation of ComputeLstmNonlinearity()
                     respectively,
                     they appear in the equations for (i_t, f_t, c_t, o_t, m_t).
                     This will be divided by 'count_in' to get the average
                     derivative value so far, for each of the nonlinearities.
 @param [in] self_repair_config
                     A vector of dimension 10, containing the configuration of
                     the self-repair to be used for the 5 nonlinearities.
                     The first 5 elements are the self_repair_lower_threshold
                     values (typically 0.05 for sigmoid and 0.2 for tanh),
                     and the next 5 elements are the corresponding
                     self-repair-scales (typically 10^-5).
 @param [in] count_in  The data-count that corresponds to the stats in
                     'deriv_sum_in' at entry to the function.
                     This function should tolerate the count being zero
                     (in that case, it is free to do the self-repair or not,
                     as this should only happen on the 1st minibatch of each
                     training job).
 @param [out] input_deriv
                     May be NULL; if not, this function writes, to this
                     location, the backpropagated derivative of the objective
                     function w.r.t. the 'input' matrix.  This matrix should
                     have the same dimension as 'input' i.e.  N by 5C.  In
                     addition to the regular backpropagated derivative, the
                     output will include small values relating to 'self-repair'.
 @param [out] params_deriv
                     May be NULL; if not, this is where this function *writes*
                     [not adds] the backpropagated derivative of the objective
                     function w.r.t. 'params'; it should have the same dimension
                     as 'params' (3 by C).  (This matrix will then be processed
                     by the natural gradient code and added to the appropriate
                     copy of the parameter matrix, outside this function).
 @param [out] value_sum_out
                     Must be NULL if params_deriv is NULL; if not, a matrix of
                     dimension 5 by C.  This function *adds* to this location
                     the total value of each of the sigmoid/tanh nonlinearities
                     that it computes (this is for diagnostic purposes).
 @param [out] deriv_sum_out
                     Must be NULL if params_deriv is NULL; if not, a matrix of
                     dimension 5 by C; this function *adds* to this location the
                     total of the derivative of each of the sigmoid/tanh
                     nonlinearities that it computes (this is for diagnostic
                     purposes and to control the self-repair).  This function
                     should tolerate the case when 'deriv_sum_out' points to the
                     same data as 'deriv_sum_in'.
 @param [out] self_repair_sum_out
                     Must be NULL if params_deriv is NULL; if not, a matrix of
                     dimension 5 by C; this function *writes* to this location
                     the sum of the number of times the self-repair code was
                     activated (integer values 0 <= k <= N).  This will be
                     processed outside this function into self-repair stats for
                     diagnostics.
// Use 2D block (8x32 threads) as we need to compute column sum.
// Use 1D grid to cover the data matrix `cell_dim`.
*/
template<typename Real>
__global__
static void _diff_lstm_nonlinearity(const int cell_dim, const int have_dropout_mask,
                                    const int num_rows,
                                    const Real* input, const int input_stride,
                                    const Real* params, const int params_stride,
                                    const Real* output_deriv,
                                    const int output_deriv_stride,
                                    const double* deriv_sum_in,
                                    const int deriv_sum_in_stride,
                                    const Real* self_repair_config,
                                    double count, Real* input_deriv,
                                    const int input_deriv_stride,
                                    Real* params_deriv,
                                    const int params_deriv_stride,
                                    double* value_sum_out,
                                    const int value_sum_out_stride,
                                    double* deriv_sum_out,
                                    const int deriv_sum_out_stride,
                                    Real* self_repair_sum_out,
                                    const int self_repair_sum_out_stride) {
  __shared__ Real smem[CU1DBLOCK];

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int grid_stride = gridDim.y * blockDim.y;
  const int i0 = blockIdx.y * blockDim.y + threadIdx.y;

  Real w_ic_deriv_sum = 0;
  Real w_fc_deriv_sum = 0;
  Real w_oc_deriv_sum = 0;

  Real i_t_value_sum = 0, i_t_deriv_sum = 0;
  Real f_t_value_sum = 0, f_t_deriv_sum = 0;
  Real c_part_value_sum = 0, c_part_deriv_sum = 0;
  Real o_t_value_sum = 0, o_t_deriv_sum = 0;
  Real c_t_value_sum = 0, c_t_deriv_sum = 0;

  bool update_sr[5];

  if (j < cell_dim) {
    const Real w_ic = params[j];
    const Real w_fc = params[params_stride + j];
    const Real w_oc = params[2 * params_stride + j];

    const Real* sr_config = self_repair_config;
#   pragma unroll
    for (int i = 0; i < 5; i++) {
      update_sr[i] = deriv_sum_in[i * deriv_sum_in_stride + j] / count
          < sr_config[i];
    }
    const Real i_t_self_repair = (update_sr[0] ? sr_config[5] : 0);
    const Real f_t_self_repair = (update_sr[1] ? sr_config[6] : 0);
    const Real c_part_self_repair = (update_sr[2] ? sr_config[7] : 0);
    const Real o_t_self_repair = (update_sr[3] ? sr_config[8] : 0);
    const Real c_t_self_repair = (update_sr[4] ? sr_config[9] : 0);


    for (int i = i0; i < num_rows; i += grid_stride) {
      const Real i_part = input[i * input_stride + j];
      const Real f_part = input[i * input_stride + j + cell_dim];
      const Real c_part = input[i * input_stride + j + 2 * cell_dim];
      const Real o_part = input[i * input_stride + j + 3 * cell_dim];
      const Real c_prev = input[i * input_stride + j + 4 * cell_dim];


      const Real i_scale = (have_dropout_mask ?
                            input[i * input_stride + cell_dim * 5] : 1),
                 f_scale = (have_dropout_mask ?
                            input[i * input_stride + cell_dim * 5 + 1] :1),
                 o_scale = (have_dropout_mask ?
                            input[i * input_stride + cell_dim * 5 + 2] :1);


      const Real i_t = Real(1) / (1 + exp(-i_part - w_ic * c_prev));
      const Real f_t = Real(1) / (1 + exp(-f_part - w_fc * c_prev));
      const Real tanh_c_part = tanh(c_part);
      const Real c_t = f_t * f_scale * c_prev + i_t * i_scale * tanh_c_part;
      const Real o_t = 1 / (1 + exp(-o_part - w_oc * c_t));
      const Real tanh_c_t = tanh(c_t);

      const Real i_t_deriv = i_t * (1 - i_t);
      const Real f_t_deriv = f_t * (1 - f_t);
      const Real c_part_deriv = 1 - tanh_c_part * tanh_c_part;
      const Real o_t_deriv = o_t * (1 - o_t);
      const Real c_t_deriv = 1 - tanh_c_t * tanh_c_t;

      if (params_deriv) {
        i_t_value_sum += i_t;
        f_t_value_sum += f_t;
        c_part_value_sum += tanh_c_part;
        o_t_value_sum += o_t;
        c_t_value_sum += tanh_c_t;

        i_t_deriv_sum += i_t_deriv;
        f_t_deriv_sum += f_t_deriv;
        c_part_deriv_sum += c_part_deriv;
        o_t_deriv_sum += o_t_deriv;
        c_t_deriv_sum += c_t_deriv;
      }

      const Real dc_t_out = output_deriv[i * output_deriv_stride + j];
      const Real dm_t = output_deriv[i * output_deriv_stride + j + cell_dim];

      const Real dtanh_c_t = o_t * o_scale * dm_t;
      const Real do_t = o_scale * tanh_c_t * dm_t;
      const Real do_t_input = (o_t_deriv * do_t
          - (2 * o_t - 1) * o_t_self_repair);

      const Real dc_t = (c_t_deriv * dtanh_c_t + dc_t_out + do_t_input * w_oc)
          - tanh_c_t * c_t_self_repair;
      const Real dtanh_c_part = i_t * i_scale * dc_t;
      const Real df_t = dc_t * f_scale * c_prev;
      const Real df_t_input = (df_t * f_t_deriv
                               - (2 * f_t - 1) * f_t_self_repair);
      const Real di_t = dc_t * i_scale * tanh_c_part;
      const Real di_t_input = (di_t * i_t_deriv
                               - (2 * i_t - 1) * i_t_self_repair);

      if (params_deriv) {
        w_ic_deriv_sum += c_prev * di_t_input;
        w_fc_deriv_sum += c_prev * df_t_input;
        w_oc_deriv_sum += c_t * do_t_input;
      }

      const Real dc_prev = w_ic * di_t_input + w_fc * df_t_input + f_t * f_scale * dc_t;
      const Real do_part = do_t_input;
      const Real dc_part = (c_part_deriv * dtanh_c_part
          - tanh_c_part * c_part_self_repair);
      const Real df_part = df_t_input;
      const Real di_part = di_t_input;

      if (input_deriv) {
        input_deriv[i * input_deriv_stride + j] = di_part;
        input_deriv[i * input_deriv_stride + j + cell_dim] = df_part;
        input_deriv[i * input_deriv_stride + j + cell_dim * 2] = dc_part;
        input_deriv[i * input_deriv_stride + j + cell_dim * 3] = do_part;
        input_deriv[i * input_deriv_stride + j + cell_dim * 4] = dc_prev;
      }
    }
  }

  if (params_deriv) {
    // compute params_deriv
    smem[tid] = w_ic_deriv_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      params_deriv[j] = smem[tid];
    }

    __syncthreads();
    smem[tid] = w_fc_deriv_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      params_deriv[params_deriv_stride + j] = smem[tid];
    }

    __syncthreads();
    smem[tid] = w_oc_deriv_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      params_deriv[2 * params_deriv_stride + j] = smem[tid];
    }

    // compute value_sum_out
    __syncthreads();
    smem[tid] = i_t_value_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      value_sum_out[j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = f_t_value_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      value_sum_out[value_sum_out_stride + j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = c_part_value_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      value_sum_out[2 * value_sum_out_stride + j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = o_t_value_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      value_sum_out[3 * value_sum_out_stride + j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = c_t_value_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      value_sum_out[4 * value_sum_out_stride + j] += smem[tid];
    }

    // need to update self_repair_sum_out before deriv_sum_out, because
    // deriv_sum_out and deriv_sum_in might point to the same memory.
    if (i0 < 5 && j < cell_dim) {
      self_repair_sum_out[i0 * self_repair_sum_out_stride + j] =
          update_sr[i0] ? num_rows : 0;
    }

    // compute derive_sum_out
    __syncthreads();
    smem[tid] = i_t_deriv_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      deriv_sum_out[j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = f_t_deriv_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      deriv_sum_out[deriv_sum_out_stride + j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = c_part_deriv_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      deriv_sum_out[2 * deriv_sum_out_stride + j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = o_t_deriv_sum;
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      deriv_sum_out[3 * deriv_sum_out_stride + j] += smem[tid];
    }

    __syncthreads();
    smem[tid] = c_t_deriv_sum;
    __syncthreads();
#   pragma unroll
    for (int shift = CU1DBLOCK / 2; shift >= warpSize; shift >>= 1) {
      __syncthreads();
      if (tid < shift) {
        smem[tid] += smem[tid + shift];
      }
    }
    if (tid < warpSize && j < cell_dim) {
      deriv_sum_out[4 * deriv_sum_out_stride + j] += smem[tid];
    }
  }
}

/***********************************************************************
 * ANSI-C wrappers of CUDA kernels
 */

/*
 * "int32"
 */
void cuda_int32_set_const(dim3 Gr, dim3 Bl, int32_cuda* mat, int32_cuda value,
                          MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d);
}
void cuda_int32_add(dim3 Gr, dim3 Bl, int32_cuda* mat, int32_cuda value,
                    MatrixDim d) {
  _add<<<Gr,Bl>>>(mat,value,d);
}

/*
 * "float"
 */

/*
 * CuMatrix
 */
void cudaF_copy_upp_low(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) {
  _copy_upp_low<<<Gr,Bl>>>(A,dimA);}
void cudaF_copy_low_upp(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) {
  _copy_low_upp<<<Gr,Bl>>>(A,dimA);}
void cudaF_add_diag_vec_mat(dim3 Gr, dim3 Bl, float alpha, float *mat,
                            MatrixDim mat_dim, const float *vec,
                            const float *mat2, int mat2_row_stride,
                            int mat2_col_stride, float beta) {
  _add_diag_vec_mat<<<Gr,Bl>>>(alpha, mat, mat_dim, vec, mat2, mat2_row_stride,
      mat2_col_stride, beta);
}

void cudaF_copy_from_tp_trans(dim3 Gr, dim3 Bl, float* A, const float* B,
                              MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}
void cudaFD_copy_from_tp_trans(dim3 Gr, dim3 Bl, float* A, const double* B,
                               MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}

void cudaF_copy_from_tp(dim3 Gr, dim3 Bl, float* A, const float* B,
                        MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}
void cudaFD_copy_from_tp(dim3 Gr, dim3 Bl, float* A, const double* B,
                         MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}

void cudaF_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}

void cudaF_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}

void cudaF_apply_pow_abs(dim3 Gr, dim3 Bl, float* mat, float power,
                         bool include_sign, MatrixDim d) {
  _apply_pow_abs<<<Gr,Bl>>>(mat, power, include_sign, d);
}

void cudaF_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_heaviside<<<Gr,Bl>>>(mat, d);
}

void cudaF_copy_cols(dim3 Gr, dim3 Bl, float* dst, const float* src,
                     const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                     int src_stride) {
  _copy_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaF_add_cols(dim3 Gr, dim3 Bl, float* dst, const float* src,
                    const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                    int src_stride) {
  _add_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaF_copy_rows(dim3 Gr, dim3 Bl, float* dst, const float* src,
                     const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                     int src_stride) {
  _copy_rows<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaF_copy_rows_direct(dim3 Gr, dim3 Bl, float* dst,
                            const float* const * src, MatrixDim dst_dim) {
  _copy_rows<<<Gr,Bl>>>(dst, src, dst_dim);
}

void cudaF_copy_to_rows_direct(dim3 Gr, dim3 Bl, float* const * dst,
                               const float* src, MatrixDim src_dim) {
  _copy_to_rows<<<Gr,Bl>>>(dst, src, src_dim);
}

void cudaF_add_rows(dim3 Gr, dim3 Bl, float alpha, float* dst, const float* src,
                    const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                    int src_stride) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, reorder, dst_dim, src_stride);
}

void cudaF_add_rows_direct(dim3 Gr, dim3 Bl, float alpha, float* dst,
                           const float* const * src, MatrixDim dst_dim) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, dst_dim);
}

void cudaF_add_to_rows_direct(dim3 Gr, dim3 Bl, float alpha, float* const * dst,
                              const float* src, MatrixDim src_dim) {
  _add_to_rows<<<Gr,Bl>>>(alpha, dst, src, src_dim);
}

void cudaF_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val,
                       MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}

void cudaF_apply_ceiling(dim3 Gr, dim3 Bl, float* mat, float ceiling_val,
                         MatrixDim d) {
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

void cudaF_mul_elements(dim3 Gr, dim3 Bl, float* mat, const float* A,
                        MatrixDim dst_d, int src_stride) {
  _mul_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaF_div_elements(dim3 Gr, dim3 Bl, float* mat, const float* A,
                        MatrixDim dst_d, int src_stride) {
  _div_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaF_max(dim3 Gr, dim3 Bl, float* mat, const float* A, MatrixDim dst_d,
               int src_stride) {
  _max<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaF_min(dim3 Gr, dim3 Bl, float* mat, const float* other,
               MatrixDim mat_d, int other_stride) {
  _min<<<Gr,Bl>>>(mat,other,mat_d,other_stride);
}

void cudaF_mul_cols_vec(dim3 Gr, dim3 Bl, float* mat, const float* scale,
                        MatrixDim d) {
  _mul_cols_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaF_mul_rows_vec(dim3 Gr, dim3 Bl, float* mat, const float* scale,
                        MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaF_mul_rows_group_mat(dim3 Gr, dim3 Bl, float *y, const float *x,
                              MatrixDim d, int src_stride, int group_size) {
  _mul_rows_group_mat<<<Gr,Bl>>>(y, x, d, src_stride, group_size);
}


void cudaF_diff_group_pnorm(dim3 Gr, dim3 Bl, float *id, const float *iv,
                            const float *ov, const float* od, MatrixDim id_dim,
                            int iv_stride, int ov_stride, int od_stride,
                            int group_size, float power) {
  _diff_group_pnorm<<<Gr, Bl>>>(id, iv, ov, od, id_dim, iv_stride, ov_stride,
      od_stride, group_size, power);
}

void cudaF_calc_group_max_deriv(dim3 Gr, dim3 Bl, float *y, const float *x1,
                                const float *x2, MatrixDim y_dim, int x1_stride,
                                int x2_stride, int group_size) {
  _calc_group_max_deriv<<<Gr,Bl>>>(y, x1, x2, y_dim, x1_stride, x2_stride,
      group_size);
}

void cudaF_div_rows_vec(dim3 Gr, dim3 Bl, float* mat, const float* vec_div,
                        MatrixDim d) {
  _div_rows_vec<<<Gr,Bl>>>(mat, vec_div, d);
}

void cudaF_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* src, float* dst,
                   MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_trans<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  } else {
    _add_mat<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  }
}

void cudaF_add_mat_blocks(dim3 Gr, dim3 Bl, float alpha, const float* src,
                          int32_cuda num_row_blocks, int32_cuda num_col_blocks,
                          float* dst, MatrixDim d, int src_stride,
                          int A_trans) {
  if (A_trans) {
    _add_mat_blocks_trans<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks,
        dst, d, src_stride);
  } else {
    _add_mat_blocks<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks, dst,
        d, src_stride);
  }
}

void cudaF_add_mat_repeated(dim3 Gr, dim3 Bl, float alpha, const float* src,
                            MatrixDim src_dim, float *dst, MatrixDim dst_dim) {
  _add_mat_repeated<<<Gr,Bl>>>(alpha, src, src_dim, dst, dst_dim);
}


void cudaF_set_mat_mat_div_mat(dim3 Gr, dim3 Bl, const float *A, const float *B,
                               const float *C, float *dst, MatrixDim d,
                               int stride_a, int stride_b, int stride_c) {
  _set_mat_mat_div_mat<<<Gr,Bl>>>(A,B,C,dst,d, stride_a, stride_b, stride_c);
}

void cudaF_sy_add_tr2(dim3 Gr, dim3 Bl, float alpha, float beta, const float* T,
                      MatrixDim tdim, float *S, MatrixDim sdim) {
  _sy_add_tr2<<<Gr,Bl>>>(alpha, beta, T, tdim, S, sdim);
}

void cudaF_add_vec_to_cols(dim3 Gr, dim3 Bl, float alpha, const float* col,
                           float beta, float* dst, MatrixDim d) {
  _add_vec_to_cols<<<Gr,Bl>>>(alpha,col,beta,dst,d);
}

void cudaF_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float* row,
                           float beta, float* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d);
}

void cudaF_add_mat_diag_vec(dim3 Gr, dim3 Bl, float alpha, float *mat,
                            MatrixDim mat_dim, const float *mat2,
                            int mat2_row_stride, int mat2_col_stride,
                            const float *vec, float beta) {
  _add_mat_diag_vec<<<Gr,Bl>>>(alpha, mat, mat_dim, mat2, mat2_row_stride,
      mat2_col_stride, vec, beta);
}

void cudaF_add_mat_mat_elements(dim3 Gr, dim3 Bl, float *data,
                                const float *srcA_data, const float *srcB_data,
                                MatrixDim dim, int srcA_stride, int srcB_stride,
                                float alpha, float beta) {
  _add_mat_mat_elements<<<Gr, Bl>>>(data, srcA_data, srcB_data, dim,
      srcA_stride, srcB_stride, alpha, beta);
}

// CURRENTLY UNUSED...
void cudaF_apply_mask(dim3 Gr, dim3 Bl, float* mat, const char* mask,
                      MatrixDim dmat, MatrixDim dmask) {
  _apply_mask<<<Gr,Bl>>>(mat,mask,dmat,dmask);
}

/*
 * CuVector
 */

void cudaF_max_mat_cols(int Gr, int Bl, float* result, const float* mat,
                        const MatrixDim d) {
  _transform_reduce_mat_cols<<<Gr,Bl>>>(result,mat,d,
      TransReduceOp<MAX,float>());
}
void cudaF_min_mat_cols(int Gr, int Bl, float* result, const float* mat,
                        const MatrixDim d) {
  _transform_reduce_mat_cols<<<Gr,Bl>>>(result,mat,d,
      TransReduceOp<MIN,float>());
}
void cudaF_sum_mat_cols(int Gr, int Bl, float* result, const float* mat,
                        const MatrixDim d) {
  _transform_reduce_mat_cols<<<Gr,Bl>>>(result,mat,d,
      TransReduceOp<SUM,float>());
}
void cudaF_add_col_sum_mat(int Gr, int Bl, float* result, const float* mat,
                           const MatrixDim d, const float alpha,
                           const float beta) {
  _transform_reduce_mat_cols<<<Gr, Bl>>>(result, mat, d,
      TransReduceOp<SUMAB, float>(alpha, beta));
}

void cudaF_replace_value(int Gr, int Bl, float *v, int dim, float orig,
                         float changed) {
  _replace_value<<<Gr,Bl>>>(v, dim, orig, changed);
}

void cudaF_set_bias_params(int Gr, int Bl, float* v, const float* a,
                           float param_1, float param_2, float param_3,
                           int* flag, int dim) {
  _set_bias_params<<<Gr,Bl>>>(v,a,param_1,param_2,param_3,flag,dim);
}

void cublas_copy_kaldi_fd(int Gr, int Bl, int n, const float* x, int incx,
                          double* y, int incy) {
  _cublas_copy_kaldi<<<Gr,Bl>>>(n, x, incx, y, incy);
}

void cublas_copy_kaldi_df(int Gr, int Bl, int n, const double* x, int incx,
                          float* y, int incy) {
  _cublas_copy_kaldi<<<Gr,Bl>>>(n, x, incx, y, incy);
}

void cudaF_vec_mul_elements(int Gr, int Bl, float* v, const float* a, int dim) {
  _vec_mul_elements<<<Gr,Bl>>>(v, a, dim);
}

void cudaF_vec_min(int Gr, int Bl, const float* v, float* value, int dim,
                   int inc) {
  _vec_transform_reduce<<<Gr,Bl>>>(v, value, dim, inc,
      TransReduceOp<MIN, float>());
}

void cudaF_vec_max(int Gr, int Bl, const float* v, float* value, int dim,
                   int inc) {
  _vec_transform_reduce<<<Gr,Bl>>>(v, value, dim, inc,
      TransReduceOp<MAX, float>());
}

void cudaF_trace_mat_mat_trans(dim3 Gr, dim3 Bl, const float* A, const float* B,
                               MatrixDim dA, int B_stride, float* value) {
  _trace_mat_mat_trans<<<Gr,Bl>>>(A,B,dA,B_stride,value);
}

void cudaF_trace_mat_mat(dim3 Gr, dim3 Bl, const float* A, const float* B,
                         MatrixDim dA, int B_stride, float* value) {
  _trace_mat_mat<32> <<<Gr,Bl>>>(A,B,dA,B_stride,value);
}

void cudaF_add_diag_mat_mat_MNT(int Gr, int Bl, const float alpha,
                                const float* M, const MatrixDim dim_M,
                                const float* N, const int stride_N,
                                const float beta, float* v) {
  _add_diag_mat_mat_MNT<<<Gr,Bl>>>(alpha,M,dim_M,N,stride_N,beta,v);
}

void cudaF_add_diag_mat_mat_MTN(dim3 Gr, dim3 Bl, const float alpha,
                                const float* M, const int stride_M,
                                const float* N, const MatrixDim dim_N,
                                const float beta, float* v) {
  if (Bl.x == 16) {
    _add_diag_mat_mat_MTN<16> <<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  } else if (Bl.x==32) {
    _add_diag_mat_mat_MTN<32><<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  }
}

void cudaF_add_diag_mat_mat_MN(dim3 Gr, dim3 Bl, const float alpha,
                               const float* M, const int stride_M,
                               const float* N, const MatrixDim dim_N,
                               const float beta, float* v) {
  if (Bl.x == 16) {
    _add_diag_mat_mat_MN<16> <<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  } else if (Bl.x==32) {
    _add_diag_mat_mat_MN<32><<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  }
}

void cudaF_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x,
                       const float* y, float beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}

void cudaF_vec_sum(int Gr, int Bl, float* v, float* value, int dim, int inc) {
  _vec_transform_reduce<<<Gr,Bl>>>(v, value, dim, inc,
      TransReduceOp<SUM, float>());
}

void cudaF_matrix_add_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim dim,
                               float alpha, MatrixElement<float>* x,
                               int num_elements) {
  _cuda_matrix_add_elements<<<Gr, Bl>>>(data, dim, alpha, x, num_elements);
}

void cudaF_matrix_add_indexed_values(dim3 Gr, dim3 Bl, MatrixDim dim,
                                     float alpha, const Int32Pair* indices,
                                     const float* x, int s, float* data) {
  _cuda_matrix_add_indexed_values<<<Gr, Bl>>>(dim, alpha, indices, x, s, data);
}

void cudaF_comp_obj_deriv(dim3 Gr, dim3 Bl, MatrixElement<float>* x, int s,
                          const float* z, MatrixDim d, float* z2, MatrixDim d2,
                          float* t) {
  _cuda_comp_obj_deriv<<<Gr,Bl>>>(x,s,z,d,z2,d2,t);
}

void cudaD_comp_obj_deriv(dim3 Gr, dim3 Bl, MatrixElement<double>* x, int s,
                          const double* z, MatrixDim d, double* z2,
                          MatrixDim d2, double* t) {
  _cuda_comp_obj_deriv<<<Gr,Bl>>>(x,s,z,d,z2,d2,t);
}

void cudaF_vec_copy_diag_from_packed(int Gr, int Bl, float *dst,
                                     const float *src, int dim) {
  _vec_copy_diag_from_packed<<<Gr,Bl>>>(dst,src,dim);
}

void cudaF_vec_apply_floor(int Gr, int Bl, float* v, float floor_val,
                           float *count, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,count,dim);
}

void cudaF_vec_apply_ceiling(int Gr, int Bl, float* v, float ceiling_val,
                             float *count, int dim) {
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

void cudaF_add_mat_blockmat(dim3 Gr, dim3 Bl, float *data, MatrixDim d,
                            const float *Adata, int A_num_rows, int A_num_cols,
                            int A_row_stride, int A_col_stride,
                            const CuBlockMatrixData *B_cu_data,
                            int B_num_blocks, float alpha, float beta,
                            int B_trans) {
  if (B_trans) {
    _add_mat_blockmat_trans<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
        A_row_stride, A_col_stride, B_cu_data, B_num_blocks, alpha, beta);
  } else {
    _add_mat_blockmat<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
        A_row_stride, A_col_stride, B_cu_data, B_num_blocks, alpha, beta);

  }
}

void cudaF_block_add_mat_mat(dim3 Gr, dim3 Bl, CuBlockMatrixData *B_cu_data,
                             int num_blocks, const float *C_data,
                             int C_num_cols, int C_row_stride, int C_col_stride,
                             const float *D_data, int D_row_stride,
                             int D_col_stride, float alpha, float beta) {
  _block_add_mat_mat<<<Gr,Bl>>>(B_cu_data, num_blocks, C_data, C_num_cols,
      C_row_stride, C_col_stride, D_data, D_row_stride, D_col_stride, alpha,
      beta);
}

/*
 * cu::
 */
void cudaF_soft_hinge(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d,
                      int src_stride) {
  _soft_hinge<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_group_pnorm(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d,
                       int src_stride, int group_size, float power) {
  _group_pnorm<<<Gr,Bl>>>(y, x, d, src_stride, group_size, power);
}

void cudaF_group_spec_pnorm(dim3 Gr, dim3 Bl, float* y, const float* x,
                            MatrixDim d, int src_stride, int group_size,
                            float power) {
  if (power == float(0)) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<L0NORM, float>());
  } else if (power == float(1)) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<L1NORM, float>());
  } else if (power == float(2)) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<L2NORM, float>());
  } else if (power == std::numeric_limits<float>::infinity()) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<LINFNORM, float>());
  } else {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<LPNORM, float>(power));
  }
}

void cudaF_group_max(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d,
                     int src_stride, int group_size) {
  _group_transform_reduce<<<Gr,Bl>>>(y, x, d, src_stride, group_size,
      TransReduceOp<MAX, float>());
}

void cudaF_sigmoid(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d,
                   int src_stride) {
  _sigmoid<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_diff_sigmoid(dim3 Gr, dim3 Bl, float* eout, const float* e,
                        const float* y, MatrixDim d, int e_stride,
                        int y_stride) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaF_tanh(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d,
                int src_stride) {
  _tanh<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_diff_tanh(dim3 Gr, dim3 Bl, float* eout, const float* e,
                     const float* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaF_parametric_relu(dim3 Gr, dim3 Bl, float* y, const float* x,
                           MatrixDim d, int src_stride,
                           const float* a, const float* b) {
  _parametric_relu<<<Gr,Bl>>>(y, x, d, src_stride, a, b);
}

void cudaF_diff_parametric_relu(dim3 Gr, dim3 Bl, float* eout, const float* e,
                                const float* y, MatrixDim d, int e_stride,
                                int y_stride, const float* a, const float* b) {
  _diff_parametric_relu<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride, a, b);
}

void cudaF_heaviside(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d,
                     int src_stride) {
  _heaviside<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_softmax_reduce(size_t Gr, size_t Bl, float* y, const float* x,
                          MatrixDim d, int src_stride) {
  _softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_log_softmax_reduce(size_t Gr, size_t Bl, float* y, const float* x,
                              MatrixDim y_dim, int x_stride) {
  _log_softmax_reduce<<<Gr,Bl>>>(y, x, y_dim, x_stride);
}

void cudaF_splice(dim3 Gr, dim3 Bl, float* y, const float* x,
                  const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in);
}

void cudaF_normalize_per_row(size_t Gr, size_t Bl, float *y, int y_stride,
                             const float *x, MatrixDim x_d, float target_rms,
                             bool add_log_stddev) {
  _normalize_per_row<<<Gr, Bl>>>(y, y_stride, x, x_d, target_rms, add_log_stddev);
}

void cudaF_one(int Gr, int Bl, float* x, int dim) {
  _one<<<Gr,Bl>>>(x,dim);
}

void cudaF_take_mean(dim3 Gr, dim3 Bl, const float* x, float* y,
                     MatrixDim d_in) {
  _take_mean<<<Gr,Bl>>>(x,y,d_in);
}

void cudaF_take_lower(dim3 Gr, dim3 Bl, const float* x, float* y,
                      MatrixDim d_in) {
  _take_lower<<<Gr,Bl>>>(x,y,d_in);
}

void cudaF_take_upper(dim3 Gr, dim3 Bl, const float* x, float* y,
                      MatrixDim d_in) {
  _take_upper<<<Gr,Bl>>>(x,y,d_in);
}

void cudaF_copy_from_sp(dim3 Gr, dim3 Bl, const float* x, float* y,
                        MatrixDim dim) {
  _copy_from_sp<<<Gr,Bl>>>(x, y, dim);
}

void cudaF_copy(dim3 Gr, dim3 Bl, float* y, const float* x,
                const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaF_randomize(dim3 Gr, dim3 Bl, float* y, const float* x,
                     const int32_cuda* copy_from, MatrixDim d_out,
                     MatrixDim d_in) {
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float* wei, float* grad, float l1,
                         float lr, MatrixDim d, int stride_grad) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d,stride_grad);
}

void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float* mat, float* vec_val,
                           int32_cuda* vec_id, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, d);
}

void cudaF_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda* vec_tgt,
                     float* mat_net_out, float* vec_log_post, MatrixDim d) {
  _diff_xent<<<Gr,Bl>>>(vec_tgt,mat_net_out,vec_log_post,d);
}

void cudaF_diff_softmax(dim3 Gr, dim3 Bl, float* x, const MatrixDim dim,
                        const float* value, const int value_stride,
                        const float* diff, const int diff_stride) {
  _diff_softmax<<<Gr, Bl>>>(x, dim, value, value_stride, diff, diff_stride);
}

void cudaF_copy_rows_from_vec(dim3 Gr, dim3 Bl, float *mat_out, MatrixDim d_out,
                              const float *v_in) {
  _copy_rows_from_vec<<<Gr,Bl>>>(mat_out, d_out, v_in);
}

void cudaF_diff_log_softmax(dim3 Gr, dim3 Bl, const MatrixDim in_deriv_dim,
                            const float* out_value, const int out_value_stride,
                            const float* out_deriv, const int out_deriv_stride,
                            float* in_deriv) {
  _diff_log_softmax<<<Gr, Bl>>>(in_deriv_dim, out_value, out_value_stride,
      out_deriv, out_deriv_stride, in_deriv);
}

void cudaF_copy_col_from_mat_df(int Gr, int Bl, double* v, int col,
                                const float* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_df<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaF_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col,
                                const float* mat, MatrixDim dmat, int dim) {
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
                              const float *mat2, float *mask,
                              MatrixDim mat1_dim, int mat2_stride,
                              int mask_stride) {
  _equal_element_mask<<<Gr,Bl>>>(mat1, mat2, mask, mat1_dim, mat2_stride,
      mask_stride);
}

/*
 * "double"
 */

/*
 * CuMatrix
 */
void cudaD_copy_upp_low(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) {
  _copy_upp_low<<<Gr,Bl>>>(A,dimA);}
void cudaD_copy_low_upp(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) {
  _copy_low_upp<<<Gr,Bl>>>(A,dimA);}
void cudaD_add_diag_vec_mat(dim3 Gr, dim3 Bl, double alpha, double *mat,
                            MatrixDim mat_dim, const double *vec,
                            const double *mat2, int mat2_row_stride,
                            int mat2_col_stride, double beta) {
  _add_diag_vec_mat<<<Gr,Bl>>>(alpha, mat, mat_dim, vec, mat2, mat2_row_stride,
      mat2_col_stride, beta);
}

void cudaD_copy_from_tp_trans(dim3 Gr, dim3 Bl, double* A, const double* B,
                              MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}
void cudaDF_copy_from_tp_trans(dim3 Gr, dim3 Bl, double* A, const float* B,
                               MatrixDim dmat) {
  _copy_from_tp_trans<<<Gr,Bl>>>(A,B,dmat);
}

void cudaD_copy_from_tp(dim3 Gr, dim3 Bl, double* A, const double* B,
                        MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}
void cudaDF_copy_from_tp(dim3 Gr, dim3 Bl, double* A, const float* B,
                         MatrixDim dmat) {
  _copy_from_tp<<<Gr,Bl>>>(A,B,dmat);
}

void cudaD_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}

void cudaD_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}

void cudaD_apply_pow_abs(dim3 Gr, dim3 Bl, double* mat, double power,
                         bool include_sign, MatrixDim d) {
  _apply_pow_abs<<<Gr,Bl>>>(mat, power, include_sign, d);
}

void cudaD_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_heaviside<<<Gr,Bl>>>(mat, d);
}

void cudaD_copy_cols(dim3 Gr, dim3 Bl, double* dst, const double* src,
                     const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                     int src_stride) {
  _copy_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaD_add_cols(dim3 Gr, dim3 Bl, double* dst, const double* src,
                    const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                    int src_stride) {
  _add_cols<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaD_copy_rows(dim3 Gr, dim3 Bl, double* dst, const double* src,
                     const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                     int src_stride) {
  _copy_rows<<<Gr,Bl>>>(dst, src, reorder, dst_dim, src_stride);
}

void cudaD_copy_rows_direct(dim3 Gr, dim3 Bl, double* dst,
                            const double* const * src, MatrixDim dst_dim) {
  _copy_rows<<<Gr,Bl>>>(dst, src, dst_dim);
}

void cudaD_copy_to_rows_direct(dim3 Gr, dim3 Bl, double* const * dst,
                               const double* src, MatrixDim src_dim) {
  _copy_to_rows<<<Gr,Bl>>>(dst, src, src_dim);
}

void cudaD_add_rows(dim3 Gr, dim3 Bl, double alpha, double* dst,
                    const double* src, const MatrixIndexT_cuda* reorder,
                    MatrixDim dst_dim, int src_stride) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, reorder, dst_dim, src_stride);
}

void cudaD_add_rows_direct(dim3 Gr, dim3 Bl, double alpha, double* dst,
                           const double* const * src, MatrixDim dst_dim) {
  _add_rows<<<Gr,Bl>>>(alpha, dst, src, dst_dim);
}

void cudaD_add_to_rows_direct(dim3 Gr, dim3 Bl, double alpha,
                              double* const * dst, const double* src,
                              MatrixDim src_dim) {
  _add_to_rows<<<Gr,Bl>>>(alpha, dst, src, src_dim);
}

void cudaD_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val,
                       MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}

void cudaD_apply_ceiling(dim3 Gr, dim3 Bl, double* mat, double ceiling_val,
                         MatrixDim d) {
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

void cudaD_scale_diag_packed(int Gr, int Bl, double* mat, double value,
                             int dim) {
  _scale_diag_packed<<<Gr,Bl>>>(mat,value,dim);
}

void cudaD_scale(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _scale<<<Gr,Bl>>>(mat,value,d);
}

void cudaD_apply_log(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d);
}

void cudaD_mul_elements(dim3 Gr, dim3 Bl, double* mat, const double* A,
                        MatrixDim dst_d, int src_stride) {
  _mul_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaD_div_elements(dim3 Gr, dim3 Bl, double* mat, const double* A,
                        MatrixDim dst_d, int src_stride) {
  _div_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaD_max(dim3 Gr, dim3 Bl, double* mat, const double* A, MatrixDim dst_d,
               int src_stride) {
  _max<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaD_min(dim3 Gr, dim3 Bl, double* mat, const double* other, MatrixDim mat_d,
               int other_stride) {
  _min<<<Gr,Bl>>>(mat,other,mat_d,other_stride);
}

void cudaD_mul_cols_vec(dim3 Gr, dim3 Bl, double* mat, const double* scale,
                        MatrixDim d) {
  _mul_cols_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaD_mul_rows_vec(dim3 Gr, dim3 Bl, double* mat, const double* scale,
                        MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaD_mul_rows_group_mat(dim3 Gr, dim3 Bl, double* y, const double* x,
                              MatrixDim d, int src_stride, int group_size) {
  _mul_rows_group_mat<<<Gr,Bl>>>(y, x, d, src_stride, group_size);
}

void cudaD_diff_group_pnorm(dim3 Gr, dim3 Bl, double *id, const double *iv,
                            const double *ov, const double* od,
                            MatrixDim id_dim, int iv_stride, int ov_stride,
                            int od_stride, int group_size, double power) {
  _diff_group_pnorm<<<Gr, Bl>>>(id, iv, ov, od, id_dim, iv_stride, ov_stride,
      od_stride, group_size, power);
}

void cudaD_calc_group_max_deriv(dim3 Gr, dim3 Bl, double*y, const double* x1,
                                const double* x2, MatrixDim y_dim,
                                int x1_stride, int x2_stride, int group_size) {
  _calc_group_max_deriv<<<Gr,Bl>>>(y, x1, x2, y_dim, x1_stride, x2_stride,
      group_size);
}

void cudaD_div_rows_vec(dim3 Gr, dim3 Bl, double* mat, const double* vec_div,
                        MatrixDim d) {
  _div_rows_vec<<<Gr,Bl>>>(mat, vec_div, d);
}

void cudaD_add_mat(dim3 Gr, dim3 Bl, double alpha, const double* src,
                   double* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_trans<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  } else {
    _add_mat<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  }
}

void cudaD_add_mat_blocks(dim3 Gr, dim3 Bl, double alpha, const double* src,
                          int32_cuda num_row_blocks, int32_cuda num_col_blocks,
                          double* dst, MatrixDim d, int src_stride,
                          int A_trans) {
  if (A_trans) {
    _add_mat_blocks_trans<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks,
        dst, d, src_stride);
  } else {
    _add_mat_blocks<<<Gr,Bl>>>(alpha, src, num_row_blocks, num_col_blocks, dst,
        d, src_stride);
  }
}

void cudaD_add_mat_repeated(dim3 Gr, dim3 Bl, double alpha, const double* src,
                            MatrixDim src_dim, double *dst, MatrixDim dst_dim) {
  _add_mat_repeated<<<Gr,Bl>>>(alpha, src, src_dim, dst, dst_dim);
}

void cudaD_set_mat_mat_div_mat(dim3 Gr, dim3 Bl, const double *A,
                               const double *B, const double *C, double *dst,
                               MatrixDim d, int stride_a, int stride_b,
                               int stride_c) {
  _set_mat_mat_div_mat<<<Gr,Bl>>>(A,B,C,dst,d,stride_a,stride_b,stride_c);
}

void cudaD_sy_add_tr2(dim3 Gr, dim3 Bl, double alpha, double beta,
                      const double* T, MatrixDim tdim, double *S,
                      MatrixDim sdim) {
  _sy_add_tr2<<<Gr,Bl>>>(alpha, beta, T, tdim, S, sdim);
}

void cudaD_add_vec_to_cols(dim3 Gr, dim3 Bl, double alpha, const double* col,
                           double beta, double* dst, MatrixDim d) {
  _add_vec_to_cols<<<Gr,Bl>>>(alpha,col,beta,dst,d);
}

void cudaD_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double* row,
                           double beta, double* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d);
}

void cudaD_add_mat_diag_vec(dim3 Gr, dim3 Bl, double alpha, double *mat,
                            MatrixDim mat_dim, const double *mat2,
                            int mat2_row_stride, int mat2_col_stride,
                            const double *vec, double beta) {
  _add_mat_diag_vec<<<Gr,Bl>>>(alpha, mat, mat_dim, mat2, mat2_row_stride,
      mat2_col_stride, vec, beta);
}

void cudaD_add_mat_mat_elements(dim3 Gr, dim3 Bl, double *data,
                                const double *srcA_data,
                                const double *srcB_data, MatrixDim dim,
                                int srcA_stride, int srcB_stride, double alpha,
                                double beta) {
  _add_mat_mat_elements<<<Gr, Bl>>>(data, srcA_data, srcB_data, dim,
      srcA_stride, srcB_stride, alpha, beta);
}

// CURRENTLY UNUSED...
void cudaD_apply_mask(dim3 Gr, dim3 Bl, double* mat, const char* mask,
                      MatrixDim dmat, MatrixDim dmask) {
  _apply_mask<<<Gr,Bl>>>(mat,mask,dmat,dmask);
}

/*
 * CuVector
 */
void cudaD_max_mat_cols(int Gr, int Bl, double* result, const double* mat,
                        const MatrixDim d) {
  _transform_reduce_mat_cols<<<Gr,Bl>>>(result,mat,d,
      TransReduceOp<MAX,double>());
}
void cudaD_min_mat_cols(int Gr, int Bl, double* result, const double* mat,
                        const MatrixDim d) {
  _transform_reduce_mat_cols<<<Gr,Bl>>>(result,mat,d,
      TransReduceOp<MIN,double>());
}
void cudaD_sum_mat_cols(int Gr, int Bl, double* result, const double* mat,
                        const MatrixDim d) {
  _transform_reduce_mat_cols<<<Gr,Bl>>>(result,mat,d,
      TransReduceOp<SUM,double>());
}
void cudaD_add_col_sum_mat(int Gr, int Bl, double* result, const double* mat,
                           const MatrixDim d, const double alpha,
                           const double beta) {
  _transform_reduce_mat_cols<<<Gr, Bl>>>(result, mat, d,
      TransReduceOp<SUMAB, double>(alpha, beta));
}

void cudaD_replace_value(int Gr, int Bl, double *v, int dim, double orig,
                         double changed) {
  _replace_value<<<Gr,Bl>>>(v, dim, orig, changed);
}

void cudaD_set_bias_params(int Gr, int Bl, double* v, const double* a,
                           double param_1, double param_2, double param_3,
                           int* flag, int dim) {
  _set_bias_params<<<Gr,Bl>>>(v,a,param_1,param_2,param_3,flag,dim);
}

void cudaD_vec_mul_elements(int Gr, int Bl, double* v, const double* a,
                            int dim) {
  _vec_mul_elements<<<Gr,Bl>>>(v, a, dim);
}

void cudaD_vec_min(int Gr, int Bl, const double* v, double* value, int dim,
                   int inc) {
  _vec_transform_reduce<<<Gr,Bl>>>(v, value, dim, inc,
      TransReduceOp<MIN, double>());
}

void cudaD_vec_max(int Gr, int Bl, const double* v, double* value, int dim,
                   int inc) {
  _vec_transform_reduce<<<Gr,Bl>>>(v, value, dim, inc,
      TransReduceOp<MAX, double>());
}

void cudaD_trace_mat_mat_trans(dim3 Gr, dim3 Bl, const double* A,
                               const double* B, MatrixDim dA, int B_stride,
                               double* value) {
  _trace_mat_mat_trans<<<Gr,Bl>>>(A,B,dA,B_stride,value);
}

void cudaD_trace_mat_mat(dim3 Gr, dim3 Bl, const double* A, const double* B,
                         MatrixDim dA, int B_stride, double* value) {
  _trace_mat_mat<32> <<<Gr,Bl>>>(A,B,dA,B_stride,value);
}

void cudaD_add_diag_mat_mat_MNT(int Gr, int Bl, const double alpha,
                                const double* M, const MatrixDim dim_M,
                                const double* N, const int stride_N,
                                const double beta, double* v) {
  _add_diag_mat_mat_MNT<<<Gr,Bl>>>(alpha,M,dim_M,N,stride_N,beta,v);
}

void cudaD_add_diag_mat_mat_MTN(dim3 Gr, dim3 Bl, const double alpha,
                                const double* M, const int stride_M,
                                const double* N, const MatrixDim dim_N,
                                const double beta, double* v) {
  if (Bl.x == 16) {
    _add_diag_mat_mat_MTN<16> <<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  } else if (Bl.x==32) {
    _add_diag_mat_mat_MTN<32><<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  }
}

void cudaD_add_diag_mat_mat_MN(dim3 Gr, dim3 Bl, const double alpha,
                               const double* M, const int stride_M,
                               const double* N, const MatrixDim dim_N,
                               const double beta, double* v) {
  if (Bl.x == 16) {
    _add_diag_mat_mat_MN<16> <<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  } else if (Bl.x==32) {
    _add_diag_mat_mat_MN<32><<<Gr,Bl>>>(alpha,M,stride_M,N,dim_N,beta,v);
  }
}

void cudaD_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x,
                       const double* y, double beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}

void cudaD_copy_col_from_mat_df(int Gr, int Bl, double* v, int col,
                                const double* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_df<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaD_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col,
                                const double* mat, MatrixDim dmat, int dim) {
  _copy_col_from_mat_fd<<<Gr,Bl>>>(v,col,mat,dmat,dim);
}

void cudaD_vec_sum(int Gr, int Bl, double* v, double* value, int dim, int inc) {
  _vec_transform_reduce<<<Gr,Bl>>>(v,value,dim,inc,
      TransReduceOp<SUM, double>());
}

void cudaD_matrix_add_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim dim,
                               double alpha, MatrixElement<double>* x,
                               int num_elements) {
  _cuda_matrix_add_elements<<<Gr, Bl>>>(data, dim, alpha, x, num_elements);
}

void cudaD_matrix_add_indexed_values(dim3 Gr, dim3 Bl, MatrixDim dim,
                                     double alpha, const Int32Pair* indices,
                                     const double* x, int s, double* data) {
  _cuda_matrix_add_indexed_values<<<Gr, Bl>>>(dim, alpha, indices, x, s, data);
}

void cudaD_vec_copy_diag_from_packed(int Gr, int Bl, double *dst,
                                     const double *src, int dim) {
  _vec_copy_diag_from_packed<<<Gr,Bl>>>(dst,src,dim);
}

void cudaD_vec_apply_floor(int Gr, int Bl, double* v, double floor_val,
                           float *count, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,count,dim);
}

void cudaD_vec_apply_ceiling(int Gr, int Bl, double* v, double ceiling_val,
                             float *count, int dim) {
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

void cudaD_add_mat_blockmat(dim3 Gr, dim3 Bl, double *data, MatrixDim d,
                            const double *Adata, int A_num_rows, int A_num_cols,
                            int A_row_stride, int A_col_stride,
                            const CuBlockMatrixData *B_cu_data,
                            int B_num_blocks, double alpha, double beta,
                            int B_trans) {
  if (B_trans) {
    _add_mat_blockmat_trans<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
        A_row_stride, A_col_stride, B_cu_data, B_num_blocks, alpha, beta);
  } else {
    _add_mat_blockmat<<<Gr,Bl>>>(data, d, Adata, A_num_rows, A_num_cols,
        A_row_stride, A_col_stride, B_cu_data, B_num_blocks, alpha, beta);
  }
}

void cudaD_block_add_mat_mat(dim3 Gr, dim3 Bl, CuBlockMatrixData *B_cu_data,
                             int num_blocks, const double *C_data,
                             int C_num_cols, int C_row_stride, int C_col_stride,
                             const double *D_data, int D_row_stride,
                             int D_col_stride, double alpha, double beta) {
  _block_add_mat_mat<<<Gr,Bl>>>(B_cu_data, num_blocks, C_data, C_num_cols,
      C_row_stride, C_col_stride, D_data, D_row_stride, D_col_stride,
      alpha, beta);
}

/*
 * cu::
 */
void cudaD_soft_hinge(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d,
                      int src_stride) {
  _soft_hinge<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_group_pnorm(dim3 Gr, dim3 Bl, double* y, const double* x,
                       MatrixDim d, int src_stride, int group_size,
                       double power) {
  _group_pnorm<<<Gr,Bl>>>(y, x, d, src_stride, group_size, power);
}

void cudaD_group_spec_pnorm(dim3 Gr, dim3 Bl, double* y, const double* x,
                            MatrixDim d, int src_stride, int group_size,
                            double power) {
  if (power == double(0)) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<L0NORM, double>());
  } else if (power == double(1)) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<L1NORM, double>());
  } else if (power == double(2)) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<L2NORM, double>());
  } else if (power == std::numeric_limits<double>::infinity()) {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<LINFNORM, double>());
  } else {
    _group_transform_reduce<<<Gr, Bl>>>(y, x, d, src_stride, group_size,
        TransReduceOp<LPNORM, double>(power));
  }
}

void cudaD_group_max(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d,
                     int src_stride, int group_size) {
  _group_transform_reduce<<<Gr,Bl>>>(y, x, d, src_stride, group_size,
      TransReduceOp<MAX, double>());
}

void cudaD_sigmoid(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d,
                   int src_stride) {
  _sigmoid<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_diff_sigmoid(dim3 Gr, dim3 Bl, double* eout, const double* e,
                        const double* y, MatrixDim d, int e_stride,
                        int y_stride) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaD_tanh(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d,
                int src_stride) {
  _tanh<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_diff_tanh(dim3 Gr, dim3 Bl, double* eout, const double* e,
                     const double* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaD_parametric_relu(dim3 Gr, dim3 Bl, double* y, const double* x,
                           MatrixDim d, int src_stride,
                           const double* a, const double* b) {
  _parametric_relu<<<Gr,Bl>>>(y, x, d, src_stride, a, b);
}

void cudaD_diff_parametric_relu(dim3 Gr, dim3 Bl, double* eout, const double* e,
                                const double* y, MatrixDim d, int e_stride,
                                int y_stride, const double* a, const double* b) {
  _diff_parametric_relu<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride, a, b);
}

void cudaD_heaviside(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d,
                     int src_stride) {
  _heaviside<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_softmax_reduce(size_t Gr, size_t Bl, double* y, const double* x,
                          MatrixDim d, int src_stride) {
  _softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaD_log_softmax_reduce(size_t Gr, size_t Bl, double* y, const double* x,
                              MatrixDim y_dim, int x_stride) {
  _log_softmax_reduce<<<Gr,Bl>>>(y, x, y_dim, x_stride);
}

void cudaD_normalize_per_row(size_t Gr, size_t Bl, double *y, int y_stride,
                             const double *x, MatrixDim x_d, double target_rms,
                             bool add_log_stddev) {
  _normalize_per_row<<<Gr, Bl>>>(y, y_stride, x, x_d, target_rms, add_log_stddev);
}

void cudaD_splice(dim3 Gr, dim3 Bl, double* y, const double* x,
                  const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in);
}

void cudaD_one(int Gr, int Bl, double* x, int dim) {
  _one<<<Gr,Bl>>>(x,dim);
}

void cudaD_take_mean(dim3 Gr, dim3 Bl, const double* x, double* y,
                     MatrixDim d_in) {
  _take_mean<<<Gr,Bl>>>(x,y,d_in);
}

void cudaD_take_lower(dim3 Gr, dim3 Bl, const double* x, double* y,
                      MatrixDim d_in) {
  _take_lower<<<Gr,Bl>>>(x,y,d_in);
}

void cudaD_take_upper(dim3 Gr, dim3 Bl, const double* x, double* y,
                      MatrixDim d_in) {
  _take_upper<<<Gr,Bl>>>(x,y,d_in);
}

void cudaD_copy_from_sp(dim3 Gr, dim3 Bl, const double* x, double* y,
                        MatrixDim d_out) {
  _copy_from_sp<<<Gr,Bl>>>(x,y,d_out);
}

void cudaD_copy(dim3 Gr, dim3 Bl, double* y, const double* x,
                const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaD_randomize(dim3 Gr, dim3 Bl, double* y, const double* x,
                     const int32_cuda* copy_from, MatrixDim d_out,
                     MatrixDim d_in) {
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaD_regularize_l1(dim3 Gr, dim3 Bl, double* wei, double* grad, double l1,
                         double lr, MatrixDim d, int stride_grad) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d,stride_grad);
}

void cudaD_find_row_max_id(dim3 Gr, dim3 Bl, const double* mat, double* vec_val,
                           int32_cuda* vec_id, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, d);
}

void cudaD_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda* vec_tgt,
                     double* mat_net_out, double* vec_log_post, MatrixDim d) {
  _diff_xent<<<Gr,Bl>>>(vec_tgt,mat_net_out,vec_log_post,d);
}

void cudaD_diff_softmax(dim3 Gr, dim3 Bl, double* x, const MatrixDim dim,
                        const double* value, const int value_stride,
                        const double* diff, const int diff_stride) {
  _diff_softmax<<<Gr, Bl>>>(x, dim, value, value_stride, diff, diff_stride);
}

void cudaD_diff_log_softmax(dim3 Gr, dim3 Bl, const MatrixDim in_deriv_dim,
                            const double* out_value, const int out_value_stride,
                            const double* out_deriv, const int out_deriv_stride,
                            double* in_deriv) {
  _diff_log_softmax<<<Gr, Bl>>>(in_deriv_dim, out_value, out_value_stride,
      out_deriv, out_deriv_stride, in_deriv);
}

void cudaD_copy_rows_from_vec(dim3 Gr, dim3 Bl, double *mat_out,
                              MatrixDim d_out, const double *v_in) {
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
                              const double *mat2, double *mask,
                              MatrixDim mat1_dim, int mat2_stride,
                              int mask_stride) {
  _equal_element_mask<<<Gr,Bl>>>(mat1, mat2, mask, mat1_dim, mat2_stride,
      mask_stride);
}

// Some conversion kernels for which it's more convenient
// to not name them F or D.

void cuda_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out,
                           const float* mat_in, MatrixDim d_out,
                           MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_ff(dim3 Gr, dim3 Bl, float* mat_out,
                           const float* mat_in, MatrixDim d_out,
                           MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_fd(dim3 Gr, dim3 Bl, float *mat_out,
                           const double* mat_in, MatrixDim d_out,
                           MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_dd(dim3 Gr, dim3 Bl, double *mat_out,
                           const double* mat_in, MatrixDim d_out,
                           MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out,
                                 const float* mat_in, MatrixDim d_out,
                                 MatrixDim d_in) {
  _copy_from_mat_trans<32> <<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_ff_trans(dim3 Gr, dim3 Bl, float* mat_out,
                                 const float* mat_in, MatrixDim d_out,
                                 MatrixDim d_in) {
  _copy_from_mat_trans<32> <<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float *mat_out,
                                 const double* mat_in, MatrixDim d_out,
                                 MatrixDim d_in) {
  _copy_from_mat_trans<32> <<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_dd_trans(dim3 Gr, dim3 Bl, double *mat_out,
                                 const double* mat_in, MatrixDim d_out,
                                 MatrixDim d_in) {
  _copy_from_mat_trans<32> <<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_smat_ff(dim3 Gr, dim3 Bl, float* mat_out,
                            const MatrixElement<float>* smat_in,
                            MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_fd(dim3 Gr, dim3 Bl, float* mat_out,
                            const MatrixElement<double>* smat_in,
                            MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_df(dim3 Gr, dim3 Bl, double* mat_out,
                            const MatrixElement<float>* smat_in,
                            MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_dd(dim3 Gr, dim3 Bl, double* mat_out,
                            const MatrixElement<double>* smat_in,
                            MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_ff_trans(dim3 Gr, dim3 Bl, float* mat_out,
                                  const MatrixElement<float>* smat_in,
                                  MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_fd_trans(dim3 Gr, dim3 Bl, float* mat_out,
                                  const MatrixElement<double>* smat_in,
                                  MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_df_trans(dim3 Gr, dim3 Bl, double* mat_out,
                                  const MatrixElement<float>* smat_in,
                                  MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}
void cuda_copy_from_smat_dd_trans(dim3 Gr, dim3 Bl, double* mat_out,
                                  const MatrixElement<double>* smat_in,
                                  MatrixDim d_out, MatrixIndexT_cuda d_in) {
  _copy_from_smat_trans<<<Gr,Bl>>>(mat_out, smat_in, d_out, d_in);
}

void cudaF_trace_mat_smat(dim3 Gr, dim3 Bl, const float* mat_in,
                          const MatrixElement<float>* smat_in,
                          MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in,
                          float* trace_vec_out) {
  _trace_mat_smat<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in,
      trace_vec_out);
}
void cudaF_trace_mat_smat_trans(dim3 Gr, dim3 Bl, const float* mat_in,
                                const MatrixElement<float>* smat_in,
                                MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in,
                                float* trace_vec_out) {
  _trace_mat_smat_trans<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in,
      trace_vec_out);
}
void cudaD_trace_mat_smat(dim3 Gr, dim3 Bl, const double* mat_in,
                          const MatrixElement<double>* smat_in,
                          MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in,
                          double* trace_vec_out) {
  _trace_mat_smat<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in,
      trace_vec_out);
}
void cudaD_trace_mat_smat_trans(dim3 Gr, dim3 Bl, const double* mat_in,
                                const MatrixElement<double>* smat_in,
                                MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in,
                                double* trace_vec_out) {
  _trace_mat_smat_trans<<<Gr,Bl>>>(mat_in, smat_in, mat_d_in, smat_d_in,
      trace_vec_out);
}
void cudaD_lstm_nonlinearity(dim3 Gr, dim3 Bl, const double* in,
                             const int in_stride, const double* params,
                             const int params_stride, const int out_stride,
                             const int cell_dim, const int have_dropout_mask,
                             const int num_rows, double* out) {
  _lstm_nonlinearity<<<Gr, Bl>>>(
      in, in_stride, params, params_stride,
      out_stride, cell_dim, have_dropout_mask, num_rows, out);
}
void cudaF_lstm_nonlinearity(dim3 Gr, dim3 Bl, const float* in,
                             const int in_stride, const float* params,
                             const int params_stride, const int out_stride,
                             const int cell_dim, const int have_dropout_mask,
                             const int num_rows, float* out) {
  _lstm_nonlinearity<<<Gr, Bl>>>(
      in, in_stride, params, params_stride,
      out_stride, cell_dim, have_dropout_mask, num_rows, out);
}
void cudaD_diff_lstm_nonlinearity(dim3 Gr, dim3 Bl, const int cell_dim,
                                  const int have_dropout_mask,
                                  const int num_rows, const double* input,
                                  const int input_stride, const double* params,
                                  const int params_stride,
                                  const double* output_deriv,
                                  const int output_deriv_stride,
                                  const double* deriv_sum_in,
                                  const int deriv_sum_in_stride,
                                  const double* self_repair_config,
                                  double count, double* input_deriv,
                                  const int input_deriv_stride,
                                  double* params_deriv,
                                  const int params_deriv_stride,
                                  double* value_sum_out,
                                  const int value_sum_out_stride,
                                  double* deriv_sum_out,
                                  const int deriv_sum_out_stride,
                                  double* self_repair_sum_out,
                                  const int self_repair_sum_out_stride) {
  _diff_lstm_nonlinearity<<<Gr, Bl>>>(
      cell_dim, have_dropout_mask, num_rows, input,
      input_stride, params, params_stride, output_deriv, output_deriv_stride,
      deriv_sum_in, deriv_sum_in_stride, self_repair_config, count, input_deriv,
      input_deriv_stride, params_deriv, params_deriv_stride, value_sum_out,
      value_sum_out_stride, deriv_sum_out, deriv_sum_out_stride,
      self_repair_sum_out, self_repair_sum_out_stride);
}
void cudaF_diff_lstm_nonlinearity(dim3 Gr, dim3 Bl, const int cell_dim,
                                  const int have_dropout_mask,
                                  const int num_rows, const float* input,
                                  const int input_stride, const float* params,
                                  const int params_stride,
                                  const float* output_deriv,
                                  const int output_deriv_stride,
                                  const double* deriv_sum_in,
                                  const int deriv_sum_in_stride,
                                  const float* self_repair_config, double count,
                                  float* input_deriv,
                                  const int input_deriv_stride,
                                  float* params_deriv,
                                  const int params_deriv_stride,
                                  double* value_sum_out,
                                  const int value_sum_out_stride,
                                  double* deriv_sum_out,
                                  const int deriv_sum_out_stride,
                                  float* self_repair_sum_out,
                                  const int self_repair_sum_out_stride) {
  _diff_lstm_nonlinearity<<<Gr, Bl>>>(
      cell_dim, have_dropout_mask, num_rows, input,
      input_stride, params, params_stride, output_deriv, output_deriv_stride,
      deriv_sum_in, deriv_sum_in_stride, self_repair_config, count, input_deriv,
      input_deriv_stride, params_deriv, params_deriv_stride, value_sum_out,
      value_sum_out_stride, deriv_sum_out, deriv_sum_out_stride,
      self_repair_sum_out, self_repair_sum_out_stride);
}


void cudaD_copy_cols_from_vec(dim3 Gr, dim3 Bl, double *mat_out,
                              MatrixDim d_out, const double *v_in) {
  _copy_cols_from_vec<<<Gr, Bl>>>(mat_out, d_out, v_in);
}
void cudaF_copy_cols_from_vec(dim3 Gr, dim3 Bl, float *mat_out, MatrixDim d_out,
                              const float *v_in) {
  _copy_cols_from_vec<<<Gr, Bl>>>(mat_out, d_out, v_in);
}

void cudaF_diff_normalize_per_row(size_t Gr, size_t Bl, float *id,
                                  int id_stride, const float *iv,
                                  MatrixDim iv_dim, const float* od,
                                  int od_stride, float target_rms,
                                  bool add_log_stddev) {
  _diff_normalize_per_row<<<Gr, Bl>>>(id, id_stride, iv, iv_dim, od, od_stride,
                                      target_rms, add_log_stddev);
}
void cudaD_diff_normalize_per_row(size_t Gr, size_t Bl, double *id,
                                  int id_stride, const double *iv,
                                  MatrixDim iv_dim, const double* od,
                                  int od_stride, double target_rms,
                                  bool add_log_stddev) {
  _diff_normalize_per_row<<<Gr, Bl>>>(id, id_stride, iv, iv_dim, od, od_stride,
                                      target_rms, add_log_stddev);
}
