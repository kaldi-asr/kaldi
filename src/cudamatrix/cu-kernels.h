// cudamatrix/cu-kernels.h

// Copyright 2009-2012  Karel Vesely
//                2013  Ehsan Variani
//                2014  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//           2013-2015  Guoguo Chen
//                2016  Shiyin Kang

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

#ifndef KALDI_CUDAMATRIX_CU_KERNELS_H_
#define KALDI_CUDAMATRIX_CU_KERNELS_H_

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "cudamatrix/cu-kernels-ansi.h"

/*
 * In this file are C++ templated wrappers
 * of the ANSI-C CUDA kernels
 */

namespace kaldi {

/*
 * CuMatrix
 */

inline void cuda_copy_upp_low(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) {
  cudaF_copy_upp_low(Gr, Bl, A, dimA);
}
inline void cuda_copy_low_upp(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) {
  cudaF_copy_low_upp(Gr, Bl, A, dimA);
}
inline void cuda_add_diag_vec_mat(dim3 Gr, dim3 Bl, float alpha, float *mat,
                                  MatrixDim mat_dim, const float *vec,
                                  const float *mat2, int mat2_row_stride,
                                  int mat2_col_stride, float beta) {
  cudaF_add_diag_vec_mat(Gr, Bl, alpha, mat, mat_dim, vec, mat2,
                         mat2_row_stride, mat2_col_stride, beta);
}
inline void cuda_copy_from_tp_trans(dim3 Gr, dim3 Bl, float* A, const float* B,
                                    MatrixDim dmat) {
  cudaF_copy_from_tp_trans(Gr, Bl, A, B, dmat);
}
inline void cuda_copy_from_tp_trans(dim3 Gr, dim3 Bl, float* A, const double* B,
                                    MatrixDim dmat) {
  cudaFD_copy_from_tp_trans(Gr, Bl, A, B, dmat);
}
inline void cuda_copy_from_tp(dim3 Gr, dim3 Bl, float* A, const float* B,
                              MatrixDim dmat) {
  cudaF_copy_from_tp(Gr, Bl, A, B, dmat);
}
inline void cuda_copy_from_tp(dim3 Gr, dim3 Bl, float* A, const double* B,
                              MatrixDim dmat) {
  cudaFD_copy_from_tp(Gr, Bl, A, B, dmat);
}

inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, float* mat_out,
                               const double* mat_in, MatrixDim d_out,
                               MatrixDim d_in) {
  cuda_copy_from_mat_fd(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, float* mat_out,
                               const float* mat_in, MatrixDim d_out,
                               MatrixDim d_in) {
  cuda_copy_from_mat_ff(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, double* mat_out,
                               const double* mat_in, MatrixDim d_out,
                               MatrixDim d_in) {
  cuda_copy_from_mat_dd(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, double* mat_out,
                               const float* mat_in, MatrixDim d_out,
                               MatrixDim d_in) {
  cuda_copy_from_mat_df(Gr, Bl, mat_out, mat_in, d_out, d_in);
}

inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, float* mat_out,
                                     const double* mat_in, MatrixDim d_out,
                                     MatrixDim d_in) {
  cuda_copy_from_mat_fd_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, float* mat_out,
                                     const float* mat_in, MatrixDim d_out,
                                     MatrixDim d_in) {
  cuda_copy_from_mat_ff_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, double* mat_out,
                                     const double* mat_in, MatrixDim d_out,
                                     MatrixDim d_in) {
  cuda_copy_from_mat_dd_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, double* mat_out,
                                     const float* mat_in, MatrixDim d_out,
                                     MatrixDim d_in) {
  cuda_copy_from_mat_df_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}

inline void cuda_copy_from_smat(dim3 Gr, dim3 Bl, float* mat_out,
                                const MatrixElement<float>* smat_in,
                                MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_ff(Gr, Bl, mat_out, smat_in, d_out, d_in);
}
inline void cuda_copy_from_smat(dim3 Gr, dim3 Bl, float* mat_out,
                                const MatrixElement<double>* smat_in,
                                MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_fd(Gr, Bl, mat_out, smat_in, d_out, d_in);
}
inline void cuda_copy_from_smat(dim3 Gr, dim3 Bl, double* mat_out,
                                const MatrixElement<float>* smat_in,
                                MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_df(Gr, Bl, mat_out, smat_in, d_out, d_in);
}
inline void cuda_copy_from_smat(dim3 Gr, dim3 Bl, double* mat_out,
                                const MatrixElement<double>* smat_in,
                                MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_dd(Gr, Bl, mat_out, smat_in, d_out, d_in);
}

inline void cuda_copy_from_smat_trans(dim3 Gr, dim3 Bl, float* mat_out,
                                      const MatrixElement<float>* smat_in,
                                      MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_ff_trans(Gr, Bl, mat_out, smat_in, d_out, d_in);
}
inline void cuda_copy_from_smat_trans(dim3 Gr, dim3 Bl, float* mat_out,
                                      const MatrixElement<double>* smat_in,
                                      MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_fd_trans(Gr, Bl, mat_out, smat_in, d_out, d_in);
}
inline void cuda_copy_from_smat_trans(dim3 Gr, dim3 Bl, double* mat_out,
                                      const MatrixElement<float>* smat_in,
                                      MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_df_trans(Gr, Bl, mat_out, smat_in, d_out, d_in);
}
inline void cuda_copy_from_smat_trans(dim3 Gr, dim3 Bl, double* mat_out,
                                      const MatrixElement<double>* smat_in,
                                      MatrixDim d_out, MatrixIndexT_cuda d_in) {
  cuda_copy_from_smat_dd_trans(Gr, Bl, mat_out, smat_in, d_out, d_in);
}

inline void cuda_trace_mat_smat(dim3 Gr, dim3 Bl, const float* mat_in,
                                const MatrixElement<float>* smat_in,
                                MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in,
                                float* trace_vec_out) {
  cudaF_trace_mat_smat(Gr, Bl, mat_in, smat_in, mat_d_in, smat_d_in,
                       trace_vec_out);
}
inline void cuda_trace_mat_smat_trans(dim3 Gr, dim3 Bl, const float* mat_in,
                                      const MatrixElement<float>* smat_in,
                                      MatrixDim mat_d_in,
                                      MatrixIndexT_cuda smat_d_in,
                                      float* trace_vec_out) {
  cudaF_trace_mat_smat_trans(Gr, Bl, mat_in, smat_in, mat_d_in, smat_d_in,
                             trace_vec_out);
}
inline void cuda_trace_mat_smat(dim3 Gr, dim3 Bl, const double* mat_in,
                                const MatrixElement<double>* smat_in,
                                MatrixDim mat_d_in, MatrixIndexT_cuda smat_d_in,
                                double* trace_vec_out) {
  cudaD_trace_mat_smat(Gr, Bl, mat_in, smat_in, mat_d_in, smat_d_in,
                       trace_vec_out);
}
inline void cuda_trace_mat_smat_trans(dim3 Gr, dim3 Bl, const double* mat_in,
                                      const MatrixElement<double>* smat_in,
                                      MatrixDim mat_d_in,
                                      MatrixIndexT_cuda smat_d_in,
                                      double* trace_vec_out) {
  cudaD_trace_mat_smat_trans(Gr, Bl, mat_in, smat_in, mat_d_in, smat_d_in,
                             trace_vec_out);
}

inline void cuda_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  cudaF_apply_exp(Gr, Bl, mat, d);
}
inline void cuda_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power,
                           MatrixDim dim) {
  cudaF_apply_pow(Gr, Bl, mat, power, dim);
}
inline void cuda_apply_pow_abs(dim3 Gr, dim3 Bl, float* mat, float power,
                               bool include_sign, MatrixDim dim) {
  cudaF_apply_pow_abs(Gr, Bl, mat, power, include_sign, dim);
}
inline void cuda_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim dim) {
  cudaF_apply_heaviside(Gr, Bl, mat, dim);
}
inline void cuda_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val,
                             MatrixDim dim) {
  cudaF_apply_floor(Gr, Bl, mat, floor_val, dim);
}
inline void cuda_apply_ceiling(dim3 Gr, dim3 Bl, float* mat, float ceiling_val,
                               MatrixDim dim) {
  cudaF_apply_ceiling(Gr, Bl, mat, ceiling_val, dim);
}
inline void cuda_copy_cols(dim3 Gr, dim3 Bl, float* dst, const float* src,
                           const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                           int src_stride) {
  cudaF_copy_cols(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_add_cols(dim3 Gr, dim3 Bl, float* dst, const float* src,
                          const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                          int src_stride) {
  cudaF_add_cols(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_copy_rows(dim3 Gr, dim3 Bl, float* dst, const float* src,
                           const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                           int src_stride) {
  cudaF_copy_rows(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_copy_rows(dim3 Gr, dim3 Bl, float* dst,
                           const float* const * src, MatrixDim dst_dim) {
  cudaF_copy_rows_direct(Gr, Bl, dst, src, dst_dim);
}
inline void cuda_copy_to_rows(dim3 Gr, dim3 Bl, float* const * dst,
                              const float* src, MatrixDim src_dim) {
  cudaF_copy_to_rows_direct(Gr, Bl, dst, src, src_dim);
}
inline void cuda_add_rows(dim3 Gr, dim3 Bl, float alpha, float* dst,
                          const float* src, const MatrixIndexT_cuda* reorder,
                          MatrixDim dst_dim, int src_stride) {
  cudaF_add_rows(Gr, Bl, alpha, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_add_rows(dim3 Gr, dim3 Bl, float alpha, float* dst,
                          const float* const * src, MatrixDim dst_dim) {
  cudaF_add_rows_direct(Gr, Bl, alpha, dst, src, dst_dim);
}
inline void cuda_add_to_rows(dim3 Gr, dim3 Bl, float alpha, float* const * dst,
                             const float* src, MatrixDim src_dim) {
  cudaF_add_to_rows_direct(Gr, Bl, alpha, dst, src, src_dim);
}
inline void cuda_trace(int Gr, int Bl, float* mat, float* value, int dim) {
  cudaF_trace(Gr, Bl, mat, value, dim);
}
inline void cuda_set_diag(int Gr, int Bl, float* mat, float value,
                          MatrixDim d) {
  cudaF_set_diag(Gr, Bl, mat, value, d);
}
inline void cuda_set_diag_packed(int Gr, int Bl, float* mat, float value,
                                 int dim) {
  cudaF_set_diag_packed(Gr, Bl, mat, value, dim);
}
inline void cuda_add_diag_packed(int Gr, int Bl, float* mat, float value,
                                 int dim) {
  cudaF_add_diag_packed(Gr, Bl, mat, value, dim);
}
inline void cuda_set_const(dim3 Gr, dim3 Bl, float *mat, float value,
                           MatrixDim d) {
  cudaF_set_const(Gr, Bl, mat, value, d);
}
inline void cuda_set_zero_above_diag(dim3 Gr, dim3 Bl, float* mat,
                                     MatrixDim d) {
  cudaF_set_zero_above_diag(Gr, Bl, mat, d);
}
inline void cuda_add(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) {
  cudaF_add(Gr, Bl, mat, value, d);
}
inline void cuda_add_vec2(dim3 Gr, dim3 Bl, float *mat, const float *vec,
                          const float alpha, int dim) {
  cudaF_add_vec2(Gr, Bl, mat, vec, alpha, dim);
}
inline void cuda_scale_diag_packed(int Gr, int Bl, float* mat, float value,
                                   int dim) {
  cudaF_scale_diag_packed(Gr, Bl, mat, value, dim);
}
inline void cuda_scale(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) {
  cudaF_scale(Gr, Bl, mat, value, d);
}
inline void cuda_apply_log(dim3 Gr, dim3 Bl, float *mat, MatrixDim d) {
  cudaF_apply_log(Gr, Bl, mat, d);
}
inline void cuda_mul_elements(dim3 Gr, dim3 Bl, float *mat, const float *A,
                              MatrixDim dst_d, int src_stride) {
  cudaF_mul_elements(Gr, Bl, mat, A, dst_d, src_stride);
}
inline void cuda_div_elements(dim3 Gr, dim3 Bl, float *mat, const float *A,
                              MatrixDim dst_d, int src_stride) {
  cudaF_div_elements(Gr, Bl, mat, A, dst_d, src_stride);
}
inline void cuda_max(dim3 Gr, dim3 Bl, float *mat, const float *A,
                     MatrixDim dst_d, int src_stride) {
  cudaF_max(Gr, Bl, mat, A, dst_d, src_stride);
}
inline void cuda_mul_cols_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale,
                              MatrixDim d) {
  cudaF_mul_cols_vec(Gr, Bl, mat, scale, d);
}
inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale,
                              MatrixDim d) {
  cudaF_mul_rows_vec(Gr, Bl, mat, scale, d);
}
inline void cuda_mul_rows_group_mat(dim3 Gr, dim3 Bl, float *y, const float *x,
                                    MatrixDim d, int src_stride,
                                    int group_size) {
  cudaF_mul_rows_group_mat(Gr, Bl, y, x, d, src_stride, group_size);
}

inline void cuda_diff_group_pnorm(dim3 Gr, dim3 Bl, float *id, const float *iv,
                                  const float *ov, const float* od,
                                  MatrixDim id_dim, int iv_stride,
                                  int ov_stride, int od_stride, int group_size,
                                  float power) {
  cudaF_diff_group_pnorm(Gr, Bl, id, iv, ov, od, id_dim, iv_stride, ov_stride,
                         od_stride, group_size, power);
}
inline void cuda_calc_group_max_deriv(dim3 Gr, dim3 Bl, float *y,
                                      const float *x1, const float *x2,
                                      MatrixDim y_dim, int x1_stride,
                                      int x2_stride, int group_size) {
  cudaF_calc_group_max_deriv(Gr, Bl, y, x1, x2, y_dim, x1_stride, x2_stride,
                             group_size);
}
inline void cuda_add_mat(dim3 Gr, dim3 Bl, float alpha, const float *src,
                         float *dst, MatrixDim d, int src_stride, int A_trans) {
  cudaF_add_mat(Gr, Bl, alpha, src, dst, d, src_stride, A_trans);
}
inline void cuda_add_mat_blocks(dim3 Gr, dim3 Bl, float alpha, const float *src,
                                int32_cuda num_row_blocks,
                                int32_cuda num_col_blocks, float *dst,
                                MatrixDim d, int src_stride, int A_trans) {
  cudaF_add_mat_blocks(Gr, Bl, alpha, src, num_row_blocks, num_col_blocks, dst,
                       d, src_stride, A_trans);
}
inline void cuda_set_mat_mat_div_mat(dim3 Gr, dim3 Bl, const float *A,
                                     const float *B, const float *C, float *dst,
                                     MatrixDim d, int stride_a, int stride_b,
                                     int stride_c) {
  cudaF_set_mat_mat_div_mat(Gr, Bl, A, B, C, dst, d, stride_a, stride_b,
                            stride_c);
}
inline void cuda_add_vec_to_cols(dim3 Gr, dim3 Bl, float alpha,
                                 const float *col, float beta, float *dst,
                                 MatrixDim d) {
  cudaF_add_vec_to_cols(Gr, Bl, alpha, col, beta, dst, d);
}
inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha,
                                 const float *row, float beta, float *dst,
                                 MatrixDim d) {
  cudaF_add_vec_to_rows(Gr, Bl, alpha, row, beta, dst, d);
}
inline void cuda_sy_add_tr2(dim3 Gr, dim3 Bl, float alpha, float beta,
                            const float* T, MatrixDim tdim, float *S,
                            MatrixDim sdim) {
  cudaF_sy_add_tr2(Gr, Bl, alpha, beta, T, tdim, S, sdim);
}
inline void cuda_add_mat_diag_vec(dim3 Gr, dim3 Bl, float alpha, float *mat,
                                  MatrixDim mat_dim, const float *mat2,
                                  int mat2_row_stride, int mat2_col_stride,
                                  const float *vec, float beta) {
  cudaF_add_mat_diag_vec(Gr, Bl, alpha, mat, mat_dim, mat2, mat2_row_stride,
                         mat2_col_stride, vec, beta);
}
inline void cuda_add_mat_mat_elements(dim3 Gr, dim3 Bl, float *data,
                                      const float *srcA_data,
                                      const float *srcB_data, MatrixDim dim,
                                      int srcA_stride, int srcB_stride,
                                      float alpha, float beta) {
  cudaF_add_mat_mat_elements(Gr, Bl, data, srcA_data, srcB_data, dim,
                             srcA_stride, srcB_stride, alpha, beta);
}

/*
 * CuVector
 */
inline void cuda_max_mat_cols(int Gr, int Bl, float* result, const float* mat,
                              const MatrixDim d) {
  cudaF_max_mat_cols(Gr, Bl, result, mat, d);
}
inline void cuda_min_mat_cols(int Gr, int Bl, float* result, const float* mat,
                              const MatrixDim d) {
  cudaF_min_mat_cols(Gr, Bl, result, mat, d);
}
inline void cuda_sum_mat_cols(int Gr, int Bl, float* result, const float* mat,
                              const MatrixDim d) {
  cudaF_sum_mat_cols(Gr, Bl, result, mat, d);
}
inline void cuda_replace_value(int Gr, int Bl, float *v, int dim, float orig,
                               float changed) {
  cudaF_replace_value(Gr, Bl, v, dim, orig, changed);
}
inline void cuda_div_rows_vec(dim3 Gr, dim3 Bl, float *mat,
                              const float *vec_div, MatrixDim d) {
  cudaF_div_rows_vec(Gr, Bl, mat, vec_div, d);
}
inline void cuda_set_bias_params(int Gr, int Bl, float* v, const float* a,
                                 float param_1, float param_2, float param_3,
                                 int* flag, int dim) {
  cudaF_set_bias_params(Gr, Bl, v, a, param_1, param_2, param_3, flag, dim);
}
inline void cuda_vec_mul_elements(int Gr, int Bl, float* v, const float* a,
                                  int dim) {
  cudaF_vec_mul_elements(Gr, Bl, v, a, dim);
}
inline void cuda_vec_soft_max(int Gr, int Bl, float* v, int dim) {
  cudaF_vec_soft_max(Gr, Bl, v, dim);
}
inline void cuda_vec_min(int Gr, int Bl, const float* v, float* value, int dim,
                         int inc) {
  cudaF_vec_min(Gr, Bl, v, value, dim, inc);
}
inline void cuda_vec_max(int Gr, int Bl, const float* v, float* value, int dim,
                         int inc) {
  cudaF_vec_max(Gr, Bl, v, value, dim, inc);
}
inline void cuda_trace_mat_mat_trans(dim3 Gr, dim3 Bl, const float* A,
                                     const float* B, MatrixDim dA, int B_stride,
                                     float* value) {
  cudaF_trace_mat_mat_trans(Gr, Bl, A, B, dA, B_stride, value);
}
inline void cuda_trace_mat_mat(dim3 Gr, dim3 Bl, const float* A, const float* B,
                               MatrixDim dA, int B_stride, float* value) {
  cudaF_trace_mat_mat(Gr, Bl, A, B, dA, B_stride, value);
}
inline void cuda_add_diag_mat_mat_MNT(int Gr, int Bl, const float alpha,
                                      const float* M, const MatrixDim dim_M,
                                      const float* N, const int stride_N,
                                      const float beta, float* v) {
  cudaF_add_diag_mat_mat_MNT(Gr, Bl, alpha, M, dim_M, N, stride_N, beta, v);
}
inline void cuda_add_diag_mat_mat_MTN(dim3 Gr, dim3 Bl, const float alpha,
                                      const float* M, const int stride_M,
                                      const float* N, const MatrixDim dim_N,
                                      const float beta, float* v) {
  cudaF_add_diag_mat_mat_MTN(Gr, Bl, alpha, M, stride_M, N, dim_N, beta, v);
}
inline void cuda_add_diag_mat_mat_MN(dim3 Gr, dim3 Bl, const float alpha,
                                     const float* M, const int stride_M,
                                     const float* N, const MatrixDim dim_N,
                                     const float beta, float* v) {
  cudaF_add_diag_mat_mat_MN(Gr, Bl, alpha, M, stride_M, N, dim_N, beta, v);
}
inline void cuda_add_vec_vec(int Gr, int Bl, float alpha, float* v,
                             const float* x, const float* y, float beta,
                             int dim) {
  cudaF_add_vec_vec(Gr, Bl, alpha, v, x, y, beta, dim);
}
inline void cuda_copy_col_from_mat_df(int Gr, int Bl, double* v, int col,
                                      const float* mat, MatrixDim dmat,
                                      int dim) {
  cudaF_copy_col_from_mat_df(Gr, Bl, v, col, mat, dmat, dim);
}
inline void cuda_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col,
                                      const float* mat, MatrixDim dmat,
                                      int dim) {
  cudaF_copy_col_from_mat_fd(Gr, Bl, v, col, mat, dmat, dim);
}
inline void cuda_vec_sum(int Gr, int Bl, float* v, float* value, int dim,
                         int inc) {
  cudaF_vec_sum(Gr, Bl, v, value, dim, inc);
}
inline void cuda_vec_copy_diag_from_packed(int Gr, int Bl, float *dst,
                                           const float *src, int dim) {
  cudaF_vec_copy_diag_from_packed(Gr, Bl, dst, src, dim);
}
inline void cuda_vec_apply_floor(int Gr, int Bl, float* v, float floor_val,
                                 float* num, int dim) {
  cudaF_vec_apply_floor(Gr, Bl, v, floor_val, num, dim);
}
inline void cuda_vec_apply_ceiling(int Gr, int Bl, float* v, float floor_val,
                                   float* num, int dim) {
  cudaF_vec_apply_ceiling(Gr, Bl, v, floor_val, num, dim);
}
inline void cuda_vec_apply_exp(int Gr, int Bl, float* v, int dim) {
  cudaF_vec_apply_exp(Gr, Bl, v, dim);
}
inline void cuda_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim) {
  cudaF_vec_apply_log(Gr, Bl, v, flag, dim);
}
inline void cuda_invert_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim d) {
  cudaF_invert_elements(Gr, Bl, data, d);
}
// B_trans nonzero if B transposed.
inline void cuda_add_mat_blockmat(dim3 Gr, dim3 Bl, float *data, MatrixDim d,
                                  const float *Adata, int A_num_rows,
                                  int A_num_cols, int A_row_stride,
                                  int A_col_stride,
                                  const CuBlockMatrixData *B_cu_data,
                                  int B_num_blocks, float alpha, float beta,
                                  int B_trans) {
  cudaF_add_mat_blockmat(Gr, Bl, data, d, Adata, A_num_rows, A_num_cols,
                         A_row_stride, A_col_stride, B_cu_data, B_num_blocks,
                         alpha, beta, B_trans);
}
inline void cuda_block_add_mat_mat(dim3 Gr, dim3 Bl,
                                   CuBlockMatrixData *B_cu_data, int num_blocks,
                                   const float *C_data, int C_num_cols,
                                   int C_row_stride, int C_col_stride,
                                   const float *D_data, int D_row_stride,
                                   int D_col_stride, float alpha, float beta) {
  cudaF_block_add_mat_mat(Gr, Bl, B_cu_data, num_blocks, C_data, C_num_cols,
                          C_row_stride, C_col_stride, D_data, D_row_stride,
                          D_col_stride, alpha, beta);
}

/*
 * cu::
 */
inline void cuda_soft_hinge(dim3 Gr, dim3 Bl, float *y, const float *x,
                            MatrixDim d, int src_stride) {
  cudaF_soft_hinge(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_group_pnorm(dim3 Gr, dim3 Bl, float *y, const float *x,
                             MatrixDim d, int src_stride, int group_size,
                             float power) {
  cudaF_group_pnorm(Gr, Bl, y, x, d, src_stride, group_size, power);
}
inline void cuda_group_spec_pnorm(dim3 Gr, dim3 Bl, float *y, const float *x,
                                  MatrixDim d, int src_stride, int group_size,
                                  float power) {
  cudaF_group_spec_pnorm(Gr, Bl, y, x, d, src_stride, group_size, power);
}
inline void cuda_group_max(dim3 Gr, dim3 Bl, float *y, const float *x,
                           MatrixDim d, int src_stride, int group_size) {
  cudaF_group_max(Gr, Bl, y, x, d, src_stride, group_size);
}
inline void cuda_sigmoid(dim3 Gr, dim3 Bl, float *y, const float *x,
                         MatrixDim d, int src_stride) {
  cudaF_sigmoid(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, float *eout, const float *e,
                              const float *y, MatrixDim d, int e_stride,
                              int y_stride) {
  cudaF_diff_sigmoid(Gr, Bl, eout, e, y, d, e_stride, y_stride);
}
inline void cuda_tanh(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d,
                      int src_stride) {
  cudaF_tanh(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, float *eout, const float *e,
                           const float *y, MatrixDim d, int e_stride,
                           int y_stride) {
  cudaF_diff_tanh(Gr, Bl, eout, e, y, d, e_stride, y_stride);
}
inline void cuda_parametric_relu(dim3 Gr, dim3 Bl, float *y, const float *x,
                                 MatrixDim d, int src_stride,
                                 const float *a, const float *b) {
  cudaF_parametric_relu(Gr,Bl,y,x,d,src_stride,a,b);
}
inline void cuda_diff_parametric_relu(dim3 Gr, dim3 Bl, float *eout,
                                      const float *e, const float *y,
                                      MatrixDim d, int e_stride, int y_stride,
                                      const float *a, const float *b) {
  cudaF_diff_parametric_relu(Gr,Bl,eout,e,y,d,e_stride,y_stride,a,b);
}
inline void cuda_heaviside(dim3 Gr, dim3 Bl, float *y, const float *x,
                           MatrixDim d, int src_stride) {
  cudaF_heaviside(Gr, Bl, y, x, d, src_stride);
}
// Bl: dimBlock value is fixed min(d.col, CU1DBLOCK), represent CU1DBLOCK
//     threads reduce a row at the same time.
// Gr: the number of rows
inline void cuda_softmax_reduce(size_t Gr, size_t Bl, float *y, const float *x,
                                MatrixDim d, int src_stride) {
  cudaF_softmax_reduce(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_log_softmax_reduce(size_t Gr, size_t Bl, float *y,
                                    const float *x, MatrixDim y_dim,
                                    int x_stride) {
  cudaF_log_softmax_reduce(Gr, Bl, y, x, y_dim, x_stride);
}

inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, float *wei, float *grad,
                               float l1, float lr, MatrixDim d,
                               int stride_grad) {
  cudaF_regularize_l1(Gr, Bl, wei, grad, l1, lr, d, stride_grad);
}
inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const float *mat,
                                 float *vec_val, int32_cuda *vec_id,
                                 MatrixDim d) {
  cudaF_find_row_max_id(Gr, Bl, mat, vec_val, vec_id, d);
}
inline void cuda_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt,
                           float *mat_net_out, float *vec_log_post,
                           MatrixDim d) {
  cudaF_diff_xent(Gr, Bl, vec_tgt, mat_net_out, vec_log_post, d);
}
inline void cuda_normalize_per_row(size_t Gr, size_t Bl, float *y, int y_stride,
                                   const float *x, MatrixDim x_d,
                                   float target_rms, bool add_log_stddev) {
  cudaF_normalize_per_row(Gr, Bl, y, y_stride, x, x_d, target_rms,
                          add_log_stddev);
}
inline void cuda_diff_softmax(dim3 Gr, dim3 Bl, float* x, const MatrixDim dim,
                              const float* value, const int value_stride,
                              const float* diff, const int diff_stride) {
  cudaF_diff_softmax(Gr, Bl, x, dim, value, value_stride, diff, diff_stride);
}
inline void cuda_diff_log_softmax(dim3 Gr, dim3 Bl,
                                  const MatrixDim in_deriv_dim,
                                  const float* out_value,
                                  const int out_value_stride,
                                  const float* out_deriv,
                                  const int out_deriv_stride, float* in_deriv) {
  cudaF_diff_log_softmax(Gr, Bl, in_deriv_dim, out_value, out_value_stride,
                         out_deriv, out_deriv_stride, in_deriv);
}
inline void cuda_copy_rows_from_vec(dim3 Gr, dim3 Bl, float *mat_out,
                                    MatrixDim d_out, const float *v_in) {
  cudaF_copy_rows_from_vec(Gr, Bl, mat_out, d_out, v_in);
}

inline void cuda_randomize(dim3 Gr, dim3 Bl, float *y, const float *x,
                           const int32_cuda *copy_from, MatrixDim d_out,
                           MatrixDim d_in) {
  cudaF_randomize(Gr, Bl, y, x, copy_from, d_out, d_in);
}

inline void cuda_splice(dim3 Gr, dim3 Bl, float *y, const float *x,
                        const int32_cuda *off, MatrixDim d_out,
                        MatrixDim d_in) {
  cudaF_splice(Gr, Bl, y, x, off, d_out, d_in);
}
inline void cuda_one(int Gr, int Bl, float* x, int dim) {
  cudaF_one(Gr, Bl, x, dim);
}
inline void cuda_copy(dim3 Gr, dim3 Bl, float *y, const float *x,
                      const int32_cuda *copy_from, MatrixDim d_out,
                      MatrixDim d_in) {
  cudaF_copy(Gr, Bl, y, x, copy_from, d_out, d_in);
}
inline void cuda_copy_from_sp(dim3 Gr, dim3 Bl, const float* x, float* y,
                              MatrixDim d_out) {
  cudaF_copy_from_sp(Gr, Bl, x, y, d_out);
}
inline void cuda_take_lower(dim3 Gr, dim3 Bl, const float* x, float* y,
                            MatrixDim d_in) {
  cudaF_take_lower(Gr, Bl, x, y, d_in);
}
inline void cuda_take_upper(dim3 Gr, dim3 Bl, const float* x, float* y,
                            MatrixDim d_in) {
  cudaF_take_upper(Gr, Bl, x, y, d_in);
}
inline void cuda_take_mean(dim3 Gr, dim3 Bl, const float* x, float* y,
                           MatrixDim d_in) {
  cudaF_take_mean(Gr, Bl, x, y, d_in);
}
inline void cuda_matrix_add_elements(dim3 Gr, dim3 Bl, float *data,
                                     MatrixDim dim, float alpha,
                                     MatrixElement<float>* x,
                                     int num_elements) {
  cudaF_matrix_add_elements(Gr, Bl, data, dim, alpha, x, num_elements);
}
inline void cuda_matrix_add_indexed_values(dim3 Gr, dim3 Bl, MatrixDim dim,
                                           float alpha,
                                           const Int32Pair* indices,
                                           const float* x, int s, float* data) {
  cudaF_matrix_add_indexed_values(Gr, Bl, dim, alpha, indices, x, s, data);
}
inline void cuda_comp_obj_deriv(dim3 Gr, dim3 Bl, MatrixElement<float>* x,
                                int32 size, const float* z, MatrixDim d,
                                float* z2, MatrixDim d2, float* t) {
  cudaF_comp_obj_deriv(Gr, Bl, x, size, z, d, z2, d2, t);
}
inline void cuda_sum_column_ranges(dim3 Gr, dim3 Bl, float *data, MatrixDim dim,
                                   const float *src_data, MatrixDim src_dim,
                                   const Int32Pair *indices) {
  cudaF_sum_column_ranges(Gr, Bl, data, dim, src_data, src_dim, indices);
}
inline void cuda_add_row_ranges(dim3 Gr, dim3 Bl, float *data, MatrixDim dim,
                                const float *src_data, MatrixDim src_dim,
                                const Int32Pair *indexes) {
  cudaF_add_row_ranges(Gr, Bl, data, dim, src_data, src_dim, indexes);
}
inline void cuda_matrix_lookup(dim3 Gr, dim3 Bl, const float *data,
                               MatrixDim dim, const Int32Pair *indices,
                               int indices_size, float *output) {
  cudaF_matrix_lookup(Gr, Bl, data, dim, indices, indices_size, output);
}

inline void cuda_equal_element_mask(dim3 Gr, dim3 Bl, const float *mat1,
                                    const float *mat2, float *mask,
                                    MatrixDim mat1_dim, int mat2_stride,
                                    int mask_stride) {
  cudaF_equal_element_mask(Gr, Bl, mat1, mat2, mask, mat1_dim, mat2_stride,
                           mask_stride);
}

// double versions

/*
 * CuMatrix
 */
inline void cuda_copy_upp_low(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) {
  cudaD_copy_upp_low(Gr, Bl, A, dimA);
}
inline void cuda_copy_low_upp(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) {
  cudaD_copy_low_upp(Gr, Bl, A, dimA);
}
inline void cuda_add_diag_vec_mat(dim3 Gr, dim3 Bl, double alpha, double *mat,
                                  MatrixDim mat_dim, const double *vec,
                                  const double *mat2, int mat2_row_stride,
                                  int mat2_col_stride, double beta) {
  cudaD_add_diag_vec_mat(Gr, Bl, alpha, mat, mat_dim, vec, mat2,
                         mat2_row_stride, mat2_col_stride, beta);
}
inline void cuda_copy_from_tp_trans(dim3 Gr, dim3 Bl, double* A,
                                    const double* B, MatrixDim dmat) {
  cudaD_copy_from_tp_trans(Gr, Bl, A, B, dmat);
}
inline void cuda_copy_from_tp_trans(dim3 Gr, dim3 Bl, double* A, const float* B,
                                    MatrixDim dmat) {
  cudaDF_copy_from_tp_trans(Gr, Bl, A, B, dmat);
}
inline void cuda_copy_from_tp(dim3 Gr, dim3 Bl, double* A, const double* B,
                              MatrixDim dmat) {
  cudaD_copy_from_tp(Gr, Bl, A, B, dmat);
}
inline void cuda_copy_from_tp(dim3 Gr, dim3 Bl, double* A, const float* B,
                              MatrixDim dmat) {
  cudaDF_copy_from_tp(Gr, Bl, A, B, dmat);
}
inline void cuda_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  cudaD_apply_exp(Gr, Bl, mat, d);
}
inline void cuda_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power,
                           MatrixDim dim) {
  cudaD_apply_pow(Gr, Bl, mat, power, dim);
}
inline void cuda_apply_pow_abs(dim3 Gr, dim3 Bl, double* mat, double power,
                               bool include_sign, MatrixDim dim) {
  cudaD_apply_pow_abs(Gr, Bl, mat, power, include_sign, dim);
}
inline void cuda_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim dim) {
  cudaD_apply_heaviside(Gr, Bl, mat, dim);
}
inline void cuda_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val,
                             MatrixDim dim) {
  cudaD_apply_floor(Gr, Bl, mat, floor_val, dim);
}
inline void cuda_apply_ceiling(dim3 Gr, dim3 Bl, double* mat,
                               double ceiling_val, MatrixDim dim) {
  cudaD_apply_ceiling(Gr, Bl, mat, ceiling_val, dim);
}
inline void cuda_copy_cols(dim3 Gr, dim3 Bl, double* dst, const double* src,
                           const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                           int src_stride) {
  cudaD_copy_cols(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_add_cols(dim3 Gr, dim3 Bl, double* dst, const double* src,
                          const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                          int src_stride) {
  cudaD_add_cols(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_copy_rows(dim3 Gr, dim3 Bl, double* dst, const double* src,
                           const MatrixIndexT_cuda* reorder, MatrixDim dst_dim,
                           int src_stride) {
  cudaD_copy_rows(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_copy_rows(dim3 Gr, dim3 Bl, double* dst,
                           const double* const * src, MatrixDim dst_dim) {
  cudaD_copy_rows_direct(Gr, Bl, dst, src, dst_dim);
}
inline void cuda_copy_to_rows(dim3 Gr, dim3 Bl, double* const * dst,
                              const double* src, MatrixDim src_dim) {
  cudaD_copy_to_rows_direct(Gr, Bl, dst, src, src_dim);
}
inline void cuda_add_rows(dim3 Gr, dim3 Bl, double alpha, double* dst,
                          const double* src, const MatrixIndexT_cuda* reorder,
                          MatrixDim dst_dim, int src_stride) {
  cudaD_add_rows(Gr, Bl, alpha, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_add_rows(dim3 Gr, dim3 Bl, double alpha, double* dst,
                          const double* const * src, MatrixDim dst_dim) {
  cudaD_add_rows_direct(Gr, Bl, alpha, dst, src, dst_dim);
}
inline void cuda_add_to_rows(dim3 Gr, dim3 Bl, double alpha,
                             double* const * dst, const double* src,
                             MatrixDim src_dim) {
  cudaD_add_to_rows_direct(Gr, Bl, alpha, dst, src, src_dim);
}
inline void cuda_trace(int Gr, int Bl, double* mat, double* value, int dim) {
  cudaD_trace(Gr, Bl, mat, value, dim);
}
inline void cuda_set_diag(int Gr, int Bl, double* mat, double value,
                          MatrixDim d) {
  cudaD_set_diag(Gr, Bl, mat, value, d);
}
inline void cuda_set_diag_packed(int Gr, int Bl, double* mat, double value,
                                 int dim) {
  cudaD_set_diag_packed(Gr, Bl, mat, value, dim);
}
inline void cuda_add_diag_packed(int Gr, int Bl, double* mat, double value,
                                 int dim) {
  cudaD_add_diag_packed(Gr, Bl, mat, value, dim);
}
inline void cuda_set_const(dim3 Gr, dim3 Bl, double *mat, double value,
                           MatrixDim d) {
  cudaD_set_const(Gr, Bl, mat, value, d);
}
inline void cuda_set_zero_above_diag(dim3 Gr, dim3 Bl, double* mat,
                                     MatrixDim d) {
  cudaD_set_zero_above_diag(Gr, Bl, mat, d);
}
inline void cuda_add(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) {
  cudaD_add(Gr, Bl, mat, value, d);
}
inline void cuda_add_vec2(dim3 Gr, dim3 Bl, double *mat, const double *vec,
                          const double alpha, int dim) {
  cudaD_add_vec2(Gr, Bl, mat, vec, alpha, dim);
}
inline void cuda_scale_diag_packed(int Gr, int Bl, double* mat, double value,
                                   int dim) {
  cudaD_scale_diag_packed(Gr, Bl, mat, value, dim);
}
inline void cuda_scale(dim3 Gr, dim3 Bl, double *mat, double value,
                       MatrixDim d) {
  cudaD_scale(Gr, Bl, mat, value, d);
}
inline void cuda_apply_log(dim3 Gr, dim3 Bl, double *mat, MatrixDim d) {
  cudaD_apply_log(Gr, Bl, mat, d);
}
inline void cuda_mul_elements(dim3 Gr, dim3 Bl, double *mat, const double *A,
                              MatrixDim dst_d, int src_stride) {
  cudaD_mul_elements(Gr, Bl, mat, A, dst_d, src_stride);
}
inline void cuda_div_elements(dim3 Gr, dim3 Bl, double *mat, const double *A,
                              MatrixDim dst_d, int src_stride) {
  cudaD_div_elements(Gr, Bl, mat, A, dst_d, src_stride);
}
inline void cuda_max(dim3 Gr, dim3 Bl, double *mat, const double *A,
                     MatrixDim dst_d, int src_stride) {
  cudaD_max(Gr, Bl, mat, A, dst_d, src_stride);
}
inline void cuda_mul_cols_vec(dim3 Gr, dim3 Bl, double *mat,
                              const double *scale, MatrixDim d) {
  cudaD_mul_cols_vec(Gr, Bl, mat, scale, d);
}
inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, double *mat,
                              const double *scale, MatrixDim d) {
  cudaD_mul_rows_vec(Gr, Bl, mat, scale, d);
}
inline void cuda_mul_rows_group_mat(dim3 Gr, dim3 Bl, double *y,
                                    const double *x, MatrixDim d,
                                    int src_stride, int group_size) {
  cudaD_mul_rows_group_mat(Gr, Bl, y, x, d, src_stride, group_size);
}

inline void cuda_diff_group_pnorm(dim3 Gr, dim3 Bl, double *id,
                                  const double *iv, const double *ov,
                                  const double* od, MatrixDim id_dim,
                                  int iv_stride, int ov_stride, int od_stride,
                                  int group_size, double power) {
  cudaD_diff_group_pnorm(Gr, Bl, id, iv, ov, od, id_dim, iv_stride, ov_stride,
                         od_stride, group_size, power);
}
inline void cuda_calc_group_max_deriv(dim3 Gr, dim3 Bl, double *y,
                                      const double *x1, const double *x2,
                                      MatrixDim y_dim, int x1_stride,
                                      int x2_stride, int group_size) {
  cudaD_calc_group_max_deriv(Gr, Bl, y, x1, x2, y_dim, x1_stride, x2_stride,
                             group_size);
}
inline void cuda_add_mat(dim3 Gr, dim3 Bl, double alpha, const double *src,
                         double *dst, MatrixDim d, int src_stride,
                         int A_trans) {
  cudaD_add_mat(Gr, Bl, alpha, src, dst, d, src_stride, A_trans);
}
inline void cuda_add_mat_blocks(dim3 Gr, dim3 Bl, double alpha,
                                const double *src, int32_cuda num_row_blocks,
                                int32_cuda num_col_blocks, double *dst,
                                MatrixDim d, int src_stride, int A_trans) {
  cudaD_add_mat_blocks(Gr, Bl, alpha, src, num_row_blocks, num_col_blocks, dst,
                       d, src_stride, A_trans);
}
inline void cuda_set_mat_mat_div_mat(dim3 Gr, dim3 Bl, const double *A,
                                     const double *B, const double *C,
                                     double *dst, MatrixDim d, int stride_a,
                                     int stride_b, int stride_c) {
  cudaD_set_mat_mat_div_mat(Gr, Bl, A, B, C, dst, d, stride_a, stride_b,
                            stride_c);
}
inline void cuda_add_vec_to_cols(dim3 Gr, dim3 Bl, double alpha,
                                 const double *col, double beta, double *dst,
                                 MatrixDim d) {
  cudaD_add_vec_to_cols(Gr, Bl, alpha, col, beta, dst, d);
}
inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha,
                                 const double *row, double beta, double *dst,
                                 MatrixDim d) {
  cudaD_add_vec_to_rows(Gr, Bl, alpha, row, beta, dst, d);
}
inline void cuda_sy_add_tr2(dim3 Gr, dim3 Bl, double alpha, double beta,
                            const double* T, MatrixDim tdim, double *S,
                            MatrixDim sdim) {
  cudaD_sy_add_tr2(Gr, Bl, alpha, beta, T, tdim, S, sdim);
}
inline void cuda_add_mat_diag_vec(dim3 Gr, dim3 Bl, double alpha, double *mat,
                                  MatrixDim mat_dim, const double *mat2,
                                  int mat2_row_stride, int mat2_col_stride,
                                  const double *vec, double beta) {
  cudaD_add_mat_diag_vec(Gr, Bl, alpha, mat, mat_dim, mat2, mat2_row_stride,
                         mat2_col_stride, vec, beta);
}
inline void cuda_add_mat_mat_elements(dim3 Gr, dim3 Bl, double *data,
                                      const double *srcA_data,
                                      const double *srcB_data, MatrixDim dim,
                                      int srcA_stride, int srcB_stride,
                                      double alpha, double beta) {
  cudaD_add_mat_mat_elements(Gr, Bl, data, srcA_data, srcB_data, dim,
                             srcA_stride, srcB_stride, alpha, beta);
}

/*
 * CuVector
 */

inline void cuda_max_mat_cols(int Gr, int Bl, double* result, const double* mat,
                              const MatrixDim d) {
  cudaD_max_mat_cols(Gr, Bl, result, mat, d);
}
inline void cuda_min_mat_cols(int Gr, int Bl, double* result, const double* mat,
                              const MatrixDim d) {
  cudaD_min_mat_cols(Gr, Bl, result, mat, d);
}
inline void cuda_sum_mat_cols(int Gr, int Bl, double* result, const double* mat,
                              const MatrixDim d) {
  cudaD_sum_mat_cols(Gr, Bl, result, mat, d);
}
inline void cuda_replace_value(int Gr, int Bl, double *v, int dim, double orig,
                               double changed) {
  cudaD_replace_value(Gr, Bl, v, dim, orig, changed);
}
inline void cuda_div_rows_vec(dim3 Gr, dim3 Bl, double *mat,
                              const double *vec_div, MatrixDim d) {
  cudaD_div_rows_vec(Gr, Bl, mat, vec_div, d);
}
inline void cuda_set_bias_params(int Gr, int Bl, double* v, const double* a,
                                 double param_1, double param_2, double param_3,
                                 int* flag, int dim) {
  cudaD_set_bias_params(Gr, Bl, v, a, param_1, param_2, param_3, flag, dim);
}
inline void cuda_vec_mul_elements(int Gr, int Bl, double* v, const double* a,
                                  int dim) {
  cudaD_vec_mul_elements(Gr, Bl, v, a, dim);
}
inline void cuda_vec_soft_max(int Gr, int Bl, double* v, int dim) {
  cudaD_vec_soft_max(Gr, Bl, v, dim);
}
inline void cuda_vec_min(int Gr, int Bl, const double* v, double* value,
                         int dim, int inc) {
  cudaD_vec_min(Gr, Bl, v, value, dim, inc);
}
inline void cuda_vec_max(int Gr, int Bl, const double* v, double* value,
                         int dim, int inc) {
  cudaD_vec_max(Gr, Bl, v, value, dim, inc);
}
inline void cuda_trace_mat_mat_trans(dim3 Gr, dim3 Bl, const double* A,
                                     const double* B, MatrixDim dA,
                                     int B_stride, double* value) {
  cudaD_trace_mat_mat_trans(Gr, Bl, A, B, dA, B_stride, value);
}
inline void cuda_trace_mat_mat(dim3 Gr, dim3 Bl, const double* A,
                               const double* B, MatrixDim dA, int B_stride,
                               double* value) {
  cudaD_trace_mat_mat(Gr, Bl, A, B, dA, B_stride, value);
}
inline void cuda_add_diag_mat_mat_MNT(int Gr, int Bl, const double alpha,
                                      const double* M, const MatrixDim dim_M,
                                      const double* N, const int stride_N,
                                      const double beta, double* v) {
  cudaD_add_diag_mat_mat_MNT(Gr, Bl, alpha, M, dim_M, N, stride_N, beta, v);
}
inline void cuda_add_diag_mat_mat_MTN(dim3 Gr, dim3 Bl, const double alpha,
                                      const double* M, const int stride_M,
                                      const double* N, const MatrixDim dim_N,
                                      const double beta, double* v) {
  cudaD_add_diag_mat_mat_MTN(Gr, Bl, alpha, M, stride_M, N, dim_N, beta, v);
}
inline void cuda_add_diag_mat_mat_MN(dim3 Gr, dim3 Bl, const double alpha,
                                     const double* M, const int stride_M,
                                     const double* N, const MatrixDim dim_N,
                                     const double beta, double* v) {
  cudaD_add_diag_mat_mat_MN(Gr, Bl, alpha, M, stride_M, N, dim_N, beta, v);
}
inline void cuda_add_vec_vec(int Gr, int Bl, double alpha, double* v,
                             const double* x, const double* y, double beta,
                             int dim) {
  cudaD_add_vec_vec(Gr, Bl, alpha, v, x, y, beta, dim);
}
inline void cuda_copy_col_from_mat_df(int Gr, int Bl, double* v, int col,
                                      const double* mat, MatrixDim dmat,
                                      int dim) {
  cudaD_copy_col_from_mat_df(Gr, Bl, v, col, mat, dmat, dim);
}
inline void cuda_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col,
                                      const double* mat, MatrixDim dmat,
                                      int dim) {
  cudaD_copy_col_from_mat_fd(Gr, Bl, v, col, mat, dmat, dim);
}
inline void cuda_vec_sum(int Gr, int Bl, double* v, double* value, int dim,
                         int inc) {
  cudaD_vec_sum(Gr, Bl, v, value, dim, inc);
}
inline void cuda_vec_copy_diag_from_packed(int Gr, int Bl, double *dst,
                                           const double *src, int dim) {
  cudaD_vec_copy_diag_from_packed(Gr, Bl, dst, src, dim);
}
inline void cuda_vec_apply_floor(int Gr, int Bl, double* v, double floor_val,
                                 float* num, int dim) {
  cudaD_vec_apply_floor(Gr, Bl, v, floor_val, num, dim);
}
inline void cuda_vec_apply_ceiling(int Gr, int Bl, double* v, double floor_val,
                                   float* num, int dim) {
  cudaD_vec_apply_ceiling(Gr, Bl, v, floor_val, num, dim);
}
inline void cuda_vec_apply_exp(int Gr, int Bl, double* v, int dim) {
  cudaD_vec_apply_exp(Gr, Bl, v, dim);
}
inline void cuda_vec_apply_log(int Gr, int Bl, double* v, double* flag,
                               int dim) {
  cudaD_vec_apply_log(Gr, Bl, v, flag, dim);
}
inline void cuda_invert_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim d) {
  cudaD_invert_elements(Gr, Bl, data, d);
}
// B_trans nonzero if B transposed.
inline void cuda_add_mat_blockmat(dim3 Gr, dim3 Bl, double *data, MatrixDim d,
                                  const double *Adata, int A_num_rows,
                                  int A_num_cols, int A_row_stride,
                                  int A_col_stride,
                                  const CuBlockMatrixData *B_cu_data,
                                  int B_num_blocks, double alpha, double beta,
                                  int B_trans) {
  cudaD_add_mat_blockmat(Gr, Bl, data, d, Adata, A_num_rows, A_num_cols,
                         A_row_stride, A_col_stride, B_cu_data, B_num_blocks,
                         alpha, beta, B_trans);
}
inline void cuda_block_add_mat_mat(dim3 Gr, dim3 Bl,
                                   CuBlockMatrixData *B_cu_data, int num_blocks,
                                   const double *C_data, int C_num_cols,
                                   int C_row_stride, int C_col_stride,
                                   const double *D_data, int D_row_stride,
                                   int D_col_stride, double alpha,
                                   double beta) {
  cudaD_block_add_mat_mat(Gr, Bl, B_cu_data, num_blocks, C_data, C_num_cols,
                          C_row_stride, C_col_stride, D_data, D_row_stride,
                          D_col_stride, alpha, beta);
}

/*
 * cu::
 */
inline void cuda_soft_hinge(dim3 Gr, dim3 Bl, double *y, const double *x,
                            MatrixDim d, int src_stride) {
  cudaD_soft_hinge(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_group_pnorm(dim3 Gr, dim3 Bl, double *y, const double *x,
                             MatrixDim d, int src_stride, int group_size,
                             double power) {
  cudaD_group_pnorm(Gr, Bl, y, x, d, src_stride, group_size, power);
}
inline void cuda_group_spec_pnorm(dim3 Gr, dim3 Bl, double *y, const double *x,
                                  MatrixDim d, int src_stride, int group_size,
                                  double power) {
  cudaD_group_spec_pnorm(Gr, Bl, y, x, d, src_stride, group_size, power);
}
inline void cuda_group_max(dim3 Gr, dim3 Bl, double *y, const double *x,
                           MatrixDim d, int src_stride, int group_size) {
  cudaD_group_max(Gr, Bl, y, x, d, src_stride, group_size);
}
inline void cuda_sigmoid(dim3 Gr, dim3 Bl, double *y, const double *x,
                         MatrixDim d, int src_stride) {
  cudaD_sigmoid(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, double *eout, const double *e,
                              const double *y, MatrixDim d, int e_stride,
                              int y_stride) {
  cudaD_diff_sigmoid(Gr, Bl, eout, e, y, d, e_stride, y_stride);
}
inline void cuda_tanh(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d,
                      int src_stride) {
  cudaD_tanh(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, double *eout, const double *e,
                           const double *y, MatrixDim d, int e_stride,
                           int y_stride) {
  cudaD_diff_tanh(Gr, Bl, eout, e, y, d, e_stride, y_stride);
}
inline void cuda_parametric_relu(dim3 Gr, dim3 Bl, double *y, const double *x,
                                 MatrixDim d, int src_stride,
                                 const double *a, const double *b) {
  cudaD_parametric_relu(Gr,Bl,y,x,d,src_stride,a,b);
}
inline void cuda_diff_parametric_relu(dim3 Gr, dim3 Bl, double *eout,
                                      const double *e, const double *y,
                                      MatrixDim d, int e_stride, int y_stride,
                                      const double *a, const double *b) {
  cudaD_diff_parametric_relu(Gr,Bl,eout,e,y,d,e_stride,y_stride,a,b);
}
inline void cuda_heaviside(dim3 Gr, dim3 Bl, double *y, const double *x,
                           MatrixDim d, int src_stride) {
  cudaD_heaviside(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_softmax_reduce(size_t Gr, size_t Bl, double *y,
                                const double *x, MatrixDim d, int src_stride) {
  cudaD_softmax_reduce(Gr, Bl, y, x, d, src_stride);
}
inline void cuda_log_softmax_reduce(size_t Gr, size_t Bl, double *y,
                                    const double *x, MatrixDim y_dim,
                                    int x_stride) {
  cudaD_log_softmax_reduce(Gr, Bl, y, x, y_dim, x_stride);
}
inline void cuda_normalize_per_row(size_t Gr, size_t Bl, double *y,
                                   int y_stride, const double *x, MatrixDim x_d,
                                   double target_rms, bool add_log_stddev) {
  cudaD_normalize_per_row(Gr, Bl, y, y_stride, x, x_d, target_rms,
                          add_log_stddev);
}

inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, double *wei, double *grad,
                               double l1, double lr, MatrixDim d,
                               int stride_grad) {
  cudaD_regularize_l1(Gr, Bl, wei, grad, l1, lr, d, stride_grad);
}
inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const double *mat,
                                 double *vec_val, int32_cuda *vec_id,
                                 MatrixDim d) {
  cudaD_find_row_max_id(Gr, Bl, mat, vec_val, vec_id, d);
}
inline void cuda_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt,
                           double *mat_net_out, double *vec_log_post,
                           MatrixDim d) {
  cudaD_diff_xent(Gr, Bl, vec_tgt, mat_net_out, vec_log_post, d);
}
inline void cuda_diff_softmax(dim3 Gr, dim3 Bl, double* x, const MatrixDim dim,
                              const double* value, const int value_stride,
                              const double* diff, const int diff_stride) {
  cudaD_diff_softmax(Gr, Bl, x, dim, value, value_stride, diff, diff_stride);
}
inline void cuda_diff_log_softmax(dim3 Gr, dim3 Bl,
                                  const MatrixDim in_deriv_dim,
                                  const double* out_value,
                                  const int out_value_stride,
                                  const double* out_deriv,
                                  const int out_deriv_stride,
                                  double* in_deriv) {
  cudaD_diff_log_softmax(Gr, Bl, in_deriv_dim, out_value, out_value_stride,
                         out_deriv, out_deriv_stride, in_deriv);
}
inline void cuda_copy_rows_from_vec(dim3 Gr, dim3 Bl, double *mat_out,
                                    MatrixDim d_out, const double *v_in) {
  cudaD_copy_rows_from_vec(Gr, Bl, mat_out, d_out, v_in);
}

inline void cuda_randomize(dim3 Gr, dim3 Bl, double *y, const double *x,
                           const int32_cuda *copy_from, MatrixDim d_out,
                           MatrixDim d_in) {
  cudaD_randomize(Gr, Bl, y, x, copy_from, d_out, d_in);
}
inline void cuda_splice(dim3 Gr, dim3 Bl, double *y, const double *x,
                        const int32_cuda *off, MatrixDim d_out,
                        MatrixDim d_in) {
  cudaD_splice(Gr, Bl, y, x, off, d_out, d_in);
}
inline void cuda_one(int Gr, int Bl, double* x, int dim) {
  cudaD_one(Gr, Bl, x, dim);
}
inline void cuda_copy(dim3 Gr, dim3 Bl, double *y, const double *x,
                      const int32_cuda *copy_from, MatrixDim d_out,
                      MatrixDim d_in) {
  cudaD_copy(Gr, Bl, y, x, copy_from, d_out, d_in);
}
inline void cuda_copy_from_sp(dim3 Gr, dim3 Bl, const double* x, double* y,
                              MatrixDim d_out) {
  cudaD_copy_from_sp(Gr, Bl, x, y, d_out);
}
inline void cuda_take_lower(dim3 Gr, dim3 Bl, const double* x, double* y,
                            MatrixDim d_in) {
  cudaD_take_lower(Gr, Bl, x, y, d_in);
}
inline void cuda_take_upper(dim3 Gr, dim3 Bl, const double* x, double* y,
                            MatrixDim d_in) {
  cudaD_take_upper(Gr, Bl, x, y, d_in);
}
inline void cuda_take_mean(dim3 Gr, dim3 Bl, const double* x, double* y,
                           MatrixDim d_in) {
  cudaD_take_mean(Gr, Bl, x, y, d_in);
}
inline void cuda_matrix_add_elements(dim3 Gr, dim3 Bl, double *data,
                                     MatrixDim dim, double alpha,
                                     MatrixElement<double>* x,
                                     int num_elements) {
  cudaD_matrix_add_elements(Gr, Bl, data, dim, alpha, x, num_elements);
}
inline void cuda_matrix_add_indexed_values(dim3 Gr, dim3 Bl, MatrixDim dim,
                                           double alpha,
                                           const Int32Pair* indices,
                                           const double* x, int s,
                                           double* data) {
  cudaD_matrix_add_indexed_values(Gr, Bl, dim, alpha, indices, x, s, data);
}
inline void cuda_comp_obj_deriv(dim3 Gr, dim3 Bl, MatrixElement<double>* x,
                                int32 size, const double* z, MatrixDim d,
                                double* z2, MatrixDim d2, double* t) {
  cudaD_comp_obj_deriv(Gr, Bl, x, size, z, d, z2, d2, t);
}
inline void cuda_sum_column_ranges(dim3 Gr, dim3 Bl, double *data,
                                   MatrixDim dim, const double *src_data,
                                   MatrixDim src_dim,
                                   const Int32Pair *indices) {
  cudaD_sum_column_ranges(Gr, Bl, data, dim, src_data, src_dim, indices);
}
inline void cuda_add_row_ranges(dim3 Gr, dim3 Bl, double *data, MatrixDim dim,
                                const double *src_data, MatrixDim src_dim,
                                const Int32Pair *indexes) {
  cudaD_add_row_ranges(Gr, Bl, data, dim, src_data, src_dim, indexes);
}
inline void cuda_matrix_lookup(dim3 Gr, dim3 Bl, const double *data,
                               MatrixDim dim, const Int32Pair *indices,
                               int indices_size, double *output) {
  cudaD_matrix_lookup(Gr, Bl, data, dim, indices, indices_size, output);
}

inline void cuda_equal_element_mask(dim3 Gr, dim3 Bl, const double *mat1,
                                    const double *mat2, double *mask,
                                    MatrixDim mat1_dim, int mat2_stride,
                                    int mask_stride) {
  cudaD_equal_element_mask(Gr, Bl, mat1, mat2, mask, mat1_dim, mat2_stride,
                           mask_stride);
}

// Also include some template-friendly wrappers of cublas functions:
inline cublasStatus_t cuda_axpy(cublasHandle_t handle, int n, float alpha,
                                const float *x, int incx, float *y, int incy) {
  return cublasSaxpy_v2(handle, n, &alpha, x, incx, y, incy);
}
inline cublasStatus_t cuda_axpy(cublasHandle_t handle, int n, double alpha,
                                const double *x, int incx, double *y,
                                int incy) {
  return cublasDaxpy_v2(handle, n, &alpha, x, incx, y, incy);
}
inline cublasStatus_t cuda_scal(cublasHandle_t handle, int n, float alpha,
                                float *x, int incx) {
  return cublasSscal_v2(handle, n, &alpha, x, incx);
}
inline cublasStatus_t cuda_scal(cublasHandle_t handle, int n, double alpha,
                                double *x, int incx) {
  return cublasDscal_v2(handle, n, &alpha, x, incx);
}

inline void cuda_lstm_nonlinearity(dim3 Gr, dim3 Bl, const double* in,
                                   const int in_stride, const double* params,
                                   const int params_stride,
                                   const int out_stride, const int cell_dim,
                                   const int num_rows, double* out) {
  cudaD_lstm_nonlinearity(Gr, Bl, in, in_stride, params, params_stride,
                          out_stride, cell_dim, num_rows, out);
}
inline void cuda_lstm_nonlinearity(dim3 Gr, dim3 Bl, const float* in,
                                   const int in_stride, const float* params,
                                   const int params_stride,
                                   const int out_stride, const int cell_dim,
                                   const int num_rows, float* out) {
  cudaF_lstm_nonlinearity(Gr, Bl, in, in_stride, params, params_stride,
                          out_stride, cell_dim, num_rows, out);
}
inline void cuda_diff_lstm_nonlinearity(dim3 Gr, dim3 Bl, const int cell_dim,
                                        const int num_rows, const double* input,
                                        const int input_stride,
                                        const double* params,
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
  cudaD_diff_lstm_nonlinearity(Gr, Bl, cell_dim, num_rows, input, input_stride,
                               params, params_stride, output_deriv,
                               output_deriv_stride, deriv_sum_in,
                               deriv_sum_in_stride, self_repair_config, count,
                               input_deriv, input_deriv_stride, params_deriv,
                               params_deriv_stride, value_sum_out,
                               value_sum_out_stride, deriv_sum_out,
                               deriv_sum_out_stride, self_repair_sum_out,
                               self_repair_sum_out_stride);
}
inline void cuda_diff_lstm_nonlinearity(dim3 Gr, dim3 Bl, const int cell_dim,
                                        const int num_rows, const float* input,
                                        const int input_stride,
                                        const float* params,
                                        const int params_stride,
                                        const float* output_deriv,
                                        const int output_deriv_stride,
                                        const double* deriv_sum_in,
                                        const int deriv_sum_in_stride,
                                        const float* self_repair_config,
                                        double count, float* input_deriv,
                                        const int input_deriv_stride,
                                        float* params_deriv,
                                        const int params_deriv_stride,
                                        double* value_sum_out,
                                        const int value_sum_out_stride,
                                        double* deriv_sum_out,
                                        const int deriv_sum_out_stride,
                                        float* self_repair_sum_out,
                                        const int self_repair_sum_out_stride) {
  cudaF_diff_lstm_nonlinearity(Gr, Bl, cell_dim, num_rows, input, input_stride,
                               params, params_stride, output_deriv,
                               output_deriv_stride, deriv_sum_in,
                               deriv_sum_in_stride, self_repair_config, count,
                               input_deriv, input_deriv_stride, params_deriv,
                               params_deriv_stride, value_sum_out,
                               value_sum_out_stride, deriv_sum_out,
                               deriv_sum_out_stride, self_repair_sum_out,
                               self_repair_sum_out_stride);
}

inline void cuda_copy_cols_from_vec(dim3 Gr, dim3 Bl, double *mat_out,
                                    MatrixDim d_out, const double *v_in) {
  cudaD_copy_cols_from_vec(Gr, Bl, mat_out, d_out, v_in);
}
inline void cuda_copy_cols_from_vec(dim3 Gr, dim3 Bl, float *mat_out,
                                    MatrixDim d_out, const float *v_in) {
  cudaF_copy_cols_from_vec(Gr, Bl, mat_out, d_out, v_in);
}

} // namespace kaldi

#endif // HAVE_CUDA

#endif
