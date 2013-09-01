// cudamatrix/cu-kernels-ansi.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)

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




#ifndef KALDI_CUDAMATRIX_CU_KERNELS_ANSI_H_
#define KALDI_CUDAMATRIX_CU_KERNELS_ANSI_H_

#include "cudamatrix/cu-matrixdim.h"

#if HAVE_CUDA == 1

extern "C" {

/*********************************************************
 * int32 CUDA kernel calls (no template wrapper)
 */
void cudaI32_set_const(dim3 Gr, dim3 Bl, int32_cuda *mat, int32_cuda value, MatrixDim d);



/*********************************************************
 * float CUDA kernel calls
 */

/*
 * CuMatrix 
 */
void cudaF_ammdm_elements(dim3 Gr, dim3 Bl, float alpha, float* mat, const float* A, const float* B, const float* C, float beta, MatrixDim d);
void cudaF_copy_from_tp_trans(int Gr, int Bl, float* A, const float* B, MatrixDim dmat);
void cudaFD_copy_from_tp_trans(int Gr, int Bl, float* A, const double* B, MatrixDim dmat);
void cudaF_copy_from_tp(int Gr, int Bl, float* A, const float* B, MatrixDim dmat);
void cudaFD_copy_from_tp(int Gr, int Bl, float* A, const double* B, MatrixDim dmat);
void cudaF_trace_sp_sp_fd(int Gr, int Bl, const float* A, const float* B, float* value, int dim);
void cudaF_trace_sp_sp_df(int Gr, int Bl, const double* A, const float* B, double* value, int dim);
void cudaF_copy_col_from_vec(int Gr, int Bl, float* mat, const float* v, int col, MatrixDim d);
void cudaF_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);
void cudaF_sum(dim3 Gr, dim3 Bl, float* mat, float* value, MatrixDim d);
void cudaF_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim d);
void cudaF_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);  
void cudaF_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim d);
void cudaF_set_diag(int Gr, int Bl, float* mat, float value, MatrixDim d);
void cudaF_set_diag_packed(int Gr, int Bl, float* mat, float value, int dim);
void cudaF_add_diag_packed(int Gr, int Bl, float* mat, float value, int dim);
void cudaF_set_const(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d);
void cudaF_set_zero_above_diag(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);
void cudaF_add(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d);
void cudaF_add_vec2(dim3 Gr, dim3 Bl, float* mat, const float* vec, const float alpha, int dim);
void cudaF_scale_diag(int Gr, int Bl, float* mat, float value, int dim);
void cudaF_scale(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d);
void cudaF_apply_log(dim3 Gr, dim3 Bl, float *mat, MatrixDim d);
void cudaF_mul_elements(dim3 Gr, dim3 Bl, float *mat, const float *A, MatrixDim d);
void cudaF_mul_cols_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d);
void cudaF_mul_rows_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d);
void cudaF_div_rows_vec(dim3 Gr, dim3 Bl, float *mat, const float *vec_div, MatrixDim d);
void cudaF_add_mat(dim3 Gr, dim3 Bl, float alpha, const float *A, float beta, float *dst, MatrixDim d);
void cudaF_add_vec_to_cols(dim3 Gr, dim3 Bl, float alpha, const float *col, float beta, float *dst, MatrixDim d);
void cudaF_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float *row, float beta, float *dst, MatrixDim d);
 
/*
 * CuVector
 */
void cudaF_set_bias_params(int Gr, int Bl, float* v, const float* a, float param_1, float param_2, float param_3, int* flag, int dim);
void cudaF_copy_from_vec_df(int Gr, int Bl, double* v_out, const float* v_in, int dim);
void cudaF_copy_from_vec_fd(int Gr, int Bl, float* v_out, const float* v_in, int dim);
void cudaF_vec_mul_elements(int Gr, int Bl, float* v, const float* a, int dim);
void cudaF_vec_soft_max(int Gr, int Bl, float* v, int dim);
void cudaF_min(int Gr, int Bl, const float* v, float* value, int dim);
void cudaF_trace_mat_mat_trans(int Gr, int Bl, const float* A, const float* B, MatrixDim dA, MatrixDim dB, float* value);
void cudaF_trace_mat_mat(int Gr, int Bl, const float* A, const float* B, MatrixDim dA, MatrixDim dB, float* value);
void cudaF_add_diag_mat_trans(int Gr, int Bl, float alpha, float* v, const float* mat, float beta, MatrixDim dmat, int dim);
void cudaF_add_diag_mat(int Gr, int Bl, float alpha, float* v, const float* mat, float beta, MatrixDim dmat, int dim);
void cudaF_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim);
void cudaF_copy_col_from_mat(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim);
void cudaF_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const float* mat, MatrixDim dmat, int dim);
void cudaF_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim);
void cudaF_vec_sum(int Gr, int Bl, float* v, float* value, int dim);
void cudaF_vec_apply_floor(int Gr, int Bl, float* v, float floor_val, int* num, int dim);
void cudaF_vec_apply_exp(int Gr, int Bl, float* v, int dim);
void cudaF_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim);
void cudaF_trace(int Gr, int Bl, float* mat, float* value, int dim);
void cudaF_add_row_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d);
void cudaF_add_col_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d);
void cudaF_invert_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim d);

/*
 * cu::
 */
void cudaF_softmax(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d);
void cudaF_softmax_part(dim3 Gr, dim3 Bl, const float *X, const int32_cuda *vec_ids, float* Y, MatrixDim d);
void cudaF_soft_hinge(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d);
void cudaF_sigmoid(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d);
void cudaF_diff_sigmoid(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d);
void cudaF_tanh(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d);
void cudaF_diff_tanh(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d);

void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float *wei, float *grad, float l1, float lr, MatrixDim d);
void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float *mat, float *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d);
void cudaF_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, float *mat_net_out, float *vec_log_post, MatrixDim d);

void cudaF_randomize(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaF_splice(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in);
void cudaF_one(int Gr, int Bl, float* x, int dim);
void cudaF_copy(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaF_copy_diag(int Gr, int Bl, float* y, const float* x, int dim);
void cudaF_copy_from_sp(int Gr, int Bl, const float* x, float* y, int d_in, MatrixDim d_out);
void cudaF_take_lower(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out);
void cudaF_take_upper(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out);
void cudaF_take_mean(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out);
/*********************************************************
 * double CUDA kernel calls
 */

/*
 * CuMatrix 
 */
void cudaD_ammdm_elements(dim3 Gr, dim3 Bl, double alpha, double* mat, const double* A, const double* B, const double* C, double beta, MatrixDim d);
void cudaD_copy_from_tp_trans(int Gr, int Bl, double* A, const double* B, MatrixDim dmat);
void cudaDF_copy_from_tp_trans(int Gr, int Bl, double* A, const float* B, MatrixDim dmat);
void cudaD_copy_from_tp(int Gr, int Bl, double* A, const double* B, MatrixDim dmat);
void cudaDF_copy_from_tp(int Gr, int Bl, double* A, const float* B, MatrixDim dmat);
void cudaD_trace_sp_sp_fd(int Gr, int Bl, const float* A, const double* B, float* value, int dim);
void cudaD_trace_sp_sp_df(int Gr, int Bl, const double* A, const double* B, double* value, int dim);
void cudaD_copy_col_from_vec(int Gr, int Bl, double* mat, const double* v, int col, MatrixDim d);
void cudaD_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d);
void cudaD_sum(dim3 Gr, dim3 Bl, double* mat, double* value, MatrixDim d);
void cudaD_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim d);
void cudaD_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim d);  
void cudaD_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim d);
void cudaD_set_diag(int Gr, int Bl, double* mat, double value, MatrixDim d);
void cudaD_set_diag_packed(int Gr, int Bl, double* mat, double value, int dim);
void cudaD_add_diag_packed(int Gr, int Bl, double* mat, double value, int dim);
void cudaD_set_const(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d);
void cudaD_set_zero_above_diag(dim3 Gr, dim3 Bl, double* mat, MatrixDim d);
void cudaD_add(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d);
void cudaD_add_vec2(dim3 Gr, dim3 Bl, double *mat, const double *vec, const double alpha, int dim);
void cudaD_scale_diag(int Gr, int Bl, double* mat, double value, int dim);
void cudaD_scale(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d);
void cudaD_apply_log(dim3 Gr, dim3 Bl, double *mat, MatrixDim d);
void cudaD_mul_elements(dim3 Gr, dim3 Bl, double *mat, const double *A, MatrixDim d);
void cudaD_mul_cols_vec(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d);
void cudaD_mul_rows_vec(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d);
void cudaD_div_rows_vec(dim3 Gr, dim3 Bl, double *mat, const double *vec_div, MatrixDim d);
void cudaD_add_mat(dim3 Gr, dim3 Bl, double alpha, const double *A, double beta, double *dst, MatrixDim d);
void cudaD_add_vec_to_cols(dim3 Gr, dim3 Bl, double alpha, const double *col, double beta, double *dst, MatrixDim d);
void cudaD_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double *row, double beta, double *dst, MatrixDim d);
 
/*
 * CuVector
 */
void cudaD_set_bias_params(int Gr, int Bl, double* v, const double* a, double param_1, double param_2, double param_3, int* flag, int dim);
void cudaD_copy_from_vec_df(int Gr, int Bl, double* v_out, const double* v_in, int dim);
void cudaD_copy_from_vec_fd(int Gr, int Bl, float* v_out, const double* v_in, int dim);
void cudaD_vec_mul_elements(int Gr, int Bl, double* v, const double* a, int dim);
void cudaD_vec_soft_max(int Gr, int Bl, double* v, int dim);
void cudaD_min(int Gr, int Bl, const double* v, double* value, int dim);
void cudaD_trace_mat_mat_trans(int Gr, int Bl, const double* A, const double* B, MatrixDim dA, MatrixDim dB, double* value);
void cudaD_trace_mat_mat(int Gr, int Bl, const double* A, const double* B, MatrixDim dA, MatrixDim dB, double* value);
void cudaD_add_diag_mat_trans(int Gr, int Bl, double alpha, double* v, const double* mat, double beta, MatrixDim dmat, int dim);
void cudaD_add_diag_mat(int Gr, int Bl, double alpha, double* v, const double* mat, double beta, MatrixDim dmat, int dim);
void cudaD_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim);
void cudaD_copy_col_from_mat(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim);
void cudaD_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim);
void cudaD_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const double* mat, MatrixDim dmat, int dim);
void cudaD_vec_sum(int Gr, int Bl, double* v, double* value, int dim);
void cudaD_vec_apply_floor(int Gr, int Bl, double* v, double floor_val, int* num, int dim);
void cudaD_vec_apply_exp(int Gr, int Bl, double* v, int dim);
void cudaD_vec_apply_log(int Gr, int Bl, double* v, double* flag, int dim);
void cudaD_trace(int Gr, int Bl, double* mat, double* value, int dim);
void cudaD_add_row_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d);
void cudaD_add_col_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d);
void cudaD_invert_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim d);

/*
 * cu::
 */
void cudaD_softmax(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d);
void cudaD_softmax_part(dim3 Gr, dim3 Bl, const double *X, const int32_cuda *vec_ids, double* Y, MatrixDim d);
void cudaD_soft_hinge(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d);
void cudaD_sigmoid(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d);
void cudaD_diff_sigmoid(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d);
void cudaD_tanh(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d);
void cudaD_diff_tanh(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d);

void cudaD_regularize_l1(dim3 Gr, dim3 Bl, double *wei, double *grad, double l1, double lr, MatrixDim d);
void cudaD_find_row_max_id(dim3 Gr, dim3 Bl, const double *mat, double *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d);
void cudaD_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, double *mat_net_out, double *vec_log_post, MatrixDim d);

void cudaD_randomize(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaD_splice(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in);
void cudaD_one(int Gr, int Bl, double* x, int dim);
void cudaD_copy(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaD_copy_diag(int Gr, int Bl, double* y, const double* x, int dim);
void cudaD_copy_from_sp(int Gr, int Bl, const double* x, double* y, int d_in, MatrixDim d_out);
void cudaD_take_lower(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out);
void cudaD_take_upper(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out);
void cudaD_take_mean(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out);


// some mostly mixed-type kernels.
void cuda_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_ff(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_fd(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_dd(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_ff_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_dd_trans(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
  
} // extern "C" 

#endif // HAVE_CUDA


#endif
