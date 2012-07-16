// cudamatrix/cu-kernels-ansi.h

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




#ifndef KALDI_CUDAMATRIX_CU_KERNELS_ANSI_H_
#define KALDI_CUDAMATRIX_CU_KERNELS_ANSI_H_

#include "cudamatrix/cu-matrixdim.h"

#if HAVE_CUDA==1

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
void cudaF_set_const(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d);
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
void cudaF_add_row_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d);
void cudaF_add_col_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d);
void cudaF_invert_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim d);

/*
 * cu::
 */
void cudaF_softmax(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d);
void cudaF_softmax_part(dim3 Gr, dim3 Bl, const float *X, const int32_cuda *vec_ids, float* Y, MatrixDim d);
void cudaF_sigmoid(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d);
void cudaF_diff_sigmoid(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d);

void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float *wei, float *grad, float l1, float lr, MatrixDim d);
void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float *mat, float *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d);
void cudaF_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, float *mat_net_out, float *vec_log_post, MatrixDim d);

void cudaF_randomize(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaF_expand(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in);
void cudaF_rearrange(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);


/*********************************************************
 * double CUDA kernel calls
 */

/*
 * CuMatrix 
 */
void cudaD_set_const(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d);
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
void cudaD_add_row_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d);
void cudaD_add_col_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d);
void cudaD_invert_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim d);

/*
 * cu::
 */
void cudaD_softmax(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d);
void cudaD_softmax_part(dim3 Gr, dim3 Bl, const double *X, const int32_cuda *vec_ids, double* Y, MatrixDim d);
void cudaD_sigmoid(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d);
void cudaD_diff_sigmoid(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d);

void cudaD_regularize_l1(dim3 Gr, dim3 Bl, double *wei, double *grad, double l1, double lr, MatrixDim d);
void cudaD_find_row_max_id(dim3 Gr, dim3 Bl, const double *mat, double *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d);
void cudaD_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, double *mat_net_out, double *vec_log_post, MatrixDim d);

void cudaD_randomize(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaD_expand(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in);
void cudaD_rearrange(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);


} // extern "C" 

#endif // HAVE_CUDA

#endif
