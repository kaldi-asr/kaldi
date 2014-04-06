// cudamatrix/cu-kernels.h

// Copyright 2009-2012  Karel Vesely
//                2013  Ehsan Variani
//                2014  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang	
//                2013  Johns Hopkins University (author: Guoguo Chen)

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

inline void cuda_copy_upp_low(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) { cudaF_copy_upp_low(Gr, Bl, A, dimA); }
inline void cuda_copy_low_upp(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA) { cudaF_copy_low_upp(Gr, Bl, A, dimA); }
inline void cuda_add_diag_vec_mat(dim3 Gr, dim3 Bl, float alpha, float *mat, MatrixDim mat_dim,
                                  const float *vec, const float *mat2, int mat2_row_stride,
                                  int mat2_col_stride, float beta) {
  cudaF_add_diag_vec_mat(Gr, Bl, alpha, mat, mat_dim, vec, mat2,
                         mat2_row_stride, mat2_col_stride, beta);
}
inline void cuda_copy_from_tp_trans(int Gr, int Bl, float* A, const float* B, MatrixDim dmat) { cudaF_copy_from_tp_trans(Gr,Bl,A,B,dmat); }
inline void cuda_copy_from_tp_trans(int Gr, int Bl, float* A, const double* B, MatrixDim dmat) { cudaFD_copy_from_tp_trans(Gr,Bl,A,B,dmat); }
inline void cuda_copy_from_tp(int Gr, int Bl, float* A, const float* B, MatrixDim dmat) { cudaF_copy_from_tp(Gr,Bl,A,B,dmat); }
inline void cuda_copy_from_tp(int Gr, int Bl, float* A, const double* B, MatrixDim dmat) { cudaFD_copy_from_tp(Gr,Bl,A,B,dmat); }

inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_fd(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_ff(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_dd(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_df(Gr, Bl, mat_out, mat_in, d_out, d_in);
}

inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_fd_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_ff_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_dd_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_df_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}

inline void cuda_copy_col_from_vec(int Gr, int Bl, float* mat, const float* v, int col, MatrixDim d) { cudaF_copy_col_from_vec(Gr,Bl,mat,v,col,d); }
inline void cuda_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) { cudaF_apply_exp(Gr,Bl,mat,d); }
inline void cuda_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim dim) { cudaF_apply_pow(Gr,Bl,mat,power,dim); }
inline void cuda_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim dim) { cudaF_apply_heaviside(Gr,Bl,mat,dim); }
inline void cuda_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim dim) { cudaF_apply_floor(Gr,Bl,mat,floor_val,dim); }
inline void cuda_apply_ceiling(dim3 Gr, dim3 Bl, float* mat, float ceiling_val, MatrixDim dim) { cudaF_apply_ceiling(Gr,Bl,mat,ceiling_val,dim); }
inline void cuda_copy_cols(dim3 Gr, dim3 Bl, float* dst, const float* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  cudaF_copy_cols(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_copy_rows(dim3 Gr, dim3 Bl, float* dst, const float* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  cudaF_copy_rows(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_trace(int Gr, int Bl, float* mat, float* value, int dim) { cudaF_trace(Gr,Bl,mat,value,dim); }
inline void cuda_set_diag(int Gr, int Bl, float* mat, float value, MatrixDim d) { cudaF_set_diag(Gr,Bl,mat,value,d); }
inline void cuda_set_diag_packed(int Gr, int Bl, float* mat, float value, int dim) { cudaF_set_diag_packed(Gr,Bl,mat,value,dim); }
inline void cuda_add_diag_packed(int Gr, int Bl, float* mat, float value, int dim) { cudaF_add_diag_packed(Gr,Bl,mat,value,dim); }
inline void cuda_set_const(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_set_const(Gr,Bl,mat,value,d); }
inline void cuda_set_zero_above_diag(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) { cudaF_set_zero_above_diag(Gr,Bl,mat,d); }
inline void cuda_add(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_add(Gr,Bl,mat,value,d); }
inline void cuda_add_vec2(dim3 Gr, dim3 Bl, float *mat, const float *vec, const float alpha, int dim) { cudaF_add_vec2(Gr,Bl,mat,vec,alpha,dim); }
inline void cuda_scale_diag(int Gr, int Bl, float* mat, float value, int dim) { cudaF_scale_diag(Gr,Bl,mat,value,dim); }
inline void cuda_scale(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_scale(Gr,Bl,mat,value,d); }
inline void cuda_apply_log(dim3 Gr, dim3 Bl, float *mat, MatrixDim d) { cudaF_apply_log(Gr,Bl,mat,d); }
inline void cuda_mul_elements(dim3 Gr, dim3 Bl, float *mat, const float *A, MatrixDim dst_d, int src_stride) {
  cudaF_mul_elements(Gr,Bl,mat,A,dst_d,src_stride);
}
inline void cuda_max(dim3 Gr, dim3 Bl, float *mat, const float *A, MatrixDim dst_d, int src_stride) {
  cudaF_max(Gr,Bl,mat,A,dst_d,src_stride);
}
inline void cuda_mul_cols_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d) { cudaF_mul_cols_vec(Gr,Bl,mat,scale,d); }
inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d) { cudaF_mul_rows_vec(Gr,Bl,mat,scale,d); }
inline void cuda_mul_rows_group_mat(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride, int group_size) { cudaF_mul_rows_group_mat(Gr, Bl, y, x, d, src_stride, group_size); }
inline void cuda_calc_pnorm_deriv(dim3 Gr, dim3 Bl, float *y, const float *x1, const float *x2,  MatrixDim d, int src_stride, int group_size, float power) {cudaF_calc_pnorm_deriv(Gr, Bl, y, x1, x2, d, src_stride, group_size, power); }
inline void cuda_add_mat(dim3 Gr, dim3 Bl, float alpha, const float *src, float *dst, MatrixDim d, int src_stride, int A_trans) { cudaF_add_mat(Gr,Bl,alpha,src,dst,d,src_stride, A_trans); }
inline void cuda_add_mat_mat_div_mat(dim3 Gr, dim3 Bl, const float *A, const float *B, const float *C, float *dst, MatrixDim d) { cudaF_add_mat_mat_div_mat(Gr,Bl,A,B,C,dst,d); }
inline void cuda_add_vec_to_cols(dim3 Gr, dim3 Bl, float alpha, const float *col, float beta, float *dst, MatrixDim d) { cudaF_add_vec_to_cols(Gr,Bl,alpha,col,beta,dst,d); }
inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float *row, float beta, float *dst, MatrixDim d) { cudaF_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }
inline void cuda_transpose_matrix(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) { cudaF_transpose_matrix(Gr, Bl, mat, d); }
inline void cuda_sy_add_tr2(dim3 Gr, dim3 Bl, float alpha, float beta, const float* T, MatrixDim tdim,
                            float *S, MatrixDim sdim) {
  cudaF_sy_add_tr2(Gr, Bl, alpha, beta, T, tdim, S, sdim);
}

 
/*
 * CuVector
 */
inline void cuda_replace_value(int Gr, int Bl, float *v, int dim, float orig, float changed) {cudaF_replace_value(Gr, Bl, v, dim, orig, changed); }
inline void cuda_div_rows_vec(dim3 Gr, dim3 Bl, float *mat, const float *vec_div, MatrixDim d) { cudaF_div_rows_vec(Gr,Bl,mat,vec_div,d); }
inline void cuda_set_bias_params(int Gr, int Bl, float* v, const float* a, float param_1, float param_2, float param_3, int* flag, int dim) { cudaF_set_bias_params(Gr,Bl,v,a,param_1,param_2,param_3,flag,dim); }
inline void cuda_copy_from_vec_df(int Gr, int Bl, double* v_out, const float* v_in, int dim) { cudaF_copy_from_vec_df(Gr,Bl,v_out,v_in,dim); }
inline void cuda_copy_from_vec_fd(int Gr, int Bl, float* v_out, const float* v_in, int dim) { cudaF_copy_from_vec_fd(Gr,Bl,v_out,v_in,dim); }
inline void cuda_vec_mul_elements(int Gr, int Bl, float* v, const float* a, int dim) { cudaF_vec_mul_elements(Gr,Bl,v,a,dim); }
inline void cuda_vec_soft_max(int Gr, int Bl, float* v, int dim) { cudaF_vec_soft_max(Gr,Bl,v,dim); }
inline void cuda_vec_min(const float* v, float* value, int dim) { cudaF_vec_min(v,value,dim); }
inline void cuda_vec_max(const float* v, float* value, int dim) { cudaF_vec_max(v,value,dim); }
inline void cuda_trace_mat_mat_trans(const float* A, const float* B, MatrixDim dA, int B_stride, float* value) { cudaF_trace_mat_mat_trans(A,B,dA,B_stride,value); }
inline void cuda_trace_mat_mat(const float* A, const float* B, MatrixDim dA, int B_stride, float* value) { cudaF_trace_mat_mat(A,B,dA,B_stride,value); }
inline void cuda_add_diag_mat_trans(int Gr, int Bl, float alpha, float* v, const float* mat, float beta, MatrixDim dmat, int dim) { cudaF_add_diag_mat_trans(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
inline void cuda_add_diag_mat_mat(int Gr, int Bl, float alpha, float* v, int v_dim, const float* M, 
                                  int M_cols, int M_row_stride, int M_col_stride, const float *N, int N_row_stride, 
                                  int N_col_stride, int threads_per_element, float beta) {
  cudaF_add_diag_mat_mat(Gr, Bl, alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride, N, N_row_stride,
                         N_col_stride, threads_per_element, beta);
}
inline void cuda_add_diag_mat(int Gr, int Bl, float alpha, float* v, const float* mat, float beta, MatrixDim dmat, int dim) { cudaF_add_diag_mat(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
inline void cuda_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim) { cudaF_add_vec_vec(Gr,Bl,alpha,v,x,y,beta,dim); }
inline void cuda_copy_col_from_mat(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim) { cudaF_copy_col_from_mat(Gr,Bl,v,col,mat,dmat,dim); }
inline void cuda_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const float* mat, MatrixDim dmat, int dim) { cudaF_copy_col_from_mat_df(Gr,Bl,v,col,mat,dmat,dim); }
inline void cuda_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim) { cudaF_copy_col_from_mat_fd(Gr,Bl,v,col,mat,dmat,dim); }
inline void cuda_vec_sum(int Gr, int Bl, float* v, float* value, int dim, int inc) { cudaF_vec_sum(Gr,Bl,v,value,dim,inc); }
inline void cuda_pvec_sum(int Gr, int Bl, float* vec, float* pvec_sum, int dim, int size) { cudaF_pvec_sum(Gr, Bl, vec, pvec_sum, dim, size); }
inline void cuda_vec_copy_diag_from_packed(int Gr, int Bl, float *dst, const float *src, int dim) { cudaF_vec_copy_diag_from_packed(Gr,Bl,dst,src,dim); }
inline void cuda_vec_apply_floor(int Gr, int Bl, float* v, float floor_val, float* num, int dim) { cudaF_vec_apply_floor(Gr,Bl,v,floor_val,num,dim); }
inline void cuda_vec_apply_exp(int Gr, int Bl, float* v, int dim) { cudaF_vec_apply_exp(Gr,Bl,v,dim); }
inline void cuda_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim) { cudaF_vec_apply_log(Gr,Bl,v,flag,dim); }
inline void cuda_add_row_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }
inline void cuda_add_col_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }
inline void cuda_invert_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim d) { cudaF_invert_elements(Gr,Bl,data,d); }
// B_trans nonzero if B transposed.
inline void cuda_add_mat_blockmat(dim3 Gr, dim3 Bl, float *data, MatrixDim d, const float *Adata,
                                  int A_num_rows, int A_num_cols, int A_row_stride, int A_col_stride,
                                  const CuBlockMatrixData *B_cu_data, int B_num_blocks,
                                  float alpha, float beta, int B_trans) {
  cudaF_add_mat_blockmat(Gr, Bl, data, d, Adata, A_num_rows, A_num_cols, A_row_stride, A_col_stride,
                         B_cu_data, B_num_blocks, alpha, beta, B_trans);
}
inline void cuda_block_add_mat_mat(dim3 Gr, dim3 Bl, CuBlockMatrixData *B_cu_data, int num_blocks,
                                   const float *C_data, int C_num_cols, int C_row_stride, int C_col_stride,
                                   const float *D_data, int D_row_stride, int D_col_stride,
                                   float alpha, float beta) {
  cudaF_block_add_mat_mat(Gr, Bl, B_cu_data, num_blocks, C_data, C_num_cols,
                          C_row_stride, C_col_stride, D_data, D_row_stride,
                          D_col_stride, alpha, beta);
}



/*
 * cu::
 */
inline void cuda_soft_hinge(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride) { cudaF_soft_hinge(Gr,Bl,y,x,d,src_stride); }
inline void cuda_group_pnorm(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride, int group_size, float power) { cudaF_group_pnorm(Gr, Bl, y, x, d, src_stride, group_size, power);}
inline void cuda_sigmoid(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride) { cudaF_sigmoid(Gr,Bl,y,x,d,src_stride); }
inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d, int src_stride) { cudaF_diff_sigmoid(Gr,Bl,eout,e,y,d,src_stride); }
inline void cuda_tanh(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride) { cudaF_tanh(Gr,Bl,y,x,d,src_stride); }
inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d) { cudaF_diff_tanh(Gr,Bl,eout,e,y,d); }
inline void cuda_softmax(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d) { cudaF_softmax(Gr,Bl,y,x,d); }
/*
Bl: dimBlock value is fixed min(d.col, CU1DBLOCK), represent CU1DBLOCK threads reduce a row at the same time.
Gr: the number of rows
*/
inline void cuda_softmax_reduce(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d, int src_stride) { cudaF_softmax_reduce(Gr,Bl,y,x,d,src_stride); }
inline void cuda_softmax_part(dim3 Gr, dim3 Bl, const float *X, const int32_cuda *vec_ids, float* Y, MatrixDim d) { cudaF_softmax_part(Gr,Bl,X,vec_ids,Y,d); }

inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, float *wei, float *grad, float l1, float lr, MatrixDim d) { cudaF_regularize_l1(Gr,Bl,wei,grad,l1,lr,d); }
inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const float *mat, float *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaF_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }
inline void cuda_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, float *mat_net_out, float *vec_log_post, MatrixDim d) { cudaF_diff_xent(Gr,Bl,vec_tgt,mat_net_out,vec_log_post,d); }
inline void cuda_copy_rows_from_vec(dim3 Gr, dim3 Bl, float *mat_out, MatrixDim d_out, const float *v_in) {
  cudaF_copy_rows_from_vec(Gr, Bl, mat_out, d_out, v_in);
}


inline void cuda_randomize(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }

inline void cuda_splice(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaF_splice(Gr,Bl,y,x,off,d_out,d_in); }
inline void cuda_one(int Gr,int Bl,float* x,int dim) { cudaF_one(Gr,Bl,x,dim); }
inline void cuda_copy(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }
inline void cuda_copy_from_sp(int Gr, int Bl, const float* x, float* y, int d_in, MatrixDim d_out) { cudaF_copy_from_sp(Gr,Bl,x,y,d_in,d_out); }
inline void cuda_take_lower(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in) { cudaF_take_lower(Gr,Bl,x,y,d_in); }
inline void cuda_take_upper(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in) { cudaF_take_upper(Gr,Bl,x,y,d_in); }
inline void cuda_take_mean(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in) { cudaF_take_mean(Gr,Bl,x,y,d_in); }
inline void cuda_matrix_add_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim dim, float alpha, MatrixElement<float>* x, int s) { cudaF_matrix_add_elements(Gr, Bl, data, dim, alpha, x, s); }
inline void cuda_comp_obj_deriv(dim3 Gr, dim3 Bl, MatrixElement<float>* x, int32 size, const float* z, MatrixDim d, float* z2, MatrixDim d2, float* t) {cudaF_comp_obj_deriv(Gr,Bl,x,size,z,d,z2,d2,t); }
inline void cuda_sum_column_ranges(dim3 Gr, dim3 Bl, float *data, MatrixDim dim,
                                   const float *src_data, MatrixDim src_dim,
                                   const Int32Pair *indices) {
  cudaF_sum_column_ranges(Gr, Bl, data, dim, src_data, src_dim, indices);
}
inline void cuda_matrix_lookup(dim3 Gr, dim3 Bl, const float *data,
                               MatrixDim dim, const Int32Pair *indices,
                               int indices_size, float *output) {
  cudaF_matrix_lookup(Gr, Bl, data, dim, indices, indices_size, output);
}

inline void cuda_equal_element_mask(dim3 Gr, dim3 Bl, const float *mat1, const float *mat2, float *mask, 
                               MatrixDim mat1_dim, int mat2_stride, int mask_stride) {
  cudaF_equal_element_mask(Gr, Bl, mat1, mat2, mask, mat1_dim, mat2_stride, mask_stride);
}



// double versions

/*
 * CuMatrix 
 */
inline void cuda_copy_upp_low(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) { cudaD_copy_upp_low(Gr, Bl, A, dimA); }
inline void cuda_copy_low_upp(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA) { cudaD_copy_low_upp(Gr, Bl, A, dimA); }
inline void cuda_add_diag_vec_mat(dim3 Gr, dim3 Bl, double alpha, double *mat, MatrixDim mat_dim,
                                  const double *vec, const double *mat2, int mat2_row_stride,
                                  int mat2_col_stride, double beta) {
  cudaD_add_diag_vec_mat(Gr, Bl, alpha, mat, mat_dim, vec, mat2,
                         mat2_row_stride, mat2_col_stride, beta);
}
inline void cuda_copy_from_tp_trans(int Gr, int Bl, double* A, const double* B, MatrixDim dmat) { cudaD_copy_from_tp_trans(Gr,Bl,A,B,dmat); }
inline void cuda_copy_from_tp_trans(int Gr, int Bl, double* A, const float* B, MatrixDim dmat) { cudaDF_copy_from_tp_trans(Gr,Bl,A,B,dmat); }
inline void cuda_copy_from_tp(int Gr, int Bl, double* A, const double* B, MatrixDim dmat) { cudaD_copy_from_tp(Gr,Bl,A,B,dmat); }
inline void cuda_copy_from_tp(int Gr, int Bl, double* A, const float* B, MatrixDim dmat) { cudaDF_copy_from_tp(Gr,Bl,A,B,dmat); }
inline void cuda_copy_col_from_vec(int Gr, int Bl, double* mat, const double* v, int col, MatrixDim d) { cudaD_copy_col_from_vec(Gr,Bl,mat,v,col,d); }
inline void cuda_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) { cudaD_apply_exp(Gr,Bl,mat,d); }
inline void cuda_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim dim) { cudaD_apply_pow(Gr,Bl,mat,power,dim); }
inline void cuda_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim dim) { cudaD_apply_heaviside(Gr,Bl,mat,dim); }
inline void cuda_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim dim) { cudaD_apply_floor(Gr,Bl,mat,floor_val,dim); }
inline void cuda_apply_ceiling(dim3 Gr, dim3 Bl, double* mat, double ceiling_val, MatrixDim dim) { cudaD_apply_ceiling(Gr,Bl,mat,ceiling_val,dim); }
inline void cuda_copy_cols(dim3 Gr, dim3 Bl, double* dst, const double* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  cudaD_copy_cols(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_copy_rows(dim3 Gr, dim3 Bl, double* dst, const double* src, const MatrixIndexT_cuda* reorder, MatrixDim dst_dim, int src_stride) {
  cudaD_copy_rows(Gr, Bl, dst, src, reorder, dst_dim, src_stride);
}
inline void cuda_trace(int Gr, int Bl, double* mat, double* value, int dim) { cudaD_trace(Gr,Bl,mat,value,dim); }
inline void cuda_set_diag(int Gr, int Bl, double* mat, double value, MatrixDim d) { cudaD_set_diag(Gr,Bl,mat,value,d); }
inline void cuda_set_diag_packed(int Gr, int Bl, double* mat, double value, int dim) { cudaD_set_diag_packed(Gr,Bl,mat,value,dim); }
inline void cuda_add_diag_packed(int Gr, int Bl, double* mat, double value, int dim) { cudaD_add_diag_packed(Gr,Bl,mat,value,dim); }
inline void cuda_set_const(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_set_const(Gr,Bl,mat,value,d); }
inline void cuda_set_zero_above_diag(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) { cudaD_set_zero_above_diag(Gr,Bl,mat,d); }
inline void cuda_add(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_add(Gr,Bl,mat,value,d); }
inline void cuda_add_vec2(dim3 Gr, dim3 Bl, double *mat, const double *vec, const double alpha, int dim) { cudaD_add_vec2(Gr,Bl,mat,vec,alpha,dim); }
inline void cuda_scale_diag(int Gr, int Bl, double* mat, double value, int dim) { cudaD_scale_diag(Gr,Bl,mat,value,dim); }
inline void cuda_scale(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_scale(Gr,Bl,mat,value,d); }
inline void cuda_apply_log(dim3 Gr, dim3 Bl, double *mat, MatrixDim d) { cudaD_apply_log(Gr,Bl,mat,d); }
inline void cuda_mul_elements(dim3 Gr, dim3 Bl, double *mat, const double *A, MatrixDim dst_d, int src_stride) {
  cudaD_mul_elements(Gr,Bl,mat,A,dst_d,src_stride);
}
inline void cuda_max(dim3 Gr, dim3 Bl, double *mat, const double *A, MatrixDim dst_d, int src_stride) {
  cudaD_max(Gr,Bl,mat,A,dst_d,src_stride);
}
inline void cuda_mul_cols_vec(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d) { cudaD_mul_cols_vec(Gr,Bl,mat,scale,d); }
inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d) { cudaD_mul_rows_vec(Gr,Bl,mat,scale,d); }
inline void cuda_mul_rows_group_mat(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride, int group_size) { cudaD_mul_rows_group_mat(Gr, Bl, y, x, d, src_stride, group_size); }
inline void cuda_calc_pnorm_deriv(dim3 Gr, dim3 Bl, double *y, const double *x1, const double *x2,  MatrixDim d, int src_stride, int group_size, double power) {cudaD_calc_pnorm_deriv(Gr, Bl, y, x1, x2, d, src_stride, group_size, power); }
inline void cuda_add_mat(dim3 Gr, dim3 Bl, double alpha, const double *src, double *dst, MatrixDim d, int src_stride, int A_trans) { cudaD_add_mat(Gr,Bl,alpha,src,dst,d,src_stride, A_trans); }
inline void cuda_add_mat_mat_div_mat(dim3 Gr, dim3 Bl, const double *A, const double *B, const double *C, double *dst, MatrixDim d) { cudaD_add_mat_mat_div_mat(Gr,Bl,A,B,C,dst,d); }
inline void cuda_add_vec_to_cols(dim3 Gr, dim3 Bl, double alpha, const double *col, double beta, double *dst, MatrixDim d) { cudaD_add_vec_to_cols(Gr,Bl,alpha,col,beta,dst,d); }
inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double *row, double beta, double *dst, MatrixDim d) { cudaD_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }
inline void cuda_transpose_matrix(dim3 Gr, dim3 Bl, double *mat, MatrixDim d) { cudaD_transpose_matrix(Gr, Bl, mat, d); }
inline void cuda_sy_add_tr2(dim3 Gr, dim3 Bl, double alpha, double beta, const double* T, MatrixDim tdim,
                            double *S, MatrixDim sdim) {
  cudaD_sy_add_tr2(Gr, Bl, alpha, beta, T, tdim, S, sdim);
}


/*
 * CuVector
 */
inline void cuda_replace_value(int Gr, int Bl, double *v, int dim, double orig, double changed) {cudaD_replace_value(Gr, Bl, v, dim, orig, changed); }
inline void cuda_div_rows_vec(dim3 Gr, dim3 Bl, double *mat, const double *vec_div, MatrixDim d) { cudaD_div_rows_vec(Gr,Bl,mat,vec_div,d); }
inline void cuda_set_bias_params(int Gr, int Bl, double* v, const double* a, double param_1, double param_2, double param_3, int* flag, int dim) { cudaD_set_bias_params(Gr,Bl,v,a,param_1,param_2,param_3,flag,dim); }
inline void cuda_copy_from_vec_df(int Gr, int Bl, double* v_out, const double* v_in, int dim) { cudaD_copy_from_vec_df(Gr,Bl,v_out,v_in,dim); }
inline void cuda_copy_from_vec_fd(int Gr, int Bl, float* v_out, const double* v_in, int dim) { cudaD_copy_from_vec_fd(Gr,Bl,v_out,v_in,dim); }
inline void cuda_vec_mul_elements(int Gr, int Bl, double* v, const double* a, int dim) { cudaD_vec_mul_elements(Gr,Bl,v,a,dim); }
inline void cuda_vec_soft_max(int Gr, int Bl, double* v, int dim) { cudaD_vec_soft_max(Gr,Bl,v,dim); }
inline void cuda_vec_min(const double* v, double* value, int dim) { cudaD_vec_min(v,value,dim); }
inline void cuda_vec_max(const double* v, double* value, int dim) { cudaD_vec_max(v,value,dim); }
inline void cuda_trace_mat_mat_trans(const double* A, const double* B, MatrixDim dA, int B_stride, double* value) { cudaD_trace_mat_mat_trans(A,B,dA,B_stride,value); }
inline void cuda_trace_mat_mat(const double* A, const double* B, MatrixDim dA, int B_stride, double* value) { cudaD_trace_mat_mat(A,B,dA,B_stride,value); }
inline void cuda_add_diag_mat_trans(int Gr, int Bl, double alpha, double* v, const double* mat, double beta, MatrixDim dmat, int dim) { cudaD_add_diag_mat_trans(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
inline void cuda_add_diag_mat_mat(int Gr, int Bl, double alpha, double* v, int v_dim, const double* M, 
                                  int M_cols, int M_row_stride, int M_col_stride, const double *N, int N_row_stride, 
                                  int N_col_stride, int threads_per_element, double beta) {
  cudaD_add_diag_mat_mat(Gr, Bl, alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride, N, N_row_stride,
                         N_col_stride, threads_per_element, beta);
}
inline void cuda_add_diag_mat(int Gr, int Bl, double alpha, double* v, const double* mat, double beta, MatrixDim dmat, int dim) { cudaD_add_diag_mat(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
inline void cuda_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim) { cudaD_add_vec_vec(Gr,Bl,alpha,v,x,y,beta,dim); }
inline void cuda_copy_col_from_mat(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim) { cudaD_copy_col_from_mat(Gr,Bl,v,col,mat,dmat,dim); }
inline void cuda_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim) { cudaD_copy_col_from_mat_df(Gr,Bl,v,col,mat,dmat,dim); }
inline void cuda_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const double* mat, MatrixDim dmat, int dim) { cudaD_copy_col_from_mat_fd(Gr,Bl,v,col,mat,dmat,dim); }
inline void cuda_vec_sum(int Gr, int Bl, double* v, double* value, int dim, int inc) { cudaD_vec_sum(Gr,Bl,v,value,dim,inc); }
inline void cuda_pvec_sum(int Gr, int Bl, double* vec, double* pvec_sum, int dim, int size) { cudaD_pvec_sum(Gr,Bl,vec,pvec_sum,dim,size); }
inline void cuda_vec_copy_diag_from_packed(int Gr, int Bl, double *dst, const double *src, int dim) { cudaD_vec_copy_diag_from_packed(Gr,Bl,dst,src,dim); }
inline void cuda_vec_apply_floor(int Gr, int Bl, double* v, double floor_val, float* num, int dim) { cudaD_vec_apply_floor(Gr,Bl,v,floor_val,num,dim); }
inline void cuda_vec_apply_exp(int Gr, int Bl, double* v, int dim) { cudaD_vec_apply_exp(Gr,Bl,v,dim); }
inline void cuda_vec_apply_log(int Gr, int Bl, double* v, double* flag, int dim) { cudaD_vec_apply_log(Gr,Bl,v,flag,dim); }
inline void cuda_add_row_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }
inline void cuda_add_col_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }
inline void cuda_invert_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim d) { cudaD_invert_elements(Gr,Bl,data,d); }
// B_trans nonzero if B transposed.
inline void cuda_add_mat_blockmat(dim3 Gr, dim3 Bl, double *data, MatrixDim d, const double *Adata,
                                  int A_num_rows, int A_num_cols, int A_row_stride, int A_col_stride,
                                  const CuBlockMatrixData *B_cu_data, int B_num_blocks,
                                  double alpha, double beta, int B_trans) {
  cudaD_add_mat_blockmat(Gr, Bl, data, d, Adata, A_num_rows, A_num_cols, A_row_stride, A_col_stride,
                         B_cu_data, B_num_blocks, alpha, beta, B_trans);
}
inline void cuda_block_add_mat_mat(dim3 Gr, dim3 Bl, CuBlockMatrixData *B_cu_data, int num_blocks,
                                   const double *C_data, int C_num_cols, int C_row_stride, int C_col_stride,
                                   const double *D_data, int D_row_stride, int D_col_stride,
                                   double alpha, double beta) {
  cudaD_block_add_mat_mat(Gr, Bl, B_cu_data, num_blocks, C_data, C_num_cols,
                          C_row_stride, C_col_stride, D_data, D_row_stride,
                          D_col_stride, alpha, beta);
}

/*
 * cu::
 */
inline void cuda_soft_hinge(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride) { cudaD_soft_hinge(Gr,Bl,y,x,d,src_stride); }
inline void cuda_group_pnorm(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride, int group_size, double power) { cudaD_group_pnorm(Gr, Bl, y, x, d, src_stride, group_size, power); }
inline void cuda_sigmoid(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride) { cudaD_sigmoid(Gr,Bl,y,x,d,src_stride); }
inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d, int src_stride) { cudaD_diff_sigmoid(Gr,Bl,eout,e,y,d,src_stride); }
inline void cuda_tanh(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride) { cudaD_tanh(Gr,Bl,y,x,d,src_stride); }
inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d) { cudaD_diff_tanh(Gr,Bl,eout,e,y,d); }
inline void cuda_softmax(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d) { cudaD_softmax(Gr,Bl,y,x,d); }
inline void cuda_softmax_reduce(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d, int src_stride) { cudaD_softmax_reduce(Gr,Bl,y,x,d,src_stride); }
inline void cuda_softmax_part(dim3 Gr, dim3 Bl, const double *X, const int32_cuda *vec_ids, double* Y, MatrixDim d) { cudaD_softmax_part(Gr,Bl,X,vec_ids,Y,d); }

inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, double *wei, double *grad, double l1, double lr, MatrixDim d) { cudaD_regularize_l1(Gr,Bl,wei,grad,l1,lr,d); }
inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const double *mat, double *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaD_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }
inline void cuda_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, double *mat_net_out, double *vec_log_post, MatrixDim d) {
  cudaD_diff_xent(Gr,Bl,vec_tgt,mat_net_out,vec_log_post,d);
}
inline void cuda_copy_rows_from_vec(dim3 Gr, dim3 Bl, double *mat_out, MatrixDim d_out, const double *v_in) {
  cudaD_copy_rows_from_vec(Gr, Bl, mat_out, d_out, v_in);
}

inline void cuda_randomize(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }
inline void cuda_splice(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaD_splice(Gr,Bl,y,x,off,d_out,d_in); }
inline void cuda_one(int Gr,int Bl,double* x,int dim) { cudaD_one(Gr,Bl,x,dim); }
inline void cuda_copy(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }
inline void cuda_copy_from_sp(int Gr, int Bl, const double* x, double* y, int d_in, MatrixDim d_out) { cudaD_copy_from_sp(Gr,Bl,x,y,d_in,d_out); }
inline void cuda_take_lower(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in) { cudaD_take_lower(Gr,Bl,x,y,d_in); }
inline void cuda_take_upper(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in) { cudaD_take_upper(Gr,Bl,x,y,d_in); }
inline void cuda_take_mean(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in) { cudaD_take_mean(Gr,Bl,x,y,d_in); }
inline void cuda_matrix_add_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim dim, double alpha, MatrixElement<double>* x, int s) { cudaD_matrix_add_elements(Gr, Bl, data, dim, alpha, x, s); }
inline void cuda_comp_obj_deriv(dim3 Gr, dim3 Bl, MatrixElement<double>* x, int32 size, const double* z, MatrixDim d, double* z2, MatrixDim d2, double* t) {cudaD_comp_obj_deriv(Gr,Bl,x,size,z,d,z2,d2,t); }
inline void cuda_sum_column_ranges(dim3 Gr, dim3 Bl, double *data, MatrixDim dim,
                                   const double *src_data, MatrixDim src_dim, const Int32Pair *indices) {
  cudaD_sum_column_ranges(Gr, Bl, data, dim, src_data, src_dim, indices);
}
inline void cuda_matrix_lookup(dim3 Gr, dim3 Bl, const double *data,
                               MatrixDim dim, const Int32Pair *indices,
                               int indices_size, double *output) {
  cudaD_matrix_lookup(Gr, Bl, data, dim, indices, indices_size, output);
}

inline void cuda_equal_element_mask(dim3 Gr, dim3 Bl, const double *mat1, const double *mat2, double *mask, 
                                    MatrixDim mat1_dim, int mat2_stride, int mask_stride) {
  cudaD_equal_element_mask(Gr, Bl, mat1, mat2, mask, mat1_dim, mat2_stride, mask_stride);
}



// Also include some template-friendly wrappers of cublas functions:
inline void cuda_axpy(int n, float alpha, const float *x, int incx, float *y, int incy) {
  cublasSaxpy(n, alpha, x, incx, y, incy);
}
inline void cuda_axpy(int n, double alpha, const double *x, int incx, double *y, int incy) {
  cublasDaxpy(n, alpha, x, incx, y, incy);
}
inline void cuda_scal(int n, float alpha, float *x, int incx) {
  cublasSscal(n, alpha, x, incx);
}
inline void cuda_scal(int n, double alpha, double *x, int incx) {
  cublasDscal(n, alpha, x, incx);
}


} // namespace kaldi



#endif // HAVE_CUDA

#endif
