// cudamatrix/cu-kernels.h

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



#ifndef KALDI_CUDAMATRIX_CU_KERNELS_H_
#define KALDI_CUDAMATRIX_CU_KERNELS_H_

#if HAVE_CUDA==1

#include "base/kaldi-error.h"
#include "cudamatrix/cu-kernels-ansi.h"

/*
 * In this file are C++ templated wrappers 
 * of the ANSI-C CUDA kernels
 */

namespace kaldi {



/*********************************************************
 * base templates
 */

/*
 * CuMatrix
 */
template<typename Real> inline void cuda_set_const(dim3 Gr, dim3 Bl, Real *mat, Real value, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_apply_log(dim3 Gr, dim3 Bl, Real *mat, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_mul_elements(dim3 Gr, dim3 Bl, Real *mat, const Real *A, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_mul_cols_vec(dim3 Gr, dim3 Bl, Real *mat, const Real *scale, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, Real *mat, const Real *scale, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_div_rows_vec(dim3 Gr, dim3 Bl, Real *mat, const Real *vec_div, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_mat(dim3 Gr, dim3 Bl, Real alpha, const Real *A, Real beta, Real *dst, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_vec_to_cols(dim3 Gr, dim3 Bl, Real alpha, const Real *col, Real beta, Real *dst, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, Real alpha, const Real *row, Real beta, Real *dst, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
 
/*
 * CuVector
 */
template<typename Real> inline void cuda_add_row_sum_mat(dim3 Gr, dim3 Bl, const Real *mat, Real *vec_sum, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_col_sum_mat(dim3 Gr, dim3 Bl, const Real *mat, Real *vec_sum, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_invert_elements(dim3 Gr, dim3 Bl, Real *data, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }

/*
 * cu::
 */
template<typename Real> inline void cuda_sigmoid(dim3 Gr, dim3 Bl, Real *y, const Real *x, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, Real *eout, const Real *e, const Real *y, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_tanh(dim3 Gr, dim3 Bl, Real *y, const Real *x, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, Real *eout, const Real *e, const Real *y, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_softmax(size_t Gr, size_t Bl, Real *y, const Real *x, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_softmax_part(dim3 Gr, dim3 Bl, const Real *X, const int32_cuda *vec_ids, Real* Y, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }

template<typename Real> inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, Real *wei, Real *grad, Real l1, Real lr, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const Real *mat, Real *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, Real *mat_net_out, Real *vec_log_post, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }

template<typename Real> inline void cuda_randomize(dim3 Gr, dim3 Bl, Real *y, const Real *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_splice(dim3 Gr, dim3 Bl, Real *y, const Real *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_copy(dim3 Gr, dim3 Bl, Real *y, const Real *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << " Not implemented!"; }



/*********************************************************
 * float specializations
 */

/*
 * CuMatrix 
 */
template<> inline void cuda_set_const<float>(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_set_const(Gr,Bl,mat,value,d); }
template<> inline void cuda_apply_log<float>(dim3 Gr, dim3 Bl, float *mat, MatrixDim d) { cudaF_apply_log(Gr,Bl,mat,d); }
template<> inline void cuda_mul_elements<float>(dim3 Gr, dim3 Bl, float *mat, const float *A, MatrixDim d) { cudaF_mul_elements(Gr,Bl,mat,A,d); }
template<> inline void cuda_mul_cols_vec<float>(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d) { cudaF_mul_cols_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_mul_rows_vec<float>(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d) { cudaF_mul_rows_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_div_rows_vec<float>(dim3 Gr, dim3 Bl, float *mat, const float *vec_div, MatrixDim d) { cudaF_div_rows_vec(Gr,Bl,mat,vec_div,d); }
template<> inline void cuda_add_mat<float>(dim3 Gr, dim3 Bl, float alpha, const float *A, float beta, float *dst, MatrixDim d) { cudaF_add_mat(Gr,Bl,alpha,A,beta,dst,d); }
template<> inline void cuda_add_vec_to_cols<float>(dim3 Gr, dim3 Bl, float alpha, const float *col, float beta, float *dst, MatrixDim d) { cudaF_add_vec_to_cols(Gr,Bl,alpha,col,beta,dst,d); }
template<> inline void cuda_add_vec_to_rows<float>(dim3 Gr, dim3 Bl, float alpha, const float *row, float beta, float *dst, MatrixDim d) { cudaF_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }
 
/*
 * CuVector
 */
template<> inline void cuda_add_row_sum_mat<float>(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_add_col_sum_mat<float>(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_invert_elements<float>(dim3 Gr, dim3 Bl, float *data, MatrixDim d) { cudaF_invert_elements(Gr,Bl,data,d); }

/*
 * cu::
 */
template<> inline void cuda_sigmoid<float>(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d) { cudaF_sigmoid(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_sigmoid<float>(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d) { cudaF_diff_sigmoid(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_tanh<float>(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d) { cudaF_tanh(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_tanh<float>(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d) { cudaF_diff_tanh(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_softmax<float>(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d) { cudaF_softmax(Gr,Bl,y,x,d); }
template<> inline void cuda_softmax_part<float>(dim3 Gr, dim3 Bl, const float *X, const int32_cuda *vec_ids, float* Y, MatrixDim d) { cudaF_softmax_part(Gr,Bl,X,vec_ids,Y,d); }

template<> inline void cuda_regularize_l1<float>(dim3 Gr, dim3 Bl, float *wei, float *grad, float l1, float lr, MatrixDim d) { cudaF_regularize_l1(Gr,Bl,wei,grad,l1,lr,d); }
template<> inline void cuda_find_row_max_id<float>(dim3 Gr, dim3 Bl, const float *mat, float *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaF_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }
template<> inline void cuda_diff_xent<float>(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, float *mat_net_out, float *vec_log_post, MatrixDim d) { cudaF_diff_xent(Gr,Bl,vec_tgt,mat_net_out,vec_log_post,d); }

template<> inline void cuda_randomize<float>(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }

template<> inline void cuda_splice<float>(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaF_splice(Gr,Bl,y,x,off,d_out,d_in); }
template<> inline void cuda_copy<float>(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }


/*********************************************************
 * double specializations
 */

/*
 * CuMatrix 
 */
template<> inline void cuda_set_const<double>(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_set_const(Gr,Bl,mat,value,d); }
template<> inline void cuda_apply_log<double>(dim3 Gr, dim3 Bl, double *mat, MatrixDim d) { cudaD_apply_log(Gr,Bl,mat,d); }
template<> inline void cuda_mul_elements<double>(dim3 Gr, dim3 Bl, double *mat, const double *A, MatrixDim d) { cudaD_mul_elements(Gr,Bl,mat,A,d); }
template<> inline void cuda_mul_cols_vec<double>(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d) { cudaD_mul_cols_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_mul_rows_vec<double>(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d) { cudaD_mul_rows_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_div_rows_vec<double>(dim3 Gr, dim3 Bl, double *mat, const double *vec_div, MatrixDim d) { cudaD_div_rows_vec(Gr,Bl,mat,vec_div,d); }
template<> inline void cuda_add_mat<double>(dim3 Gr, dim3 Bl, double alpha, const double *A, double beta, double *dst, MatrixDim d) { cudaD_add_mat(Gr,Bl,alpha,A,beta,dst,d); }
template<> inline void cuda_add_vec_to_cols<double>(dim3 Gr, dim3 Bl, double alpha, const double *col, double beta, double *dst, MatrixDim d) { cudaD_add_vec_to_cols(Gr,Bl,alpha,col,beta,dst,d); }
template<> inline void cuda_add_vec_to_rows<double>(dim3 Gr, dim3 Bl, double alpha, const double *row, double beta, double *dst, MatrixDim d) { cudaD_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }
 
/*
 * CuVector
 */
template<> inline void cuda_add_row_sum_mat<double>(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_add_col_sum_mat<double>(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_invert_elements<double>(dim3 Gr, dim3 Bl, double *data, MatrixDim d) { cudaD_invert_elements(Gr,Bl,data,d); }

/*
 * cu::
 */
template<> inline void cuda_sigmoid<double>(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d) { cudaD_sigmoid(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_sigmoid<double>(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d) { cudaD_diff_sigmoid(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_tanh<double>(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d) { cudaD_tanh(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_tanh<double>(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d) { cudaD_diff_tanh(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_softmax<double>(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d) { cudaD_softmax(Gr,Bl,y,x,d); }
template<> inline void cuda_softmax_part<double>(dim3 Gr, dim3 Bl, const double *X, const int32_cuda *vec_ids, double* Y, MatrixDim d) { cudaD_softmax_part(Gr,Bl,X,vec_ids,Y,d); }

template<> inline void cuda_regularize_l1<double>(dim3 Gr, dim3 Bl, double *wei, double *grad, double l1, double lr, MatrixDim d) { cudaD_regularize_l1(Gr,Bl,wei,grad,l1,lr,d); }
template<> inline void cuda_find_row_max_id<double>(dim3 Gr, dim3 Bl, const double *mat, double *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaD_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }
template<> inline void cuda_diff_xent<double>(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, double *mat_net_out, double *vec_log_post, MatrixDim d) { cudaD_diff_xent(Gr,Bl,vec_tgt,mat_net_out,vec_log_post,d); }

template<> inline void cuda_randomize<double>(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }
template<> inline void cuda_splice<double>(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaD_splice(Gr,Bl,y,x,off,d_out,d_in); }
template<> inline void cuda_copy<double>(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }

} // namespace



#endif // HAVE_CUDA

#endif


