#ifndef KALDI_CUDAMATRIX_CU_KERNELS_H_
#define KALDI_CUDAMATRIX_CU_KERNELS_H_

#include "cudamatrix/cu-matrixdim.h"

extern "C" {

/*
 * float CUDA functions
 */

/*
 * CuMatrix 
 */
void cudaF_set_const(dim3 Gr, dim3 Bl, float*mat, float value, MatrixDim d);
void cudaF_apply_log(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);
void cudaF_scale_cols(dim3 Gr, dim3 Bl, float*mat, const float* scale, MatrixDim d);
void cudaF_scale_rows(dim3 Gr, dim3 Bl, float*mat, const float* scale, MatrixDim d);
void cudaF_div_rows_vec(dim3 Gr, dim3 Bl, float*mat, const float* vec_div, MatrixDim d);
void cudaF_add_scaled(dim3 Gr, dim3 Bl, float alpha, const float* A, float beta, float* dst, MatrixDim d);
void cudaF_add_scaled_row(dim3 Gr, dim3 Bl, float alpha, const float* row, float beta, float* dst, MatrixDim d);
void cudaF_mul_elem(dim3 Gr, dim3 Bl, float*mat, const float*A, MatrixDim d);
 
/*
 * CuVector
 */
void cudaF_add_col_sum(size_t Gr, size_t Bl, float alpha, const float* mat, float beta, float* vec, MatrixDim d);
void cudaF_add_col_sum_reduce(dim3 Gr, dim3 Bl, float alpha, const float* mat, float beta, float* vec, MatrixDim d);
void cudaF_invert_elements(dim3 Gr, dim3 Bl, float* data, MatrixDim d);

/*
 * cu::
 */
void cudaF_softmax      (size_t Gr, size_t Bl, float*y, const float*x, MatrixDim d);
void cudaF_softmax_reduce (dim3 Gr, dim3 Bl, float*y, const float*x, MatrixDim d); 
void cudaF_sigmoid      (dim3 Gr, dim3 Bl, float*y, const float*x, MatrixDim d);
void cudaF_diff_sigmoid (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d);

void cudaF_expand(dim3 Gr, dim3 Bl, float* y, const float* x, const int* off, MatrixDim d_out, MatrixDim d_in);
void cudaF_rearrange(dim3 Gr, dim3 Bl, float* y, const float* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaF_randomize(dim3 Gr, dim3 Bl, float* y, const float* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in);

void cudaF_check_class(size_t Gr, size_t Bl, const float* out, const float* des, float* match, MatrixDim d);
void cudaF_check_class_reduce(dim3 Gr, dim3 Bl, const float* out, const float* des, float* match, MatrixDim d);

void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float* wei, float* grad, float l1, float lr, MatrixDim d);

void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float* mat, float* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d);

void cudaF_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda* vec_tgt, float* mat_net_out, float* vec_log_post, MatrixDim d);

void cudaF_softmax_part(dim3 Gr, dim3 Bl, const float* X, const int32_cuda* vec_ids, float* Y, MatrixDim d);

void cudaF_sum_rows_vec(dim3 Gr, dim3 Bl, const float* mat, float* vec_sum, MatrixDim d);


/*
 * int32 CUDA functions
 */
void cudaI32_set_const(dim3 Gr, dim3 Bl, int32_cuda*mat, int32_cuda value, MatrixDim d);


} //extern "C" 

#endif
