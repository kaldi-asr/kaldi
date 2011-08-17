#ifndef KALDI_CUDAMATRIX_CU_KERNELS_H_
#define KALDI_CUDAMATRIX_CU_KERNELS_H_

#include "cudamatrix/cu-matrixdim.h"

extern "C" {

/*************
 * Float instances
 */
//CuMatrix 
void cudaF_set_const(dim3 Gr, dim3 Bl, float*mat, float value, MatrixDim d);
void cudaF_apply_log(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);
void cudaF_scale_cols(dim3 Gr, dim3 Bl, float*mat, const float* scale, MatrixDim d);
void cudaF_scale_rows(dim3 Gr, dim3 Bl, float*mat, const float* scale, MatrixDim d);
void cudaF_add_scaled(dim3 Gr, dim3 Bl, float alpha, const float* A, float beta, float* dst, MatrixDim d);
void cudaF_add_scaled_row(dim3 Gr, dim3 Bl, float alpha, const float* row, float beta, float* dst, MatrixDim d);
void cudaF_mul_elem(dim3 Gr, dim3 Bl, float*mat, const float*A, MatrixDim d);
 
//CuVector
void cudaF_add_col_sum(size_t Gr, size_t Bl, float alpha, const float* mat, float beta, float* vec, MatrixDim d);
void cudaF_add_col_sum_reduce(dim3 Gr, dim3 Bl, float alpha, const float* mat, float beta, float* vec, MatrixDim d);

//CuMath
void cudaF_softmax      (size_t Gr, size_t Bl, float*y, const float*x, MatrixDim d);
void cudaF_softmax_reduce (dim3 Gr, dim3 Bl, float*y, const float*x, MatrixDim d); 
void cudaF_sigmoid      (dim3 Gr, dim3 Bl, float*y, const float*x, MatrixDim d);
void cudaF_diff_sigmoid (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d);

void cudaF_expand(dim3 Gr, dim3 Bl, float* y, const float* x, const int* off, MatrixDim d_out, MatrixDim d_in);
void cudaF_rearrange(dim3 Gr, dim3 Bl, float* y, const float* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaF_randomize(dim3 Gr, dim3 Bl, float* y, const float* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in);

void cudaF_check_class(size_t Gr, size_t Bl, const float* out, const float* des, float* match, MatrixDim d);
void cudaF_check_class_reduce(dim3 Gr, dim3 Bl, const float* out, const float* des, float* match, MatrixDim d);



/*************
 * Double instances
 */
//CuMatrix 
void cudaD_set_const(dim3 Gr, dim3 Bl, double*mat, double value, MatrixDim d);
void cudaD_apply_log(dim3 Gr, dim3 Bl, double* mat, MatrixDim d);
void cudaD_scale_cols(dim3 Gr, dim3 Bl, double*mat, const double* scale, MatrixDim d);
void cudaD_scale_rows(dim3 Gr, dim3 Bl, double*mat, const double* scale, MatrixDim d);
void cudaD_add_scaled(dim3 Gr, dim3 Bl, double alpha, const double* A, double beta, double* dst, MatrixDim d);
void cudaD_add_scaled_row(dim3 Gr, dim3 Bl, double alpha, const double* row, double beta, double* dst, MatrixDim d);
void cudaD_mul_elem(dim3 Gr, dim3 Bl, double*mat, const double*A, MatrixDim d);
 
//CuVector
void cudaD_add_col_sum(size_t Gr, size_t Bl, double alpha, const double* mat, double beta, double* vec, MatrixDim d);

//CuMath
void cudaD_softmax      (size_t Gr, size_t Bl, double*y, const double*x, MatrixDim d);
void cudaD_sigmoid      (dim3 Gr, dim3 Bl, double*y, const double*x, MatrixDim d);
void cudaD_diff_sigmoid (dim3 Gr, dim3 Bl, double* eout, const double* e, const double* y, MatrixDim d);

void cudaD_expand(dim3 Gr, dim3 Bl, double* y, const double* x, const int* off, MatrixDim d_out, MatrixDim d_in);
void cudaD_rearrange(dim3 Gr, dim3 Bl, double* y, const double* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaD_randomize(dim3 Gr, dim3 Bl, double* y, const double* x, const int* copy_from, MatrixDim d_out, MatrixDim d_in);

void cudaD_check_class(size_t Gr, size_t Bl, const double* out, const double* des, float* match, MatrixDim d);


}

#endif
