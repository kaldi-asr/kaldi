
#include "cudamatrix/cu-vector.h"

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-kernels.h"

namespace kaldi {


/*
 * implement float specialized methdos
 */
template<>
void CuVector<float>::Set(float value) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaF_set_const(dimGrid, dimBlock, data_, value, d);
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    vec_.Set(value);
  }
}


template<>
void CuVector<float>::AddVec(float alpha, const CuVector<float>& vec, float beta) {
  assert(vec.Dim() == Dim());
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;


    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaF_add_scaled(dimGrid, dimBlock, alpha, vec.Data(), beta, data_, d);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    if (beta != 1.0) { vec_.Scale(beta); }
    vec_.AddVec(alpha, vec.Vec());
  }
}


template<>
void CuVector<float>::AddColSum(float alpha, const CuMatrix<float>& mat, float beta) {
  assert(mat.NumCols() == Dim());
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    
    /**
     * Rows()<=512 limit due to limited shared memory
     * Cols()<=256 limit due to coalesced memory alignment:
     *             matrices with huge strides have slow access!!!
     */
    if (mat.NumRows() > 512 || mat.NumCols() > 256) {
      size_t dimBlock = CUBLOCK*2;
      size_t dimGrid = n_blocks(Dim(), CUBLOCK*2); 

      cudaF_add_col_sum(dimGrid, dimBlock, alpha, mat.Data(), beta, data_, mat.Dim());
      cuSafeCall(cudaGetLastError());
    } else {
      dim3 dimBlock(mat.NumRows(), 1);
      dim3 dimGrid(1, Dim()); 

      cudaF_add_col_sum_reduce(dimGrid, dimBlock, alpha, mat.Data(), beta, data_, mat.Dim());
      cuSafeCall(cudaGetLastError());
    }

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Vector<BaseFloat> tmp(mat.NumCols());
    for(MatrixIndexT r=0; r<mat.NumRows(); r++) {
      tmp.AddVec(1.0, mat.Mat().Row(r));
    }
    if (beta != 1.0) { vec_.Scale(beta); }
    vec_.AddVec(alpha, tmp);
  }
}


template<> 
void CuVector<float>::InvertElements() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    
    dim3 dimBlock(CUBLOCK*8, 1);
    dim3 dimGrid(n_blocks(dim_, CUBLOCK*8));
    MatrixDim d = {1, dim_, dim_};

    cudaF_invert_elements(dimGrid, dimBlock, data_, d);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    vec_.InvertElements();
  }
}



} // namespace
