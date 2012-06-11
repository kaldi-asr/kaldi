
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
void CuVector<float>::AddVec(float alpha, const CuVector<float> &vec, float beta) {
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
void CuVector<float>::AddRowSumMat(float alpha, const CuMatrix<float> &mat, float beta) {
  assert(mat.NumCols() == Dim());
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
   
    CuVector<float> tmp(Dim()); // create a buffer
    tmp.SetZero();
    
    MatrixDim d = mat.Dim(); // only stride will be used!
  
    // process per 256 row blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumRows(); block++) {
      // 1st dim ... rows, 2nd dim ... cols
      dim3 dimBlock(256, 1); 
      dim3 dimGrid(1, mat.NumCols());
      int32 offset = block*256*d.stride;

      cudaF_add_row_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    
    // process the remainder
    int32 div = mat.NumRows() / 256;
    int32 mod = mat.NumRows() % 256;
    if (mod != 0) {
      // 1st dim ... rows, 2nd dim ... cols
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumCols());
      int32 offset = div*256*d.stride;
      
      cudaF_add_row_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    // now we have the sum!
    
    // add buffer rmp to this vector using alpha and beta
    this->AddVec(alpha,tmp,beta);

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Vector<float> tmp(mat.NumCols());
    tmp.AddRowSumMat(mat.Mat());
    if(beta != 1.0) vec_.Scale(beta);
    vec_.AddVec(alpha,tmp);
  }
}


template<>
void CuVector<float>::AddColSumMat(float alpha, const CuMatrix<float> &mat, float beta) {
  assert(mat.NumRows() == Dim());
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    CuVector<float> tmp(Dim()); // create a buffer
    
    MatrixDim d = mat.Dim(); // only stride will be used!
  
    // process per 256 column blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumCols(); block++) {
      // 1st dim ... cols, 2nd dim ... rows
      dim3 dimBlock(256, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset = block*256;

      cudaF_add_col_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    
    // process the remainder
    int32 div = mat.NumCols() / 256;
    int32 mod = mat.NumCols() % 256;
    if (mod != 0) {
      // 1st dim ... cols, 2nd dim ... rows
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset=div*256;
      
      cudaF_add_col_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    // now we have the sum!
    
    // add buffer rmp to this vector using alpha and beta
    this->AddVec(alpha,tmp,beta);
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Vector<float> tmp(mat.NumRows());
    tmp.AddColSumMat(mat.Mat());
    if(beta != 1.0) vec_.Scale(beta);
    vec_.AddVec(alpha,tmp);
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
