#include "cudamatrix/cu-matrix.h"

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-kernels.h"


namespace kaldi {

/*
 * implement float specialized methdos
 */
 /*
template<> 
void CuMatrix<float>::Set(float value) { 
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_set_const(dimGrid, dimBlock, data_, value, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.Set(value);
  }
}

template<> 
void CuMatrix<float>::ApplyLog() { 
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cudaF_apply_log(dimGrid, dimBlock, data_, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.ApplyLog();
  }
}


template<>
void CuMatrix<float>::MulElements(const CuMatrix<float>& A) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(num_cols_ == A.NumCols());
    assert(num_rows_ == A.NumRows());
    assert(stride_ == A.Stride());
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cudaF_mul_elem(dimGrid, dimBlock, data_, A.Data(), Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.MulElements(A.mat_);
  }
}


template<>
void CuMatrix<float>::MulColsVec(const CuVector<float> &scale) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    assert(scale.Dim() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cudaF_scale_cols(dimGrid, dimBlock, data_, scale.Data(), Dim());
    cuSafeCall(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.MulColsVec(scale.Vec());
  }
}


template<>
void CuMatrix<float>::MulRowsVec(const CuVector<float> &scale) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(scale.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cudaF_scale_rows(dimGrid, dimBlock, data_, scale.Data(), Dim());
    cuSafeCall(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    mat_.MulRowsVec(scale.Vec());
  }
}


template<>
void CuMatrix<float>::DivRowsVec(const CuVector<float> &div) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(div.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cudaF_div_rows_vec(dimGrid, dimBlock, data_, div.Data(), Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    Vector<float> tmp(div.Vec());
    tmp.InvertElements();
    mat_.MulRowsVec(tmp);
  }
}


template<>
void CuMatrix<float>::AddMat(float alpha, const CuMatrix<float>& A, float beta) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(A.NumRows() == NumRows());
    assert(A.NumCols() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cudaF_add_scaled(dimGrid, dimBlock, alpha, A.Data(), beta, data_, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.Scale(beta);
    mat_.AddMat(alpha, A.mat_);
  }
}


template<>
void CuMatrix<float>::AddScaledRow(float alpha, const CuVector<float> &row, float beta) { 
  
  if (row.Dim() != NumCols()) {
    KALDI_ERR << "Non matching dimensions: Cols:" << NumCols() << " VectorDim:" << row.Dim();
  }

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cudaF_add_scaled_row(dimGrid, dimBlock, alpha, row.Data(), beta, data_, Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    for(MatrixIndexT r=0; r<NumRows(); r++) {
      mat_.Row(r).Scale(beta);
      mat_.Row(r).AddVec(alpha, row.Vec());
    }
  }
}


template<>
void CuMatrix<float>::AddMatMat(
               float alpha, const CuMatrix<float>& A, MatrixTransposeType transA,
               const CuMatrix<float>& B, MatrixTransposeType transB, float beta) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    // CUBLAS is col major, C++ is row major
    // keep trans..., just swap A&B matrices: A->B B->A
    MatrixIndexT m = ((transB==kTrans)? B.NumRows() : B.NumCols()); 
    MatrixIndexT n = ((transA==kTrans)? A.NumCols() : A.NumRows());
    MatrixIndexT k = ((transB==kTrans)? B.NumCols() : B.NumRows());
    MatrixIndexT k1 = ((transA==kTrans)? A.NumRows() : A.NumCols());

    assert(m == NumCols());
    assert(n == NumRows());
    assert(k == k1);

    Timer tim;

    cublas_gemm((transB==kTrans?'T':'N'), (transA==kTrans?'T':'N'), m, n, k, 
                alpha, B.Data(), B.Stride(), A.Data(), A.Stride(), 
                beta, data_, Stride());

    cuSafeCall(cublasGetError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.AddMatMat(alpha, A.mat_, transA, B.mat_, transB, beta);
  }
}

*/


} // namespace kaldi

