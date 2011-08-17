


#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-kernels.h"


namespace kaldi {
namespace cu {

/*
 * Float specializations
 */
void Sigmoid(const CuMatrix<float>& X, CuMatrix<float>* Y) {
  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(X.NumCols(),CUBLOCK), n_blocks(X.NumRows(), CUBLOCK));

    cudaF_sigmoid(dimGrid, dimBlock, Y->Data(), X.Data(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__,tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float>& y = Y->Mat();
    const MatrixBase<float>& x = X.Mat();
    for(MatrixIndexT r=0; r<x.NumRows(); r++) {
      for(MatrixIndexT c=0; c<x.NumCols(); c++) {
        y(r,c) = 1.0/(1.0+exp(-x(r,c)));
      }
    }
  }
}


void DiffSigmoid(const CuMatrix<float>& Ein, const CuMatrix<float>& Y, CuMatrix<float>* Eout) {
  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(Eout->NumCols(), CUBLOCK), n_blocks(Eout->NumRows(),CUBLOCK));

    cudaF_diff_sigmoid(dimGrid, dimBlock, Eout->Data(), Ein.Data(), Y.Data(), Eout->Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__,tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float>& eout = Eout->Mat();
    const MatrixBase<float>& ein = Ein.Mat();
    const MatrixBase<float>& y = Y.Mat();
    for(MatrixIndexT r=0; r<eout.NumRows(); r++) {
      for(MatrixIndexT c=0; c<eout.NumCols(); c++) {
        eout(r,c) = ein(r,c) * y(r,c)*(1.0-y(r,c));
      }
    }
  }
}

  
void Softmax(const CuMatrix<float>& X, CuMatrix<float>* Y) {
  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    #if 0
    //disable 'reduce' functions
    size_t dimBlock = CUBLOCK;
    size_t dimGrid  = n_blocks(X.NumRows(),CUBLOCK);

    cudaF_softmax(dimGrid, dimBlock, Y.Data(), X.Data(), X.Dim());
    cuSafeCall(cudaGetLastError());
    #else
    if(X.NumCols() > 256) {
      //use old implementation (can't use reduction due to 
      //limited size of shared memory)
      size_t dimBlock = CUBLOCK;
      size_t dimGrid  = n_blocks(X.NumRows(),CUBLOCK);

      cudaF_softmax(dimGrid, dimBlock, Y->Data(), X.Data(), X.Dim());
      cuSafeCall(cudaGetLastError());
    } else {
      //use implementation with reduction
      dim3 dimBlock(X.NumCols(),1);
      dim3 dimGrid(1,X.NumRows());

      cudaF_softmax_reduce(dimGrid, dimBlock, Y->Data(), X.Data(), X.Dim());
      cuSafeCall(cudaGetLastError());
    }
    #endif

    CuDevice::Instantiate().AccuProfile(__func__,tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float>& y = Y->Mat();
    const MatrixBase<float>& x = X.Mat();
    y.CopyFromMat(x);
    for(MatrixIndexT r=0; r<x.NumRows(); r++) {
      y.Row(r).ApplySoftMax();
    }

  }
}



void CheckClass(const CuMatrix<float>& out, const CuMatrix<float> &des, CuVector<float>* match) {
  assert(out.NumCols() == des.NumCols());
  assert(out.NumRows() == des.NumRows());
  assert(out.Stride() == des.Stride());
  assert(match->Dim() == out.NumRows());

  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    if(out.NumCols() > 256) {
      size_t dimBlock = CUBLOCK;
      size_t dimGrid = n_blocks(out.NumRows(),CUBLOCK);

      cudaF_check_class(dimGrid, dimBlock, out.Data(), des.Data(), match->Data(), out.Dim());
      cuSafeCall(cudaGetLastError());
    } else {
      dim3 dimBlock(out.NumCols(),1);
      dim3 dimGrid(1,out.NumRows());

      cudaF_check_class_reduce(dimGrid, dimBlock, out.Data(), des.Data(), match->Data(), out.Dim());
      cuSafeCall(cudaGetLastError());
    }
    
    CuDevice::Instantiate().AccuProfile(__func__,tim.Elapsed());
  } else
  #endif
  {
    const MatrixBase<float>& mout = out.Mat();
    const MatrixBase<float>& mdes = des.Mat();
    VectorBase<float>& vmatch = match->Vec();
    vmatch.Set(0);

    for(MatrixIndexT r=0; r<mout.NumRows(); r++) {
      MatrixIndexT i1=-1, i2=-1; 
      float v1=0.0, v2=0.0;
      for(MatrixIndexT c=0; c<mout.NumCols(); c++) {
        if(v1 < mout(r,c)) { v1=mout(r,c); i1=c; }
        if(v2 < mdes(r,c)) { v2=mdes(r,c); i2=c; }
      }
      if(i1==i2) { vmatch(r) = 1; }
    }
  }
}


void RegularizeL1(CuMatrix<float>* wei, CuMatrix<float>* grad, float l1, float lr) {
  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(wei->NumCols(),CUBLOCK), n_blocks(wei->NumRows(), CUBLOCK));

    cudaF_regularize_l1(dimGrid, dimBlock, wei->Data(), grad->Data(), l1, lr, wei->Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__,tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float>& wei2 = wei->Mat();
    MatrixBase<float>& grad2 = grad->Mat();
    for(MatrixIndexT r=0; r<wei2.NumRows(); r++) {
      for(MatrixIndexT c=0; c<wei2.NumCols(); c++) {
        
        if(wei2(r,c)==0.0) continue; //skip L1 if zero weight!

        BaseFloat l1_signed = l1;
        if(wei2(r,c) < 0.0) 
          l1_signed = -l1;

        BaseFloat before = wei2(r,c);
        BaseFloat after = wei2(r,c) -lr*grad2(r,c) -l1_signed;
        if((after > 0.0) ^ (before > 0.0)) {
          wei2(r,c) = 0.0;
          grad2(r,c) = 0.0;
        } else {
          wei2(r,c) -= l1_signed;
        }
      }
    }
  }
}


} //namespace cu

} //namespace kaldi
