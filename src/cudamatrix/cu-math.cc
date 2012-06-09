


#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-kernels.h"


namespace kaldi {
namespace cu {

/*
 * Float specializations
 */
void Sigmoid(const CuMatrix<float>& X, CuMatrix<float>* Y) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(X.NumCols(), CUBLOCK), n_blocks(X.NumRows(), CUBLOCK));

    cudaF_sigmoid(dimGrid, dimBlock, Y->Data(), X.Data(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float> &y = Y->Mat();
    const MatrixBase<float> &x = X.Mat();
    for(MatrixIndexT r=0; r<x.NumRows(); r++) {
      for(MatrixIndexT c=0; c<x.NumCols(); c++) {
        y(r, c) = 1.0/(1.0+exp(-x(r, c)));
      }
    }
  }
}


void DiffSigmoid(const CuMatrix<float>& Ein, const CuMatrix<float>& Y, CuMatrix<float>* Eout) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(Eout->NumCols(), CUBLOCK), n_blocks(Eout->NumRows(), CUBLOCK));

    cudaF_diff_sigmoid(dimGrid, dimBlock, Eout->Data(), Ein.Data(), Y.Data(), Eout->Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float> &eout = Eout->Mat();
    const MatrixBase<float> &ein = Ein.Mat();
    const MatrixBase<float> &y = Y.Mat();
    for(MatrixIndexT r=0; r<eout.NumRows(); r++) {
      for(MatrixIndexT c=0; c<eout.NumCols(); c++) {
        eout(r, c) = ein(r, c) * y(r, c)*(1.0-y(r, c));
      }
    }
  }
}

  
void Softmax(const CuMatrix<float>& X, CuMatrix<float>* Y) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    #if 0 
    // disable 'tree-reduce' functions, 
    // slower, but can be used for debugging
    size_t dimBlock = CUBLOCK;
    size_t dimGrid  = n_blocks(X.NumRows(), CUBLOCK);

    cudaF_softmax(dimGrid, dimBlock, Y.Data(), X.Data(), X.Dim());
    cuSafeCall(cudaGetLastError());
    #endif

    #if 1 
    // enable 'tree-reduce' functions, 
    //find maximum in each row (tree reduction)
    CuStlVector<int32> max_id;
    FindRowMaxId(X, &max_id); 
    //in each row subtract maximum, apply exp (grid kernel)
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(X.NumCols(), CUBLOCK), n_blocks(X.NumRows(), CUBLOCK));
    cudaF_softmax_part(dimGrid, dimBlock, X.Data(), max_id.Data(), Y->Data(), X.Dim()); 
    //sum the rows to get normalizers (tree reduction) 
    CuVector<float> sum(X.NumRows());
    sum.AddRowSum(1.0, *Y, 0.0);
    //divide by normalizers to get posteriors (grid kernel)
    Y->DivRowsVec(sum);
    #endif

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float> &y = Y->Mat();
    const MatrixBase<float> &x = X.Mat();
    y.CopyFromMat(x);
    for(MatrixIndexT r=0; r<x.NumRows(); r++) {
      y.Row(r).ApplySoftMax();
    }

  }
}


void RegularizeL1(CuMatrix<float> *wei, CuMatrix<float> *grad, float l1, float lr) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(wei->NumCols(), CUBLOCK), n_blocks(wei->NumRows(), CUBLOCK));

    cudaF_regularize_l1(dimGrid, dimBlock, wei->Data(), grad->Data(), l1, lr, wei->Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<float> &wei2 = wei->Mat();
    MatrixBase<float> &grad2 = grad->Mat();
    for(MatrixIndexT r=0; r<wei2.NumRows(); r++) {
      for(MatrixIndexT c=0; c<wei2.NumCols(); c++) {
        
        if(wei2(r,c)==0.0) continue; // skip L1 if zero weight!

        float l1_signed = l1;
        if (wei2(r, c) < 0.0) 
          l1_signed = -l1;

        float before = wei2(r, c);
        float after = wei2(r, c) -lr*grad2(r, c) -l1_signed;
        if ((after > 0.0) ^ (before > 0.0)) {
          wei2(r, c) = 0.0;
          grad2(r, c) = 0.0;
        } else {
          wei2(r, c) -= l1_signed;
        }
      }
    }
  }
}


void FindRowMaxId(const CuMatrix<float> &mat, CuStlVector<int32> *id) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
     
    // initialize the vectors
    CuVector<float> max(mat.NumRows());
    max.Set(-1e21);
    id->Resize(mat.NumRows());
    id->Set(-1);

    MatrixDim d=mat.Dim();// only stride will be used!
   
    // process per 256 column blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumCols(); block++) {
      dim3 dimBlock(256, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset=block*256;

      cudaF_find_row_max_id(dimGrid, dimBlock, mat.Data()+offset, max.Data(), id->Data(), offset, d);
    }
    
    // process the remainder
    int32 div = mat.NumCols() / 256;
    int32 mod = mat.NumCols() % 256;
    if (mod != 0) {
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset=div*256;
      
      cudaF_find_row_max_id(dimGrid, dimBlock, mat.Data()+offset, max.Data(), id->Data(), offset, d);
    }
    // now we have the indices!
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // allocate index buffer
    id->Resize(mat.NumRows());
    id->Set(-1);
    // find maxima
    for(int32 r=0; r<mat.NumRows(); r++) {
      float max = -1e21;
      int32 max_id = -1;
      for(int32 c=0; c<mat.NumCols(); c++) {
        if (max < mat.Mat()(r, c)) {
          max = mat.Mat()(r, c);
          max_id = c;
        }
      }
      id->Vec()[r] = max_id;
    }
  }
}


void DiffXent(const CuStlVector<int32> &tgt, CuMatrix<float> *net_out_or_diff, CuVector<float> *log_post_tgt) {

  assert(tgt.Dim() == net_out_or_diff->NumRows());
  log_post_tgt->Resize(tgt.Dim());

  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(1, CUBLOCK*8);
    dim3 dimGrid(1, n_blocks(tgt.Dim(), CUBLOCK*8));
    cudaF_diff_xent(dimGrid, dimBlock, tgt.Data(), net_out_or_diff->Data(), log_post_tgt->Data(), net_out_or_diff->Dim());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    for(int32 r=0; r<net_out_or_diff->NumRows(); r++) {
      int32 col_tgt = tgt.Vec()[r];
      log_post_tgt->Vec()(r) = log(net_out_or_diff->Mat()(r, col_tgt));
      net_out_or_diff->Mat()(r, col_tgt) -= 1.0;
    }
  }
}



void Randomize(const CuMatrix<float> &src, const CuStlVector<int32> &copy_from_idx, CuMatrix<float> *tgt) {

  assert(src.NumCols() == tgt->NumCols());
  assert(src.NumRows() == tgt->NumRows());
  assert(copy_from_idx.Dim() <= tgt->NumRows());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CUBLOCK), n_blocks(copy_from_idx.Dim(), CUBLOCK));
    
    MatrixDim dimsrc = src.Dim(); dimsrc.rows=copy_from_idx.Dim();
    MatrixDim dimtgt = tgt->Dim(); dimtgt.rows=copy_from_idx.Dim();

    cudaF_randomize(dimGrid, dimBlock, tgt->Data(), src.Data(), copy_from_idx.Data(), dimtgt, dimsrc);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // randomize in CPU
    const MatrixBase<float> &srcmat = src.Mat();
    const std::vector<int32> &copy_from_idxvec = copy_from_idx.Vec();
    MatrixBase<float> &tgtmat = tgt->Mat();
    for(int32 i=0; i<copy_from_idx.Dim(); i++) {
      tgtmat.Row(i).CopyFromVec(srcmat.Row(copy_from_idxvec[i]));
    }
  }
} 



} // namespace cu

} // namespace kaldi
