// cudamatrix/cu-math-inl.h

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



#ifndef KALDI_CUDAMATRIX_CUMATH_INL_H_
#define KALDI_CUDAMATRIX_CUMATH_INL_H_

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"

namespace kaldi {

/**
 * Hide the CUDA kernel ANSI-C wrappers to subnamespace cu::
 */
namespace cu {

/*
 * templated functions wrapping the ANSI-C CUDA kernel functions 
 */
template<typename Real>
void Sigmoid(const CuMatrix<Real>& X, CuMatrix<Real>* Y) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(X.NumCols(), CUBLOCK), n_blocks(X.NumRows(), CUBLOCK));

    cuda_sigmoid(dimGrid, dimBlock, Y->Data(), X.Data(), X.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &y = Y->Mat();
    const MatrixBase<Real> &x = X.Mat();
    for(MatrixIndexT r=0; r<x.NumRows(); r++) {
      for(MatrixIndexT c=0; c<x.NumCols(); c++) {
        y(r, c) = 1.0/(1.0+exp(-x(r, c)));
      }
    }
  }
}



template<typename Real>
void DiffSigmoid(const CuMatrix<Real>& Ein, const CuMatrix<Real>& Y, CuMatrix<Real>* Eout) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(Eout->NumCols(), CUBLOCK), n_blocks(Eout->NumRows(), CUBLOCK));

    cuda_diff_sigmoid(dimGrid, dimBlock, Eout->Data(), Ein.Data(), Y.Data(), Eout->Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &eout = Eout->Mat();
    const MatrixBase<Real> &ein = Ein.Mat();
    const MatrixBase<Real> &y = Y.Mat();
    for(MatrixIndexT r=0; r<eout.NumRows(); r++) {
      for(MatrixIndexT c=0; c<eout.NumCols(); c++) {
        eout(r, c) = ein(r, c) * y(r, c)*(1.0-y(r, c));
      }
    }
  }
}


  
template<typename Real>
void Softmax(const CuMatrix<Real>& X, CuMatrix<Real>* Y) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    #if 0 
    // disable 'tree-reduce' functions, 
    // slower, but can be used for debugging
    size_t dimBlock = CUBLOCK;
    size_t dimGrid  = n_blocks(X.NumRows(), CUBLOCK);

    cuda_softmax(dimGrid, dimBlock, Y.Data(), X.Data(), X.Dim());
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
    cuda_softmax_part(dimGrid, dimBlock, X.Data(), max_id.Data(), Y->Data(), X.Dim()); 
    //sum the rows to get normalizers (tree reduction) 
    CuVector<Real> sum(X.NumRows());
    sum.AddColSumMat(1.0, *Y, 0.0);
    //divide by normalizers to get posteriors (grid kernel)
    Y->DivRowsVec(sum);
    #endif

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &y = Y->Mat();
    const MatrixBase<Real> &x = X.Mat();
    y.CopyFromMat(x);
    for(MatrixIndexT r=0; r<x.NumRows(); r++) {
      y.Row(r).ApplySoftMax();
    }
  }
}



template<typename Real>
void RegularizeL1(CuMatrix<Real> *wei, CuMatrix<Real> *grad, Real l1, Real lr) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(wei->NumCols(), CUBLOCK), n_blocks(wei->NumRows(), CUBLOCK));

    cuda_regularize_l1(dimGrid, dimBlock, wei->Data(), grad->Data(), l1, lr, wei->Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &wei2 = wei->Mat();
    MatrixBase<Real> &grad2 = grad->Mat();
    for(MatrixIndexT r=0; r<wei2.NumRows(); r++) {
      for(MatrixIndexT c=0; c<wei2.NumCols(); c++) {
        
        if(wei2(r,c)==0.0) continue; // skip L1 if zero weight!

        Real l1_signed = l1;
        if (wei2(r, c) < 0.0) 
          l1_signed = -l1;

        Real before = wei2(r, c);
        Real after = wei2(r, c) -lr*grad2(r, c) -l1_signed;
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



template<typename Real>
void FindRowMaxId(const CuMatrix<Real> &mat, CuStlVector<int32> *id) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
     
    // initialize the vectors
    CuVector<Real> max(mat.NumRows());
    max.Set(-1e21);
    id->Resize(mat.NumRows());
    id->Set(-1);

    MatrixDim d=mat.Dim();// only stride will be used!
   
    // process per 256 column blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumCols(); block++) {
      dim3 dimBlock(256, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset=block*256;

      cuda_find_row_max_id(dimGrid, dimBlock, mat.Data()+offset, max.Data(), id->Data(), offset, d);
    }
    
    // process the remainder
    int32 div = mat.NumCols() / 256;
    int32 mod = mat.NumCols() % 256;
    if (mod != 0) {
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset=div*256;
      
      cuda_find_row_max_id(dimGrid, dimBlock, mat.Data()+offset, max.Data(), id->Data(), offset, d);
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
      Real max = -1e21;
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



template<typename Real>
void DiffXent(const CuStlVector<int32> &tgt, CuMatrix<Real> *net_out_or_diff, CuVector<Real> *log_post_tgt) {

  assert(tgt.Dim() == net_out_or_diff->NumRows());
  log_post_tgt->Resize(tgt.Dim());

  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(1, CUBLOCK*8);
    dim3 dimGrid(1, n_blocks(tgt.Dim(), CUBLOCK*8));
    cuda_diff_xent(dimGrid, dimBlock, tgt.Data(), net_out_or_diff->Data(), log_post_tgt->Data(), net_out_or_diff->Dim());

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



template<typename Real>
void Randomize(const CuMatrix<Real> &src, const CuStlVector<int32> &copy_from_idx, CuMatrix<Real> *tgt) {

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

    cuda_randomize(dimGrid, dimBlock, tgt->Data(), src.Data(), copy_from_idx.Data(), dimtgt, dimsrc);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // randomize in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const std::vector<int32> &copy_from_idxvec = copy_from_idx.Vec();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    for(int32 i=0; i<copy_from_idx.Dim(); i++) {
      tgtmat.Row(i).CopyFromVec(srcmat.Row(copy_from_idxvec[i]));
    }
  }
} 



template<typename Real>
void Expand(const CuMatrix<Real> &src, const CuStlVector<int32> &frame_offsets, CuMatrix<Real> *tgt) {

  assert(src.NumCols()*frame_offsets.Dim() == tgt->NumCols());
  assert(src.NumRows() == tgt->NumRows());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CUBLOCK), n_blocks(tgt->NumRows(), CUBLOCK));
    
    cuda_expand(dimGrid, dimBlock, tgt->Data(), src.Data(), frame_offsets.Data(), tgt->Dim(), src.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // expand in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const std::vector<int32> &frame_offsetvec = frame_offsets.Vec();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    //
    for(int32 r=0; r < tgtmat.NumRows(); r++) {
      for(int32 off=0; off < frame_offsetvec.size(); off++) {
        int32 r_off = r + frame_offsetvec[off];
        if(r_off < 0) r_off = 0;
        if(r_off >= srcmat.NumRows()) r_off = srcmat.NumRows()-1;
        memcpy(tgtmat.RowData(r)+off*srcmat.NumCols(),srcmat.RowData(r_off),sizeof(Real)*srcmat.NumCols());
      }
    }
  }
}



template<typename Real>
void Copy(const CuMatrix<Real> &src, const CuStlVector<int32> &copy_from_indices, CuMatrix<Real> *tgt) { 

  assert(copy_from_indices.Dim() == tgt->NumCols());
  assert(src.NumRows() == tgt->NumRows());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CUBLOCK), n_blocks(tgt->NumRows(), CUBLOCK));
    
    cuda_copy(dimGrid, dimBlock, tgt->Data(), src.Data(), copy_from_indices.Data(), tgt->Dim(), src.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // expand in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const std::vector<int32> &copy_from_indicesvec = copy_from_indices.Vec();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    //
    for(int32 r=0; r < tgtmat.NumRows(); r++) {
      for(int32 c=0; c < copy_from_indicesvec.size(); c++) {
        tgtmat(r,c) = srcmat(r,copy_from_indicesvec[c]);
      }
    }
  }
}




} //namespace cu

} //namespace kaldi


#endif
