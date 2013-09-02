// cudamatrix/cu-rand-inl.h

// Copyright 2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CU_RAND_INL_H_
#define KALDI_CUDAMATRIX_CU_RAND_INL_H_

#include "base/kaldi-math.h"

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-rand.h"
#include "cudamatrix/cu-randkernels.h"


namespace kaldi {


template<typename Real> 
void CuRand<Real>::SeedGpu(MatrixIndexT state_size) {
  if(NULL != host_) delete[] host_; 
  host_ = new uint32[state_size]; 
  host_size_ = state_size;

  SeedBuffer(&z1_, state_size);
  SeedBuffer(&z2_, state_size);
  SeedBuffer(&z3_, state_size);
  SeedBuffer(&z4_, state_size);
  state_size_ = state_size;

  delete[] host_;
  host_ = NULL;
  host_size_ = 0;
}



template<typename Real> 
void CuRand<Real>::SeedBuffer(uint32* *tgt, MatrixIndexT state_size) {
  // generate random state
  for(MatrixIndexT i = 0; i < host_size_; i++) {
    host_[i] = RandInt(128, RAND_MAX);
  }
  #if HAVE_CUDA == 1
  // push it to the GPU
  if (CuDevice::Instantiate().Enabled()) {
    int32 state_size_in_bytes = state_size*sizeof(uint32);
    // resize the GPU buffer
    if (state_size_ != state_size) {
      cudaFree(*tgt);
      cudaMalloc((void**)tgt, state_size_in_bytes);
    }
    // copy the values
    cudaMemcpy(*tgt, host_, state_size_in_bytes, cudaMemcpyHostToDevice);
  } else
  #endif
  { // use back-off host buffer
    if (state_size_ != state_size) {
      delete[] (*tgt);
      *tgt = new uint32[state_size];
    }
    int32 state_size_in_bytes = state_size*sizeof(uint32);
    memcpy(*tgt, host_, state_size_in_bytes);
  }
}



template<typename Real> void CuRand<Real>::RandUniform(CuMatrix<Real> *tgt) {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    int32 tgt_size = tgt->NumRows()*tgt->Stride();
    if (tgt_size != state_size_) SeedGpu(tgt_size);

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->num_cols_, CU2DBLOCK), n_blocks(tgt->num_rows_, CU2DBLOCK));

    cuda_rand(dimGrid, dimBlock, tgt->data_, z1_, z2_, z3_, z4_, tgt->Dim());
    CU_SAFE_CALL(cudaGetLastError());
  
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    for(int32 r=0; r<tgt->NumRows(); r++) {
      for(int32 c=0; c<tgt->num_cols_; c++) {
        tgt->Mat()(r, c) = kaldi::RandUniform();
      }
    }
  }
}



template<typename Real> void CuRand<Real>::RandGaussian(CuMatrixBase<Real> *tgt) {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    int32 tgt_size = tgt->NumRows()*tgt->Stride();
    if (tgt_size != state_size_) SeedGpu(tgt_size);
    
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->num_cols_, CU2DBLOCK), n_blocks(tgt->num_rows_, CU2DBLOCK));
    
    cuda_gauss_rand(dimGrid, dimBlock, tgt->data_, z1_, z2_, z3_, z4_, tgt->Dim());
    CU_SAFE_CALL(cudaGetLastError());
  
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    for(int32 r=0; r<tgt->NumRows(); r++) {
      for(int32 c=0; c<tgt->num_cols_; c++) {
        tgt->Mat()(r, c) = RandGauss();
      }
    }
  }
}


template<typename Real> void CuRand<Real>::RandGaussian(CuVectorBase<Real> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    int32 tgt_size = tgt->Dim();
    if (tgt_size != state_size_) SeedGpu(tgt_size);

    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(tgt->Dim(), CU1DBLOCK));
    
    cuda_vec_gauss_rand(dimGrid, dimBlock, tgt->Data(), z1_, z2_, z3_, z4_, tgt->Dim());

    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    
  } else
#endif
  {
    tgt->Vec().SetRandn();
  }
}


template<typename Real> void CuRand<Real>::BinarizeProbs(const CuMatrix<Real> &probs, CuMatrix<Real> *states) {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    // optionally re-seed the inner state 
    // (this is done in host, for good performance it is better to avoid re-seeding)
    int32 tgt_size = probs.num_rows_ * probs.stride_;
    if (tgt_size != state_size_) SeedGpu(tgt_size);

    // prepare the output matrix
    if (states != &probs)
      states->Resize(probs.num_rows_, probs.num_cols_, kUndefined);
    // prepare the temporary matrix of uniform random numbers (0,1)
    tmp_.Resize(probs.num_rows_, probs.num_cols_, kUndefined);
    RandUniform(&tmp_);

    // use the uniform random numbers to compute discrete 0/1 states
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(states->num_cols_, CU2DBLOCK), n_blocks(states->num_rows_, CU2DBLOCK));

    cuda_binarize_probs(dimGrid, dimBlock, states->data_, probs.data_, tmp_.data_, states->Dim());
    CU_SAFE_CALL(cudaGetLastError());
  
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    for(int32 r=0; r<states->num_rows_; r++) {
      for(int32 c=0; c<states->num_cols_; c++) {
        states->Mat()(r, c) = ((kaldi::RandUniform() < probs.Mat()(r, c))? 1 : 0 );
      }
    }
  }
}



template<typename Real> void CuRand<Real>::AddGaussNoise(CuMatrix<Real> *tgt, Real gscale) {
  tmp_.Resize(tgt->num_rows_, tgt->num_cols_);
  RandGaussian(&tmp_);
  tgt->AddMat(gscale, tmp_, 1.0);
}



} // namespace

#endif


