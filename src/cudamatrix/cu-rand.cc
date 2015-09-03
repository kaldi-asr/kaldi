// cudamatrix/cu-rand.cc

// Copyright 2012  Karel Vesely
//           2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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


#include "base/kaldi-math.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "cudamatrix/cu-randkernels.h"


namespace kaldi {


template<typename Real> 
void CuRand<Real>::SeedGpu(MatrixIndexT state_size) {
  KALDI_ASSERT(state_size >= 0);
  state_size_ = state_size;
  SeedBuffer(state_size, &z1_);
  SeedBuffer(state_size, &z2_);
  SeedBuffer(state_size, &z3_);
  SeedBuffer(state_size, &z4_);
}


template<typename Real> 
void CuRand<Real>::SeedBuffer(MatrixIndexT state_size, uint32 **tgt) {
#if HAVE_CUDA == 1
  CuDevice &device = CuDevice::Instantiate();
  if (device.Enabled()) {
    if (*tgt != NULL) {
      device.Free(*tgt);
      *tgt = NULL;
    }
    if (state_size == 0) return; // Nothing to do.
    std::vector<uint32> temp_rand_data(state_size);
    for(MatrixIndexT i = 0; i < state_size; i++)
      temp_rand_data[i] = RandInt(128, RAND_MAX);
    int32 state_size_in_bytes = state_size * sizeof(uint32);
    *tgt = static_cast<uint32*>(device.Malloc(state_size_in_bytes));
    CU_SAFE_CALL(cudaMemcpy(*tgt, &(temp_rand_data[0]),
                            state_size_in_bytes, cudaMemcpyHostToDevice));
  }
#endif
}

template<class Real>
CuRand<Real>::~CuRand() {
  SeedBuffer(0, &z1_);
  SeedBuffer(0, &z2_);
  SeedBuffer(0, &z3_);
  SeedBuffer(0, &z4_);
}



template<typename Real> void CuRand<Real>::RandUniform(CuMatrixBase<Real> *tgt) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    int32 tgt_size = tgt->NumRows() * tgt->Stride();
    if (tgt_size != state_size_) SeedGpu(tgt_size);

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->num_cols_, CU2DBLOCK), n_blocks(tgt->num_rows_, CU2DBLOCK));

    cuda_rand(dimGrid, dimBlock, tgt->data_, z1_, z2_, z3_, z4_, tgt->Dim());
    CU_SAFE_CALL(cudaGetLastError());
  
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->SetRandUniform();
  }
}



template<typename Real> void CuRand<Real>::RandGaussian(CuMatrixBase<Real> *tgt) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    int32 tgt_size = tgt->NumRows() * tgt->Stride();
    if (tgt_size == 0)
      return;
    if (tgt_size > state_size_) SeedGpu(tgt_size);
    
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(tgt->num_cols_, CU2DBLOCK), n_blocks(tgt->num_rows_, CU2DBLOCK));
    
    cuda_gauss_rand(dimGrid, dimBlock, tgt->data_, z1_, z2_, z3_, z4_, tgt->Dim());
    CU_SAFE_CALL(cudaGetLastError());
  
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->SetRandn();
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
  tgt->AddMat(gscale, tmp_);
}

// Instantiate the class for float and double.
template class CuRand<float>;
template class CuRand<double>;

} // namespace




