// cudafeat/online-ivector-feature-cuda.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef CUDAFEAT_ONLINE_IVECTOR_FEATURE_CUDA_H_
#define CUDAFEAT_ONLINE_IVECTOR_FEATURE_CUDA_H_

#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "cudafeat/feature-online-cmvn-cuda.h"
#include "cudamatrix/cu-matrix.h"
#include "online2/online-ivector-feature.h"

namespace kaldi {

class IvectorExtractorFastCuda {
 public:
  IvectorExtractorFastCuda(const OnlineIvectorExtractionConfig &config)
      : b_(0), tot_post_(2) {
    if (config.use_most_recent_ivector == false) {
      KALDI_WARN
          << "IvectorExractorFastCuda: Ignoring use_most_recent_ivector=false.";
    }
    if (config.greedy_ivector_extractor == false) {
      KALDI_WARN << "IvectorExractorFastCuda: Ignoring "
                    "greedy_ivector_extractor=false.";
    }

    info_.Init(config);
    naive_cmvn_state_ = OnlineCmvnState(info_.global_cmvn_stats);
    Read(config);
    cu_lda_.Resize(info_.lda_mat.NumRows(), info_.lda_mat.NumCols());
    cu_lda_.CopyFromMat(info_.lda_mat);

    // The last col in the LDA matrix may be an affine offset
    // copy that column to offset_ now.  This may or may not be used
    // when getting the features later
    offset_.Resize(cu_lda_.NumRows());
    offset_.CopyColFromMat(cu_lda_, cu_lda_.NumCols() - 1);
    d_info_ = static_cast<int *>(CuDevice::Instantiate().Malloc(sizeof(int)));
  }
  ~IvectorExtractorFastCuda() {
    KALDI_ASSERT(d_info_);
    CuDevice::Instantiate().Free(d_info_);
  }

  // This function goes directly from features to an i-vector
  // which makes the computation easier to port to GPU
  // and make it run more efficiently
  //
  // It is roughly the replacement for the following in kaldi:
  //
  // DiagGmm.LogLikelihoods(), VectorToPosteriorEntry()
  // IvectorExtractorUtteranceStats.AccStats()
  // IvectorExtractor.GetIvectorDistribution()
  //
  // Also note we only do single precision (float)
  // which will *NOT* give same results as kaldi
  // i-vector extractor which is float precision
  // however, in practice, the differences do *NOT*
  // affect overall accuracy
  //
  // This function is thread safe as all class variables
  // are read-only
  //
  void GetIvector(const CuMatrixBase<float> &feats, CuVector<float> *ivector);

  int32 FeatDim() const { return feat_dim_; }
  int32 IvectorDim() const { return ivector_dim_; }
  int32 NumGauss() const { return num_gauss_; }

 private:
  OnlineIvectorExtractionInfo info_;

  IvectorExtractorFastCuda(IvectorExtractorFastCuda const &);
  IvectorExtractorFastCuda &operator=(IvectorExtractorFastCuda const &);

  void Read(const kaldi::OnlineIvectorExtractionConfig &config);

  void SpliceFeats(const CuMatrixBase<BaseFloat> &feats,
                   CuMatrix<BaseFloat> *spliced_feats);

  void ComputePosteriors(const CuMatrixBase<float> &feats,
                         CuMatrix<float> *posteriors);

  void ComputeIvectorStats(const CuMatrixBase<float> &feats,
                           const CuMatrixBase<float> &posteriors,
                           CuVector<float> *gamma, CuMatrix<float> *X);

  void ComputeIvectorFromStats(const CuVector<float> &gamma,
                               const CuMatrix<float> &X,
                               CuVector<float> *ivector);

  CudaOnlineCmvnState naive_cmvn_state_;

  int32 feat_dim_;
  int32 ivector_dim_;
  int32 num_gauss_;

  // ubm variables
  CuVector<BaseFloat> ubm_gconsts_;
  CuMatrix<BaseFloat> ubm_means_inv_vars_;
  CuMatrix<BaseFloat> ubm_inv_vars_;
  CuMatrix<BaseFloat> cu_lda_;
  CuVector<BaseFloat> offset_;
  // extractor variables
  CuMatrix<BaseFloat> ie_U_;

  // Batched matrix which sotres this:
  CuMatrix<BaseFloat> ie_Sigma_inv_M_f_;

  // double buffer to store total posteriors.
  // double buffering avoids extra calls to intitialize buffer
  int b_;
  CuVector<BaseFloat> tot_post_;
  float prior_offset_;

  // Buffer used by cusolver
  int *d_info_;
};
}  // namespace kaldi

#endif  // IVECTOR_IVECTOR_EXTRACTOR_FAST_CUDA_H_
