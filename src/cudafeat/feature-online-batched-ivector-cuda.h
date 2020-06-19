// cudafeat/feature-online-batched-ivector-cuda.h
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef KALDI_CUDAFEAT_BATCHED_FEATURE_ONLINE_IVECTOR_CUDA_H_
#define KALDI_CUDAFEAT_BATCHED_FEATURE_ONLINE_IVECTOR_CUDA_H_

#include "cudafeat/feature-online-batched-cmvn-cuda.h"
#include "cudafeat/lane-desc.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/online-feature.h"
#include "online2/online-ivector-feature.h"

namespace kaldi {

class BatchedIvectorExtractorCuda {
 public:
  BatchedIvectorExtractorCuda(const OnlineIvectorExtractionConfig &config,
                              int32_t chunk_size, int32_t num_lanes,
                              int32_t num_channels);
  ~BatchedIvectorExtractorCuda();

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
  // i-vector extractor which is double precision
  // however, in practice, the differences do *NOT*
  // affect overall accuracy
  //
  // This function is thread safe as all class variables
  // are read-only
  //
  void GetIvectors(const CuMatrixBase<BaseFloat> &feats,
                   CuVectorBase<BaseFloat> *ivectors, const LaneDesc *lanes,
                   int32_t num_lanes);

  int32 FeatDim() const { return feat_dim_; }
  int32 IvectorDim() const { return ivector_dim_; }
  int32 NumGauss() const { return num_gauss_; }

 private:
  OnlineIvectorExtractionInfo info_;

  BatchedIvectorExtractorCuda(BatchedIvectorExtractorCuda const &);
  BatchedIvectorExtractorCuda &operator=(BatchedIvectorExtractorCuda const &);

  void Read(const kaldi::OnlineIvectorExtractionConfig &config);

  void InitializeChannels(const LaneDesc *lanes, int32_t num_lanes);

  // Reads from feats, splice based on left/right contex,
  // and writes to spliced_feats
  void SpliceFeats(const CuMatrixBase<BaseFloat> &feats,
                   const CuMatrix<BaseFloat> &feats_stash,
                   CuMatrix<BaseFloat> *spliced_feats, const LaneDesc *lanes,
                   int32_t num_lanes);

  // Stores the left context of features for use in the
  // next chunk.
  void StashFeats(const CuMatrixBase<BaseFloat> &feats,
                  CuMatrix<BaseFloat> *feats_stash, const LaneDesc *lanes,
                  int32_t num_lanes);

  // Performs LDA transform on spliced_feat and writes
  // to lda_feats
  void LDATransform(const CuMatrix<BaseFloat> &feats,
                    CuMatrix<BaseFloat> *lda_feats, const LaneDesc *lanes,
                    int32_t num_lanes);

  // Computes posteriors_ based on feats.  This
  // is destructive on feats
  void ComputePosteriors(CuMatrix<BaseFloat> &feats, const LaneDesc *lanes,
                         int32_t num_lanes);

  // Computes Ivector stats based on posteriors_ and feats
  void ComputeIvectorStats(const CuMatrix<BaseFloat> &feats,
                           const LaneDesc *lanes, int32_t num_lanes);

  // Computes Ivectors based on precomputed stats
  void ComputeIvectorsFromStats(CuVectorBase<BaseFloat> *ivectors,
                                const LaneDesc *lanes, int32_t num_lanes);

  CudaOnlineCmvnState naive_cmvn_state_;
  CudaOnlineBatchedCmvn *cmvn_;
  int32_t feat_dim_;
  int32_t ivector_dim_;
  int32_t num_gauss_;

  // ubm variables
  CuVector<BaseFloat> ubm_gconsts_;
  CuMatrix<BaseFloat> ubm_means_inv_vars_;
  CuMatrix<BaseFloat> ubm_inv_vars_;
  CuMatrix<BaseFloat> cu_lda_;
  CuVector<BaseFloat> offset_;
  // extractor variables
  CuMatrix<BaseFloat> ie_U_;
  // Batched matrix which stores this:
  CuMatrix<BaseFloat> ie_Sigma_inv_M_f_;

  // temporary memory unique per batch element
  CuMatrix<BaseFloat> spliced_feats_;
  CuMatrix<BaseFloat> tmp_feats_;
  CuMatrix<BaseFloat> posteriors_;
  CuMatrix<BaseFloat> X_;
  CuVector<BaseFloat> gamma_;
  CuVector<BaseFloat> linear_;
  CuMatrix<BaseFloat> quadratic_;
  CuMatrix<BaseFloat> sp_quadratic_;

  // Stash for features
  CuMatrix<BaseFloat> feats_stash_;
  CuMatrix<BaseFloat> norm_feats_stash_;
  CuMatrix<BaseFloat> X_stash_;
  CuVector<BaseFloat> gamma_stash_;

  // Buffers used by cusolver
  int *d_infoArray_;
  BaseFloat **quad_array_;
  BaseFloat **ivec_array_;

  float prior_offset_;
  int32_t chunk_size_;
  int32_t max_lanes_;
  int32_t num_channels_;
};

}  // namespace kaldi

#endif
