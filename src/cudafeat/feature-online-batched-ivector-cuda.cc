// cudafeat/feature-online-batched-ivector-cuda.cc
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

#include "cudafeat/feature-online-batched-ivector-cuda.h"
#include "cudafeat/feature-online-batched-ivector-cuda-kernels.h"

namespace kaldi {
BatchedIvectorExtractorCuda::BatchedIvectorExtractorCuda(
    const OnlineIvectorExtractionConfig &config, int32_t chunk_size,
    int32_t num_lanes, int32_t num_channels)
    : cmvn_(NULL),
      chunk_size_(chunk_size),
      max_lanes_(num_lanes),
      num_channels_(num_channels) {
#if CUDA_VERSION < 9010
  // some components require newer cuda versions.  If you see this error
  // upgrade to a more recent CUDA version.
  KALDI_ERR << "BatchedIvectorExtractorCuda requires CUDA 9.1 or newer.";
#endif
  info_.Init(config);
  Read(config);

  naive_cmvn_state_ = OnlineCmvnState(info_.global_cmvn_stats);
  // TODO parameterize coarsening factor?
  cmvn_ = new CudaOnlineBatchedCmvn(info_.cmvn_opts, naive_cmvn_state_,
                                    feat_dim_, chunk_size_, num_channels_, 1);
  cu_lda_.Resize(info_.lda_mat.NumRows(), info_.lda_mat.NumCols());
  cu_lda_.CopyFromMat(info_.lda_mat);

  // The last col in the LDA matrix may be an affine offset
  // copy that column to offset_ now.  This may or may not be used
  // when getting the features later
  offset_.Resize(cu_lda_.NumRows());
  offset_.CopyColFromMat(cu_lda_, cu_lda_.NumCols() - 1);

  int left = info_.splice_opts.left_context;
  int right = info_.splice_opts.right_context;
  int size = right + left + 1;

  // resize temporary memory
  feats_stash_.Resize(num_channels_ * (left + right), feat_dim_, kUndefined);
  norm_feats_stash_.Resize(num_channels_ * (left + right), feat_dim_,
                           kUndefined);
  spliced_feats_.Resize(num_lanes * chunk_size, feat_dim_ * size, kUndefined);
  tmp_feats_.Resize(num_lanes * chunk_size, feat_dim_, kUndefined);
  posteriors_.Resize(num_lanes * chunk_size, num_gauss_, kUndefined);

  gamma_.Resize(num_lanes * num_gauss_, kUndefined);
  gamma_stash_.Resize(num_channels * num_gauss_, kUndefined);

  X_.Resize(num_lanes * num_gauss_, feat_dim_, kUndefined);
  X_stash_.Resize(num_channels * num_gauss_, feat_dim_, kUndefined);

  linear_.Resize(num_lanes * ivector_dim_);
  sp_quadratic_.Resize(num_lanes, ivector_dim_ * (ivector_dim_ + 1) / 2);
  quadratic_.Resize(num_lanes, ivector_dim_ * ivector_dim_);

  d_infoArray_ = static_cast<int *>(
      CuDevice::Instantiate().Malloc(num_lanes * sizeof(int)));
  quad_array_ = static_cast<BaseFloat **>(
      CuDevice::Instantiate().Malloc(num_lanes * sizeof(BaseFloat *)));
  ivec_array_ = static_cast<BaseFloat **>(
      CuDevice::Instantiate().Malloc(num_lanes * sizeof(BaseFloat *)));

  std::vector<BaseFloat *> h_quad_array(num_lanes), h_ivec_array(num_lanes);
  int32_t qstride = quadratic_.Stride();
  int32_t istride = ivector_dim_;
  for (int lane = 0; lane < num_lanes; lane++) {
    h_quad_array[lane] = quadratic_.Data() + lane * qstride;
    h_ivec_array[lane] = linear_.Data() + lane * istride;
  }
  cudaMemcpyAsync(quad_array_, &h_quad_array[0],
                  sizeof(BaseFloat *) * num_lanes, cudaMemcpyHostToDevice,
                  cudaStreamPerThread);
  cudaMemcpyAsync(ivec_array_, &h_ivec_array[0],
                  sizeof(BaseFloat *) * num_lanes, cudaMemcpyHostToDevice,
                  cudaStreamPerThread);
}
BatchedIvectorExtractorCuda::~BatchedIvectorExtractorCuda() {
  delete cmvn_;
  CuDevice::Instantiate().Free(d_infoArray_);
  CuDevice::Instantiate().Free(quad_array_);
  CuDevice::Instantiate().Free(ivec_array_);
}

void BatchedIvectorExtractorCuda::Read(
    const kaldi::OnlineIvectorExtractionConfig &config) {
  // read ubm
  DiagGmm gmm;
  ReadKaldiObject(config.diag_ubm_rxfilename, &gmm);
  ubm_gconsts_.Resize(gmm.NumGauss());
  ubm_gconsts_.CopyFromVec(gmm.gconsts());
  ubm_means_inv_vars_.Resize(gmm.NumGauss(), gmm.Dim());
  ubm_means_inv_vars_.CopyFromMat(gmm.means_invvars());
  ubm_inv_vars_.Resize(gmm.NumGauss(), gmm.Dim());
  ubm_inv_vars_.CopyFromMat(gmm.inv_vars());
  num_gauss_ = gmm.NumGauss();

  // read extractor (copied from ivector/ivector-extractor.cc)
  bool binary;
  Input input(config.ivector_extractor_rxfilename, &binary);
  Matrix<float> w;
  Vector<float> w_vec;
  std::vector<Matrix<float> > ie_M;
  std::vector<SpMatrix<float> > ie_Sigma_inv;

  ExpectToken(input.Stream(), binary, "<IvectorExtractor>");
  ExpectToken(input.Stream(), binary, "<w>");
  w.Read(input.Stream(), binary);
  ExpectToken(input.Stream(), binary, "<w_vec>");
  w_vec.Read(input.Stream(), binary);
  ExpectToken(input.Stream(), binary, "<M>");
  int32 size;
  ReadBasicType(input.Stream(), binary, &size);
  KALDI_ASSERT(size > 0);
  ie_M.resize(size);
  for (int32 i = 0; i < size; i++) {
    ie_M[i].Read(input.Stream(), binary);
  }
  ExpectToken(input.Stream(), binary, "<SigmaInv>");
  ie_Sigma_inv.resize(size);
  for (int32 i = 0; i < size; i++) {
    ie_Sigma_inv[i].Read(input.Stream(), binary);
  }
  ExpectToken(input.Stream(), binary, "<IvectorOffset>");
  ReadBasicType(input.Stream(), binary, &prior_offset_);
  ExpectToken(input.Stream(), binary, "</IvectorExtractor>");

  // compute derived variables
  ivector_dim_ = ie_M[0].NumCols();
  feat_dim_ = ie_M[0].NumRows();

  ie_Sigma_inv_M_f_.Resize(num_gauss_ * feat_dim_, ivector_dim_, kUndefined);

  ie_U_.Resize(num_gauss_, ivector_dim_ * (ivector_dim_ + 1) / 2);

  SpMatrix<float> tmp_sub_U(ivector_dim_);
  Matrix<float> tmp_Sigma_inv_M(feat_dim_, ivector_dim_);
  for (int32 i = 0; i < num_gauss_; i++) {
    // compute matrix ie_Sigma_inv_M[i]
    tmp_sub_U.AddMat2Sp(1, ie_M[i], kTrans, ie_Sigma_inv[i], 0);
    SubVector<float> tmp_U_vec(tmp_sub_U.Data(),
                               ivector_dim_ * (ivector_dim_ + 1) / 2);
    ie_U_.Row(i).CopyFromVec(tmp_U_vec);

    tmp_Sigma_inv_M.AddSpMat(1, ie_Sigma_inv[i], ie_M[i], kNoTrans, 0);

    // copy into global matrix
    CuSubMatrix<float> window(ie_Sigma_inv_M_f_, i * feat_dim_, feat_dim_, 0,
                              ivector_dim_);
    window.CopyFromMat(tmp_Sigma_inv_M);
  }
}

void BatchedIvectorExtractorCuda::GetIvectors(
    const CuMatrixBase<BaseFloat> &feats, CuVectorBase<BaseFloat> *ivectors,
    const LaneDesc *lanes, int32_t num_lanes) {
  InitializeChannels(lanes, num_lanes);

  // normalized pipeline
  {
    // cmvn feats and store in tmp_feats_
    cmvn_->ComputeFeaturesBatched(num_lanes, lanes, feats, &tmp_feats_);

    // splice normalized feats
    SpliceFeats(tmp_feats_, norm_feats_stash_, &spliced_feats_, lanes,
                num_lanes);

    // Stash feats
    StashFeats(tmp_feats_, &norm_feats_stash_, lanes, num_lanes);

    // LDA transform spliced feats back into tmp_feats
    LDATransform(spliced_feats_, &tmp_feats_, lanes, num_lanes);

    // compute posteriors based normalized lda feats
    ComputePosteriors(tmp_feats_, lanes, num_lanes);
  }

  // non-normalized pipeline
  {
    // splice non-normalized feats into spliced feats_
    SpliceFeats(feats, feats_stash_, &spliced_feats_, lanes, num_lanes);

    // Stash feats
    StashFeats(feats, &feats_stash_, lanes, num_lanes);

    // LDA transform spliced feats back into tmp_feats
    LDATransform(spliced_feats_, &tmp_feats_, lanes, num_lanes);
  }

  // compute ivector stats
  ComputeIvectorStats(tmp_feats_, lanes, num_lanes);

  // compute ivectors for the stats
  ComputeIvectorsFromStats(ivectors, lanes, num_lanes);
}

void BatchedIvectorExtractorCuda::InitializeChannels(const LaneDesc *lanes,
                                                     int32_t num_lanes) {
  initialize_channels(num_gauss_, feat_dim_, gamma_stash_.Data(), num_gauss_,
                      X_stash_.Data(), X_stash_.Stride(),
                      X_stash_.Stride() * num_gauss_, lanes, num_lanes);
}

void BatchedIvectorExtractorCuda::SpliceFeats(
    const CuMatrixBase<BaseFloat> &feats,
    const CuMatrix<BaseFloat> &feats_stash, CuMatrix<BaseFloat> *spliced_feats,
    const LaneDesc *lanes, int32_t num_lanes) {
  int left = info_.splice_opts.left_context;
  int right = info_.splice_opts.right_context;

  splice_features_batched(
      chunk_size_, feat_dim_, left, right, feats.Data(), feats.Stride(),
      feats.Stride() * chunk_size_, feats_stash.Data(), feats_stash.Stride(),
      feats_stash.Stride() * (left + right), spliced_feats->Data(),
      spliced_feats->Stride(), spliced_feats->Stride() * chunk_size_, lanes,
      num_lanes);
}

void BatchedIvectorExtractorCuda::StashFeats(
    const CuMatrixBase<BaseFloat> &feats, CuMatrix<BaseFloat> *feats_stash,
    const LaneDesc *lanes, int32_t num_lanes) {
  int left = info_.splice_opts.left_context;
  int right = info_.splice_opts.right_context;

  stash_feats(chunk_size_, feats.Data(), feat_dim_, feats.Stride(),
              feats.Stride() * chunk_size_, feats_stash->Data(), left + right,
              feats_stash->Stride(), feats_stash->Stride() * (left + right),
              lanes, num_lanes);
}

void BatchedIvectorExtractorCuda::LDATransform(const CuMatrix<BaseFloat> &feats,
                                               CuMatrix<BaseFloat> *lda_feats,
                                               const LaneDesc *lanes,
                                               int32_t num_lanes) {
  if (feats.NumCols() == cu_lda_.NumCols()) {
    // linear transformation
    lda_feats->AddMatMat(1.0, feats, kNoTrans, cu_lda_, kTrans, 0.0);
  } else {
    // affine transformation
    int lda_rows = cu_lda_.NumRows();
    int lda_cols = cu_lda_.NumCols();
    // create submatrix which removes last column
    CuSubMatrix<BaseFloat> cu_lda(cu_lda_, 0, lda_rows, 0, lda_cols - 1);
    lda_feats->CopyRowsFromVec(offset_);
    lda_feats->AddMatMat(1.0, feats, kNoTrans, cu_lda, kTrans, 1.0);
  }
}

void BatchedIvectorExtractorCuda::ComputePosteriors(CuMatrix<BaseFloat> &feats,
                                                    const LaneDesc *lanes,
                                                    int32_t num_lanes) {
  int right = info_.splice_opts.right_context;

  // inititalize posteriors
  posteriors_.CopyRowsFromVec(ubm_gconsts_);

  // add in normamalized feats * umb_means_inv
  posteriors_.AddMatMat(1.0, feats, kNoTrans, ubm_means_inv_vars_, kTrans, 1.0);

  // square feats
  square_batched_matrix(chunk_size_, feat_dim_, feats.Data(), feats.Stride(),
                        feats.Stride() * chunk_size_, feats.Data(),
                        feats.Stride(), feats.Stride() * chunk_size_, lanes,
                        num_lanes);

  // add in feats .^2 * umb_inv_vars
  posteriors_.AddMatMat(-0.5, feats, kNoTrans, ubm_inv_vars_, kTrans, 1.0);

  posteriors_.ApplySoftMaxPerRow();

  // At this point some rows of posteriors are invalid because they
  // didn't have valid input rows.  Zero those out now so that
  // they don't impact stats
  zero_invalid_posteriors(
      chunk_size_, num_gauss_, posteriors_.Data(), posteriors_.Stride(),
      posteriors_.Stride() * chunk_size_, right, lanes, num_lanes);
}

void BatchedIvectorExtractorCuda::ComputeIvectorStats(
    const CuMatrix<BaseFloat> &feats, const LaneDesc *lanes,
    int32_t num_lanes) {
  batched_sum_posteriors(chunk_size_, num_gauss_, posteriors_.Data(),
                         posteriors_.Stride(),
                         posteriors_.Stride() * chunk_size_, gamma_.Data(),
                         num_gauss_, info_.posterior_scale, lanes, num_lanes);

#if CUDA_VERSION >= 9010
  int32_t m = feat_dim_;
  int32_t n = num_gauss_;
  int32_t k = chunk_size_;
  float alpha = info_.posterior_scale;
  float beta = 0.0f;
  const float *A = feats.Data();
  int32_t lda = feats.Stride();
  int32_t strideA = lda * chunk_size_;
  const float *B = posteriors_.Data();
  int32_t ldb = posteriors_.Stride();
  int32_t strideB = ldb * chunk_size_;
  float *C = X_.Data();
  int32_t ldc = X_.Stride();
  int32_t strideC = ldc * num_gauss_;

  // multiplying X = post * feats
  CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(
      GetCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A,
      CUDA_R_32F, lda, strideA, B, CUDA_R_32F, ldb, strideB, &beta, C,
      CUDA_R_32F, ldc, strideC, num_lanes, CUDA_R_32F, CUBLAS_GEMM_DEFAULT))
#endif

  apply_and_update_stash(
      num_gauss_, feat_dim_, gamma_.Data(), gamma_stash_.Data(), num_gauss_,
      X_.Data(), X_.Stride(), X_.Stride() * num_gauss_, X_stash_.Data(),
      X_stash_.Stride(), X_stash_.Stride() * num_gauss_, lanes, num_lanes);
}

void BatchedIvectorExtractorCuda::ComputeIvectorsFromStats(
    CuVectorBase<BaseFloat> *ivectors, const LaneDesc *lanes,
    int32_t num_lanes) {
  // Computing Linear Term
  {
    // need to set this term to zero because batched_compute_linear_term
    // uses atomics with a +=
    linear_.SetZero();
    batched_compute_linear_term(num_gauss_, feat_dim_, ivector_dim_,
                                ie_Sigma_inv_M_f_.Data(),
                                ie_Sigma_inv_M_f_.Stride(), X_.Data(),
                                X_.Stride(), X_.Stride() * num_gauss_,
                                linear_.Data(), ivector_dim_, lanes, num_lanes);
  }  // end linear term

  // Computing Quadratic Term
  {
    // Convert  gamma from Vector to Matrix
    CuSubMatrix<BaseFloat> gamma(gamma_.Data(), num_lanes, num_gauss_,
                                 num_gauss_);
    CuSubMatrix<BaseFloat> sp_quadratic(sp_quadratic_.RowRange(0, num_lanes));
    //  compute quadratic (batch_size x (ivector_dim * (ivector_dim + 1) / 2))
    sp_quadratic.AddMatMat(1.0f, gamma, kNoTrans, ie_U_, kNoTrans, 0.0f);

    // copy a result sp_quadratic into quadratic_
    batched_convert_sp_to_dense(
        ivector_dim_, sp_quadratic_.Data(), sp_quadratic_.Stride(),
        quadratic_.Data(), ivector_dim_, quadratic_.Stride(), lanes, num_lanes);
  }

  // compute and apply prior offset to linear and quadraditic terms
  batched_update_linear_and_quadratic_terms(
      ivector_dim_, prior_offset_, info_.posterior_scale, info_.max_count,
      quadratic_.Data(), ivector_dim_, quadratic_.Stride(), linear_.Data(),
      ivector_dim_, lanes, num_lanes);

#if CUDA_VERSION >= 9010
  int nrhs = 1;
  // perform factorization in batched
  CUSOLVER_SAFE_CALL(cusolverDnSpotrfBatched(
      GetCusolverDnHandle(), CUBLAS_FILL_MODE_LOWER, ivector_dim_, quad_array_,
      ivector_dim_, d_infoArray_, num_lanes));

  // solve for rhs in batched
  CUSOLVER_SAFE_CALL(cusolverDnSpotrsBatched(
      GetCusolverDnHandle(), CUBLAS_FILL_MODE_LOWER, ivector_dim_, nrhs,
      quad_array_, ivector_dim_, ivec_array_, ivector_dim_, d_infoArray_,
      num_lanes));
#endif

  // cusolver solves in place.  Ivectors are now in linear_

  // Create a submatrix which points to the first element of each ivector
  CuSubMatrix<BaseFloat> ivector0(linear_.Data(), num_lanes, 1, ivector_dim_);
  // remove prior
  ivector0.Add(-prior_offset_);

  // output was written to ivectors_ now copy that into output array
  cudaMemcpyAsync(ivectors->Data(), linear_.Data(),
                  ivector_dim_ * num_lanes * sizeof(BaseFloat),
                  cudaMemcpyDeviceToDevice, cudaStreamPerThread);
}

}  // namespace kaldi
