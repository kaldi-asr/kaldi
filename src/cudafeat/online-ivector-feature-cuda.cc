// cudafeat/online-ivector-feature-cuda.cc
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

#if HAVE_CUDA == 1
#include <nvToolsExt.h>
#endif
#include <iostream>

#include "base/io-funcs.h"
#include "base/kaldi-common.h"
#include "base/timer.h"
#include "cudafeat/feature-online-cmvn-cuda.h"
#include "cudafeat/online-ivector-feature-cuda-kernels.h"
#include "cudafeat/online-ivector-feature-cuda.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "gmm/diag-gmm.h"
#include "util/kaldi-io.h"
#include "util/table-types.h"
namespace kaldi {

void IvectorExtractorFastCuda::GetIvector(const CuMatrixBase<BaseFloat> &feats,
                                          CuVector<BaseFloat> *ivector) {
  nvtxRangePushA("GetIvector");
  CuMatrix<BaseFloat> posteriors, X;
  CuVector<BaseFloat> gamma;
  int rows = feats.NumRows();
  int cols = feats.NumCols();

  int lda_rows = cu_lda_.NumRows();
  int lda_cols = cu_lda_.NumCols();

  // normalized pipeline
  CuMatrix<BaseFloat> lda_feats_normalized(rows, lda_rows, kUndefined);
  {
    CudaOnlineCmvn cmvn(info_.cmvn_opts, naive_cmvn_state_);
    CuMatrix<BaseFloat> cmvn_feats(rows, cols, kUndefined);
    CuMatrix<BaseFloat> spliced_feats_normalized;

    // Normalize
    cmvn.ComputeFeatures(feats, &cmvn_feats);

    // Splice
    SpliceFeats(cmvn_feats, &spliced_feats_normalized);

    // Transform by LDA matrix
    if (spliced_feats_normalized.NumCols() == lda_cols) {
      // Linear transformation
      lda_feats_normalized.AddMatMat(1.0, spliced_feats_normalized, kNoTrans,
                                     cu_lda_, kTrans, 0.0);
    } else if (spliced_feats_normalized.NumCols() + 1 == lda_cols) {
      // Affine transformation

      // create submatrix which removes last column
      CuSubMatrix<BaseFloat> cu_lda(cu_lda_, 0, lda_rows, 0, lda_cols - 1);

      // Add offset
      lda_feats_normalized.CopyRowsFromVec(offset_);
      lda_feats_normalized.AddMatMat(1.0, spliced_feats_normalized, kNoTrans,
                                   cu_lda, kTrans, 1.0);

    } else {
      KALDI_ERR << "Dimension mismatch: source features have dimension "
                << spliced_feats_normalized.NumCols() << " and LDA #cols is "
                << lda_cols;
    }
  }

  // non-normalized pipeline
  CuMatrix<BaseFloat> lda_feats(rows, lda_rows, kUndefined);
  {
    CuMatrix<BaseFloat> spliced_feats;

    // Splice feats
    SpliceFeats(feats, &spliced_feats);

    // Transform by LDA matrix
    if (spliced_feats.NumCols() == lda_cols) {
      // Linear transformation
      lda_feats.AddMatMat(1.0, spliced_feats, kNoTrans, cu_lda_, kTrans, 0.0);
    } else if (spliced_feats.NumCols() + 1 == lda_cols) {
      // Affine transformation

      // create submatrix which removes last column
      CuSubMatrix<BaseFloat> cu_lda(cu_lda_, 0, lda_rows, 0, lda_cols - 1);

      // Add offset
      lda_feats.CopyRowsFromVec(offset_);
      lda_feats.AddMatMat(1.0, spliced_feats, kNoTrans, cu_lda, kTrans, 1.0);

    } else {
      KALDI_ERR << "Dimension mismatch: source features have dimension "
                << spliced_feats.NumCols() << " and LDA #cols is "
                << lda_cols;
    }
  }

  // based on normalized feats
  ComputePosteriors(lda_feats_normalized, &posteriors);

  // based on non-normalized feats
  ComputeIvectorStats(lda_feats, posteriors, &gamma, &X);

  ComputeIvectorFromStats(gamma, X, ivector);

  nvtxRangePop();
}

void IvectorExtractorFastCuda::Read(
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

  ie_Sigma_inv_M_f_.Resize(num_gauss_ * feat_dim_, ivector_dim_);

  ie_U_.Resize(num_gauss_, ivector_dim_ * (ivector_dim_ + 1) / 2);

  SpMatrix<float> tmp_sub_U(ivector_dim_);
  Matrix<float> tmp_Sigma_inv_M(feat_dim_, ivector_dim_);
  for (int32 i = 0; i < num_gauss_; i++) {
    // compute matrix ie_Sigma_inv_M[i[
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

void IvectorExtractorFastCuda::SpliceFeats(const CuMatrixBase<BaseFloat> &feats,
                                           CuMatrix<BaseFloat> *spliced_feats) {
  int left = -info_.splice_opts.left_context;
  int right = info_.splice_opts.right_context;
  int size = right - left + 1;
  spliced_feats->Resize(feats.NumRows(), feats.NumCols() * size, kUndefined);

  splice_features(feats.NumRows(), feats.NumCols(), left, size, feats.Data(),
                  feats.Stride(), spliced_feats->Data(),
                  spliced_feats->Stride());
}

void IvectorExtractorFastCuda::ComputePosteriors(
    const CuMatrixBase<float> &feats, CuMatrix<float> *posteriors) {
  int num_frames = feats.NumRows();

  posteriors->Resize(num_frames, num_gauss_, kUndefined);

  posteriors->CopyRowsFromVec(ubm_gconsts_);

  CuMatrix<float> feats_sq(feats.NumRows(), feats.NumCols(), kUndefined);

  // using our own kernel here to avoid an extra memcpy.
  // ApplyPow unfortunately only works in place.
  square_matrix(feats.NumRows(), feats.NumCols(), feats.Data(), feats.Stride(),
                feats_sq.Data(), feats_sq.Stride());

  posteriors->AddMatMat(1.0, feats, kNoTrans, ubm_means_inv_vars_, kTrans, 1.0);
  posteriors->AddMatMat(-0.5, feats_sq, kNoTrans, ubm_inv_vars_, kTrans, 1.0);

  // apply scaling factor
  posteriors->ApplySoftMaxPerRow();

  if (info_.max_count > 0) {
    // when max count > 0 we need to know the total posterior sum to adjust
    // the prior offset.  So calculate that here.
    get_matrix_sum_double_buffer(
        b_, posteriors->NumRows(), posteriors->NumCols(), posteriors->Data(),
        posteriors->Stride(), info_.posterior_scale, tot_post_.Data());
  }
}

void IvectorExtractorFastCuda::ComputeIvectorStats(
    const CuMatrixBase<float> &feats, const CuMatrixBase<float> &posteriors,
    CuVector<float> *gamma, CuMatrix<float> *X) {
  gamma->Resize(num_gauss_, kUndefined);
  X->Resize(num_gauss_, feat_dim_, kUndefined);

  gamma->AddRowSumMat(info_.posterior_scale, posteriors, 0.0f);
  X->AddMatMat(info_.posterior_scale, posteriors, kTrans, feats, kNoTrans,
               0.0f);
}

void IvectorExtractorFastCuda::ComputeIvectorFromStats(
    const CuVector<float> &gamma, const CuMatrix<float> &X,
    CuVector<float> *ivector) {
  CuVector<float> &linear = *ivector;
  linear.Resize(ivector_dim_, kUndefined);
  // Initialize to zero as batched kernel is +=
  linear.SetZero();

  CuSpMatrix<float> quadratic(ivector_dim_, kUndefined);

  batched_gemv_reduce(num_gauss_, feat_dim_, ivector_dim_,
                      ie_Sigma_inv_M_f_.Stride(), ie_Sigma_inv_M_f_.Data(),
                      X.Stride(), X.Data(), linear.Data());

  CuSubVector<float> q_vec(quadratic.Data(),
                           ivector_dim_ * (ivector_dim_ + 1) / 2);
  q_vec.AddMatVec(1.0f, ie_U_, kTrans, gamma, 0.0f);

  // TODO for online this needs to be stored and passed forward
  // For offline this is always zero.
  float old_num_frames = 0.0f;

  // compute and apply prior offset to linear and quadraditic terms
  // offset tot_post_ by correct buffer
  update_linear_and_quadratic_terms(
      quadratic.NumRows(), old_num_frames, prior_offset_, tot_post_.Data() + b_,
      info_.max_count, quadratic.Data(), linear.Data());
  // advance double buffer
  b_ = (b_ + 1) % 2;

  // We are computing a solution to this linear system:
  // x = quadratic^-1 * linear
  // ivector+=x

  // Inverting the matrix is unneccessary.  We are only solving a single
  // linear system.  So just use choleskey's to solve for a single ivector
  // Equation being solved: quadratic * ivector = linear

#if CUDA_VERSION >= 9010
  // Comment this out to use LU decomposistion instead.
  // CHOLESKY's should be faster and more accurate so this is preffered.
#define CHOLESKY
  int nrhs = 1;
  // Forming new non-SP matrix for cusolver.
  CuMatrix<float> A(quadratic);

#ifdef CHOLESKY
  // query temp buffer size
  int L_work;
  CUSOLVER_SAFE_CALL(
      cusolverDnSpotrf_bufferSize(GetCusolverDnHandle(), CUBLAS_FILL_MODE_LOWER,
                                  A.NumRows(), A.Data(), A.Stride(), &L_work));

  // allocate temp buffer
  float *workspace = static_cast<float *>(
      CuDevice::Instantiate().Malloc(L_work * sizeof(float)));

  // perform factorization
  CUSOLVER_SAFE_CALL(cusolverDnSpotrf(
      GetCusolverDnHandle(), CUBLAS_FILL_MODE_LOWER, A.NumRows(), A.Data(),
      A.Stride(), workspace, L_work, d_info_));

  // solve for rhs
  CUSOLVER_SAFE_CALL(cusolverDnSpotrs(
      GetCusolverDnHandle(), CUBLAS_FILL_MODE_LOWER, A.NumRows(), nrhs,
      A.Data(), A.Stride(), ivector->Data(), ivector_dim_, d_info_));

  CuDevice::Instantiate().Free(workspace);
#else
  // query temp buffer size
  int L_work;
  CUSOLVER_SAFE_CALL(
      cusolverDnSgetrf_bufferSize(GetCusolverDnHandle(), A.NumRows(),
                                  A.NumCols(), A.Data(), A.Stride(), &L_work));

  // allocate temp buffer
  float *workspace = static_cast<float *>(
      CuDevice::Instantiate().Malloc(L_work * sizeof(float)));
  int *devIpiv =
      static_cast<int *>(CuDevice::Instantiate().Malloc(L_work * sizeof(int)));

  // perform factorization
  CUSOLVER_SAFE_CALL(cusolverDnSgetrf(GetCusolverDnHandle(), A.NumRows(),
                                      A.NumCols(), A.Data(), A.Stride(),
                                      workspace, devIpiv, d_info_));

  // solve for rhs
  CUSOLVER_SAFE_CALL(cusolverDnSgetrs(
      GetCusolverDnHandle(), CUBLAS_OP_N, A.NumRows(), nrhs, A.Data(),
      A.Stride(), devIpiv, ivector->Data(), ivector_dim_, d_info_));

  CuDevice::Instantiate().Free(workspace);
  CuDevice::Instantiate().Free(devIpiv);
#endif
#else
  // Cuda version is too old for cu-solver.
  // Use Kaldi built-in inversion routine.
  quadratic.Invert();
  CuVector<float> linear_tmp(linear);
  ivector->Resize(ivector_dim_, kUndefined);
  ivector->AddSpVec(1.0, quadratic, linear_tmp, 0.0);
#endif

  // remove prior from ivector
  CuSubVector<float> ivector0(*ivector, 0, 1);
  ivector0.Add(-prior_offset_);
}

};  // namespace kaldi
