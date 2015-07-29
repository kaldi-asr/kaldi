// gmm/model-test-common.cc

// Copyright 2009-2011  Microsoft Corporation;  Jan Silovsky;
//                      Saarland University

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

#include <algorithm>
#include <vector>

#include "matrix/matrix-lib.h"
#include "gmm/model-test-common.h"

namespace kaldi {
namespace unittest {

void RandPosdefSpMatrix(int32 dim, SpMatrix<BaseFloat> *matrix,
                        TpMatrix<BaseFloat> *matrix_sqrt, BaseFloat *logdet) {
  // generate random (non-singular) matrix
  Matrix<BaseFloat> tmp(dim, dim);
  while (1) {
    tmp.SetRandn();
    if (tmp.Cond() < 100) break;
    KALDI_LOG << "Condition number of random matrix large "
              << static_cast<float>(tmp.Cond())
              << ", trying again (this is normal)\n";
  }
  // tmp * tmp^T will give positive definite matrix
  matrix->AddMat2(1.0, tmp, kNoTrans, 0.0);

  if (matrix_sqrt != NULL) matrix_sqrt->Cholesky(*matrix);
  if (logdet != NULL) *logdet = matrix->LogPosDefDet();
  if ((matrix_sqrt == NULL) && (logdet == NULL)) {
    TpMatrix<BaseFloat> sqrt(dim);
    sqrt.Cholesky(*matrix);
  }
}

void RandDiagGaussFeatures(int32 num_samples,
                           const VectorBase<BaseFloat> &mean,
                           const VectorBase<BaseFloat> &sqrt_var,
                           MatrixBase<BaseFloat> *feats) {
  int32 dim = mean.Dim();
  KALDI_ASSERT(feats != NULL);
  KALDI_ASSERT(feats->NumRows() == num_samples &&
               feats->NumCols() == dim);
  KALDI_ASSERT(sqrt_var.Dim() == dim);

  Vector<BaseFloat> rnd_vec(dim);
  for (int32 counter = 0; counter < num_samples; counter++) {
    for (int32 d = 0; d < dim; d++) {
      rnd_vec(d) = RandGauss();
    }
    feats->Row(counter).CopyFromVec(mean);
    feats->Row(counter).AddVecVec(1.0, sqrt_var, rnd_vec, 1.0);
  }
}

void RandFullGaussFeatures(int32 num_samples,
                           const VectorBase<BaseFloat> &mean,
                           const TpMatrix<BaseFloat> &sqrt_var,
                           MatrixBase<BaseFloat> *feats) {
  int32 dim = mean.Dim();
  KALDI_ASSERT(feats != NULL);
  KALDI_ASSERT(feats->NumRows() == num_samples && feats->NumCols() == dim);
  KALDI_ASSERT(sqrt_var.NumRows() == dim);

  Vector<BaseFloat> rnd_vec(dim);
  for (int32 counter = 0; counter < num_samples; counter++) {
    for (int32 d = 0; d < dim; d++) {
      rnd_vec(d) = RandGauss();
    }
    feats->Row(counter).CopyFromVec(mean);
    feats->Row(counter).AddTpVec(1.0, sqrt_var, kNoTrans, rnd_vec, 1.0);
  }
}

void InitRandDiagGmm(int32 dim, int32 num_comp, DiagGmm *gmm) {
  Vector<BaseFloat> weights(num_comp);
  Matrix<BaseFloat> means(num_comp, dim), inv_vars(num_comp, dim);

  for (int32 m = 0; m < num_comp; m++) {
    weights(m) = Exp(RandGauss());
    for (int32 d= 0; d < dim; d++) {
      means(m, d) = RandGauss() / (1 + d);
      inv_vars(m, d) = Exp(RandGauss() / (1 + d)) + 1e-2;
    }
  }
  weights.Scale(1.0 / weights.Sum());

  gmm->Resize(num_comp, dim);
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(inv_vars, means);
  gmm->ComputeGconsts();
}

void InitRandFullGmm(int32 dim, int32 num_comp, FullGmm *gmm) {
  Vector<BaseFloat> weights(num_comp);
  Matrix<BaseFloat> means(num_comp, dim);
  std::vector< SpMatrix<BaseFloat> > invcovars(num_comp);
  for (int32 mix = 0; mix < num_comp; mix++) {
    invcovars[mix].Resize(dim);
  }

  BaseFloat tot_weight = 0.0;
  for (int32 m = 0; m < num_comp; m++) {
    weights(m) = RandUniform() + 1e-2;
    for (int32 d= 0; d < dim; d++) {
      means(m, d) = RandGauss();
    }
    RandPosdefSpMatrix(dim, &invcovars[m], NULL, NULL);
    invcovars[m].InvertDouble();
    tot_weight += weights(m);
  }
  weights.Scale(1/tot_weight);

  gmm->Resize(num_comp, dim);
  gmm->SetWeights(weights);
  gmm->SetInvCovarsAndMeans(invcovars, means);
  gmm->ComputeGconsts();
}

}  // End namespace unittests
}  // End namespace kaldi
