// gmm/full-gmm-test.cc

// Copyright 2009-2011  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation

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

#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/model-test-common.h"
#include "util/stl-utils.h"
#include "util/kaldi-io.h"
#include "gmm/full-gmm-normal.h"
#include "gmm/mle-full-gmm.h"

using namespace kaldi;

void RandPosdefSpMatrix(size_t dim, SpMatrix<BaseFloat> *matrix,
                          TpMatrix<BaseFloat> *matrix_sqrt = NULL,
                          BaseFloat *logdet = NULL) {
  // generate random (non-singular) matrix
  Matrix<BaseFloat> tmp(dim, dim);
  while (1) {
    tmp.SetRandn();
    if (tmp.Cond() < 100) break;
    std::cout << "Condition number of random matrix large "
      << static_cast<float>(tmp.Cond()) << ", trying again (this is normal)"
      << '\n';
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

void init_rand_diag_gmm(DiagGmm *gmm) {
  size_t num_comp = gmm->NumGauss(), dim = gmm->Dim();
  Vector<BaseFloat> weights(num_comp);
  Matrix<BaseFloat> means(num_comp, dim), vars(num_comp, dim);

  BaseFloat tot_weight = 0.0;
  for (size_t m = 0; m < num_comp; m++) {
    weights(m) = kaldi::RandUniform();
    for (size_t d= 0; d < dim; d++) {
      means(m, d) = kaldi::RandGauss();
      vars(m, d) = exp(kaldi::RandGauss()) + 1e-5;
    }
    tot_weight += weights(m);
  }

  // normalize weights
  for (size_t m = 0; m < num_comp; m++) {
    weights(m) /= tot_weight;
  }

  vars.InvertElements();
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(vars, means);
  gmm->Perturb(0.5 * RandUniform());
  gmm->ComputeGconsts();  // this is unnecassary; computed in Perturb
}

void UnitTestFullGmmEst() {
  FullGmm fgmm;
  int32 dim = 10 + Rand() % 10, num_comp = 1 + Rand() % 10;
  unittest::InitRandFullGmm(dim, num_comp, &fgmm);
  int32 num_frames = 5000;
  Matrix<BaseFloat> feats(num_frames, dim);
  FullGmmNormal fgmm_normal(fgmm);
  fgmm_normal.Rand(&feats);

  AccumFullGmm acc(fgmm, kGmmAll);
  for (int32 t = 0; t < num_frames; t++)
    acc.AccumulateFromFull(fgmm, feats.Row(t), 1.0);
  BaseFloat objf_change, count;

  MleFullGmmOptions opts;

  MleFullGmmUpdate(opts, acc, kGmmAll, &fgmm, &objf_change, &count);
  BaseFloat change = objf_change / count,
      num_params = (num_comp * (dim + 1 + (dim*(dim+1)/2))),
      predicted_change = 0.5 * num_params / num_frames; // Was there
  KALDI_LOG << "Objf change per frame was " << change << " vs. predicted "
            << predicted_change;
  KALDI_ASSERT(change < 2.0 * predicted_change && change > 0.0)
}


void
UnitTestFullGmm() {
  // random dimension of the gmm
  size_t dim = 1 + kaldi::RandInt(0, 9);
  // random number of mixtures
  size_t nMix = 1 + kaldi::RandInt(0, 9);

  std::cout << "Testing NumGauss: " << nMix << ", " << "Dim: " << dim
    << '\n';

  // generate random feature vector and
  // random mean vectors and covariance matrices
  Vector<BaseFloat> feat(dim);
  Vector<BaseFloat> weights(nMix);
  Vector<BaseFloat> loglikes(nMix);
  Matrix<BaseFloat> means(nMix, dim);
  std::vector<SpMatrix<BaseFloat> > invcovars(nMix);
  for (size_t mix = 0; mix < nMix; mix++) {
    invcovars[mix].Resize(dim);
  }
  Vector<BaseFloat> covars_logdet(nMix);

  for (size_t d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }

  float tot_weight = 0.0;
  for (size_t m = 0; m < nMix; m++) {
    weights(m) = kaldi::RandUniform();
    for (size_t d = 0; d < dim; d++) {
      means(m, d) = kaldi::RandGauss();
    }
    SpMatrix<BaseFloat> covar(dim);
    RandPosdefSpMatrix(dim, &covar, NULL, &covars_logdet(m));
    invcovars[m].CopyFromSp(covar);
    invcovars[m].InvertDouble();
    tot_weight += weights(m);
  }

  // normalize weights and compute loglike for feature vector
  for (size_t m = 0; m < nMix; m++) {
    weights(m) /= tot_weight;
  }

  // compute loglike for feature vector
  float loglike = 0.0;
  for (size_t m = 0; m < nMix; m++) {
    loglikes(m) += -0.5 * (M_LOG_2PI * dim
      + covars_logdet(m)
      + VecSpVec(means.Row(m), invcovars[m], means.Row(m))
      + VecSpVec(feat, invcovars[m], feat))
      + VecSpVec(means.Row(m), invcovars[m], feat);
    loglikes(m) += log(weights(m));
  }

  loglike = loglikes.LogSumExp();


  // new GMM
  FullGmm *gmm = new FullGmm();
  gmm->Resize(nMix, dim);
  gmm->SetWeights(weights);
  gmm->SetInvCovarsAndMeans(invcovars, means);
  gmm->ComputeGconsts();

  Vector<BaseFloat> posterior1(nMix);
  float loglike1 = gmm->ComponentPosteriors(feat, &posterior1);

  // std::cout << "LogLike: " << loglike << '\n';
  // std::cout << "LogLike1: " << loglike1 << '\n';

  AssertEqual(loglike, loglike1, 0.01);

  KALDI_ASSERT(fabs(1.0 - posterior1.Sum()) < 0.001);

  {  // Test various accessors / mutators
    Vector<BaseFloat> weights_bak(nMix);
    Matrix<BaseFloat> means_bak(nMix, dim);
    std::vector<SpMatrix<BaseFloat> > invcovars_bak(nMix);
    for (size_t i = 0; i < nMix; i++) {
      invcovars_bak[i].Resize(dim);
    }

    weights_bak.CopyFromVec(gmm->weights());
    gmm->GetMeans(&means_bak);
    gmm->GetCovars(&invcovars_bak);
    for (size_t i = 0; i < nMix; i++) {
      invcovars_bak[i].InvertDouble();
    }

    // set all params one-by-one to new model
    FullGmm gmm2;
    gmm2.Resize(gmm->NumGauss(), gmm->Dim());
    gmm2.SetWeights(weights_bak);
    gmm2.SetMeans(means_bak);
    gmm2.SetInvCovars(invcovars_bak);
    gmm2.ComputeGconsts();
    BaseFloat loglike_gmm2 = gmm2.LogLikelihood(feat);
    AssertEqual(loglike1, loglike_gmm2);
    {
      Vector<BaseFloat> loglikes;
      gmm2.LogLikelihoods(feat, &loglikes);
      AssertEqual(loglikes.LogSumExp(), loglike_gmm2);
    }
    {
      std::vector<int32> indices;
      for (int32 i = 0; i < gmm2.NumGauss(); i++)
        indices.push_back(i);
      Vector<BaseFloat> loglikes;
      gmm2.LogLikelihoodsPreselect(feat, indices, &loglikes);
      AssertEqual(loglikes.LogSumExp(), loglike_gmm2);
    }


    // single component mean accessor + mutator
    FullGmm gmm3;
    gmm3.Resize(gmm->NumGauss(), gmm->Dim());
    gmm3.SetWeights(weights_bak);
    means_bak.SetZero();
    for (size_t i = 0; i < nMix; i++) {
      SubVector<BaseFloat> tmp = means_bak.Row(i);
      gmm->GetComponentMean(i, &tmp);
    }
    gmm3.SetMeans(means_bak);
    gmm3.SetInvCovars(invcovars_bak);
    gmm3.ComputeGconsts();
    float loglike_gmm3 = gmm3.LogLikelihood(feat);
    AssertEqual(loglike1, loglike_gmm3, 0.01);

    // set all params one-by-one to new model
    FullGmm gmm4;
    gmm4.Resize(gmm->NumGauss(), gmm->Dim());
    gmm4.SetWeights(weights_bak);
    gmm->GetCovarsAndMeans(&invcovars_bak, &means_bak);
    for (size_t i = 0; i < nMix; i++) {
      invcovars_bak[i].InvertDouble();
    }
    gmm4.SetInvCovarsAndMeans(invcovars_bak, means_bak);
    gmm4.ComputeGconsts();
    BaseFloat loglike_gmm4 = gmm4.LogLikelihood(feat);
    AssertEqual(loglike1, loglike_gmm4, 0.001);

  }  // Test various accessors / mutators end

   // First, non-binary write
  gmm->Write(Output("tmpf", false).Stream(), false);

  {  // I/O tests
    bool binary_in;
    FullGmm *gmm2 = new FullGmm();
    Input ki("tmpf", &binary_in);
    gmm2->Read(ki.Stream(), binary_in);

    float loglike3 = gmm2->ComponentPosteriors(feat, &posterior1);
    AssertEqual(loglike, loglike3, 0.01);

    // binary write
    gmm2->Write(Output("tmpfb", true).Stream(), true);
    delete gmm2;

    // binary read
    FullGmm *gmm3;
    gmm3 = new FullGmm();

    Input ki2("tmpfb", &binary_in);
    gmm3->Read(ki2.Stream(), binary_in);

    AssertEqual(loglike, loglike3, 0.01);

    delete gmm3;
  }

  {  // CopyFromFullGmm
    FullGmm gmm4;
    gmm4.CopyFromFullGmm(*gmm);
    float loglike5 = gmm4.ComponentPosteriors(feat, &posterior1);
    AssertEqual(loglike, loglike5, 0.01);
  }

  {  // test copy from DiagGmm and back to DiagGmm
    DiagGmm gmm_diag;
    gmm_diag.Resize(nMix, dim);
    init_rand_diag_gmm(&gmm_diag);
    float loglike_diag = gmm_diag.LogLikelihood(feat);

    FullGmm gmm_full;
    gmm_full.CopyFromDiagGmm(gmm_diag);
    float loglike_full = gmm_full.LogLikelihood(feat);

    DiagGmm gmm_diag2;
    gmm_diag2.CopyFromFullGmm(gmm_full);
    float loglike_diag2 = gmm_diag2.LogLikelihood(feat);

    AssertEqual(loglike_diag, loglike_full, 0.01);
    AssertEqual(loglike_diag, loglike_diag2, 0.01);
  }

  {  // split and merge test for 1 component GMM (doesn't test the merge crit.)
    FullGmm gmm1;
    Vector<BaseFloat> weights1(1);
    Matrix<BaseFloat> means1(1, dim);
    std::vector<SpMatrix<BaseFloat> > invcovars1(1);
    weights1(0) = 1.0;
    means1.CopyFromMat(means.Range(0, 1, 0, dim));
    invcovars1[0].Resize(dim);
    invcovars1[0].CopyFromSp(invcovars[0]);
    gmm1.Resize(1, dim);
    gmm1.SetWeights(weights1);
    gmm1.SetInvCovarsAndMeans(invcovars1, means1);
    gmm1.ComputeGconsts();
    FullGmm gmm2;
    gmm2.CopyFromFullGmm(gmm1);
    gmm2.Split(2, 0.001);
    gmm2.Merge(1);
    float loglike1 = gmm1.LogLikelihood(feat);
    float loglike2 = gmm2.LogLikelihood(feat);
    AssertEqual(loglike1, loglike2, 0.01);
  }

  delete gmm;
}

int
main() {
  // repeat the test ten times
  for (int i = 0; i < 2; i++) {
    UnitTestFullGmm();
    UnitTestFullGmmEst();
  }
  std::cout << "Test OK.\n";
}
