// gmm/diag-gmm-test.cc

// Copyright 2009-2011  Microsoft Corporation;  Georg Stemmer;  Jan Silovsky;
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

#include "gmm/diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "util/kaldi-io.h"

namespace kaldi {

void InitRandomGmm(DiagGmm *gmm_in) {
  int32 num_gauss = 10 + Rand() % 5;
  int32 dim = 20 + Rand() % 15;
  DiagGmm &gmm(*gmm_in);
  gmm.Resize(num_gauss, dim);
  Matrix<BaseFloat> inv_vars(num_gauss, dim),
      means(num_gauss, dim);
  Vector<BaseFloat> weights(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    for (int32 j = 0; j < dim; j++) {
      inv_vars(i, j) = exp(RandGauss() * (1.0 / (1 + j)));
      means(i, j) = RandGauss() * (1.0 / (1 + j));
    }
    weights(i) = exp(RandGauss());
  }
  weights.Scale(1.0 / weights.Sum());
  gmm.SetWeights(weights);
  gmm.SetInvVarsAndMeans(inv_vars, means);
  gmm.Perturb(0.5 * RandUniform());
  gmm.ComputeGconsts();  // this is unnecassary; computed in Perturb
}



// This tests the Generate function and also the HMM-update.
// it relies on some statistical ideas related to the Aikake
// criterion.
void UnitTestDiagGmmGenerate() {
  DiagGmm gmm;
  InitRandomGmm(&gmm);
  int32 dim =  gmm.Dim();
  int32 npoints = 100 * gmm.NumGauss();
  Matrix<BaseFloat> rand_points(npoints, dim);
  for (int32 i = 0; i < npoints; i++) {
    SubVector<BaseFloat> row(rand_points, i);
    gmm.Generate(&row);
  }
  int32 niters = 15;
  BaseFloat objf_change_tot = 0.0, objf_change, count;
  for (int32 j = 0; j < niters; j++) {
    MleDiagGmmOptions opts;
    AccumDiagGmm stats(gmm, kGmmAll);  // all update flags.
    for (int32 i = 0; i < npoints; i++) {
      SubVector<BaseFloat> row(rand_points, i);
      stats.AccumulateFromDiag(gmm, row, 1.0);
    }
    MleDiagGmmUpdate(opts, stats, kGmmAll, &gmm, &objf_change, &count);
    objf_change_tot += objf_change;
  }
  AssertEqual(count, npoints, 1e-6);
  int32 num_params = gmm.NumGauss() * (gmm.Dim()*2 + 1);
  BaseFloat expected_objf_change = 0.5 * num_params;
  KALDI_LOG << "Expected objf change is: not much more than "
            << expected_objf_change <<", seen: " << objf_change_tot;
  KALDI_ASSERT(objf_change_tot < 2.0 * expected_objf_change);  // way too much.
  // This test relies on statistical laws and if it fails it does not
  // *necessarily* mean that something is wrong.
}

void UnitTestDiagGmm() {
  // random dimension of the gmm
  size_t dim = 1 + kaldi::RandInt(0, 9);
  // random number of mixtures
  size_t nMix = 1 + kaldi::RandInt(0, 9);

  std::cout << "Testing NumGauss: " << nMix << ", " << "Dim: " << dim
    << '\n';

  // generate random feature vector and
  // random mean and variance vectors
  Vector<BaseFloat> feat(dim), weights(nMix), loglikes(nMix);
  Matrix<BaseFloat> means(nMix, dim), vars(nMix, dim), invvars(nMix, dim);

  float loglike = 0.0;
  for (size_t d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }

  float tot_weight = 0.0;
  for (size_t m = 0; m < nMix; m++) {
    weights(m) = kaldi::RandUniform();
    for (size_t d= 0; d < dim; d++) {
      means(m, d) = kaldi::RandGauss();
      vars(m, d) = exp(kaldi::RandGauss()) + 1e-5;
    }
    tot_weight += weights(m);
  }

  // normalize weights
  for (size_t m = 0; m < nMix; m++) {
    weights(m) /= tot_weight;
    for (size_t d= 0; d < dim; d++) {
      loglikes(m) += -0.5 * (M_LOG_2PI + log(vars(m, d)) + (feat(d) -
          means(m, d)) * (feat(d) - means(m, d)) / vars(m, d));
    }
    loglikes(m) += log(weights(m));
  }

  loglike = loglikes.LogSumExp();

  // new GMM
  DiagGmm *gmm = new DiagGmm();
  gmm->Resize(nMix, dim);
  invvars.CopyFromMat(vars);
  invvars.InvertElements();
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(invvars, means);
  gmm->ComputeGconsts();

  Vector<BaseFloat> posterior1(nMix);
  float loglike1 = gmm->ComponentPosteriors(feat, &posterior1);

  std::cout << "LogLike: " << loglike << '\n';
  std::cout << "LogLike1: " << loglike1 << '\n';
  AssertEqual(loglike, loglike1, 0.01);

  AssertEqual(1.0, posterior1.Sum(), 0.01);

  {  // Test various accessors / mutators
    Vector<BaseFloat> weights_bak(nMix);
    Matrix<BaseFloat> means_bak(nMix, dim);
    Matrix<BaseFloat> invvars_bak(nMix, dim);

    weights_bak.CopyFromVec(gmm->weights());
    gmm->GetMeans(&means_bak);
    gmm->GetVars(&invvars_bak);   // get vars
    invvars_bak.InvertElements();  // compute invvars

    // set all params one-by-one to new model
    DiagGmm gmm2;
    gmm2.Resize(gmm->NumGauss(), gmm->Dim());
    gmm2.SetWeights(weights_bak);
    gmm2.SetMeans(means_bak);
    gmm2.SetInvVars(invvars_bak);
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
    DiagGmm gmm3;
    gmm3.Resize(gmm->NumGauss(), gmm->Dim());
    gmm3.SetWeights(weights_bak);
    means_bak.SetZero();
    for (size_t i = 0; i < nMix; i++) {
      SubVector<BaseFloat> row(means_bak, i);
      gmm->GetComponentMean(i, &row);
      gmm3.SetComponentMean(i, row);
    }
    gmm3.SetInvVars(invvars_bak);
    gmm3.ComputeGconsts();
    BaseFloat loglike_gmm3 = gmm3.LogLikelihood(feat);
    AssertEqual(loglike1, loglike_gmm3, 0.01);
  }  // Test various accessors / mutators end


  // First, non-binary write.
  gmm->Write(Output("tmpf", false).Stream(), false);

  delete gmm;

  {
    bool binary_in;
    DiagGmm *gmm2 = new DiagGmm();
    Input ki("tmpf", &binary_in);
    gmm2->Read(ki.Stream(), binary_in);

    float loglike4 = gmm2->ComponentPosteriors(feat, &posterior1);
    AssertEqual(loglike, loglike4, 0.01);

    // binary write
    gmm2->Write(Output("tmpfb", true).Stream(), true);
    delete gmm2;

    // binary read
    DiagGmm *gmm3;
    gmm3 = new DiagGmm();
    Input ki2("tmpfb", &binary_in);
    gmm3->Read(ki2.Stream(), binary_in);

    float loglike5 = gmm3->ComponentPosteriors(feat, &posterior1);
    AssertEqual(loglike, loglike5, 0.01);
    delete gmm3;
  }

  {  // split and merge test for 1 component GMM (doesn't test the merge crit.)
    DiagGmm gmm1;
    Vector<BaseFloat> weights1(1);
    Matrix<BaseFloat> means1(1, dim), vars1(1, dim), invvars1(1, dim);
    weights1(0) = 1.0;
    means1.CopyFromMat(means.Range(0, 1, 0, dim));
    vars1.CopyFromMat(vars.Range(0, 1, 0, dim));
    invvars1.CopyFromMat(vars1);
    invvars1.InvertElements();
    gmm1.Resize(1, dim);
    gmm1.SetWeights(weights1);
    gmm1.SetInvVarsAndMeans(invvars1, means1);
    gmm1.ComputeGconsts();
    DiagGmm gmm2;
    gmm2.CopyFromDiagGmm(gmm1);
    gmm2.Split(2, 0.001);
    gmm2.Merge(1);
    float loglike1 = gmm1.LogLikelihood(feat);
    float loglike2 = gmm2.LogLikelihood(feat);
    AssertEqual(loglike1, loglike2, 0.01);
  }


  {  // split and merge test for 1 component GMM, this time using K-means algorithm.
    DiagGmm gmm1;
    Vector<BaseFloat> weights1(1);
    Matrix<BaseFloat> means1(1, dim), vars1(1, dim), invvars1(1, dim);
    weights1(0) = 1.0;
    means1.CopyFromMat(means.Range(0, 1, 0, dim));
    vars1.CopyFromMat(vars.Range(0, 1, 0, dim));
    invvars1.CopyFromMat(vars1);
    invvars1.InvertElements();
    gmm1.Resize(1, dim);
    gmm1.SetWeights(weights1);
    gmm1.SetInvVarsAndMeans(invvars1, means1);
    gmm1.ComputeGconsts();
    DiagGmm gmm2;
    gmm2.CopyFromDiagGmm(gmm1);
    gmm2.Split(2, 0.001);
    gmm2.MergeKmeans(1);
    float loglike1 = gmm1.LogLikelihood(feat);
    float loglike2 = gmm2.LogLikelihood(feat);
    AssertEqual(loglike1, loglike2, 0.01);
  }

    {  // Duplicate Gaussians using initializer that takes a vector, and
      // check like is unchanged.
    DiagGmm gmm1;
    Vector<BaseFloat> weights1(1);
    Matrix<BaseFloat> means1(1, dim), vars1(1, dim), invvars1(1, dim);
    weights1(0) = 1.0;
    means1.CopyFromMat(means.Range(0, 1, 0, dim));
    vars1.CopyFromMat(vars.Range(0, 1, 0, dim));
    invvars1.CopyFromMat(vars1);
    invvars1.InvertElements();
    gmm1.Resize(1, dim);
    gmm1.SetWeights(weights1);
    gmm1.SetInvVarsAndMeans(invvars1, means1);
    gmm1.ComputeGconsts();

    std::vector<std::pair<BaseFloat, const DiagGmm*> > vec;
    vec.push_back(std::make_pair(0.4, (const DiagGmm*)(&gmm1)));
    vec.push_back(std::make_pair(0.6, (const DiagGmm*)(&gmm1)));
    
    DiagGmm gmm2(vec);

    float loglike1 = gmm1.LogLikelihood(feat);
    float loglike2 = gmm2.LogLikelihood(feat);
    AssertEqual(loglike1, loglike2, 0.01);
  }

  unlink("tmpf");
  unlink("tmpfb");
}

}  // end namespace kaldi

int main() {
  // repeat the test ten times
  for (int i = 0; i < 2; i++) {
    kaldi::UnitTestDiagGmm();
    kaldi::UnitTestDiagGmmGenerate();
  }
  std::cout << "Test OK.\n";
}

