// gmm/mle-full-gmm-test.cc

// Copyright 2009-2011  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation;   Yanmin Qian;  Georg Stemmer

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
#include "gmm/model-common.h"
#include "gmm/mle-full-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "util/stl-utils.h"
#include "util/kaldi-io.h"

using namespace kaldi;

void TestComponentAcc(const FullGmm &gmm, const Matrix<BaseFloat> &feats) {
  MleFullGmmOptions config;
  AccumFullGmm est_atonce;    // updates all components
  AccumFullGmm est_compwise;  // updates single components

  // Initialize estimators
  est_atonce.Resize(gmm.NumGauss(), gmm.Dim(), kGmmAll);
  est_atonce.SetZero(kGmmAll);
  est_compwise.Resize(gmm.NumGauss(),
      gmm.Dim(), kGmmAll);
  est_compwise.SetZero(kGmmAll);

  // accumulate estimators
  for (int32 i = 0; i < feats.NumRows(); i++) {
    est_atonce.AccumulateFromFull(gmm, feats.Row(i), 1.0F);
    Vector<BaseFloat> post(gmm.NumGauss());
    gmm.ComponentPosteriors(feats.Row(i), &post);
    for (int32 m = 0; m < gmm.NumGauss(); m++) {
      est_compwise.AccumulateForComponent(feats.Row(i), m, post(m));
    }
  }

  FullGmm gmm_atonce;    // model with all components accumulated together
  FullGmm gmm_compwise;  // model with each component accumulated separately
  gmm_atonce.Resize(gmm.NumGauss(), gmm.Dim());
  gmm_compwise.Resize(gmm.NumGauss(), gmm.Dim());

  MleFullGmmUpdate(config, est_atonce, kGmmAll, &gmm_atonce, NULL, NULL);
  MleFullGmmUpdate(config, est_compwise, kGmmAll, &gmm_compwise, NULL, NULL);

  // the two ways of updating should result in the same model
  double loglike0 = 0.0;
  double loglike1 = 0.0;
  double loglike2 = 0.0;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    loglike0 += static_cast<double>(gmm.LogLikelihood(feats.Row(i)));
    loglike1 += static_cast<double>(gmm_atonce.LogLikelihood(feats.Row(i)));
    loglike2 += static_cast<double>(gmm_compwise.LogLikelihood(feats.Row(i)));
  }

  std::cout << "Per-frame log-likelihood before update = "
      << (loglike0/feats.NumRows()) << '\n';
  std::cout << "Per-frame log-likelihood (accumulating at once) = "
      << (loglike1/feats.NumRows()) << '\n';
  std::cout << "Per-frame log-likelihood (accumulating component-wise) = "
      << (loglike2/feats.NumRows()) << '\n';

  AssertEqual(loglike1, loglike2, 1.0e-6);

  if (est_atonce.NumGauss() != gmm.NumGauss()) {
    KALDI_WARN << "Unable to pass test_update_flags() test because of "
      "component removal during Update() call (this is normal)";
    return;
  } else {
    AssertGeq(loglike1, loglike0, 1.0e-6);
    AssertGeq(loglike2, loglike0, 1.0e-6);
  }
}

void rand_posdef_spmatrix(size_t dim, SpMatrix<BaseFloat> *matrix,
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

BaseFloat GetLogLikeTest(const FullGmm &gmm,
                         const VectorBase<BaseFloat> &feats,
                         bool print_eigs) {
  BaseFloat log_like_sum = -1.0e+10;
  Matrix<BaseFloat> means;
  gmm.GetMeans(&means);
  const std::vector<SpMatrix<BaseFloat> > inv_covars = gmm.inv_covars();

  if (print_eigs)
    for (size_t i = 0; i < inv_covars.size(); i++) {
      SpMatrix<BaseFloat> cov(inv_covars[i]);
      size_t dim = cov.NumRows();
      cov.Invert();
      std::cout << i << "'th component eigs are: ";
      Vector<BaseFloat> s(dim);
      Matrix<BaseFloat> P(dim, dim);
      cov.SymPosSemiDefEig(&s, &P);
      std::cout << s;
    }

  for (int32 i = 0; i < gmm.NumGauss(); i++) {
    BaseFloat logdet = -(inv_covars[i].LogPosDefDet());
    BaseFloat log_like = log(gmm.weights()(i))
      -0.5 * (gmm.Dim() * M_LOG_2PI + logdet);
    Vector<BaseFloat> offset(feats);
    offset.AddVec(-1.0, means.Row(i));
    log_like -= 0.5 * VecSpVec(offset, inv_covars[i], offset);
    log_like_sum = LogAdd(log_like_sum, log_like);
  }
  return log_like_sum;
}

void test_flags_driven_update(const FullGmm &gmm,
                              const Matrix<BaseFloat> &feats,
                              GmmFlagsType flags) {
  MleFullGmmOptions config;
  AccumFullGmm est_gmm_allp;   // updates all params
  // let's trust that all-params update works
  AccumFullGmm est_gmm_somep;  // updates params indicated by flags

  // warm-up estimators
  est_gmm_allp.Resize(gmm.NumGauss(), gmm.Dim(), kGmmAll);
  est_gmm_allp.SetZero(kGmmAll);
  
  est_gmm_somep.Resize(gmm.NumGauss(), gmm.Dim(), flags);
  est_gmm_somep.SetZero(flags);

  // accumulate estimators
  for (int32 i = 0; i < feats.NumRows(); i++) {
    est_gmm_allp.AccumulateFromFull(gmm, feats.Row(i), 1.0F);
    est_gmm_somep.AccumulateFromFull(gmm, feats.Row(i), 1.0F);
  }

  FullGmm gmm_all_update;   // model with all params updated
  FullGmm gmm_some_update;  // model with some params updated
  gmm_all_update.CopyFromFullGmm(gmm);   // init with orig. model
  gmm_some_update.CopyFromFullGmm(gmm);  // init with orig. model

  MleFullGmmUpdate(config, est_gmm_allp, kGmmAll, &gmm_all_update, NULL, NULL);
  MleFullGmmUpdate(config, est_gmm_somep, flags, &gmm_some_update, NULL, NULL);

  if (gmm_all_update.NumGauss() != gmm.NumGauss()) {
    KALDI_WARN << "Unable to pass test_update_flags() test because of "
      "component removal during Update() call (this is normal)";
    return;
  }

  // now back-off the gmm_all_update params that were not updated
  // in gmm_some_update to orig.
  if (~flags & kGmmWeights)
    gmm_all_update.SetWeights(gmm.weights());
  if (~flags & kGmmMeans) {
    Matrix<BaseFloat> means(gmm.NumGauss(), gmm.Dim());
    gmm.GetMeans(&means);
    gmm_all_update.SetMeans(means);
  }
  if (~flags & kGmmVariances) {
    std::vector<SpMatrix<BaseFloat> > vars(gmm.NumGauss());
    for (int32 i = 0; i < gmm.NumGauss(); i++)
      vars[i].Resize(gmm.Dim());
    gmm.GetCovars(&vars);
    for (int32 i = 0; i < gmm.NumGauss(); i++)
      vars[i].InvertDouble();
    gmm_all_update.SetInvCovars(vars);
  }
  gmm_some_update.ComputeGconsts();
  gmm_all_update.ComputeGconsts();

  // now both models gmm_all_update, gmm_all_update have the same params updated
  // compute loglike for models for check
  double loglike0 = 0.0;
  double loglike1 = 0.0;
  double loglike2 = 0.0;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    loglike0 += static_cast<double>(
      gmm.LogLikelihood(feats.Row(i)));
    loglike1 += static_cast<double>(
      gmm_all_update.LogLikelihood(feats.Row(i)));
    loglike2 += static_cast<double>(
      gmm_some_update.LogLikelihood(feats.Row(i)));
  }
  KALDI_LOG << "loglike1 = " << loglike1 << " loglike2 = " << loglike2;
  AssertEqual(loglike1, loglike2, 0.01);
}

void
test_io(const FullGmm &gmm, const AccumFullGmm &est_gmm, bool binary,
        const Matrix<BaseFloat> &feats) {
  std::cout << "Testing I/O, binary = " << binary << '\n';

  est_gmm.Write(Output("tmp_stats", binary).Stream(), binary);

  bool binary_in;
  AccumFullGmm est_gmm2;
  est_gmm2.Resize(gmm.NumGauss(),
    gmm.Dim(), kGmmAll);
  Input ki("tmp_stats", &binary_in);
  est_gmm2.Read(ki.Stream(), binary_in, false);  // not adding

  Input ki2("tmp_stats", &binary_in);
  est_gmm2.Read(ki2.Stream(), binary_in, true);  // adding

  est_gmm2.Scale(0.5, kGmmAll);
    // 0.5 -> make it same as what it would have been if we read just once.
    // [may affect it due to removal of components with small counts].

  MleFullGmmOptions config;
  FullGmm gmm1;
  FullGmm gmm2;
  gmm1.CopyFromFullGmm(gmm);
  gmm2.CopyFromFullGmm(gmm);
  MleFullGmmUpdate(config, est_gmm, est_gmm.Flags(), &gmm1, NULL, NULL);
  MleFullGmmUpdate(config, est_gmm2, est_gmm2.Flags(), &gmm2, NULL, NULL);

  BaseFloat loglike1 = 0.0;
  BaseFloat loglike2 = 0.0;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    loglike1 += gmm1.LogLikelihood(feats.Row(i));
    loglike2 += gmm2.LogLikelihood(feats.Row(i));
  }

  AssertEqual(loglike1, loglike2, 0.01);
}

void
UnitTestEstimateFullGmm() {
  // using namespace kaldi;

  // dimension of the gmm
  int32 dim = 10;

  // number of mixtures in the data
  int32 nMix = 7;

  // number of iterations for estimation
  int32 maxiterations = 20;

  // maximum number of densities in the GMM
  // larger than the number of mixtures in the data
  // so that we can test the removal of unseen components
  int32 maxcomponents = 50;

  // generate random feature vectors
  // first, generate parameters of vectors distribution
  // (mean and covariance matrices)
  Matrix<BaseFloat> means_f(nMix, dim);
  std::vector<SpMatrix<BaseFloat> > vars_f(nMix);
  std::vector<TpMatrix<BaseFloat> > vars_f_sqrt(nMix);
  for (int32 mix = 0; mix < nMix; mix++) {
    vars_f[mix].Resize(dim);
    vars_f_sqrt[mix].Resize(dim);
  }

  for (int32 m = 0; m < nMix; m++) {
    for (int32 d = 0; d < dim; d++) {
      means_f(m, d) = kaldi::RandGauss();
    }
    rand_posdef_spmatrix(dim, &vars_f[m], &vars_f_sqrt[m], NULL);
  }

  // second, generate 1000 feature vectors for each of the mixture components
  int32 counter = 0, multiple = 200;
  Matrix<BaseFloat> feats(nMix*200, dim);
  Vector<BaseFloat> rnd_vec(dim);
  for (int32 m = 0; m < nMix; m++) {
    for (int32 i = 0; i < multiple; i++) {
      for (int32 d = 0; d < dim; d++) {
        rnd_vec(d) = RandGauss();
      }
      feats.Row(counter).CopyFromVec(means_f.Row(m));
      feats.Row(counter).AddTpVec(1.0, vars_f_sqrt[m], kNoTrans, rnd_vec, 1.0);
      ++counter;
    }
  }

  {
    // Work out "perfect" log-like w/ one component.
    Matrix<BaseFloat> cov(dim, dim);
    Vector<BaseFloat> mean(dim);
    cov.AddMatMat(1.0, feats, kTrans, feats, kNoTrans, 0.0);
    cov.Scale(1.0 / feats.NumRows());
    mean.AddRowSumMat(1.0, feats);
    mean.Scale(1.0 / feats.NumRows());
    cov.AddVecVec(-1.0, mean, mean);
    BaseFloat logdet = cov.LogDet();
    BaseFloat avg_log = -0.5*(logdet + dim*(M_LOG_2PI + 1));
    std::cout << "Avg log-like per frame [full-cov, 1-mix] should be: "
      << avg_log << '\n';
    std::cout << "Total log-like [full-cov, 1-mix] should be: "
      << (feats.NumRows()*avg_log) << '\n';

    Vector<BaseFloat> s(dim);
    Matrix<BaseFloat> P(dim, dim);
    cov.SymPosSemiDefEig(&s, &P);
    std::cout << "Cov eigs are " << s;
  }

  // write the feature vectors to a file
  //  std::ofstream of("tmpfeats");
  //  of.precision(10);
  //  of << feats;
  //  of.close();

  // now generate randomly initial values for the GMM
  Vector<BaseFloat> weights(1);
  Matrix<BaseFloat> means(1, dim);
  std::vector<SpMatrix<BaseFloat> > invcovars(1);
  invcovars[0].Resize(dim);

  for (int32 d= 0; d < dim; d++) {
    means(0, d) = kaldi::RandGauss()*5.0F;
  }
  SpMatrix<BaseFloat> covar(dim);
  rand_posdef_spmatrix(dim, &covar, NULL, NULL);
  invcovars[0].CopyFromSp(covar);
  invcovars[0].InvertDouble();
  weights(0) = 1.0F;

  // new GMM
  FullGmm *gmm = new FullGmm();
  gmm->Resize(1, dim);
  gmm->SetWeights(weights);
  gmm->SetInvCovarsAndMeans(invcovars, means);
  gmm->ComputeGconsts();

  {
    KALDI_LOG << "Testing natural<>normal conversion";
    FullGmmNormal ngmm(*gmm);
    FullGmm rgmm;
    rgmm.Resize(1, dim);
    ngmm.CopyToFullGmm(&rgmm, kGmmAll);
    
    // check contents
    KALDI_ASSERT(ApproxEqual(weights(0), 1.0F, 1e-6));
    KALDI_ASSERT(ApproxEqual(gmm->weights()(0), rgmm.weights()(0), 1e-6));
    double prec_m = 1e-3;
    double prec_v = 1e-3;
    for (int32 d = 0; d < dim; d++) {
      KALDI_ASSERT(ApproxEqual(means.Row(0)(d), ngmm.means_.Row(0)(d), prec_m));
      KALDI_ASSERT(ApproxEqual(gmm->means_invcovars().Row(0)(d), rgmm.means_invcovars().Row(0)(d), prec_v));
      for (int32 d2 = d; d2 < dim; ++d2) {
        KALDI_ASSERT(ApproxEqual(covar(d, d2), ngmm.vars_[0](d, d2), prec_v));
        KALDI_ASSERT(ApproxEqual(gmm->inv_covars()[0](d, d2), rgmm.inv_covars()[0](d, d2), prec_v));
      }
    }
    KALDI_LOG << "OK";
  } 

  MleFullGmmOptions config;
  GmmFlagsType flags_all = kGmmAll;


  AccumFullGmm est_gmm;
  est_gmm.Resize(gmm->NumGauss(), gmm->Dim(), flags_all);

  // iterate
  int32 iteration = 0;
  float lastloglike = 0.0;
  int32 lastloglike_nM = 0;

  while (iteration < maxiterations) {
    // First, resize accums for the case of component splitting
    est_gmm.Resize(gmm->NumGauss(),
      gmm->Dim(), flags_all);
    est_gmm.SetZero(flags_all);
    double loglike = 0.0;
    double loglike_test = 0.0;
    for (int32 i = 0; i < counter; i++) {
      loglike += static_cast<double>(
        est_gmm.AccumulateFromFull(*gmm, feats.Row(i), 1.0F));
      if (iteration < 4) {
        loglike_test += GetLogLikeTest(*gmm, feats.Row(i), (i == 0));
        AssertEqual(loglike, loglike_test);
      }
    }

    std::cout << "Loglikelihood before iteration "
      << iteration << " : " << std::scientific << loglike
      << " number of components: " << gmm->NumGauss() << '\n';

    // std::cout << "Model is: " << *gmm;

    // every 5th iteration check loglike change and update lastloglike
    if (iteration % 5 == 0) {
      // likelihood should be increasing on the long term
      if ((iteration > 0) && (gmm->NumGauss() >= lastloglike_nM)) {
        KALDI_ASSERT(loglike > lastloglike);
      }
      lastloglike = loglike;
      lastloglike_nM = gmm->NumGauss();
    }

    BaseFloat obj, count;
    MleFullGmmUpdate(config, est_gmm, flags_all, gmm, &obj, &count);
    KALDI_LOG << "ML objective function change = " << (obj/count)
              << " per frame, over " << (count) << " frames.";

    // split components to double count at second iteration
    // and every next 3rd iteration
    // stop splitting when maxcomponents reached
    if ( (iteration < maxiterations - 3) && (iteration % 4 == 1)
        && (gmm->NumGauss() * 2 <= maxcomponents)) {
      gmm->Split(gmm->NumGauss() * 2, 0.01);
    }

    if (iteration == 5) {  // run following tests with not too overfitted model
      std::cout << "Testing flags-driven updates kGmmAll" << '\n';
      test_flags_driven_update(*gmm, feats, kGmmAll);
      std::cout << "Testing flags-driven updates kGmmWeights" << '\n';
      test_flags_driven_update(*gmm, feats, kGmmWeights);
      std::cout << "Testing flags-driven kGmmMeans" << '\n';
      test_flags_driven_update(*gmm, feats, kGmmMeans);
      std::cout << "Testing flags-driven kGmmVariances" << '\n';
      test_flags_driven_update(*gmm, feats, kGmmVariances);
      std::cout << "Testing flags-driven kGmmWeights | kGmmMeans" << '\n';
      test_flags_driven_update(*gmm, feats, kGmmWeights | kGmmMeans);
      std::cout << "Testing component-wise accumulation" << '\n';
      TestComponentAcc(*gmm, feats);
    }

    iteration++;
  }

  {  // I/O tests
    GmmFlagsType flags_all = kGmmAll;
    est_gmm.Resize(gmm->NumGauss(),
      gmm->Dim(), flags_all);
    est_gmm.SetZero(flags_all);
    float loglike = 0.0;
    for (int32 i = 0; i < counter; i++) {
      loglike += est_gmm.AccumulateFromFull(*gmm, feats.Row(i), 1.0F);
    }
    test_io(*gmm, est_gmm, false, feats);
    test_io(*gmm, est_gmm, true, feats);
  }

  delete gmm;
  gmm = NULL;
}

int
main() {
  // repeat the test five times
  for (int i = 0; i < 2; i++)
    UnitTestEstimateFullGmm();
  std::cout << "Test OK.\n";
}
