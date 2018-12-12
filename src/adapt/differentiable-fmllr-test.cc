// adapt/differentiable-fmllr-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

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

#include "adapt/differentiable-fmllr.h"
#include "matrix/sp-matrix.h"

namespace kaldi {
namespace differentiable_transform {



// Test derivatives produced by the Estimator object for K.
void TestCoreFmllrEstimatorKDeriv(
    BaseFloat gamma,
    const Matrix<BaseFloat> &G,
    const Matrix<BaseFloat> &K,
    const Matrix<BaseFloat> &A,
    CoreFmllrEstimator *estimator) {

  int32 num_directions = 4;
  Vector<BaseFloat> expected_changes(num_directions),
      actual_changes(num_directions);

  int32 dim = G.NumRows();
  BaseFloat epsilon = 1.0e-03 * gamma;
  Matrix<BaseFloat> A_deriv(dim, dim);
  // A_deriv defines the objective function: a random linear function in A.
  A_deriv.SetRandn();
  A_deriv.Add(0.1);  // Introduce some asymmetry.

  Matrix<BaseFloat> G_deriv(dim, dim),
      K_deriv(dim, dim);
  estimator->Backward(A_deriv, &G_deriv, &K_deriv);

  for (int32 i = 0; i < num_directions; i++) {
    Matrix<BaseFloat> K_new(dim, dim);
    K_new.SetRandn();
    K_new.Scale(epsilon);
    expected_changes(i) = TraceMatMat(K_new, K_deriv, kTrans);
    K_new.AddMat(1.0, K);
    FmllrEstimatorOptions opts;
    Matrix<BaseFloat> A_new(dim, dim);
    CoreFmllrEstimator estimator2(opts, gamma, G, K_new, &A_new);
    estimator2.Forward();
    A_new.AddMat(-1.0, A);
    // compute the change in our random linear objective function defined by
    // A_deriv, that would be produced by taking some small random change in K
    // and computing the A that results from that.
    actual_changes(i) = TraceMatMat(A_new, A_deriv, kTrans);
  }

  KALDI_LOG << "Expected changes: " << expected_changes
            << ", actual changes: " << actual_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
               << expected_changes << " vs. "
               << actual_changes;
  }
}

// Test derivatives produced by the Estimator object for G.
void TestCoreFmllrEstimatorGDeriv(
    BaseFloat gamma,
    const Matrix<BaseFloat> &G,
    const Matrix<BaseFloat> &K,
    const Matrix<BaseFloat> &A,
    CoreFmllrEstimator *estimator) {

  int32 num_directions = 4;
  Vector<BaseFloat> expected_changes(num_directions),
      actual_changes(num_directions);

  int32 dim = G.NumRows();
  BaseFloat epsilon = 1.0e-03 * gamma;
  Matrix<BaseFloat> A_deriv(dim, dim);
  // A_deriv defines the objective function: a random linear function in A.
  A_deriv.SetRandn();
  A_deriv.Add(0.1);  // Introduce some asymmetry.

  Matrix<BaseFloat> G_deriv(dim, dim),
      K_deriv(dim, dim);
  estimator->Backward(A_deriv, &G_deriv, &K_deriv);

  KALDI_ASSERT(G_deriv.IsSymmetric());

  for (int32 i = 0; i < num_directions; i++) {
    Matrix<BaseFloat> G_new(dim, dim);
    {
      SpMatrix<BaseFloat> s(dim);
      s.SetRandn();
      G_new.CopyFromSp(s);
    }
    G_new.Scale(epsilon);
    expected_changes(i) = TraceMatMat(G_new, G_deriv, kTrans);
    G_new.AddMat(1.0, G);
    FmllrEstimatorOptions opts;
    Matrix<BaseFloat> A_new(dim, dim);
    CoreFmllrEstimator estimator2(opts, gamma, G_new, K, &A_new);
    estimator2.Forward();
    A_new.AddMat(-1.0, A);
    // compute the change in our random linear objective function defined by
    // A_deriv, that would be produced by taking some small random change in K
    // and computing the A that results from that.
    actual_changes(i) = TraceMatMat(A_new, A_deriv, kTrans);
  }

  KALDI_LOG << "Expected changes: " << expected_changes
            << ", actual changes: " << actual_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
               << expected_changes << " vs. "
               << actual_changes;
  }
}



void UnitTestCoreFmllrEstimatorSimple() {
  int32 dim = RandInt(10, 20);
  BaseFloat gamma = RandInt(5, 10);
  Matrix<BaseFloat> G(dim, dim),
      K(dim, dim), A(dim, dim, kUndefined);
  G.AddToDiag(1.234 * gamma);
  K.AddToDiag(0.234 * gamma);
  FmllrEstimatorOptions opts;
  CoreFmllrEstimator estimator(opts, gamma, G, K, &A);
  BaseFloat objf_impr = estimator.Forward();
  KALDI_LOG << "A is " << A;
  KALDI_ASSERT(A.IsUnit(0.01));
  KALDI_ASSERT(fabs(objf_impr) < 0.01);
  for (int32 i = 0; i < 5; i++) {
    TestCoreFmllrEstimatorKDeriv(gamma, G, K, A, &estimator);
    TestCoreFmllrEstimatorGDeriv(gamma, G, K, A, &estimator);
  }
}

static void InitRandNonsingular(MatrixBase<BaseFloat> *M) {
  do {
    M->SetRandn();
  } while (M->Cond() > 50.0);
}


void UnitTestCoreFmllrEstimatorGeneral() {
  int32 dim = RandInt(10, 20);
  BaseFloat gamma = RandInt(5, 10);
  Matrix<BaseFloat> G(dim, dim),
      K(dim, dim), A(dim, dim, kUndefined);

  {
    // make sure G is symmetric and +ve definite.
    Matrix<BaseFloat> A(dim, dim + 10);
    A.SetRandn();
    G.AddMatMat(gamma, A, kNoTrans, A, kTrans, 0.0);
  }

  InitRandNonsingular(&K);
  K.Scale(gamma);
  FmllrEstimatorOptions opts;
  CoreFmllrEstimator estimator(opts, gamma, G, K, &A);
  BaseFloat objf_impr = estimator.Forward();
  KALDI_LOG << "A is " << A << ", objf impr is " << objf_impr;
  for (int32 i = 0; i < 5; i++) {
    TestCoreFmllrEstimatorKDeriv(gamma, G, K, A, &estimator);
    TestCoreFmllrEstimatorGDeriv(gamma, G, K, A, &estimator);
  }
}

void TestGaussianEstimatorDerivs(const MatrixBase<BaseFloat> &feats,
                                 const Posterior &post,
                                 const FmllrEstimatorOptions &opts,
                                 GaussianEstimator *g) {
  int32 n = 4;  // number of delta-params we use.
  Vector<BaseFloat> expected_changes(n),
      actual_changes(n);

  // if !test_mean_deriv, then we test the var deriv.
  bool test_mean_deriv = (RandInt(0, 1) == 0);

  int32 num_classes = g->NumClasses(), dim = g->Dim();

  Matrix<BaseFloat> mean_derivs(num_classes, dim);
  Vector<BaseFloat> var_derivs(num_classes);
  if (test_mean_deriv) {
    KALDI_LOG << "Testing mean derivs.";
    mean_derivs.SetRandn();
  } else {
    KALDI_LOG << "Testing var derivs.";
    var_derivs.SetRandn();
    var_derivs.Add(0.2);  // Nonzero mean makes the test easier to pass
  }
  g->AddToOutputDerivs(mean_derivs, var_derivs);
  Matrix<BaseFloat> feats_deriv(feats.NumRows(), feats.NumCols());
  g->AccStatsBackward(feats, post, &feats_deriv);

  BaseFloat epsilon = 1.0e-04;

  for (int32 i = 0; i < n; i++) {
    Matrix<BaseFloat> new_feats(feats.NumRows(),
                                feats.NumCols());
    new_feats.SetRandn();
    new_feats.Scale(epsilon);

    expected_changes(i) = TraceMatMat(feats_deriv, new_feats, kTrans);

    new_feats.AddMat(1.0, feats);

    GaussianEstimator g2(num_classes, dim);
    g2.AccStats(new_feats, post);
    g2.Estimate(opts);

    actual_changes(i) =
        TraceMatMat(mean_derivs, g2.GetMeans(), kTrans) -
        TraceMatMat(mean_derivs, g->GetMeans(), kTrans) +
        VecVec(var_derivs, g2.GetVars()) -
        VecVec(var_derivs, g->GetVars());
  }
  KALDI_LOG << "Actual changes are " << actual_changes
            << " vs. predicted " << expected_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
              << expected_changes << " vs. "
              << actual_changes;
  }
}

void TestFmllrEstimatorMeanDerivs(const MatrixBase<BaseFloat> &feats,
                                  const Posterior &post,
                                  const GaussianEstimator &g) {
  const MatrixBase<BaseFloat> &mu(g.GetMeans());
  const VectorBase<BaseFloat> &s(g.GetVars());

  int32 T = feats.NumRows(), dim = feats.NumCols(),
      num_classes = mu.NumRows();

  FmllrEstimatorOptions opts;

  FmllrEstimator f(opts, mu, s);

  Matrix<BaseFloat> adapted_feats(T, dim, kUndefined);
  BaseFloat objf_impr = f.ForwardCombined(feats, post, &adapted_feats);
  KALDI_LOG << "Forward objf-impr per frame (with same features) is "
            << objf_impr;

  // adapted_feats_deriv is the deriv of a random objective function
  // w.r.t the output (adapted) features.
  Matrix<BaseFloat> adapted_feats_deriv(T, dim),
      feats_deriv(T, dim);
  adapted_feats_deriv.SetRandn();
  adapted_feats_deriv.Add(0.1);  // Introduce some asymmetry.

  f.BackwardCombined(feats, post, adapted_feats_deriv, &feats_deriv);

  KALDI_LOG << "2-norm of adapted_feats_deriv is "
            << adapted_feats_deriv.FrobeniusNorm()
            << ", of feats_deriv is "
            << feats_deriv.FrobeniusNorm();

  const MatrixBase<BaseFloat> &mu_deriv = f.GetMeanDeriv();

  // measure the accuracy of the deriv in 4 random directions.
  int32 n = 4;
  BaseFloat epsilon = 1.0e-04;
  Vector<BaseFloat> expected_changes(n), actual_changes(n);
  for (int32 i = 0; i < n; i++) {
    Matrix<BaseFloat> new_mu(num_classes, dim, kUndefined),
        new_adapted_feats(T, dim, kUndefined);
    new_mu.SetRandn();
    // adding a systematic component helps the test to succeed in low precision.
    for (int32 c = 0; c < num_classes; c++) {
      new_mu.Row(c).Add(0.1 * RandInt(-1, 1));
    }
    new_mu.Scale(epsilon);
    expected_changes(i) = TraceMatMat(new_mu, mu_deriv, kTrans);
    new_mu.AddMat(1.0, mu);
    FmllrEstimator f2(opts, new_mu, s);
    f2.ForwardCombined(feats, post, &new_adapted_feats);
    actual_changes(i) =
        TraceMatMat(new_adapted_feats, adapted_feats_deriv, kTrans) -
        TraceMatMat(adapted_feats, adapted_feats_deriv, kTrans);
  }
  KALDI_LOG << "Expected changes are " << expected_changes
            << " vs. actual " << actual_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
              << expected_changes << " vs. "
               << actual_changes;
  }
}

void TestFmllrEstimatorVarDerivs(const MatrixBase<BaseFloat> &feats,
                                 const Posterior &post,
                                 const GaussianEstimator &g) {
  const MatrixBase<BaseFloat> &mu(g.GetMeans());
  const VectorBase<BaseFloat> &s(g.GetVars());

  int32 T = feats.NumRows(), dim = feats.NumCols(),
      num_classes = mu.NumRows();

  FmllrEstimatorOptions opts;

  FmllrEstimator f(opts, mu, s);

  Matrix<BaseFloat> adapted_feats(T, dim, kUndefined);
  BaseFloat objf_impr = f.ForwardCombined(feats, post, &adapted_feats);
  KALDI_LOG << "Forward objf-impr per frame (with same features) is "
            << objf_impr;

  // adapted_feats_deriv is the deriv of a random objective function
  // w.r.t the output (adapted) features.
  Matrix<BaseFloat> adapted_feats_deriv(T, dim),
      feats_deriv(T, dim);
  adapted_feats_deriv.SetRandn();
  // Adding a systematic component to the derivative makes the test easier
  // to pass, as the derivs are less random.
  adapted_feats_deriv.AddMat(0.1, feats);

  f.BackwardCombined(feats, post, adapted_feats_deriv, &feats_deriv);

  KALDI_LOG << "2-norm of adapted_feats_deriv is "
            << adapted_feats_deriv.FrobeniusNorm()
            << ", of feats_deriv is "
            << feats_deriv.FrobeniusNorm();

  const VectorBase<BaseFloat> &s_deriv = f.GetVarDeriv();

  // measure the accuracy of the deriv in 10 random directions
  int32 n = 10;
  BaseFloat epsilon = 0.001;
  Vector<BaseFloat> expected_changes(n), actual_changes(n);
  for (int32 i = 0; i < n; i++) {
    Vector<BaseFloat> new_s(num_classes, kUndefined);
    Matrix<BaseFloat> new_adapted_feats(T, dim, kUndefined);
    new_s.SetRandn();
    new_s.Scale(epsilon);
    expected_changes(i) = VecVec(new_s, s_deriv);
    new_s.AddVec(1.0, s);
    FmllrEstimator f2(opts, mu, new_s);
    f2.ForwardCombined(feats, post, &new_adapted_feats);
    actual_changes(i) =
        TraceMatMat(new_adapted_feats, adapted_feats_deriv, kTrans) -
        TraceMatMat(adapted_feats, adapted_feats_deriv, kTrans);
  }
  KALDI_LOG << "Expected changes are " << expected_changes
            << " vs. actual " << actual_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
              << expected_changes << " vs. "
               << actual_changes;
  }
}


void TestFmllrEstimatorSequence(const MatrixBase<BaseFloat> &feats,
                                const Posterior &post,
                                const GaussianEstimator &g) {
  // Do two fMLLR's in a row and see if the change in objf decreases.

  int32 T = feats.NumRows(), dim = feats.NumCols();
  const MatrixBase<BaseFloat> &mu(g.GetMeans());
  const VectorBase<BaseFloat> &s(g.GetVars());
  FmllrEstimatorOptions opts;

  FmllrEstimator f(opts, mu, s);

  Matrix<BaseFloat> adapted_feats(T, dim, kUndefined);
  BaseFloat objf_impr = f.ForwardCombined(feats, post, &adapted_feats);
  KALDI_LOG << "Forward objf-impr per frame (first time) is "
            << objf_impr;


  Matrix<BaseFloat> adapted_feats2(T, dim, kUndefined);
  FmllrEstimator f2(opts, mu, s);
  BaseFloat objf_impr2 = f.ForwardCombined(adapted_feats, post, &adapted_feats2);
  KALDI_LOG << "Forward objf-impr per frame (second time) is "
            << objf_impr2;
}

void TestFmllrEstimatorFeatDerivs(const MatrixBase<BaseFloat> &feats,
                                  const Posterior &post,
                                  const GaussianEstimator &g) {
  int32 T = feats.NumRows(), dim = feats.NumCols();
  const MatrixBase<BaseFloat> &mu(g.GetMeans());
  const VectorBase<BaseFloat> &s(g.GetVars());

  FmllrEstimatorOptions opts;

  FmllrEstimator f(opts, mu, s);

  Matrix<BaseFloat> adapted_feats(T, dim, kUndefined);
  BaseFloat objf_impr = f.ForwardCombined(feats, post, &adapted_feats);
  KALDI_LOG << "Forward objf-impr per frame (with same features) is "
            << objf_impr;

  // adapted_feats_deriv is the deriv of a random objective function
  // w.r.t the output (adapted) features.
  Matrix<BaseFloat> adapted_feats_deriv(T, dim),
      feats_deriv(T, dim);
  adapted_feats_deriv.SetRandn();
  adapted_feats_deriv.Add(0.1);  // Introduce some asymmetry.

  f.BackwardCombined(feats, post, adapted_feats_deriv, &feats_deriv);

  KALDI_LOG << "2-norm of adapted_feats_deriv is "
            << adapted_feats_deriv.FrobeniusNorm()
            << ", of feats_deriv is "
            << feats_deriv.FrobeniusNorm();

  // measure the accuracy of the deriv in 4 random directions.
  int32 n = 4;
  BaseFloat epsilon = 1.0e-03;
  Vector<BaseFloat> expected_changes(n), actual_changes(n);
  for (int32 i = 0; i < n; i++) {
    Matrix<BaseFloat> new_feats(T, dim, kUndefined),
        new_adapted_feats(T, dim, kUndefined);
    new_feats.SetRandn();
    new_feats.Add(RandGauss());  // will help to test whether the indirect
                                 // part of the derivative is accurate.
    new_feats.Scale(epsilon);
    expected_changes(i) = TraceMatMat(new_feats, feats_deriv, kTrans);
    new_feats.AddMat(1.0, feats);
    FmllrEstimator f2(opts, mu, s);
    f2.ForwardCombined(new_feats, post, &new_adapted_feats);
    actual_changes(i) =
        TraceMatMat(new_adapted_feats, adapted_feats_deriv, kTrans) -
        TraceMatMat(adapted_feats, adapted_feats_deriv, kTrans);
  }
  KALDI_LOG << "Expected changes are " << expected_changes
            << " vs. actual " << actual_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
              << expected_changes << " vs. "
               << actual_changes;
  }
}


void TestMeanOnlyTransformEstimatorMeanDerivs(
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post,
    const GaussianEstimator &g) {
  const MatrixBase<BaseFloat> &mu(g.GetMeans());

  int32 T = feats.NumRows(), dim = feats.NumCols(),
      num_classes = mu.NumRows();

  MeanOnlyTransformEstimator m(mu);

  Matrix<BaseFloat> adapted_feats(T, dim, kUndefined);
  m.ForwardCombined(feats, post, &adapted_feats);

  // adapted_feats_deriv is the deriv of a random objective function
  // w.r.t the output (adapted) features.
  Matrix<BaseFloat> adapted_feats_deriv(T, dim),
      feats_deriv(T, dim);
  adapted_feats_deriv.SetRandn();
  adapted_feats_deriv.Add(0.1);  // Introduce some asymmetry.

  m.BackwardCombined(feats, post, adapted_feats_deriv, &feats_deriv);

  KALDI_LOG << "2-norm of adapted_feats_deriv is "
            << adapted_feats_deriv.FrobeniusNorm()
            << ", of feats_deriv is "
            << feats_deriv.FrobeniusNorm();

  const MatrixBase<BaseFloat> &mu_deriv = m.GetMeanDeriv();

  // measure the accuracy of the deriv in 4 random directions.
  int32 n = 4;
  BaseFloat epsilon = 1.0e-03;
  Vector<BaseFloat> expected_changes(n), actual_changes(n);
  for (int32 i = 0; i < n; i++) {
    Matrix<BaseFloat> new_mu(num_classes, dim, kUndefined),
        new_adapted_feats(T, dim, kUndefined);
    new_mu.SetRandn();
    // adding a systematic component helps the test to succeed in low precision.
    for (int32 c = 0; c < num_classes; c++) {
      new_mu.Row(c).Add(0.1 * RandInt(-1, 1));
    }
    new_mu.Scale(epsilon);
    expected_changes(i) = TraceMatMat(new_mu, mu_deriv, kTrans);
    new_mu.AddMat(1.0, mu);
    MeanOnlyTransformEstimator m2(new_mu);
    m2.ForwardCombined(feats, post, &new_adapted_feats);
    actual_changes(i) =
        TraceMatMat(new_adapted_feats, adapted_feats_deriv, kTrans) -
        TraceMatMat(adapted_feats, adapted_feats_deriv, kTrans);
  }
  KALDI_LOG << "Expected changes are " << expected_changes
            << " vs. actual " << actual_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
              << expected_changes << " vs. "
               << actual_changes;
  }
}


void TestMeanOnlyTransformEstimatorFeatDerivs(
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post,
    const GaussianEstimator &g) {
  int32 T = feats.NumRows(), dim = feats.NumCols();
  const MatrixBase<BaseFloat> &mu(g.GetMeans());


  MeanOnlyTransformEstimator m(mu);

  Matrix<BaseFloat> adapted_feats(T, dim, kUndefined);
  m.ForwardCombined(feats, post, &adapted_feats);

  // adapted_feats_deriv is the deriv of a random objective function
  // w.r.t the output (adapted) features.
  Matrix<BaseFloat> adapted_feats_deriv(T, dim),
      feats_deriv(T, dim);
  adapted_feats_deriv.SetRandn();
  adapted_feats_deriv.Add(0.1);  // Introduce some asymmetry.

  m.BackwardCombined(feats, post, adapted_feats_deriv, &feats_deriv);

  KALDI_LOG << "2-norm of adapted_feats_deriv is "
            << adapted_feats_deriv.FrobeniusNorm()
            << ", of feats_deriv is "
            << feats_deriv.FrobeniusNorm();

  // measure the accuracy of the deriv in 4 random directions.
  int32 n = 4;
  BaseFloat epsilon = 1.0e-03;
  Vector<BaseFloat> expected_changes(n), actual_changes(n);
  for (int32 i = 0; i < n; i++) {
    Matrix<BaseFloat> new_feats(T, dim, kUndefined),
        new_adapted_feats(T, dim, kUndefined);
    new_feats.SetRandn();
    new_feats.Scale(epsilon);
    expected_changes(i) = TraceMatMat(new_feats, feats_deriv, kTrans);
    new_feats.AddMat(1.0, feats);
    MeanOnlyTransformEstimator m2(mu);
    m2.ForwardCombined(new_feats, post, &new_adapted_feats);
    actual_changes(i) =
        TraceMatMat(new_adapted_feats, adapted_feats_deriv, kTrans) -
        TraceMatMat(adapted_feats, adapted_feats_deriv, kTrans);
  }
  KALDI_LOG << "Expected changes are " << expected_changes
            << " vs. actual " << actual_changes;
  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
              << expected_changes << " vs. "
               << actual_changes;
  }
}


void UnitTestGaussianAndEstimators() {
  // It's important that the number of classes be greater than the dimension, or
  // we would get a low-rank K.
  int32 num_classes = RandInt(30, 40),
      dim = RandInt(10, 20),
      num_frames = RandInt(20 * num_classes, 40 * num_classes);

  GaussianEstimator g(num_classes, dim);

  Matrix<BaseFloat> feats(num_frames, dim);
  feats.SetRandn();
  feats.Add(0.2);  // Nonzero offset tests certain aspects of the code better.
  Posterior post(num_frames);
  for (int32 t = 0; t < num_frames; t++) {
    int32 n = RandInt(0, 2);
    for (int32 j = 0; j < n; j++) {
      int32 i = RandInt(0, num_classes - 1);
      BaseFloat p = 0.25 * RandInt(1, 5);
      post[t].push_back(std::pair<int32, BaseFloat>(i, p));
    }
  }
  g.AccStats(feats, post);
  FmllrEstimatorOptions opts;
  // avoid setting variance_sharing_weight to 1.0; it's hard for the tests to
  // succeed then, and there are valid reasons for that
  opts.variance_sharing_weight = 0.25 * RandInt(0, 2);
  g.Estimate(opts);
  KALDI_LOG << "Means are: "
            << g.GetMeans() << ", vars are: "
            << g.GetVars();

  TestGaussianEstimatorDerivs(feats, post, opts, &g);

  if (RandInt(0, 1) == 0) {
    opts.smoothing_count = 500.0;
  }

  {  // test FmllrEstimator
    TestFmllrEstimatorSequence(feats, post, g);
    TestFmllrEstimatorMeanDerivs(feats, post, g);
    TestFmllrEstimatorFeatDerivs(feats, post, g);
    TestFmllrEstimatorVarDerivs(feats, post, g);
  }

  {  // test MeanOnlyTransformEstimator.
    TestMeanOnlyTransformEstimatorMeanDerivs(feats, post, g);
    TestMeanOnlyTransformEstimatorFeatDerivs(feats, post, g);
  }




}



}  // namespace kaldi
}  // namespace differentiable_transform



int main() {
  using namespace kaldi::differentiable_transform;

  for (int32 i = 0; i < 50; i++) {
    UnitTestCoreFmllrEstimatorSimple();
    UnitTestCoreFmllrEstimatorGeneral();
    UnitTestGaussianAndEstimators();
  }
  std::cout << "Test OK.\n";
}
