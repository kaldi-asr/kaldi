// ivector/plda-test.cc

// Copyright 2013  Daniel Povey

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

#include "ivector/plda.h"


namespace kaldi {

void UnitTestPldaEstimation(int32 dim) {
  int32 num_classes = 1000 + Rand() % 10;
  Matrix<double> between_proj(dim, dim);
  while (between_proj.Cond() > 100)
    between_proj.SetRandn();
  Matrix<double> within_proj(dim, dim);
  while (within_proj.Cond() > 100)
    within_proj.SetRandn();


  Vector<double> global_mean(dim);
  global_mean.SetRandn();
  global_mean.Scale(10.0);

  PldaStats stats;

  for (int32 n = 0; n < num_classes; n++) {
    int32 num_egs = 1 + Rand() % 30;
    Vector<double> rand_vec(dim);
    rand_vec.SetRandn();
    Vector<double> class_mean(global_mean);
    class_mean.AddMatVec(1.0, between_proj, kNoTrans, rand_vec, 1.0);

    Matrix<double> rand_mat(num_egs, dim);
    rand_mat.SetRandn();
    Matrix<double> offset_mat(num_egs, dim);
    offset_mat.AddMatMat(1.0, rand_mat, kNoTrans, within_proj,
                         kTrans, 0.0);
    offset_mat.AddVecToRows(1.0, class_mean);

    double weight = 1.0 + (0.1 * (Rand() % 30));
    stats.AddSamples(weight,
                     offset_mat);
  }



  SpMatrix<double> between_var(dim), within_var(dim);
  between_var.AddMat2(1.0, between_proj, kNoTrans, 0.0);
  within_var.AddMat2(1.0, within_proj, kNoTrans, 0.0);

  stats.Sort();
  PldaEstimator estimator(stats);
  Plda plda;
  PldaEstimationConfig config;
  estimator.Estimate(config, &plda);

  KALDI_LOG << "Trace of true within-var is " << within_var.Trace();
  KALDI_LOG << "Trace of true between-var is " << between_var.Trace();

  {
    TpMatrix<double> C(dim);
    C.Cholesky(within_var);
    C.Invert();
    SpMatrix<double> between_var_proj(dim);
    between_var_proj.AddTp2Sp(1.0, C, kNoTrans, between_var, 0.0);
    Vector<double> s(dim);
    between_var_proj.Eig(&s);
    s.Scale(-1.0);
    std::sort(s.Data(), s.Data() + s.Dim());
    s.Scale(-1.0);
    KALDI_LOG << "Diagonal of between-class variance in normalized space "
              << "should be: " << s;
  }

}

}


/*
  This test is really just making sure that the PLDA estimation does not
  crash.  As for testing that it's working: I did this by eyeballing the
  output where it says "Trace of true within-var is XX" or "Trace of true
  between-var is XX" and comparing with the output from the estimation
  that says, "Trace of within-class variance is XX" and "Trace of betweeen-class
  variance is XX" (on the last iteration).  I make sure they are similar.
  I also checked that the objective function (where it says
  "Objective function is XX" is non-decreasing, and seems to be converging.
*/
int main() {
  using namespace kaldi;
  SetVerboseLevel(3);
  for (int i = 0; i < 5; i++)
    UnitTestPldaEstimation(i + 1);

  // UnitTestPldaEstimation(400);
  UnitTestPldaEstimation(40);
  std::cout << "Test OK.\n";
  return 0;
}
