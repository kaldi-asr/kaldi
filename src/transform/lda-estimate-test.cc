// transform/lda-estimate-test.cc

// Copyright 2009-2011  Jan Silovsky;  Saarland University

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

#include "transform/lda-estimate.h"
#include "util/common-utils.h"

using namespace kaldi;

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

void
test_io(const LdaEstimate &lda_est, bool binary) {
  std::cout << "Testing I/O, binary = " << binary << '\n';

  size_t dim = lda_est.Dim();

  lda_est.Write(Output("tmp_stats", binary).Stream(), binary);

  bool binary_in;
  LdaEstimate lda_est2;
  lda_est2.Init(lda_est.NumClasses(), lda_est.Dim());
  Input ki("tmp_stats", &binary_in);
  lda_est2.Read(ki.Stream(),
                binary_in, false);  // not adding

  Input ki2("tmp_stats", &binary_in);
  lda_est2.Read(ki2.Stream(),
                binary_in, true);  // adding

  lda_est2.Scale(0.5);
  // 0.5 -> make it same as what it would have been if we read just once.

  Matrix<BaseFloat> m1;
  Matrix<BaseFloat> m2;
  
  LdaEstimateOptions opts;
  opts.dim = dim;
  lda_est.Estimate(opts, &m1);
  lda_est2.Estimate(opts, &m2);
  
  m1.AddMat(-1.0, m2, kNoTrans);
  KALDI_ASSERT(m1.IsZero(1.0e-02));
}

void
UnitTestEstimateLda() {
  // using namespace kaldi;

  // dimension of the gmm
  size_t dim = kaldi::RandInt(10, 20);

  // number of mixtures in the data
  size_t num_class = dim + kaldi::RandInt(1, 10);  // must be at least dim + 1

  std::cout << "Running test with " << num_class << " classes and "
    << dim << " dimensional vectors" << '\n';

  // generate random feature vectors
  // first, generate parameters of vectors distribution
  // (mean and covariance matrices)
  Matrix<BaseFloat> means_f(num_class, dim);
  std::vector<SpMatrix<BaseFloat> > vars_f(num_class);
  std::vector<TpMatrix<BaseFloat> > vars_f_sqrt(num_class);
  for (size_t mix = 0; mix < num_class; mix++) {
    vars_f[mix].Resize(dim);
    vars_f_sqrt[mix].Resize(dim);
  }

  for (size_t m = 0; m < num_class; m++) {
    for (size_t d = 0; d < dim; d++) {
      means_f(m, d) = kaldi::RandGauss();
    }
    rand_posdef_spmatrix(dim, &vars_f[m], &vars_f_sqrt[m], NULL);
  }

  // second, generate X feature vectors for each of the mixture components
  size_t counter = 0;
  size_t vec_count = 1000;
  Matrix<BaseFloat> feats(num_class * vec_count, dim);
  std::vector<int32> feats_class(num_class * vec_count);
  Vector<BaseFloat> rnd_vec(dim);
  for (size_t m = 0; m < num_class; m++) {
    for (size_t i = 0; i < vec_count; i++) {
      for (size_t d = 0; d < dim; d++) {
        rnd_vec(d) = RandGauss();
      }
      feats.Row(counter).CopyFromVec(means_f.Row(m));
      feats.Row(counter).AddTpVec(1.0, vars_f_sqrt[m], kNoTrans, rnd_vec, 1.0);
      feats_class[counter] = m;
      ++counter;
    }
  }

  // Compute total covar and means for classes.
  Vector<double> total_mean(dim);
  Matrix<double> class_mean(num_class, dim);
  SpMatrix<double> total_covar(dim);
  Vector<double> tmp_vec_d(dim);
  for (size_t i = 0; i < counter; i++) {
    tmp_vec_d.CopyFromVec(feats.Row(i));
    class_mean.Row(feats_class[i]).AddVec(1.0, tmp_vec_d);
    total_mean.AddVec(1.0, tmp_vec_d);
    total_covar.AddVec2(1.0, tmp_vec_d);
  }
  total_mean.Scale(1/static_cast<double>(counter));
  total_covar.Scale(1/static_cast<double>(counter));
  total_covar.AddVec2(-1.0, total_mean);
  // Compute between-class covar.
  SpMatrix<double> bc_covar(dim);
  for (size_t c = 0; c < num_class; c++) {
    class_mean.Row(c).Scale(1/static_cast<double>(vec_count));
    bc_covar.AddVec2(static_cast<double>(vec_count)/counter, class_mean.Row(c));
  }
  bc_covar.AddVec2(-1.0, total_mean);
  // Compute within-class covar.
  SpMatrix<double> wc_covar(total_covar);
  wc_covar.AddSp(-1.0, bc_covar);

  // Estimate LDA transform matrix
  LdaEstimate lda_est;
  lda_est.Init(num_class, dim);
  lda_est.ZeroAccumulators();
  for (size_t i = 0; i < counter; i++) {
    lda_est.Accumulate(feats.Row(i), feats_class[i]);
  }
  LdaEstimateOptions opts;
  opts.dim = dim;

  Matrix<BaseFloat> lda_mat_bf,
      lda_mat_bf_mean_remove;
  lda_est.Estimate(opts, &lda_mat_bf);
  opts.remove_offset = true;
  lda_est.Estimate(opts, &lda_mat_bf_mean_remove);

  {
    Vector<BaseFloat> mean_ext(total_mean);
    mean_ext.Resize(mean_ext.Dim() + 1, kCopyData);
    mean_ext(mean_ext.Dim() - 1) = 1.0;
    Vector<BaseFloat> zero(mean_ext.Dim() - 1);
    zero.AddMatVec(1.0, lda_mat_bf_mean_remove, kNoTrans, mean_ext, 0.0);
    KALDI_ASSERT(zero.IsZero(0.001));
  }
  
  // Check lda_mat
  Matrix<double> lda_mat(lda_mat_bf);
  Matrix<double> tmp_mat(dim, dim);
  Matrix<double> wc_covar_mat(wc_covar);
  Matrix<double> bc_covar_mat(bc_covar);
  // following product should give unit matrix
  tmp_mat.AddMatMatMat(1.0, lda_mat, kNoTrans, wc_covar_mat, kNoTrans,
    lda_mat, kTrans, 0.0);
  KALDI_ASSERT(tmp_mat.IsUnit());
  // following product should give diagonal matrix with ordered diagonal (desc)
  tmp_mat.AddMatMatMat(1.0, lda_mat, kNoTrans, bc_covar_mat, kNoTrans,
    lda_mat, kTrans, 0.0);
  KALDI_ASSERT(tmp_mat.IsDiagonal());
  for (int32 i = 1; i < static_cast<int32>(dim); i++) {
    if (tmp_mat(i, i) < 1.0e-10) { tmp_mat(i, i) = 0.0; }
    KALDI_ASSERT(tmp_mat(i - 1, i - 1) >= tmp_mat(i, i));
  }

  // test I/O
  test_io(lda_est, false);
  test_io(lda_est, true);
}

int
main() {
  // repeat the test X times
  for (int i = 0; i < 2; i++)
    UnitTestEstimateLda();
  std::cout << "Test OK.\n";
}
