// transform/regtree-fmllr-diag-gmm-test.cc

// Copyright 2009-2011  Georg Stemmer;  Saarland University

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

#include "util/common-utils.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "gmm/model-test-common.h"
#include "transform/regtree-fmllr-diag-gmm.h"

namespace kaldi {

static void
RandFullCova(Matrix<BaseFloat> *matrix) {
  size_t dim = matrix->NumCols();
  KALDI_ASSERT(matrix->NumCols() == matrix->NumRows());

  size_t iter = 0;
  size_t max_iter = 10000;
  // generate random (non-singular) matrix
  // until condition
  Matrix<BaseFloat> tmp(dim, dim);
  SpMatrix<BaseFloat> tmp2(dim);
  while (iter < max_iter) {
    tmp.SetRandn();
    if (tmp.Cond() < 100) break;
    iter++;
  }
  if (iter >= max_iter) {
    KALDI_ERR << "Internal error: found no random covariance matrix.";
  }
  // tmp * tmp^T will give positive definite matrix
  tmp2.AddMat2(1.0, tmp, kNoTrans, 0.0);
  matrix->CopyFromSp(tmp2);
}


/// Generate features for a certain covariance type
/// covariance_type == 0: full covariance
/// covariance_type == 1: diagonal covariance

enum cova_type {
  full,
  diag
};

static void
generate_features(cova_type covariance_type,
                  size_t n_gaussians,
                  size_t dim,
                  Matrix<BaseFloat> &trans_mat,
                  size_t frames_per_gaussian,
                  std::vector<Vector<BaseFloat>*> & train_feats,
                  std::vector<Vector<BaseFloat>*> & adapt_feats
                  ) {
  // compute inverse of the transformation matrix
  Matrix<BaseFloat> inv_trans_mat(dim, dim);
  inv_trans_mat.CopyFromMat(trans_mat, kNoTrans);
  inv_trans_mat.Invert();
  // the untransformed means are random
  Matrix<BaseFloat> untransformed_means(dim, n_gaussians);
  untransformed_means.SetRandn();
  untransformed_means.Scale(10);

  // the actual means result from
  // transformation with inv_trans_mat
  Matrix<BaseFloat> actual_means(dim, n_gaussians);

  // actual_means = inv_trans_mat * untransformed_means
  actual_means.AddMatMat(1.0, inv_trans_mat, kNoTrans,
                         untransformed_means, kNoTrans, 0.0);

  size_t train_counter = 0;

  // temporary variables
  Vector<BaseFloat> randomvec(dim);
  Matrix<BaseFloat> Sj(dim, dim);

  // loop over all gaussians
  for (size_t j = 0; j < n_gaussians; j++) {
    if (covariance_type == diag) {
      // random diagonal covariance for gaussian j
      Sj.SetZero();
      for (size_t d = 0; d < dim; d++) {
        Sj(d, d) = 2*exp(RandGauss());
      }
    }
    if (covariance_type == full) {
      // random full covariance for gaussian j
      RandFullCova(&Sj);
    }
    // compute inv_trans_mat * Sj
    Matrix<BaseFloat> tmp_matrix(dim, dim);
    tmp_matrix.AddMatMat(1.0, inv_trans_mat, kNoTrans, Sj, kNoTrans, 0.0);

    // compute features
    for (size_t i = 0; i < frames_per_gaussian; i++) {
      train_feats[train_counter] = new Vector<BaseFloat>(dim);
      adapt_feats[train_counter] = new Vector<BaseFloat>(dim);

      // initalize feature vector with mean of class j
      train_feats[train_counter]->CopyColFromMat(untransformed_means, j);
      adapt_feats[train_counter]->CopyColFromMat(actual_means, j);

      // determine random vector and
      // multiply the random vector with SJ
      // and add it to train_feats:
      // train_feats = train_feats + SJ * random
      // for adapt_feats we include the invtrans_mat:
      // adapt_feats = adapt_feats + invtrans_mat * SJ * random
      for (size_t d = 0; d < dim; d++) {
        randomvec(d) = RandGauss();
      }
      train_feats[train_counter]->AddMatVec(1.0, Sj, kNoTrans,
                                            randomvec, 1.0);
      adapt_feats[train_counter]->AddMatVec(1.0, tmp_matrix, kNoTrans,
                                            randomvec, 1.0);
      train_counter++;
    }
  }
  return;
}

void UnitTestRegtreeFmllrDiagGmm(cova_type feature_type, size_t max_bclass) {
  // dimension of the feature space
  size_t dim = 5 + rand() % 3;

  // number of components in the data
  size_t n_gaussians = 8;

  // number of data points to generate for every gaussian
  size_t frames_per_gaussian = 100;

  // generate random transformation matrix trans_mat
  Matrix<BaseFloat> trans_mat(dim, dim);
  int i = 0;
  while (i < 10000) {
    trans_mat.SetRandn();
    if (trans_mat.Cond() < 100) break;
    i++;
  }
  std::cout << "Condition of original Trans_Mat: " << trans_mat.Cond() << '\n';

  // generate many feature vectors for each of the mixture components
  std::vector<Vector<BaseFloat>*>
      train_feats(n_gaussians * frames_per_gaussian);
  std::vector<Vector<BaseFloat>*>
      adapt_feats(n_gaussians * frames_per_gaussian);

  generate_features(feature_type,
                    n_gaussians,
                    dim,
                    trans_mat,
                    frames_per_gaussian,
                    train_feats,
                    adapt_feats);

  // initial values for a GMM
  Vector<BaseFloat> weights(1);
  Matrix<BaseFloat> means(1, dim), vars(1, dim), invvars(1, dim);
  for (size_t d= 0; d < dim; d++) {
    means(0, d) = 0.0F;
    vars(0, d) = 1.0F;
  }
  weights(0) = 1.0F;
  invvars.CopyFromMat(vars);
  invvars.InvertElements();

  // new HMM with 1 state
  DiagGmm *gmm = new DiagGmm();
  gmm->Resize(1, dim);
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(invvars, means);
  gmm->ComputeGconsts();
  GmmFlagsType flags = kGmmAll;
  MleDiagGmmOptions opts;

  AmDiagGmm *am = new AmDiagGmm();
  am->AddPdf(*gmm);
  AccumAmDiagGmm *est_am = new AccumAmDiagGmm();

  // train HMM
  size_t iteration = 0;
  size_t maxiterations = 10;
  int32 maxcomponents = n_gaussians;
  BaseFloat loglike = 0;
  while (iteration < maxiterations) {
    est_am->Init(*am, flags);

    loglike = 0;
    for (size_t j = 0; j < train_feats.size(); j++) {
      loglike += est_am->AccumulateForGmm(*am, *train_feats[j], 0, 1.0);
    }
    MleAmDiagGmmUpdate(opts, *est_am, flags, am, NULL, NULL);

    std::cout << "Loglikelihood before iteration " << iteration << " : "
              << std::scientific << loglike << " number of components: "
              << am->NumGaussInPdf(0) << '\n';

    if ((iteration % 3 == 1) &&
        (am->NumGaussInPdf(0) * 2 <= maxcomponents)) {
      size_t n = am->NumGaussInPdf(0)*2;
      am->SplitPdf(0, n, 0.001);
    }
    iteration++;
  }

  // adapt HMM to the transformed feature vectors
  iteration = 0;
  RegtreeFmllrDiagGmmAccs * fmllr_accs = new RegtreeFmllrDiagGmmAccs();
  RegressionTree regtree;

  RegtreeFmllrOptions xform_opts;
  xform_opts.min_count = 100 * (1 + rand() % 10);
  xform_opts.use_regtree = (RandUniform() < 0.5)? false : true;

  size_t num_pdfs = 1;
  Vector<BaseFloat> occs(num_pdfs);
  for (int32 i = 0; i < static_cast<int32>(num_pdfs); i++) {
    occs(i) = 1.0/static_cast<BaseFloat>(num_pdfs);
  }
  std::vector<int32> silphones;
  regtree.BuildTree(occs, silphones, *am, max_bclass);
  maxiterations = 10;
  std::vector<Vector<BaseFloat>*> logdet(adapt_feats.size());
  for (size_t j = 0; j < adapt_feats.size(); j++) {
    logdet[j] = new Vector<BaseFloat>(1);
    logdet[j]->operator()(0) = 0.0;
  }
  while (iteration < maxiterations) {
    fmllr_accs->Init(regtree.NumBaseclasses(), dim);
    fmllr_accs->SetZero();
    RegtreeFmllrDiagGmm *new_fmllr = new RegtreeFmllrDiagGmm();
    loglike = 0;
    for (size_t j = 0; j < adapt_feats.size(); j++) {
      loglike += fmllr_accs->AccumulateForGmm(regtree, *am, *adapt_feats[j], 0, 1.0);
      loglike += logdet[j]->operator()(0);
    }
    std::cout << "FMLLR: Loglikelihood before iteration " << iteration << " : "
              << std::scientific << loglike << '\n';

    fmllr_accs->Update(regtree, xform_opts, new_fmllr, NULL, NULL);
    std::cout << "Got " << new_fmllr->NumBaseClasses() << " baseclasses\n";
    bool binary = (RandUniform() < 0.5)? true : false;
    std::cout << "Writing the transform to disk.\n";
    new_fmllr->Write(Output("tmpf", binary).Stream(), binary);
    RegtreeFmllrDiagGmm *fmllr_read = new RegtreeFmllrDiagGmm();
    bool binary_in;
    Input ki("tmpf", &binary_in);
    std::cout << "Reading the transform from disk.\n";
    fmllr_read->Read(ki.Stream(), binary_in);
    fmllr_read->Validate();

    // transform features
    std::vector<Vector<BaseFloat> > trans_feats(1);
    Vector<BaseFloat> trans_logdet;
//    new_fmllr->ComputeLogDets();
    trans_logdet.Resize(fmllr_read->NumRegClasses());
    fmllr_read->GetLogDets(&trans_logdet);
    for (size_t j = 0; j < adapt_feats.size(); j++) {
      fmllr_read->TransformFeature(*adapt_feats[j], &trans_feats);
      logdet[j]->operator()(0) += trans_logdet(0);
      adapt_feats[j]->CopyFromVec(trans_feats[0]);
    }
    iteration++;
    delete new_fmllr;
    delete fmllr_read;
  }

//  // transform features with empty transform
//  std::vector<Vector<BaseFloat> > trans_feats(1);
//  RegtreeFmllrDiagGmm *empty_fmllr = new RegtreeFmllrDiagGmm();
//  empty_fmllr->Init(0, 0);
//  for (size_t j = 0; j < adapt_feats.size(); j++) {
//    empty_fmllr->TransformFeature(*adapt_feats[j], &trans_feats);
//  }
//  delete empty_fmllr;

  // clean up
  delete fmllr_accs;
  delete est_am;
  delete am;
  delete gmm;
  DeletePointers(&logdet);
  DeletePointers(&train_feats);
  DeletePointers(&adapt_feats);
}
}  // namespace kaldi ends here

int main() {
  for (int i = 0; i <= 8; i+=2) {  // test is too slow so can't do too many
    std::cout << "--------------------------------------" << '\n';
    std::cout << "Test number " << i << '\n';
    std::cout << "--\nfeatures = full\n";
    kaldi::UnitTestRegtreeFmllrDiagGmm(kaldi::full, (i%10+1));
    std::cout << "--\nfeatures = diag\n";
    kaldi::UnitTestRegtreeFmllrDiagGmm(kaldi::diag, (i%10+1));
    std::cout << "--------------------------------------" << '\n';
  }
  std::cout << "Test OK.\n";
}

