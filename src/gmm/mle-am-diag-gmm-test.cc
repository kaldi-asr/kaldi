// gmm/mle-am-diag-gmm-test.cc

// Copyright 2009-2012  Arnab Ghoshal

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

#include "gmm/model-test-common.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "util/kaldi-io.h"

using kaldi::AmDiagGmm;
using kaldi::AccumAmDiagGmm;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;
using namespace kaldi;

// Tests the Read() and Write() methods for the accumulators, in both binary
// and ASCII mode.
void TestAmDiagGmmAccsIO(const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &feats) {
  kaldi::GmmFlagsType flags = kaldi::kGmmAll;
  AccumAmDiagGmm accs;
  accs.Init(am_gmm, flags);
  BaseFloat loglike = 0.0;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    int32 state = RandInt(0, am_gmm.NumPdfs()-1);
    loglike += accs.AccumulateForGmm(am_gmm, feats.Row(i), state, 1.0);
  }
  KALDI_LOG << "Data log-likelihood = " << loglike << " over "
            << feats.NumRows() << " frames.";
  KALDI_LOG << "Accumulated values: log-like = " << accs.TotLogLike()
            << ", # frames = " << accs.TotCount();
  AssertEqual(accs.TotLogLike(), loglike, 1e-5);
  AssertEqual(accs.TotCount(), static_cast<BaseFloat>(feats.NumRows()), 1e-5);

  MleDiagGmmOptions config;
  AmDiagGmm *am_gmm1 = new AmDiagGmm();
  am_gmm1->CopyFromAmDiagGmm(am_gmm);
  MleAmDiagGmmUpdate(config, accs, flags, am_gmm1, NULL, NULL);

  int32 check_pdf = RandInt(0, am_gmm.NumPdfs()-1),
      check_frame = RandInt(0, feats.NumRows()-1);
  BaseFloat loglike1 = am_gmm1->LogLikelihood(check_pdf, feats.Row(check_frame));
  delete am_gmm1;

  // First, non-binary write
  accs.Write(kaldi::Output("tmpf", false).Stream(), false);
  bool binary_in;
  AccumAmDiagGmm *accs1 = new AccumAmDiagGmm();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  accs1->Read(ki1.Stream(), binary_in, false);
  AmDiagGmm *am_gmm2 = new AmDiagGmm();
  am_gmm2->CopyFromAmDiagGmm(am_gmm);
  MleAmDiagGmmUpdate(config, accs, flags, am_gmm2, NULL, NULL);
  BaseFloat loglike2 = am_gmm2->LogLikelihood(check_pdf, feats.Row(check_frame));
  kaldi::AssertEqual(loglike1, loglike2, 1e-6);
  delete am_gmm2;
  delete accs1;

  // Next, binary write
  accs.Write(kaldi::Output("tmpfb", true).Stream(), true);
  AccumAmDiagGmm *accs2 = new AccumAmDiagGmm();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  accs2->Read(ki2.Stream(), binary_in, false);
  AmDiagGmm *am_gmm3 = new AmDiagGmm();
  am_gmm3->CopyFromAmDiagGmm(am_gmm);
  MleAmDiagGmmUpdate(config, accs, flags, am_gmm3, NULL, NULL);
  BaseFloat loglike3 = am_gmm3->LogLikelihood(check_pdf, feats.Row(check_frame));
  kaldi::AssertEqual(loglike1, loglike3, 1e-6);
  delete am_gmm3;
  delete accs2;
}

void UnitTestMleAmDiagGmm() {
  int32 dim = 1 + kaldi::RandInt(0, 9),  // random dimension of the gmm
      num_pdfs = 5 + kaldi::RandInt(0, 9);  // random number of states

  AmDiagGmm am_gmm;
  int32 total_num_comp = 0;
  for (int32 i = 0; i < num_pdfs; i++) {
    int32 num_comp = 1 + kaldi::RandInt(0, 9);  // random mixture size
    kaldi::DiagGmm gmm;
    ut::InitRandDiagGmm(dim, num_comp, &gmm);
    am_gmm.AddPdf(gmm);
    total_num_comp += num_comp;
  }

  kaldi::Matrix<BaseFloat> feats;

  {  // First, generate random means and variances
    int32 num_feat_comp = total_num_comp + kaldi::RandInt(-total_num_comp/2,
                                                          total_num_comp/2);
    kaldi::Matrix<BaseFloat> means(num_feat_comp, dim),
        vars(num_feat_comp, dim);
    for (int32 m = 0; m < num_feat_comp; m++) {
      for (int32 d= 0; d < dim; d++) {
        means(m, d) = kaldi::RandGauss();
        vars(m, d) = exp(kaldi::RandGauss()) + 1e-2;
      }
    }
    // Now generate random features with those means and variances.
    feats.Resize(num_feat_comp * 200, dim);
    for (int32 m = 0; m < num_feat_comp; m++) {
      kaldi::SubMatrix<BaseFloat> tmp(feats, m*200, 200, 0, dim);
      ut::RandDiagGaussFeatures(200, means.Row(m), vars.Row(m), &tmp);
    }
  }
  TestAmDiagGmmAccsIO(am_gmm, feats);
}


int main() {
//  std::srand(time(NULL));
  for (int i = 0; i < 10; i++)
    UnitTestMleAmDiagGmm();
  std::cout << "Test OK.\n";
  return 0;
}
