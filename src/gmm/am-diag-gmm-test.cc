// gmm/am-diag-gmm-test.cc

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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
#include "util/kaldi-io.h"

using kaldi::AmDiagGmm;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

// Tests the Read() and Write() methods, in both binary and ASCII mode, as well
// as Check(), CopyFromSgmm(), and methods in likelihood computations.
void TestAmDiagGmmIO(const AmDiagGmm &am_gmm) {
  int32 dim = am_gmm.Dim();

  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }

  BaseFloat loglike = 0.0;
  for (int32 i = 0; i < am_gmm.NumPdfs(); i++)
    loglike += am_gmm.LogLikelihood(i, feat);
  // First, non-binary write
  am_gmm.Write(kaldi::Output("tmpf", false).Stream(), false);

  bool binary_in;
  AmDiagGmm *am_gmm1 = new AmDiagGmm();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  am_gmm1->Read(ki1.Stream(), binary_in);
  BaseFloat loglike1 = 0.0;
  for (int32 i = 0; i < am_gmm1->NumPdfs(); i++)
    loglike1 += am_gmm1->LogLikelihood(i, feat);
  kaldi::AssertEqual(loglike, loglike1, 1e-4);

  // Next, binary write
  am_gmm1->Write(kaldi::Output("tmpfb", true).Stream(), true);
  delete am_gmm1;

  AmDiagGmm *am_gmm2 = new AmDiagGmm();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  am_gmm2->Read(ki2.Stream(), binary_in);
  BaseFloat loglike2 = 0.0;
  for (int32 i = 0; i < am_gmm2->NumPdfs(); i++)
    loglike2 += am_gmm2->LogLikelihood(i, feat);
  kaldi::AssertEqual(loglike, loglike2, 1e-4);
  delete am_gmm2;

  unlink("tmpf");
  unlink("tmpfb");
}

void TestSplitStates(const AmDiagGmm &am_gmm) {
  int32 target_comp = 2 * am_gmm.NumGauss();
  kaldi::Vector<BaseFloat> occs(am_gmm.NumPdfs());
  for (int32 i = 0; i < occs.Dim(); i++)
    occs(i) = std::fabs(kaldi::RandGauss()) * (kaldi::RandUniform()+1) * 4;
  AmDiagGmm *am_gmm1 = new AmDiagGmm();
  am_gmm1->CopyFromAmDiagGmm(am_gmm);
  am_gmm1->SplitByCount(occs, target_comp, 0.01, 0.2, 0.0);

  int32 dim = am_gmm.Dim();
  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }
  BaseFloat loglike = am_gmm.LogLikelihood(0, feat);
  BaseFloat loglike1 = am_gmm1->LogLikelihood(0, feat);
  kaldi::AssertEqual(loglike, loglike1, 1e-2);

  delete am_gmm1;
}

void TestClustering(const AmDiagGmm &am_gmm) {
  int32 target_comp = am_gmm.NumGauss() / 5,
      interm_comp = am_gmm.NumGauss() / 2;
  kaldi::Vector<BaseFloat> occs(am_gmm.NumPdfs());
  for (int32 i = 0; i < occs.Dim(); i++)
    occs(i) = std::fabs(kaldi::RandGauss()) * (kaldi::RandUniform()+1) * 4;

  kaldi::UbmClusteringOptions ubm_opts(target_comp, 0.2, interm_comp, 0.01, 30);
  kaldi::DiagGmm ubm;
  ClusterGaussiansToUbm(am_gmm, occs, ubm_opts, &ubm);
}

void UnitTestAmDiagGmm() {
  int32 dim = 1 + kaldi::RandInt(0, 9),  // random dimension of the gmm
      num_pdfs = 5 + kaldi::RandInt(0, 9);  // random number of states

  AmDiagGmm am_gmm;
  for (int32 i = 0; i < num_pdfs; i++) {
    int32 num_comp = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
    kaldi::DiagGmm gmm;
    ut::InitRandDiagGmm(dim, num_comp, &gmm);
    am_gmm.AddPdf(gmm);
  }

  TestAmDiagGmmIO(am_gmm);
  TestSplitStates(am_gmm);
  TestClustering(am_gmm);
}

int main() {
  for (int i = 0; i < 5; i++)
    UnitTestAmDiagGmm();
  std::cout << "Test OK.\n";
  return 0;
}
