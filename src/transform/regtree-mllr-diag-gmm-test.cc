// transform/regtree-mllr-diag-gmm-test.cc

// Copyright 2009-2011   Saarland University
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

#include "util/common-utils.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "gmm/model-test-common.h"
#include "transform/regtree-mllr-diag-gmm.h"

using kaldi::int32;
using kaldi::BaseFloat;
using kaldi::RegtreeMllrDiagGmmAccs;
namespace ut = kaldi::unittest;

void TestMllrAccsIO(const kaldi::AmDiagGmm &am_gmm,
                    const kaldi::RegressionTree &regtree,
                    const RegtreeMllrDiagGmmAccs &accs,
                    const kaldi::Matrix<BaseFloat> adapt_data) {
  // First, non-binary write
  accs.Write(kaldi::Output("tmpf", false).Stream(), false);

  kaldi::RegtreeMllrDiagGmm mllr;
  kaldi::RegtreeMllrOptions opts;
  opts.min_count = 100;
  opts.use_regtree = false;
  accs.Update(regtree, opts, &mllr, NULL, NULL);
  kaldi::AmDiagGmm am1;
  am1.CopyFromAmDiagGmm(am_gmm);
  mllr.TransformModel(regtree, &am1);

  BaseFloat loglike = 0;
  int32 npoints = adapt_data.NumRows();
  for (int32 j = 0; j < npoints; j++) {
    loglike += am1.LogLikelihood(0, adapt_data.Row(j));
  }
  KALDI_LOG << "Per-frame loglike after adaptation = " << (loglike/npoints)
            << " over " << npoints << " frames.";

  size_t num_comp2 = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
  int32 dim = am_gmm.Dim();
  kaldi::DiagGmm gmm2;
  ut::InitRandDiagGmm(dim, num_comp2, &gmm2);
  kaldi::Vector<BaseFloat> data(dim);
  gmm2.Generate(&data);
  BaseFloat loglike1 = am1.LogLikelihood(0, data);
//  KALDI_LOG << "LL0 = " << loglike0 << "; LL1 = " << loglike1;

  KALDI_LOG << "Test ASCII IO.";
  bool binary_in;
  kaldi::RegtreeMllrDiagGmm mllr1;
  RegtreeMllrDiagGmmAccs *accs1 = new RegtreeMllrDiagGmmAccs();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  accs1->Read(ki1.Stream(), binary_in, false);
  accs1->Update(regtree, opts, &mllr1, NULL, NULL);
  delete accs1;
  kaldi::AmDiagGmm am2;
  am2.CopyFromAmDiagGmm(am_gmm);
  mllr.TransformModel(regtree, &am2);
  BaseFloat loglike2 = am2.LogLikelihood(0, data);
//  KALDI_LOG << "LL1 = " << loglike1 << "; LL2 = " << loglike2;
  kaldi::AssertEqual(loglike1, loglike2, 1e-6);

  kaldi::RegtreeMllrDiagGmm mllr2;
  // Next, binary write
  KALDI_LOG << "Test Binary IO.";
  accs.Write(kaldi::Output("tmpfb", true).Stream(), true);
  RegtreeMllrDiagGmmAccs *accs2 = new RegtreeMllrDiagGmmAccs();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  accs2->Read(ki2.Stream(), binary_in, false);
  accs2->Update(regtree, opts, &mllr2, NULL, NULL);
  delete accs2;
  kaldi::AmDiagGmm am3;
  am3.CopyFromAmDiagGmm(am_gmm);
  mllr.TransformModel(regtree, &am3);
  BaseFloat loglike3 = am3.LogLikelihood(0, data);
//  KALDI_LOG << "LL1 = " << loglike1 << "; LL3 = " << loglike3;
  kaldi::AssertEqual(loglike1, loglike3, 1e-6);
  
  unlink("tmpf");
  unlink("tmpfb");
}

void TestXformMean(const kaldi::AmDiagGmm &am_gmm,
                   const kaldi::RegressionTree &regtree,
                   const RegtreeMllrDiagGmmAccs &accs,
                   const kaldi::Matrix<BaseFloat> adapt_data) {
  kaldi::RegtreeMllrDiagGmm mllr;
  kaldi::RegtreeMllrOptions opts;
  opts.min_count = 100;
  opts.use_regtree = false;
  accs.Update(regtree, opts, &mllr, NULL, NULL);

  kaldi::AmDiagGmm am1;
  am1.CopyFromAmDiagGmm(am_gmm);
  mllr.TransformModel(regtree, &am1);

  kaldi::DiagGmm tmp_pdf;
  tmp_pdf.CopyFromDiagGmm(am_gmm.GetPdf(0));
  kaldi::Matrix<BaseFloat> tmp_means(am_gmm.GetPdf(0).NumGauss(), am_gmm.Dim());
  mllr.GetTransformedMeans(regtree, am_gmm, 0, &tmp_means);
  tmp_pdf.SetInvVarsAndMeans(tmp_pdf.inv_vars(), tmp_means);
  tmp_pdf.ComputeGconsts();

  BaseFloat loglike0 = 0, loglike = 0;
  int32 npoints = adapt_data.NumRows();
  for (int32 j = 0; j < npoints; j++) {
    loglike0 += am1.LogLikelihood(0, adapt_data.Row(j));
    loglike += tmp_pdf.LogLikelihood(adapt_data.Row(j));
  }
  KALDI_LOG << "Per-frame loglike after adaptation = " << (loglike0/npoints)
            << " over " << npoints << " frames.";
//  KALDI_LOG << "LL0 = " << loglike0 << "; LL = " << loglike;
  kaldi::AssertEqual(loglike0, loglike, 1e-6);

  kaldi::Matrix<BaseFloat> tmp_means2(am_gmm.GetPdf(0).NumGauss(), am_gmm.Dim());
  mllr.GetTransformedMeans(regtree, am_gmm, 0, &tmp_means2);
  tmp_pdf.SetInvVarsAndMeans(tmp_pdf.inv_vars(), tmp_means2);
  tmp_pdf.ComputeGconsts();

  BaseFloat loglike1 = 0;
  for (int32 j = 0; j < npoints; j++) {
    loglike1 += tmp_pdf.LogLikelihood(adapt_data.Row(j));
  }
//  KALDI_LOG << "LL = " << loglike << "; LL1 = " << loglike1;
  kaldi::AssertEqual(loglike, loglike1, 1e-6);
}


void UnitTestRegtreeMllrDiagGmm() {
  size_t dim = 1 + kaldi::RandInt(1, 9);  // random dimension of the gmm
  size_t num_comp = 1 + kaldi::RandInt(0, 5);  // random number of mixtures
  kaldi::DiagGmm gmm;
  ut::InitRandDiagGmm(dim, num_comp, &gmm);
  kaldi::AmDiagGmm am_gmm;
  am_gmm.Init(gmm, 1);

  size_t num_comp2 = 1 + kaldi::RandInt(0, 5);  // random number of mixtures
  kaldi::DiagGmm gmm2;
  ut::InitRandDiagGmm(dim, num_comp2, &gmm2);
  int32 npoints = dim*(dim+1)*10 + 500;
  kaldi::Matrix<BaseFloat> adapt_data(npoints, dim);
  for (int32 j = 0; j < npoints; j++) {
    kaldi::SubVector<BaseFloat> row(adapt_data, j);
    gmm2.Generate(&row);
  }

  kaldi::RegressionTree regtree;
  std::vector<int32> sil_indices;
  kaldi::Vector<BaseFloat> state_occs(1);
  state_occs(0) = npoints;
  regtree.BuildTree(state_occs, sil_indices, am_gmm, 1);
  int32 num_bclass = regtree.NumBaseclasses();

  kaldi::RegtreeMllrDiagGmmAccs accs;
  BaseFloat loglike = 0;
  accs.Init(num_bclass, dim);
  for (int32 j = 0; j < npoints; j++) {
    loglike += accs.AccumulateForGmm(regtree, am_gmm, adapt_data.Row(j),
                                     0, 1.0);
  }
  KALDI_LOG << "Per-frame loglike during accumulations = " << (loglike/npoints)
            << " over " << npoints << " frames.";

  TestMllrAccsIO(am_gmm, regtree, accs, adapt_data);
  TestXformMean(am_gmm, regtree, accs, adapt_data);
}

int main() {
  kaldi::g_kaldi_verbose_level = 5;
  for (int i = 0; i <= 10; i++)
    UnitTestRegtreeMllrDiagGmm();
  std::cout << "Test OK.\n";
}

