// sgmm2/fmllr-sgmm2-test.cc

// Copyright 2009-2011  Saarland University (author:  Arnab Ghoshal)
//           2012  Johns Hopkins University (author: Daniel Povey)

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

#include <vector>

#include "gmm/model-test-common.h"
#include "sgmm2/am-sgmm2.h"
#include "sgmm2/fmllr-sgmm2.h"
#include "util/kaldi-io.h"

using kaldi::AmSgmm2;
using kaldi::int32;
using kaldi::BaseFloat;
using kaldi::Vector;
using kaldi::Matrix;
namespace ut = kaldi::unittest;

void ApplyFmllrXform(const kaldi::VectorBase<BaseFloat> &in,
                     const Matrix<BaseFloat> &xf,
                     Vector<BaseFloat> *out) {
  int32 dim = in.Dim();
  KALDI_ASSERT(xf.NumRows() == dim && xf.NumCols() == dim + 1);
  Vector<BaseFloat> tmp(dim + 1);
  tmp.Range(0, dim).CopyFromVec(in);
  tmp(dim) = 1.0;
  out->Resize(dim, kaldi::kSetZero);
  out->AddMatVec(1.0, xf, kaldi::kNoTrans, tmp, 0.0);
}

// Tests the Read() and Write() methods for the accumulators, in both binary
// and ASCII mode, as well as Check().
void TestSgmm2FmllrAccsIO(const AmSgmm2 &sgmm,
                         const kaldi::Matrix<BaseFloat> &feats) {
  KALDI_LOG << "Test IO start.";
  using namespace kaldi;
  int32 dim = sgmm.FeatureDim();
  kaldi::Sgmm2PerFrameDerivedVars frame_vars;
  kaldi::Sgmm2PerSpkDerivedVars empty;
  kaldi::Sgmm2FmllrGlobalParams fmllr_globals;
  kaldi::Sgmm2GselectConfig sgmm_config;

  frame_vars.Resize(sgmm.NumGauss(), dim, sgmm.PhoneSpaceDim());
  sgmm_config.full_gmm_nbest = std::min(sgmm_config.full_gmm_nbest,
                                        sgmm.NumGauss());
  kaldi::Vector<BaseFloat> occs(sgmm.NumPdfs());
  occs.Set(feats.NumRows());
  sgmm.ComputeFmllrPreXform(occs, &fmllr_globals.pre_xform_,
                            &fmllr_globals.inv_xform_,
                            &fmllr_globals.mean_scatter_);
  if (fmllr_globals.mean_scatter_.Min() == 0.0) {
    KALDI_WARN << "Global covariances low rank!";
    KALDI_WARN << "Diag-scatter = " << fmllr_globals.mean_scatter_;
    return;
  }

//  std::cout << "Pre-Xform = " << fmllr_globals.pre_xform_;
//  std::cout << "Inv-Xform = " << fmllr_globals.inv_xform_;

  FmllrSgmm2Accs accs;
  accs.Init(sgmm.FeatureDim(), sgmm.NumGauss());
  BaseFloat loglike = 0.0;
  std::vector<int32> gselect;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    sgmm.GaussianSelection(sgmm_config, feats.Row(i), &gselect);
    sgmm.ComputePerFrameVars(feats.Row(i), gselect, empty, &frame_vars);
    loglike += accs.Accumulate(sgmm, feats.Row(i), frame_vars, 0, 1.0,
                               &empty);
  }

  kaldi::Sgmm2FmllrConfig update_opts;
  update_opts.fmllr_min_count = 999; // Make sure it doesn't
  // divide 200, because the test can fail when we cross the boundary
  // of 1000 due to roundoff.  Actually it's weird because 1000 should
  // be exactly representable in float and in text.  But something's going wrong.
  kaldi::Matrix<BaseFloat> xform_mat(dim, dim+1);
  xform_mat.SetUnit();
  BaseFloat frames, impr;
  accs.Update(sgmm, fmllr_globals, update_opts, &xform_mat, &frames, &impr);

  Vector<BaseFloat> xformed_feat(dim);
  ApplyFmllrXform(feats.Row(0), xform_mat, &xformed_feat);
  sgmm.GaussianSelection(sgmm_config, xformed_feat, &gselect);
  sgmm.ComputePerFrameVars(xformed_feat, gselect, empty, &frame_vars);

  Sgmm2LikelihoodCache like_cache(sgmm.NumGroups(), sgmm.NumPdfs());
  BaseFloat loglike1 = sgmm.LogLikelihood(frame_vars, 0,
                                          &like_cache, &empty);

  bool binary_in;
  // First, non-binary write
  KALDI_LOG << "Test ASCII IO.";
  accs.Write(kaldi::Output("tmpf", false).Stream(), false);
  FmllrSgmm2Accs *accs1 = new FmllrSgmm2Accs();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  accs1->Read(ki1.Stream(), binary_in, false);
  xform_mat.SetUnit();
  accs1->Update(sgmm, fmllr_globals, update_opts, &xform_mat, NULL, NULL);
  ApplyFmllrXform(feats.Row(0), xform_mat, &xformed_feat);
  sgmm.GaussianSelection(sgmm_config, xformed_feat, &gselect);
  sgmm.ComputePerFrameVars(xformed_feat, gselect, empty, &frame_vars);
  like_cache.NextFrame();
  BaseFloat loglike2 = sgmm.LogLikelihood(frame_vars, 0,
                                          &like_cache, &empty);
  std::cout << "LL1 = " << loglike1 << ", LL2 = " << loglike2 << std::endl;
  
  kaldi::AssertEqual(loglike1, loglike2, 1e-2);
  delete accs1;

  // Next, binary write
  KALDI_LOG << "Test Binary IO.";
  accs.Write(kaldi::Output("tmpfb", true).Stream(), true);
  FmllrSgmm2Accs *accs2 = new FmllrSgmm2Accs();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  accs2->Read(ki2.Stream(), binary_in, false);
  xform_mat.SetUnit();
  accs2->Update(sgmm, fmllr_globals, update_opts, &xform_mat, NULL, NULL);
  ApplyFmllrXform(feats.Row(0), xform_mat, &xformed_feat);
  sgmm.GaussianSelection(sgmm_config, xformed_feat, &gselect);
  sgmm.ComputePerFrameVars(xformed_feat, gselect, empty,  &frame_vars);
  BaseFloat loglike3 = sgmm.LogLikelihood(frame_vars, 0,
                                          &like_cache, &empty);
  std::cout << "LL1 = " << loglike1 << ", LL3 = " << loglike3 << std::endl;
  kaldi::AssertEqual(loglike1, loglike3, 1e-4);
  delete accs2;
  KALDI_LOG << "Test IO end.";
}

void TestSgmm2FmllrSubspace(const AmSgmm2 &sgmm,
                         const kaldi::Matrix<BaseFloat> &feats) {
  KALDI_LOG << "Test Subspace start.";
  using namespace kaldi;
  int32 dim = sgmm.FeatureDim();
  kaldi::Sgmm2PerFrameDerivedVars frame_vars;
  kaldi::Sgmm2PerSpkDerivedVars empty;
  kaldi::Sgmm2FmllrGlobalParams fmllr_globals;
  kaldi::Sgmm2GselectConfig sgmm_config;

  frame_vars.Resize(sgmm.NumGauss(), dim, sgmm.PhoneSpaceDim());
  sgmm_config.full_gmm_nbest = std::min(sgmm_config.full_gmm_nbest,
                                        sgmm.NumGauss());
  kaldi::Vector<BaseFloat> occs(sgmm.NumPdfs());
  occs.Set(feats.NumRows());
  sgmm.ComputeFmllrPreXform(occs, &fmllr_globals.pre_xform_,
                            &fmllr_globals.inv_xform_,
                            &fmllr_globals.mean_scatter_);
  if (fmllr_globals.mean_scatter_.Min() == 0.0) {
    KALDI_WARN << "Global covariances low rank!";
    KALDI_WARN << "Diag-scatter = " << fmllr_globals.mean_scatter_;
    return;
  }

  FmllrSgmm2Accs accs;
  accs.Init(sgmm.FeatureDim(), sgmm.NumGauss());
  BaseFloat loglike = 0.0;
  std::vector<int32> gselect;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    sgmm.GaussianSelection(sgmm_config, feats.Row(i), &gselect);
    sgmm.ComputePerFrameVars(feats.Row(i), gselect, empty, &frame_vars);
    loglike += accs.Accumulate(sgmm, feats.Row(i), frame_vars, 0, 1.0,
                               &empty);
  }

  SpMatrix<double> grad_scatter(dim * (dim+1));
  accs.AccumulateForFmllrSubspace(sgmm, fmllr_globals, &grad_scatter);
  kaldi::Sgmm2FmllrConfig update_opts;
  EstimateSgmm2FmllrSubspace(grad_scatter, update_opts.num_fmllr_bases, dim,
                            &fmllr_globals);
//  update_opts.fmllr_min_count = 100;
  kaldi::Matrix<BaseFloat> xform_mat(dim, dim+1);
  xform_mat.SetUnit();
  accs.Update(sgmm, fmllr_globals, update_opts, &xform_mat, NULL, NULL);
  KALDI_LOG << "Test Subspace end.";
}

void TestSgmm2Fmllr() {
  // srand(time(NULL));
  int32 dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  int32 num_comp = 2 + kaldi::RandInt(0, 9);  // random number of mixtures
  kaldi::FullGmm full_gmm;
  ut::InitRandFullGmm(dim, num_comp, &full_gmm);

  AmSgmm2 sgmm;
  kaldi::Sgmm2GselectConfig config;
  std::vector<int32> pdf2group;
  pdf2group.push_back(0);
  sgmm.InitializeFromFullGmm(full_gmm, pdf2group, dim+1, dim, true, 0.9);
  sgmm.ComputeNormalizers();

  kaldi::Matrix<BaseFloat> feats;

  {  // First, generate random means and variances
    int32 num_feat_comp = num_comp + kaldi::RandInt(-num_comp/2, num_comp/2);
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
  TestSgmm2FmllrAccsIO(sgmm, feats);
  TestSgmm2FmllrSubspace(sgmm, feats);
}

int main() {
  kaldi::g_kaldi_verbose_level = 5;
  for (int i = 0; i < 10; i++)
    TestSgmm2Fmllr();
  std::cout << "Test OK.\n";
  return 0;
}
