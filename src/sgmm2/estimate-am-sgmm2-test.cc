// sgmm2/estimate-am-sgmm2-test.cc

// Copyright 2009-2011  Saarland University (author:  Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (author: Daniel Povey)
//                      Arnab Ghoshal

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
#include "sgmm2/am-sgmm2.h"
#include "sgmm2/estimate-am-sgmm2.h"
#include "util/kaldi-io.h"

using kaldi::AmSgmm2;
using kaldi::MleAmSgmm2Accs;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

// Tests the Read() and Write() methods for the accumulators, in both binary
// and ASCII mode, as well as Check().
void TestSgmm2AccsIO(const AmSgmm2 &sgmm,
                     const kaldi::Matrix<BaseFloat> &feats) {
  using namespace kaldi;
  kaldi::SgmmUpdateFlagsType flags = kaldi::kSgmmAll & ~kSgmmSpeakerWeightProjections;
  kaldi::Sgmm2PerFrameDerivedVars frame_vars;
  kaldi::Sgmm2PerSpkDerivedVars empty;
  frame_vars.Resize(sgmm.NumGauss(), sgmm.FeatureDim(),
                    sgmm.PhoneSpaceDim());
  kaldi::Sgmm2GselectConfig sgmm_config;
  sgmm_config.full_gmm_nbest = std::min(sgmm_config.full_gmm_nbest,
                                        sgmm.NumGauss());
  MleAmSgmm2Accs accs(sgmm, flags, true);
  BaseFloat loglike = 0.0;

  for (int32 i = 0; i < feats.NumRows(); i++) {
    std::vector<int32> gselect;
    sgmm.GaussianSelection(sgmm_config, feats.Row(i), &gselect);
    sgmm.ComputePerFrameVars(feats.Row(i), gselect, empty, &frame_vars);
    loglike += accs.Accumulate(sgmm, frame_vars, 0, 1.0, &empty);
  }
  accs.CommitStatsForSpk(sgmm, empty);

  kaldi::MleAmSgmm2Options update_opts;
  update_opts.check_v = (rand()%2 == 0);
  AmSgmm2 *sgmm1 = new AmSgmm2();
  sgmm1->CopyFromSgmm2(sgmm, false, false);
  kaldi::MleAmSgmm2Updater updater(update_opts);
  updater.Update(accs, sgmm1, flags);
  sgmm1->ComputeDerivedVars();
  std::vector<int32> gselect;
  Sgmm2LikelihoodCache like_cache(sgmm.NumGroups(), sgmm.NumPdfs());
  
  sgmm1->GaussianSelection(sgmm_config, feats.Row(0), &gselect);
  sgmm1->ComputePerFrameVars(feats.Row(0), gselect, empty, &frame_vars);
  BaseFloat loglike1 = sgmm1->LogLikelihood(frame_vars, 0, &like_cache, &empty);
  delete sgmm1;

  // First, non-binary write
  accs.Write(kaldi::Output("tmpf", false).Stream(), false);
  bool binary_in;
  MleAmSgmm2Accs *accs1 = new MleAmSgmm2Accs();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  accs1->Read(ki1.Stream(), binary_in, false);
  accs1->Check(sgmm, true);
  AmSgmm2 *sgmm2 = new AmSgmm2();
  sgmm2->CopyFromSgmm2(sgmm, false, false);
  updater.Update(*accs1, sgmm2, flags);
  sgmm2->ComputeDerivedVars();
  sgmm2->GaussianSelection(sgmm_config, feats.Row(0), &gselect);
  sgmm2->ComputePerFrameVars(feats.Row(0), gselect, empty, &frame_vars);
  Sgmm2LikelihoodCache like_cache2(sgmm2->NumGroups(), sgmm2->NumPdfs());
  BaseFloat loglike2 = sgmm2->LogLikelihood(frame_vars, 0, &like_cache2, &empty);
  kaldi::AssertEqual(loglike1, loglike2, 1e-4);
  delete accs1;

  // Next, binary write
  accs.Write(kaldi::Output("tmpfb", true).Stream(), true);
  MleAmSgmm2Accs *accs2 = new MleAmSgmm2Accs();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  accs2->Read(ki2.Stream(), binary_in, false);
  accs2->Check(sgmm, true);
  AmSgmm2 *sgmm3 = new AmSgmm2();
  sgmm3->CopyFromSgmm2(sgmm, false, false);
  updater.Update(*accs2, sgmm3, flags);
  sgmm3->ComputeDerivedVars();
  sgmm3->GaussianSelection(sgmm_config, feats.Row(0), &gselect);
  sgmm3->ComputePerFrameVars(feats.Row(0), gselect, empty, &frame_vars);
  Sgmm2LikelihoodCache like_cache3(sgmm3->NumGroups(), sgmm3->NumPdfs());
  BaseFloat loglike3 = sgmm3->LogLikelihood(frame_vars, 0, &like_cache3, &empty);
  kaldi::AssertEqual(loglike1, loglike3, 1e-6);

  // Testing the MAP update of M
  update_opts.tau_map_M = 10;
  update_opts.full_col_cov = (RandUniform() > 0.5)? true : false;
  update_opts.full_row_cov = (RandUniform() > 0.5)? true : false;
  kaldi::MleAmSgmm2Updater updater_map(update_opts);
  sgmm3->CopyFromSgmm2(sgmm, false, false);
  updater_map.Update(*accs2, sgmm3, flags);

  delete accs2;
  delete sgmm2;
  delete sgmm3;
}

void UnitTestEstimateSgmm2() {
  int32 dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  int32 num_comp = 2 + kaldi::RandInt(0, 9);  // random mixture size
  kaldi::FullGmm full_gmm;
  ut::InitRandFullGmm(dim, num_comp, &full_gmm);

  AmSgmm2 sgmm;
  kaldi::Sgmm2GselectConfig config;
  std::vector<int32> pdf2group;
  pdf2group.push_back(0);
  sgmm.InitializeFromFullGmm(full_gmm, pdf2group, dim+1, dim, false, 0.9); // TODO-- make this true!
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
  sgmm.ComputeDerivedVars();
  TestSgmm2AccsIO(sgmm, feats);
}

int main() {
  for (int i = 0; i < 10; i++)
    UnitTestEstimateSgmm2();
  std::cout << "Test OK.\n";
  return 0;
}
