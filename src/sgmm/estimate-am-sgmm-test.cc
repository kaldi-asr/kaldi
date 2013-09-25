// sgmm/estimate-am-sgmm-test.cc

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
#include "sgmm/am-sgmm.h"
#include "sgmm/estimate-am-sgmm.h"
#include "util/kaldi-io.h"

using kaldi::AmSgmm;
using kaldi::MleAmSgmmAccs;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

// Tests the Read() and Write() methods for the accumulators, in both binary
// and ASCII mode, as well as Check().
void TestUpdateAndAccsIO(const AmSgmm &sgmm,
                         const kaldi::Matrix<BaseFloat> &feats) {
  using namespace kaldi;
  kaldi::SgmmUpdateFlagsType flags = kaldi::kSgmmAll;
  kaldi::SgmmPerFrameDerivedVars frame_vars;
  kaldi::SgmmPerSpkDerivedVars empty;
  frame_vars.Resize(sgmm.NumGauss(), sgmm.FeatureDim(),
                    sgmm.PhoneSpaceDim());
  kaldi::SgmmGselectConfig sgmm_config;
  sgmm_config.full_gmm_nbest = std::min(sgmm_config.full_gmm_nbest,
                                        sgmm.NumGauss());
  MleAmSgmmAccs accs(sgmm, flags);
  BaseFloat loglike = 0.0;
  Vector<BaseFloat> empty_spk;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    std::vector<int32> gselect;
    sgmm.GaussianSelection(sgmm_config, feats.Row(i), &gselect);
    sgmm.ComputePerFrameVars(feats.Row(i), gselect, empty, 0.0, &frame_vars);
    loglike += accs.Accumulate(sgmm, frame_vars, empty_spk, 0, 1.0, flags);
  }
  accs.CommitStatsForSpk(sgmm, empty_spk);

  kaldi::MleAmSgmmOptions update_opts;
  update_opts.check_v = (rand()%2 == 0);
  AmSgmm *sgmm1 = new AmSgmm();
  sgmm1->CopyFromSgmm(sgmm, false);
  kaldi::MleAmSgmmUpdater updater(update_opts);
  updater.Update(accs, sgmm1, flags);
  std::vector<int32> gselect;

  sgmm1->GaussianSelection(sgmm_config, feats.Row(0), &gselect);
  sgmm1->ComputePerFrameVars(feats.Row(0), gselect, empty, 0.0, &frame_vars);
  BaseFloat loglike1 = sgmm1->LogLikelihood(frame_vars, 0);
  delete sgmm1;

  // First, non-binary write
  accs.Write(kaldi::Output("tmpf", false).Stream(), false);
  bool binary_in;
  MleAmSgmmAccs *accs1 = new MleAmSgmmAccs();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  accs1->Read(ki1.Stream(), binary_in, false);
  accs1->Check(sgmm, true);
  AmSgmm *sgmm2 = new AmSgmm();
  sgmm2->CopyFromSgmm(sgmm, false);
  updater.Update(*accs1, sgmm2, flags);

  sgmm2->GaussianSelection(sgmm_config, feats.Row(0), &gselect);
  sgmm2->ComputePerFrameVars(feats.Row(0), gselect, empty, 0.0, &frame_vars);
  BaseFloat loglike2 = sgmm2->LogLikelihood(frame_vars, 0);
  kaldi::AssertEqual(loglike1, loglike2, 1e-4);
  delete accs1;

  // Next, binary write
  accs.Write(kaldi::Output("tmpfb", true).Stream(), true);
  MleAmSgmmAccs *accs2 = new MleAmSgmmAccs();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  accs2->Read(ki2.Stream(), binary_in, false);
  accs2->Check(sgmm, true);
  AmSgmm *sgmm3 = new AmSgmm();
  sgmm3->CopyFromSgmm(sgmm, false);
  updater.Update(*accs2, sgmm3, flags);
  sgmm3->GaussianSelection(sgmm_config, feats.Row(0), &gselect);
  sgmm3->ComputePerFrameVars(feats.Row(0), gselect, empty, 0.0, &frame_vars);
  BaseFloat loglike3 = sgmm3->LogLikelihood(frame_vars, 0);
  kaldi::AssertEqual(loglike1, loglike3, 1e-6);

  // Testing the MAP update of M
  update_opts.tau_map_M = 100;
  update_opts.full_col_cov = (RandUniform() > 0.5)? true : false;
  update_opts.full_row_cov = (RandUniform() > 0.5)? true : false;
  kaldi::MleAmSgmmUpdater updater_map(update_opts);
  BaseFloat impr = updater_map.Update(*accs2, sgmm3, flags);
  KALDI_ASSERT(impr >= 0);

  delete accs2;
  delete sgmm2;
  delete sgmm3;
}

void UnitTestEstimateSgmm() {
  int32 dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  int32 num_comp = 2 + kaldi::RandInt(0, 9);  // random mixture size
  kaldi::FullGmm full_gmm;
  ut::InitRandFullGmm(dim, num_comp, &full_gmm);

  int32 num_states = 1;
  AmSgmm sgmm;
  kaldi::SgmmGselectConfig config;
  sgmm.InitializeFromFullGmm(full_gmm, num_states, dim+1, dim);
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
  TestUpdateAndAccsIO(sgmm, feats);
}

int main() {
  for (int i = 0; i < 10; i++)
    UnitTestEstimateSgmm();
  std::cout << "Test OK.\n";
  return 0;
}
