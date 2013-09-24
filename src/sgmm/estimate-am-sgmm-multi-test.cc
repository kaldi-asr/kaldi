// sgmm/estimate-am-sgmm-multi-test.cc

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
#include "sgmm/am-sgmm.h"
#include "sgmm/estimate-am-sgmm.h"
#include "sgmm/estimate-am-sgmm-multi.h"
#include "util/kaldi-io.h"

using kaldi::AmSgmm;
using kaldi::MleAmSgmmAccs;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

// Tests the MleAmSgmmUpdaterMulti (and MleAmSgmmGlobalAccs) classes.
void TestMultiSgmmEst(const std::vector<AmSgmm*> &models,
                      const std::vector< kaldi::Matrix<BaseFloat> > &feats,
                      kaldi::SgmmUpdateFlagsType flags) {
  using namespace kaldi;
  int32 num_gauss = models[0]->NumGauss(),
      feat_dim = models[0]->FeatureDim(),
      phn_dim = models[0]->PhoneSpaceDim(),
      spk_dim = models[0]->SpkSpaceDim(),
      num_models = models.size();
  SgmmPerFrameDerivedVars frame_vars;
  SgmmPerSpkDerivedVars spk_vars;
  spk_vars.v_s.Resize(spk_dim);
  spk_vars.v_s.SetRandn();
  SgmmGselectConfig sgmm_config;
  frame_vars.Resize(num_gauss, feat_dim, phn_dim);
  sgmm_config.full_gmm_nbest = std::min(sgmm_config.full_gmm_nbest, num_gauss);

  std::vector<MleAmSgmmAccs*> accs(num_models);
  BaseFloat loglike = 0.0;
  for (int32 i = 0; i < num_models; ++i) {
    MleAmSgmmAccs* acc = new MleAmSgmmAccs(*models[i], flags);
    models[i]->ComputePerSpkDerivedVars(&spk_vars);
    for (int32 f = 0; f < feats[i].NumRows(); ++f) {
      std::vector<int32> gselect;
      models[i]->GaussianSelection(sgmm_config, feats[i].Row(f), &gselect);
      models[i]->ComputePerFrameVars(feats[i].Row(f), gselect, spk_vars, 0.0,
                                     &frame_vars);
      loglike += acc->Accumulate(*models[i], frame_vars, spk_vars.v_s, 0, 1.0,
                                 flags);
    }
    acc->CommitStatsForSpk(*models[i], spk_vars.v_s);
    accs[i] = acc;
  }

  std::vector<AmSgmm*> new_models(num_models);
  kaldi::MleAmSgmmOptions update_opts;
  for (int32 i = 0; i < num_models; ++i) {
    AmSgmm *sgmm1 = new AmSgmm();
    sgmm1->CopyFromSgmm(*models[i], false);
    new_models[i] = sgmm1;
  }

  // Updater class stores globals parameters; OK to initialize with any model
  // since it is assumed that they have the same global parameters.
  kaldi::MleAmSgmmUpdaterMulti updater(*models[0], update_opts);
  updater.Update(accs, new_models, flags);

  BaseFloat loglike1 = 0.0;
  for (int32 i = 0; i < num_models; ++i) {
    new_models[i]->ComputePerSpkDerivedVars(&spk_vars);
    for (int32 f = 0; f < feats[i].NumRows(); ++f) {
      std::vector<int32> gselect;
      new_models[i]->GaussianSelection(sgmm_config, feats[i].Row(f), &gselect);
      new_models[i]->ComputePerFrameVars(feats[i].Row(f), gselect, spk_vars, 0.0,
                                     &frame_vars);
      loglike1 += new_models[i]->LogLikelihood(frame_vars, 0);
    }
  }
  KALDI_LOG << "LL = " << loglike << "; LL1 = " << loglike1;
  AssertGeq(loglike1, loglike, 1e-6);

  DeletePointers(&accs);
  DeletePointers(&new_models);
}

void UnitTestEstimateSgmm() {
  int32 dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  int32 num_comp = 2 + kaldi::RandInt(0, 9);  // random mixture size
  kaldi::FullGmm full_gmm;
  ut::InitRandFullGmm(dim, num_comp, &full_gmm);

  int32 num_states = 1;
  int32 num_models = kaldi::RandInt(2, 9);
  std::vector<AmSgmm*> models(num_models);
  for (int32 i =0; i < num_models; ++i) {
    AmSgmm* sgmm = new AmSgmm();
    sgmm->InitializeFromFullGmm(full_gmm, num_states, dim+1, dim);
    sgmm->ComputeNormalizers();
    models[i] = sgmm;
  }

  std::vector< kaldi::Matrix<BaseFloat> > feats(num_models);
  for (int32 i = 0; i < num_models; ++i) {
    // First, generate random means and variances
    int32 num_feat_comp = num_comp + kaldi::RandInt(-num_comp/2, num_comp/2);
    kaldi::Matrix<BaseFloat> means(num_feat_comp, dim),
        vars(num_feat_comp, dim);
    for (int32 m = 0; m < num_feat_comp; ++m) {
      for (int32 d= 0; d < dim; d++) {
        means(m, d) = kaldi::RandGauss();
        vars(m, d) = exp(kaldi::RandGauss()) + 1e-2;
      }
    }
    // Now generate random features with those means and variances.
    feats[i].Resize(num_feat_comp * 200, dim);
    for (int32 m = 0; m < num_feat_comp; ++m) {
      kaldi::SubMatrix<BaseFloat> tmp(feats[i], m*200, 200, 0, dim);
      ut::RandDiagGaussFeatures(200, means.Row(m), vars.Row(m), &tmp);
    }
  }
  kaldi::SgmmUpdateFlagsType flags = kaldi::kSgmmAll;
  TestMultiSgmmEst(models, feats, flags);
  flags = (kaldi::kSgmmPhoneProjections | kaldi::kSgmmPhoneWeightProjections |
           kaldi::kSgmmCovarianceMatrix);
  TestMultiSgmmEst(models, feats, flags);
  flags = (kaldi::kSgmmSpeakerProjections | kaldi::kSgmmCovarianceMatrix |
           kaldi::kSgmmPhoneVectors);
  TestMultiSgmmEst(models, feats, flags);
  kaldi::DeletePointers(&models);
}

int main() {
  for (int i = 0; i < 10; ++i)
    UnitTestEstimateSgmm();
  std::cout << "Test OK.\n";
  return 0;
}
