// sgmm/am-sgmm-test.cc

// Copyright 2012   Arnab Ghoshal
// Copyright 2009-2011  Saarland University

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
#include "util/kaldi-io.h"

using kaldi::AmSgmm;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

// Tests the initialization routines: InitializeFromFullGmm(), CopyFromSgmm()
// and CopyGlobalsInitVecs().
void TestSgmmInit(const AmSgmm &sgmm) {
  using namespace kaldi;
  int32 dim = sgmm.FeatureDim();
  kaldi::SgmmGselectConfig config;
  config.full_gmm_nbest = std::min(config.full_gmm_nbest, sgmm.NumGauss());

  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }
  kaldi::SgmmPerFrameDerivedVars frame_vars;
  frame_vars.Resize(sgmm.NumGauss(), sgmm.FeatureDim(),
                    sgmm.PhoneSpaceDim());

  std::vector<int32> gselect;
  sgmm.GaussianSelection(config, feat, &gselect);
  SgmmPerSpkDerivedVars empty;
  SgmmPerFrameDerivedVars per_frame;
  sgmm.ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike = sgmm.LogLikelihood(per_frame, 0);

  // First, test the CopyFromSgmm() method:
  AmSgmm *sgmm1 = new AmSgmm();
  sgmm1->CopyFromSgmm(sgmm, true);
  sgmm1->GaussianSelection(config, feat, &gselect);
  sgmm1->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike1 = sgmm1->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike1, 1e-4);
  delete sgmm1;

  AmSgmm *sgmm2 = new AmSgmm();
  sgmm2->CopyFromSgmm(sgmm, false);
  sgmm2->ComputeNormalizers();
  sgmm2->GaussianSelection(config, feat, &gselect);
  sgmm2->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike2 = sgmm2->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike2, 1e-4);
  delete sgmm2;

  // Next, initialize using the UBM from the current model
  AmSgmm *sgmm3 = new AmSgmm();
  sgmm3->InitializeFromFullGmm(sgmm.full_ubm(), sgmm.NumPdfs(),
                               sgmm.PhoneSpaceDim(), sgmm.SpkSpaceDim());
  sgmm3->ComputeNormalizers();
  sgmm3->GaussianSelection(config, feat, &gselect);
  sgmm3->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike3 = sgmm3->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike3, 1e-4);
  delete sgmm3;

  // Finally, copy the global parameters from the current model
  AmSgmm *sgmm4 = new AmSgmm();
  sgmm4->CopyGlobalsInitVecs(sgmm, sgmm.PhoneSpaceDim(), sgmm.SpkSpaceDim(),
                             sgmm.NumPdfs());
  sgmm4->ComputeNormalizers();
  sgmm4->GaussianSelection(config, feat, &gselect);
  sgmm4->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike4 = sgmm4->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike4, 1e-4);
  delete sgmm4;
}

// Tests the Read() and Write() methods, in both binary and ASCII mode, as well
// as Check(), and methods in likelihood computations.
void TestSgmmIO(const AmSgmm &sgmm) {
  using namespace kaldi;
  int32 dim = sgmm.FeatureDim();
  kaldi::SgmmGselectConfig config;
  config.full_gmm_nbest = std::min(config.full_gmm_nbest, sgmm.NumGauss());

  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }
  kaldi::SgmmPerFrameDerivedVars frame_vars;
  frame_vars.Resize(sgmm.NumGauss(), sgmm.FeatureDim(),
                    sgmm.PhoneSpaceDim());

  std::vector<int32> gselect;
  sgmm.GaussianSelection(config, feat, &gselect);
  SgmmPerSpkDerivedVars empty;
  SgmmPerFrameDerivedVars per_frame;
  sgmm.ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike = sgmm.LogLikelihood(per_frame, 0);

  // First, non-binary write
  sgmm.Write(kaldi::Output("tmpf", false).Stream(), false,
      kaldi::kSgmmWriteAll);

  bool binary_in;
  AmSgmm *sgmm1 = new AmSgmm();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  sgmm1->Read(ki1.Stream(), binary_in);
  sgmm1->Check(true);
  sgmm1->GaussianSelection(config, feat, &gselect);
  sgmm1->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);

  BaseFloat loglike1 = sgmm1->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike1, 1e-4);

  // Next, binary write
  sgmm1->Write(kaldi::Output("tmpfb", true).Stream(), true,
      kaldi::kSgmmWriteAll);
  delete sgmm1;

  AmSgmm *sgmm2 = new AmSgmm();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  sgmm2->Read(ki2.Stream(), binary_in);
  sgmm2->Check(true);
  sgmm2->GaussianSelection(config, feat, &gselect);
  sgmm2->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike2 = sgmm2->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike2, 1e-4);
  delete sgmm2;

  unlink("tmpf");
  unlink("tmpfb");
}

void TestSgmmSubstates(const AmSgmm &sgmm) {
  using namespace kaldi;
  int32 target_substates = 2 * sgmm.NumPdfs();
  kaldi::Vector<BaseFloat> occs(sgmm.NumPdfs());
  for (int32 i = 0; i < occs.Dim(); i++)
    occs(i) = std::fabs(kaldi::RandGauss()) * (kaldi::RandUniform()+1);
  AmSgmm *sgmm1 = new AmSgmm();
  sgmm1->CopyFromSgmm(sgmm, false);
  sgmm1->SplitSubstates(occs, target_substates, 0.01, 0.2, 1000);
  sgmm1->ComputeNormalizers();
  sgmm1->Check(true);
  int32 dim = sgmm.FeatureDim();
  kaldi::SgmmGselectConfig config;
  config.full_gmm_nbest = std::min(config.full_gmm_nbest, sgmm.NumGauss());
  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }

  std::vector<int32> gselect;
  sgmm.GaussianSelection(config, feat, &gselect);

  SgmmPerSpkDerivedVars empty;
  SgmmPerFrameDerivedVars per_frame;
  sgmm.ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike = sgmm.LogLikelihood(per_frame, 0);

  sgmm1->GaussianSelection(config, feat, &gselect);
  sgmm1->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike1 = sgmm1->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike1, 1e-2);

  delete sgmm1;
}

void TestSgmmIncreaseDim(const AmSgmm &sgmm) {
  using namespace kaldi;
  int32 target_phn_dim = static_cast<int32>(1.5 * sgmm.PhoneSpaceDim());
  int32 target_spk_dim = sgmm.PhoneSpaceDim() - 1;

  int32 dim = sgmm.FeatureDim();
  kaldi::SgmmGselectConfig config;
  config.full_gmm_nbest = std::min(config.full_gmm_nbest, sgmm.NumGauss());
  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }
  kaldi::SgmmPerFrameDerivedVars frame_vars;

  std::vector<int32> gselect;
  sgmm.GaussianSelection(config, feat, &gselect);
  SgmmPerSpkDerivedVars empty;
  SgmmPerFrameDerivedVars per_frame;
  sgmm.ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike = sgmm.LogLikelihood(per_frame, 0);

  kaldi::Matrix<BaseFloat> norm_xform;
  kaldi::ComputeFeatureNormalizer(sgmm.full_ubm(), &norm_xform);
  AmSgmm *sgmm1 = new AmSgmm();
  sgmm1->CopyFromSgmm(sgmm, false);
  sgmm1->Check(true);
  sgmm1->IncreasePhoneSpaceDim(target_phn_dim, norm_xform);
  sgmm1->ComputeNormalizers();
  sgmm1->Check(true);


  sgmm1->GaussianSelection(config, feat, &gselect);
  sgmm1->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike1 = sgmm1->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike1, 1e-4);

  sgmm1->IncreaseSpkSpaceDim(target_spk_dim, norm_xform);
  sgmm1->Check(true);
  sgmm1->GaussianSelection(config, feat, &gselect);
  sgmm1->ComputePerFrameVars(feat, gselect, empty, 0.0, &per_frame);
  BaseFloat loglike2 = sgmm1->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike2, 1e-4);
  delete sgmm1;
}

void TestSgmmPreXform(const AmSgmm &sgmm) {
  kaldi::Matrix<BaseFloat> xform, inv_xform;
  kaldi::Vector<BaseFloat> diag_scatter;
  kaldi::Vector<BaseFloat> occs(sgmm.NumPdfs());
  occs.Set(100);
  sgmm.ComputeFmllrPreXform(occs, &xform, &inv_xform, &diag_scatter);
  int32 dim = xform.NumRows();
  kaldi::SubMatrix<BaseFloat> a_pre(xform, 0, dim, 0, dim),
      a_inv(inv_xform, 0, dim, 0, dim);
  kaldi::Vector<BaseFloat> b_pre(dim), b_inv(dim);
  b_pre.CopyColFromMat(xform, dim);
  b_inv.CopyColFromMat(inv_xform, dim);
  kaldi::Matrix<BaseFloat> res_mat(dim, dim, kaldi::kSetZero);
  res_mat.AddMatMat(1.0, a_pre, kaldi::kNoTrans, a_inv, kaldi::kNoTrans, 0.0);
  KALDI_ASSERT(res_mat.IsUnit(1.0e-6));
  kaldi::Vector<BaseFloat> res_vec(dim, kaldi::kSetZero);
  res_vec.AddMatVec(1.0, a_inv, kaldi::kNoTrans, b_pre, 0.0);
  res_vec.AddVec(1.0, b_inv);
  KALDI_ASSERT(res_vec.IsZero(1.0e-6));
}

void UnitTestSgmm() {
  size_t dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  size_t num_comp = 3 + kaldi::RandInt(0, 9);  // random number of mixtures;
  // make sure it's more than one or we get errors initializing the SGMM.
  kaldi::FullGmm full_gmm;
  ut::InitRandFullGmm(dim, num_comp, &full_gmm);

  size_t num_states = 1;
  AmSgmm sgmm;
  kaldi::SgmmGselectConfig config;
  sgmm.InitializeFromFullGmm(full_gmm, num_states, dim+1, 0);
  sgmm.ComputeNormalizers();
  TestSgmmInit(sgmm);
  TestSgmmIO(sgmm);
  TestSgmmSubstates(sgmm);
  TestSgmmIncreaseDim(sgmm);
  TestSgmmPreXform(sgmm);
}

int main() {
  for (int i = 0; i < 10; i++)
    UnitTestSgmm();
  std::cout << "Test OK.\n";
  return 0;
}
