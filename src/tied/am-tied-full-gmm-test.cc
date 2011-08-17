// tied/am-tied-diag-gmm-test.cc

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#include "gmm/diag-gmm.h"
#include "gmm/model-test-common.h"
#include "tied/tied-gmm.h"
#include "tied/am-tied-diag-gmm.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"

using kaldi::AmTiedDiagGmm;
using kaldi::TiedGmmPerFrameVars;
using kaldi::int32;
using kaldi::BaseFloat;
using kaldi::TiedGmm;
using kaldi::DiagGmm;
using kaldi::Vector;
namespace ut = kaldi::unittest;

// Tests the Read() and Write() methods, in both binary and ASCII mode, as well
// as Check(), CopyFromSgmm(), and methods in likelihood computations.
void TestAmTiedDiagGmmIO(const AmTiedDiagGmm &am_gmm) {
  int32 dim = am_gmm.Dim();
  TiedGmmPerFrameVars pfv;

  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; ++d) {
    feat(d) = kaldi::RandGauss();
  }

  BaseFloat loglike = 0.0;
  
  am_gmm.SetupPerFrameVars(&pfv);
  am_gmm.ComputePerFrameVars(feat, &pfv);
  
  for (int32 i = 0; i < am_gmm.NumPdfs(); ++i)
    loglike += am_gmm.LogLikelihood(pfv, i);
 
  // First, non-binary write
  am_gmm.Write(kaldi::Output("tmpf", false).Stream(), false);

  bool binary_in;
  AmTiedDiagGmm *am_gmm1 = new AmTiedDiagGmm();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  am_gmm1->Read(ki1.Stream(), binary_in);

  BaseFloat loglike1 = 0.0;
  
  am_gmm1->SetupPerFrameVars(&pfv);
  am_gmm1->ComputePerFrameVars(feat, &pfv);

  for (int32 i = 0; i < am_gmm1->NumPdfs(); ++i)
    loglike1 += am_gmm1->LogLikelihood(pfv, i);
  
  kaldi::AssertEqual(loglike, loglike1, 1e-4);

  // Next, binary write
  am_gmm1->Write(kaldi::Output("tmpfb", true).Stream(), true);
  delete am_gmm1;

  AmTiedDiagGmm *am_gmm2 = new AmTiedDiagGmm();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  am_gmm2->Read(ki2.Stream(), binary_in);
  
  BaseFloat loglike2 = 0.0;
  
  am_gmm2->SetupPerFrameVars(&pfv);
  am_gmm2->ComputePerFrameVars(feat, &pfv);
  
  for (int32 i = 0; i < am_gmm2->NumPdfs(); ++i)
    loglike2 += am_gmm2->LogLikelihood(pfv, i);

  kaldi::AssertEqual(loglike, loglike2, 1e-4);
  delete am_gmm2;
}

void UnitTestAmTiedDiagGmm() {
  int32 dim = 1 + kaldi::RandInt(0, 5);  // random dimension of the gmm
  int32 num_pdfs = 2 + kaldi::RandInt(0, 5);  // random number of codebooks
  int32 num_tied_pdfs = num_pdfs + kaldi::RandInt(0, 20);
  
  DiagGmm diag;
  int32 num_comp = 4 + kaldi::RandInt(0, 12);
  ut::InitRandDiagGmm(dim, num_comp, &diag);
  
  AmTiedDiagGmm am_gmm;
  am_gmm.Init(diag);
  
  // add codebooks
  for (int32 i = 1; i < num_pdfs; ++i) {
    num_comp = 4 + kaldi::RandInt(0, 12);  // random number of mixtures
    ut::InitRandDiagGmm(dim, num_comp, &diag);
    am_gmm.AddPdf(diag);
  }

  // add tied mixtures, round robin with codebooks
  for (int32 i = 0; i < num_tied_pdfs; ++i) {
    TiedGmm tied;
    int32 pdf_index = i % num_pdfs;
    tied.Setup(pdf_index, am_gmm.GetPdf(pdf_index).NumGauss());
    
    // generate random weights
    Vector<BaseFloat> wts(am_gmm.GetPdf(pdf_index).NumGauss());
    for (int32 j = 0; j < wts.Dim(); ++j)
      wts(j) = kaldi::RandInt(1, 1024);
    
    wts.Scale(1./wts.Sum());
    
    tied.SetWeights(wts);
    
    am_gmm.AddTiedPdf(tied);
  }
  
  am_gmm.ComputeGconsts();

  TestAmTiedDiagGmmIO(am_gmm);
}

int main() {
  for (int i = 0; i < 10; ++i) {
    UnitTestAmTiedDiagGmm();
  }
  std::cout << "Test OK.\n";
  return 0;
}
