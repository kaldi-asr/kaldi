// transform/regtree-mllr-diag-gmm-test.cc

// Copyright 2009-2011   Arnab Ghoshal (Saarland University)

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
#include "gmm/estimate-diag-gmm.h"
#include "gmm/estimate-am-diag-gmm.h"
#include "gmm/model-test-common.h"
#include "transform/regtree-mllr-diag-gmm.h"

using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

namespace kaldi {


void UnitTestRegtreeMllrDiagGmm() {
  size_t dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  size_t num_comp = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
  kaldi::DiagGmm gmm;
  ut::InitRandDiagGmm(dim, num_comp, &gmm);
  kaldi::AmDiagGmm am_gmm;
  am_gmm.Init(gmm, 1);

  size_t num_comp2 = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
  kaldi::DiagGmm gmm2;
  ut::InitRandDiagGmm(dim, num_comp2, &gmm2);
  int32 npoints = dim*(dim+1)*2 + rand() % 100;
  Matrix<BaseFloat> adapt_data(npoints, dim);
  for (int32 j = 0; j < npoints; j++) {
    SubVector<BaseFloat> row(adapt_data, j);
    gmm2.Generate(&row);
  }

  kaldi::RegressionTree regtree;
  std::vector<int32> sil_indices;
  Vector<BaseFloat> state_occs(1);
  state_occs(0) = 100;
  regtree.BuildTree(state_occs, sil_indices, am_gmm, 2);
  int32 num_bclass = regtree.NumBaseclasses();

  kaldi::RegtreeMllrDiagGmmAccs accs;
  accs.Init(num_bclass, dim);
  for (int32 j = 0; j < npoints; j++) {
    accs.AccumulateForGmm(regtree, am_gmm, adapt_data.Row(j), 0, 1.0);
  }

  kaldi::RegtreeMllrDiagGmm mllr;
  kaldi::RegtreeMllrOptions opts;
  opts.min_count = 100;
  accs.Update(regtree, opts, &mllr, NULL, NULL);

}
}  // namespace kaldi ends here

int main() {
  for (int i = 0; i <= 8; i+=2)
    kaldi::UnitTestRegtreeMllrDiagGmm();
  std::cout << "Test OK.\n";
}

