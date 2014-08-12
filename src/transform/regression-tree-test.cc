// transform/regression-tree-test.cc

// Copyright 2009-2011  Jan Silovsky;   Saarland University

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

#include "transform/regression-tree.h"
#include "util/common-utils.h"

using namespace kaldi;

void
test_io(const RegressionTree &regtree,
        const AmDiagGmm &acmodel,
        bool binary) {
  std::cout << "Testing I/O, binary = " << binary << '\n';

  regtree.Write(Output("tmp_regtree", binary).Stream(),
                binary);

  bool binary_in;
  RegressionTree regtree2;

  Input ki("tmp_regtree", &binary_in);
  regtree2.Read(ki.Stream(),
                binary_in, acmodel);

  std::ostringstream s1, s2;
  regtree.Write(s1, false);
  regtree2.Write(s2, false);
  KALDI_ASSERT(s1.str() == s2.str());
}

// void
// join_gmm(const DiagGmm &gmm1, const DiagGmm &gmm2, DiagGmm *gmm) {
//  KALDI_ASSERT(gmm1.Dimension() == gmm2.Dimension());
//  size_t num_comp = gmm1.NumGauss() + gmm2.NumGauss();
//  size_t dim = gmm1.Dimension();
//
//  Matrix<BaseFloat> means1(gmm1.NumGauss());
//  size_t num_comp
// }

void
rand_diag_gmm(size_t num_comp, size_t dim, DiagGmm *gmm) {
  Vector<BaseFloat> weights(num_comp);
  Matrix<BaseFloat> means(num_comp, dim);
  Matrix<BaseFloat> vars(num_comp, dim);

  BaseFloat tot_weight = 0.0;
  for (size_t m = 0; m < num_comp; m++) {
    weights(m) = kaldi::RandUniform();
    for (size_t d= 0; d < dim; d++) {
      means(m, d) = kaldi::RandGauss();
      vars(m, d) = exp(kaldi::RandGauss()) + 1e-5;
    }
    tot_weight += weights(m);
  }
  weights.Scale(1.0/tot_weight);

  vars.InvertElements();
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(vars, means);
  gmm->ComputeGconsts();
}

void
UnitTestRegressionTree() {
  // using namespace kaldi;

  // dimension of the gmm
  // size_t dim = kaldi::RandInt(5, 20);
  size_t dim = 2;

  // number of mixtures in the data
  size_t num_comp = kaldi::RandInt(2, 2);;

  std::cout << "Running test with " << num_comp << " components and "
    << dim << " dimensional vectors" << '\n';

  // generate random gmm
  DiagGmm gmm1;
  gmm1.Resize(num_comp, dim);
  rand_diag_gmm(num_comp, dim, &gmm1);

  // shift means for components
  Matrix<BaseFloat> means2(num_comp, dim);
  Vector<BaseFloat> tmp_vec(dim);
  gmm1.GetMeans(&means2);
  for (int32 c = 0; c < static_cast<int32>(num_comp); c++) {
    // tmp_vec.SetRandn();
    // tmp_vec.Scale(0.01);
    tmp_vec.Set(0.001 * means2.Row(c).Max());
    means2.Row(c).AddVec(1.0, tmp_vec);
  }

  // let's have another gmm with shifted means
  DiagGmm gmm2;
  gmm2.CopyFromDiagGmm(gmm1);
  gmm2.SetMeans(means2);

  AmDiagGmm acmodel;
  acmodel.AddPdf(gmm1);
  acmodel.AddPdf(gmm2);

  // let's have uniform occupancies
  size_t num_pdfs = 2;
  Vector<BaseFloat> occs(num_pdfs);
  for (int32 i = 0; i < static_cast<int32>(num_pdfs); i++) {
    occs(i) = 1.0/static_cast<BaseFloat>(num_pdfs*num_comp);
  }

  for (int32 i = 0; i < gmm1.NumGauss(); i++) {
    gmm1.GetComponentMean(i, &tmp_vec);
    tmp_vec.Write(std::cout, false);
    gmm2.GetComponentMean(i, &tmp_vec);
    tmp_vec.Write(std::cout, false);
  }

  RegressionTree regtree;
  std::vector<int32> sil_pdfs;
  if (Rand() % 2 == 0)
    sil_pdfs.push_back(Rand() % 2);
  regtree.BuildTree(occs, sil_pdfs, acmodel, 2);

  // test I/O
  test_io(regtree, acmodel, false);
  // test_io(regtree, acmodel, true);
}

int
main() {
  // repeat the test X times
  for (int i = 0; i < 4; i++)
    UnitTestRegressionTree();
  std::cout << "Test OK.\n";
}
