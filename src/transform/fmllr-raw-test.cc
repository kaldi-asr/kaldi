// transform/fmllr-raw-test.cc

// Copyright  2009-2011 Microsoft Corporation
//            2013  Johns Hopkins University (author: Daniel Povey)

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
#include "transform/fmllr-diag-gmm.h"
#include "transform/fmllr-raw.h"

namespace kaldi {


void InitRandomGmm (DiagGmm *gmm_in) {
  int32 num_gauss = 5 + rand () % 4;
  int32 dim = 6 + Rand() % 5;
  DiagGmm &gmm(*gmm_in);
  gmm.Resize(num_gauss, dim);
  Matrix<BaseFloat> inv_vars(num_gauss, dim),
      means(num_gauss, dim);
  Vector<BaseFloat> weights(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    for (int32 j = 0; j < dim; j++) {
      inv_vars(i, j) = exp(RandGauss() * (1.0 / (1 + j)));
      means(i, j) = RandGauss() * (1.0 / (1 + j));
    }
    weights(i) = exp(RandGauss());
  }
  weights.Scale(1.0 / weights.Sum());
  gmm.SetWeights(weights);
  gmm.SetInvVarsAndMeans(inv_vars, means);
  gmm.ComputeGconsts();
}

void UnitTestFmllrRaw(bool use_offset) {
  using namespace kaldi;
  DiagGmm gmm;
  InitRandomGmm(&gmm);
  int32 model_dim =  gmm.Dim();

  int32 raw_dim = 10 + Rand() % 5;
  int32 num_splice = 1 + Rand() % 5;
  while (num_splice * raw_dim < model_dim) {
    num_splice++;
  }

  int32 full_dim = num_splice * raw_dim;
  int32 npoints = raw_dim*(raw_dim+1)*10;

  Matrix<BaseFloat> rand_points(npoints, full_dim);
  rand_points.SetRandn();
  
  Matrix<BaseFloat> lda_mllt(full_dim, full_dim + (use_offset ? 1 : 0)); // This is the full LDA+MLLT
  // matrix.  TODO: test with offset.
  lda_mllt.SetRandn();

  FmllrRawAccs accs(raw_dim, model_dim, lda_mllt);

  BaseFloat prev_objf_impr;
  for (int32 iter = 0; iter < 4; iter++) {
      
    for (int32 i = 0; i < npoints; i++) {
      SubVector<BaseFloat> sample(rand_points, i);
      accs.AccumulateForGmm(gmm, sample, 1.0);
    }

    Matrix<BaseFloat> fmllr_mat(raw_dim, raw_dim + 1);
    fmllr_mat.SetUnit(); // sets diagonal elements to one.
    
    FmllrRawOptions opts;
    BaseFloat objf_impr, count;
    accs.Update(opts, &fmllr_mat, &objf_impr, &count);

    KALDI_ASSERT(objf_impr > 0.0);

    if (iter != 0) {
      // This is not something provable, but is always true
      // in practice.
      KALDI_ASSERT(objf_impr < prev_objf_impr);
    }
    prev_objf_impr = objf_impr;
    
    
    // Now transform the raw features.
    for (int32 splice = 0; splice < num_splice; splice++) {
      SubMatrix<BaseFloat> raw_feats(rand_points,
                                     0, npoints,
                                     splice * raw_dim, raw_dim);
      for (int32 t = 0; t < npoints; t++) {
        SubVector<BaseFloat> this_feat(raw_feats, t);
        ApplyAffineTransform(fmllr_mat, &this_feat);
      }
    }
    accs.SetZero();
  }
}


}  // namespace kaldi ends here

int main() {
  kaldi::g_kaldi_verbose_level = 5;
    
  for (int i = 0; i < 2; i++) {  // did more iterations when first testing...
    kaldi::UnitTestFmllrRaw(i % 2 == 0);
  }
  std::cout << "Test OK.\n";
}
