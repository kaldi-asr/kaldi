// transform/fmllr-diag-gmm-test.cc

// Copyright 2009-2011 Microsoft Corporation

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

namespace kaldi {


void InitRandomGmm (DiagGmm *gmm_in) {
  int32 num_gauss = 5 + rand () % 4;
  int32 dim = 10 + Rand() % 10;
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

// This test is statistical and relies on some identities
// related to the Aikake criterion.
void UnitTestFmllrDiagGmm() {
  using namespace kaldi;
  DiagGmm gmm;
  InitRandomGmm(&gmm);
  int32 dim =  gmm.Dim();
  int32 npoints = dim*(dim+1)*5;
  Matrix<BaseFloat> rand_points(npoints, dim);
  for (int32 i = 0; i < npoints; i++) {
    SubVector<BaseFloat> row(rand_points, i);
    gmm.Generate(&row);
  }
  Matrix<BaseFloat> cur_xform(dim, dim+1);
  cur_xform.SetUnit();  // set diag to unit.

  int32 niters = 5;
  BaseFloat objf_change_tot = 0.0, objf_change, count;
  for (int32 j = 0; j < niters; j++) {
    FmllrOptions opts;
    FmllrDiagGmmAccs stats(dim, j % 2 == 0 ? opts : FmllrOptions());
    for (int32 i = 0; i < npoints; i++) {
      SubVector<BaseFloat> row(rand_points, i);
      if (j == 0) {  // split this case off to exercise more of the code.
        stats.AccumulateForGmm(gmm, row, 1.0);
      } else {
        Vector<BaseFloat> xformed_row(row);
        ApplyAffineTransform(cur_xform, &xformed_row);
        Vector<BaseFloat> posteriors(gmm.NumGauss());
        gmm.ComponentPosteriors(xformed_row, &posteriors);
        stats.AccumulateFromPosteriors(gmm, row, posteriors);
      }
    }
    stats.Update(opts, &cur_xform, &objf_change, &count);
    {  // Test for ApplyFeatureTransformToStats:
      BaseFloat objf_change_tmp, count_tmp;
      ApplyFeatureTransformToStats(cur_xform, &stats);
      Matrix<BaseFloat> mat(dim, dim+1);
      mat.SetUnit();
      stats.Update(opts, &mat, &objf_change_tmp, &count_tmp);
      // After we apply this transform to the stats, there should
      // be nothing to gain from further transforming the data.
      KALDI_ASSERT(objf_change_tmp/count_tmp < 0.01);
    }
    KALDI_LOG << "Objf change on iter " << j << " is " << objf_change;
    objf_change_tot += objf_change;
  }
  KALDI_ASSERT(ApproxEqual(count, npoints));
  int32 num_params = dim*(dim+1);
  BaseFloat expected_objf_change = 0.5 * num_params;
  KALDI_LOG << "Expected objf change is: not much more than " << expected_objf_change
            <<", seen: " << objf_change_tot;
  KALDI_ASSERT(objf_change_tot < 2.0 * expected_objf_change);  // or way too much.
  // This test relies on statistical laws and if it fails it does not *necessarily*
  // mean that something is wrong.
}


// This is a test for the diagonal update and also of ApplyModelTransformToStats().
void UnitTestFmllrDiagGmmDiagonal() {
  using namespace kaldi;
  DiagGmm gmm;
  InitRandomGmm(&gmm);
  int32 dim =  gmm.Dim();
  int32 npoints = dim*(dim+1)*5;
  Matrix<BaseFloat> rand_points(npoints, dim);
  for (int32 i = 0; i < npoints; i++) {
    SubVector<BaseFloat> row(rand_points, i);
    gmm.Generate(&row);
  }
  Matrix<BaseFloat> cur_xform(dim, dim+1);
  cur_xform.SetUnit();  // set diag to unit.

  int32 niters = 2;
  BaseFloat objf_change_tot = 0.0, objf_change, count;
  FmllrOptions opts;
  opts.update_type = "diag";

  for (int32 j = 0; j < niters; j++) {
    FmllrDiagGmmAccs stats(dim, j % 2 == 0 ? opts : FmllrOptions());
    for (int32 i = 0; i < npoints; i++) {
      SubVector<BaseFloat> row(rand_points, i);
      if (j == 0) {  // split this case off to exercise more of the code.
        stats.AccumulateForGmm(gmm, row, 1.0);
      } else {
        Vector<BaseFloat> xformed_row(row);
        ApplyAffineTransform(cur_xform, &xformed_row);
        Vector<BaseFloat> posteriors(gmm.NumGauss());
        gmm.ComponentPosteriors(xformed_row, &posteriors);
        stats.AccumulateFromPosteriors(gmm, row, posteriors);
      }
    }

    stats.Update(opts, &cur_xform, &objf_change, &count);
    {  // Test for ApplyModelTransformToStats:
      BaseFloat objf_change_tmp, count_tmp;
      ApplyModelTransformToStats(cur_xform, &stats);
      Matrix<BaseFloat> mat(dim, dim+1);
      mat.SetUnit();
      stats.Update(opts, &mat, &objf_change_tmp, &count_tmp);
      // After we apply this transform to the stats, there should
      // be nothing to gain from further transforming the data.
      KALDI_ASSERT(objf_change_tmp/count_tmp < 0.01);
    }
    KALDI_LOG << "Objf change on iter " << j << " is " << objf_change;
    objf_change_tot += objf_change;
  }
  KALDI_ASSERT(ApproxEqual(count, npoints));
  int32 num_params = dim*2;
  BaseFloat expected_objf_change = 0.5 * num_params;
  KALDI_LOG << "Expected objf change is: not much more than " << expected_objf_change
            <<", seen: " << objf_change_tot;
  KALDI_ASSERT(objf_change_tot < 2.0 * expected_objf_change);  // or way too much.
  // This test relies on statistical laws and if it fails it does not *necessarily*
  // mean that something is wrong.
}


// This is a test for the offset-only update and also of ApplyModelTransformToStats().
void UnitTestFmllrDiagGmmOffset() {
  using namespace kaldi;
  DiagGmm gmm;
  InitRandomGmm(&gmm);
  int32 dim =  gmm.Dim();
  int32 npoints = dim*(dim+1)*5;
  Matrix<BaseFloat> rand_points(npoints, dim);
  for (int32 i = 0; i < npoints; i++) {
    SubVector<BaseFloat> row(rand_points, i);
    gmm.Generate(&row);
  }
  Matrix<BaseFloat> cur_xform(dim, dim+1);
  cur_xform.SetUnit();  // set diag to unit.

  int32 niters = 2;
  BaseFloat objf_change_tot = 0.0, objf_change, count;
  FmllrOptions opts;
  opts.update_type = "offset";

  for (int32 j = 0; j < niters; j++) {
    FmllrDiagGmmAccs stats(dim, j % 2 == 0 ? opts : FmllrOptions());
    for (int32 i = 0; i < npoints; i++) {
      SubVector<BaseFloat> row(rand_points, i);
      if (j == 0) {  // split this case off to exercise more of the code.
        stats.AccumulateForGmm(gmm, row, 1.0);
      } else {
        Vector<BaseFloat> xformed_row(row);
        ApplyAffineTransform(cur_xform, &xformed_row);
        Vector<BaseFloat> posteriors(gmm.NumGauss());
        gmm.ComponentPosteriors(xformed_row, &posteriors);
        stats.AccumulateFromPosteriors(gmm, row, posteriors);
      }
    }

    stats.Update(opts, &cur_xform, &objf_change, &count);
    {  // Test for ApplyModelTransformToStats:
      BaseFloat objf_change_tmp, count_tmp;
      ApplyModelTransformToStats(cur_xform, &stats);
      Matrix<BaseFloat> mat(dim, dim+1);
      mat.SetUnit();
      stats.Update(opts, &mat, &objf_change_tmp, &count_tmp);
      // After we apply this transform to the stats, there should
      // be nothing to gain from further transforming the data.
      KALDI_ASSERT(objf_change_tmp/count_tmp < 0.01);
    }
    KALDI_LOG << "Objf change on iter " << j << " is " << objf_change;
    objf_change_tot += objf_change;
  }
  KALDI_ASSERT(ApproxEqual(count, npoints));
  int32 num_params = dim;
  BaseFloat expected_objf_change = 0.5 * num_params;
  KALDI_LOG << "Expected objf change is: not much more than " << expected_objf_change
            <<", seen: " << objf_change_tot;
  KALDI_ASSERT(objf_change_tot < 2.0 * expected_objf_change);  // or way too much.
  // This test relies on statistical laws and if it fails it does not *necessarily*
  // mean that something is wrong.
}

}  // namespace kaldi ends here

int main() {
  for (int i = 0; i < 2; i++) {  // did more iterations when first testing...
    kaldi::UnitTestFmllrDiagGmmOffset();
    kaldi::UnitTestFmllrDiagGmmDiagonal();
    kaldi::UnitTestFmllrDiagGmm();
  }
  std::cout << "Test OK.\n";
}

