// transform/exponential-transform-test.cc

// Copyright 2009-2011 Microsoft Corporation

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
#include "transform/exponential-transform.h"
#include "transform/mllt.h"

namespace kaldi {



static void InitRandomGmm (DiagGmm *gmm_in) {
  int32 dim = 8 + rand() % 3;  // would have a larger dim if time wasn't an issue.
  int32 num_gauss = dim + 7;
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

template<class T> void TestIo(const T &t) {
  for (int32 i = 0; i < 2; i++) {
    bool binary = (i == 0);
    std::ostringstream s;
    t.Write(s, binary);
    std::istringstream s2(s.str());
    T t2;
    t2.Read(s2, binary);
    std::ostringstream s3;
    t2.Write(s3, binary);
    KALDI_ASSERT(s.str() == s3.str());
  }
}

// This test is statistical and relies on some identities
// related to the Aikake criterion.
// Also testing convergence of ET update.
void UnitTestExponentialTransformUpdate(EtNormalizeType norm_type,
                                        bool update_a,
                                        bool update_b) {
  KALDI_LOG << "Unit-test: norm-type = " << norm_type << ", update_a = " << update_a
            << ", update_b = " << update_b;
  using namespace kaldi;
  DiagGmm gmm;
  InitRandomGmm(&gmm);
  DiagGmm gmm_orig;
  gmm_orig.CopyFromDiagGmm(gmm);
  int32 dim =  gmm.Dim();
  int32 npoints = dim*(dim+1)*5;
  int32 nblocks = 3;
  Matrix<BaseFloat> rand_points(npoints*nblocks, dim);
  for (int32 i = 0; i < npoints*nblocks; i++) {
    SubVector<BaseFloat> row(rand_points, i);
    gmm.Generate(&row);
  }

  std::cout << "Dim = " << dim << "\n";
  ExponentialTransform et(dim, norm_type, rand());


  std::vector<Matrix<BaseFloat> > cur_xforms(nblocks);
  std::vector<Matrix<BaseFloat> > cur_Ds(nblocks);  // needed for update of B.
  for (int32 i = 0; i < nblocks; i++) {
    cur_xforms[i].Resize(dim, dim+1);
    cur_xforms[i].SetUnit();  // set diag to unit.
    cur_Ds[i].Resize(dim, dim+1);
    cur_Ds[i].SetUnit();
  }

  Matrix<BaseFloat> no_xform(cur_xforms[0]);

  double last_like_tot, orig_like_tot = 0.0;
  int32 niters = 10;
  double like_tot;
  for (int32 j = 0; j < niters; j++) {
    double objf_change_tot = 0.0;
    like_tot = 0.0;
    ExponentialTransformAccsA accs_a(dim);
    MlltAccs accs_b(dim);
    for (int32 k = 0; k < nblocks; k++) {
      Matrix<BaseFloat> &cur_xform(cur_xforms[k]);
      FmllrOptions opts;
      FmllrDiagGmmAccs stats(dim);
      for (int32 i = 0; i < npoints; i++) {
        SubVector<BaseFloat> row(rand_points, i+(npoints*k));
        Vector<BaseFloat> posteriors(gmm.NumGauss());
        BaseFloat like = gmm_orig.ComponentPosteriors(row, &posteriors);
        if (j == 0) orig_like_tot += like;
        stats.AccumulateFromPosteriors(gmm, row, posteriors);
      }
      // if (update_b && (!update_a || j%2 == 1))
      //            accs_b.AccumulateFromPosteriors(gmm, posteriors,
      // xformed_row, cur_Ds[k]);

      BaseFloat t;
      et.ComputeTransform(stats,
                          &cur_xform,
                          &t,
                          &(cur_Ds[k]));

      if (update_a && (!update_b || j%2 == 0))
        accs_a.AccumulateForSpeaker(stats, et, cur_Ds[k], t);  // just one "speaker".


      objf_change_tot
          += FmllrAuxFuncDiagGmm(cur_xform, stats)
          - FmllrAuxFuncDiagGmm(no_xform, stats);


      {  // Now update like_tot, and do some accumulation for B if needed.
        SubMatrix<BaseFloat> cur_xform_part(cur_xform, 0, dim, 0, dim);
        like_tot += npoints * cur_xform_part.LogDet();

        for (int32 i = 0; i < npoints; i++) {
          SubVector<BaseFloat> row(rand_points, i+(npoints*k));
            Vector<BaseFloat> xformed_row(row);
            ApplyAffineTransform(cur_xform, &xformed_row);
            Vector<BaseFloat> posteriors(gmm.NumGauss());
            like_tot += gmm.ComponentPosteriors(xformed_row, &posteriors);
            if (update_b && j%2 == 1)
              accs_b.AccumulateFromPosteriors(gmm,
                                              xformed_row,
                                              posteriors);
        }
      }
    }

    if (update_a && (!update_b || j%2 == 0)) {
      ExponentialTransformUpdateAOptions opts;
      BaseFloat count, objf_impr;
      accs_a.Update(opts, &et, &objf_impr, &count);
      TestIo(accs_a);
      KALDI_LOG << "Count is " << count << " and objf impr is " << objf_impr << " updating A";
    }
    if (update_b && j%2 == 1) {
      BaseFloat count, objf_impr;
      Matrix<BaseFloat> C(dim, dim);  // to transform GMM means.
      C.SetUnit();
      accs_b.Update(&C, &objf_impr, &count);
      et.ApplyC(C);
      TestIo(et);
      KALDI_LOG << "Count is " << count << " and objf impr is " << objf_impr << " updating B";
      // update the GMM means:
      Matrix<BaseFloat> means;
      gmm.GetMeans(&means);
      Matrix<BaseFloat> new_means(means.NumRows(), means.NumCols());
      new_means.AddMatMat(1.0, means, kNoTrans, C, kTrans, 0.0);
      gmm.SetMeans(new_means);
      gmm.ComputeGconsts();
    }

    if (j > 0)
      KALDI_LOG << "Objf change up to iter " << j << " is " << (like_tot - orig_like_tot);

    KALDI_LOG << "Objf-change-tot on iter " << j << " is " << objf_change_tot << "Note: this is an auxf improvement that may be relative to mismatched stats, and may not always be meaningful; see previous line for always-meaningful number.\n";

    if (j>0 && like_tot<last_like_tot-1.0)  // like decreased.
      KALDI_ERR << "Likelihood decreased: " << last_like_tot << " to "
                << like_tot;
    last_like_tot = like_tot;
  }
  int32 num_params = nblocks*( 1 + (norm_type== kEtNormalizeMeanAndVar ? 2*dim :
                                    norm_type == kEtNormalizeMean ? dim : 0))
      + (update_a ? dim*(dim+1) : 0)
      + (update_b ? dim*dim : 0);

  BaseFloat expected_objf_change = 0.5 * num_params;
  BaseFloat objf_change = like_tot - orig_like_tot;
  KALDI_LOG << "Expected objf change is: not much more than " << expected_objf_change
            <<", seen: " << objf_change;
  if (objf_change < 0.5 * expected_objf_change
     || objf_change > 2.0 * expected_objf_change)
    KALDI_WARN << "Objf change not the same as statistically expected (to within factor of 2)";
  // This test relies on statistical laws and if we get this warning it does not *necessarily*
  // mean that something is wrong.
}


}  // namespace kaldi ends here

int main() {
  bool long_test = false;
  using namespace kaldi;
  g_kaldi_verbose_level = 2;
  for (int i = 0; i < (long_test ? 5 : 1); i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        UnitTestExponentialTransformUpdate(kEtNormalizeMean, j != 0, k != 0);
        if ((j != 0 && k != 0) || long_test) {  // trying only a subset of these tests as it's taking
          // too long
          UnitTestExponentialTransformUpdate(kEtNormalizeMeanAndVar, j != 0, k != 0);
          UnitTestExponentialTransformUpdate(kEtNormalizeNone, j != 0, k != 0);
        }
      }
    }
  }

  std::cout << "Test OK.\n";
}

