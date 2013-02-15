// feat/balanced-cmn-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)
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


#include "transform/balanced-cmn.h"

namespace kaldi {
template<class Real> static void AssertEqual(const Matrix<Real> &A,
                                             const Matrix<Real> &B,
                                             float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j < A.NumCols();j++) {
	  KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
}

template<class Real> static void AssertEqual(const VectorBase<Real> &A,
                                             const VectorBase<Real> &B,
                                             float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0;i < A.Dim(); i++) 
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol*std::max(1.0, (double) (std::abs(A(i))+std::abs(B(i)))));
}

void BalancedCmnTest() {
  BalancedCmnConfig config;
  config.silence_proportion = 0.1 * (rand() % 11);
  config.count_cutoff = 100.0;

  int32 dim = 5 + rand() % 20;
  Vector<double> sil_average(dim), nonsil_average(dim);
  sil_average.SetRandn();
  nonsil_average.SetRandn();

  double global_sil_count = 1000.0 * (1 + rand() % 20),
      global_nonsil_count = 1000.0 * (1 + rand() % 20);
  // only 1st row matters in matrices below.
  Matrix<double> global_sil_stats(2, dim + 1),
      global_nonsil_stats(2, dim + 1);
  global_sil_stats.Row(0).Range(0, dim).AddVec(global_sil_count, sil_average);
  global_sil_stats(0, dim) = global_sil_count;

  global_nonsil_stats.Row(0).Range(0, dim).AddVec(global_nonsil_count, nonsil_average);
  global_nonsil_stats(0, dim) = global_nonsil_count;

  for (int32 i = 0; i < 20; i++) {  // for each speaker..
    bool use_exact_feats = false;
    Vector<double> offset(dim); // only relevant if use_exact_feats set
    
    BalancedCmn cmn(config, global_sil_stats, global_nonsil_stats);

    int32 num_frames = 50 * (1 + rand() % 5);
    Matrix<double> feats(num_frames, dim);
    feats.SetRandn();

    Vector<BaseFloat> nonsil_prob(num_frames);    
    if (i < 4) { // Let the stats be all classified as silence.
      nonsil_prob.Set(0.0);
      cmn.AccStats(Matrix<BaseFloat>(feats), nonsil_prob);
    } else if (i < 8) { // Let the stats be all classified as nonsilence.
      nonsil_prob.Set(1.0);
      cmn.AccStats(Matrix<BaseFloat>(feats), nonsil_prob);
    } else {
      // set nonsil_prob randomly..
      for (int32 j = 0; j < nonsil_prob.Dim(); j++)
        nonsil_prob(j) = 0.1 * (rand() % 11);
      if (rand() % 2 == 0) {
        use_exact_feats = true;
        offset.SetRandn();
        for (int32 j = 0; j < num_frames; j++) {
          SubVector<double> x(feats, j);
          BaseFloat p_nonsil = nonsil_prob(j), p_sil = 1.0 - p_nonsil;
          x.SetZero();
          x.AddVec(p_sil, sil_average);
          x.AddVec(p_nonsil, nonsil_average);
          x.AddVec(1.0, offset);
        }
      }
    }

    cmn.AccStats(Matrix<BaseFloat>(feats), nonsil_prob);

    // Normalize with what it produces...
    const Matrix<double> &stats = cmn.GetStats();
    KALDI_ASSERT(stats(0, dim) == 1.0);
    Vector<double> cmn_offset(dim); // the negative of how much
    // the cmn will move the features.
    cmn_offset.CopyFromVec(stats.Row(0).Range(0, dim));

    // Work out how much we the orginal speech and silence are offset from
    // their zero-mean normalized form.
    // The global offset is the appropriately weighted,
    // global nonsil and sil stats.
    Vector<double> global_offset(dim);
    global_offset.AddVec(1.0 * config.silence_proportion,
                         sil_average);
    global_offset.AddVec(1.0 *(1.0 - config.silence_proportion),
                         nonsil_average);

    Vector<double> expected_offset(global_offset);
    expected_offset.AddVec(1.0, offset); // what we added to the global stats..

    if (use_exact_feats) {
      KALDI_LOG << "cmn_offset is " << cmn_offset
                << ", expected_offset is " << expected_offset;
      AssertEqual(cmn_offset, expected_offset);
    }    
  }
}


} // end namespace kaldi


int main() {
  kaldi::g_kaldi_verbose_level = 10;
  for (kaldi::int32 i = 0; i < 20; i++)
    kaldi::BalancedCmnTest();
}


