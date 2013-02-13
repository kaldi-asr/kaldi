// feat/cmvn-test.cc

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


#include "transform/cmvn.h"

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

namespace cmvn_utils {


void TestInitStandardStats() {
  Matrix<double> temp;
  int32 dim = 10 + rand() % 5;
  InitStandardStats(dim, &temp);
  KALDI_ASSERT(temp.NumRows() == 2 && temp.NumCols() == dim + 1 &&
               temp.Row(0).Sum() == 1.0 && temp.Row(1).Sum() == dim);
}


void TestAddFrameToCmvnStats() {
  int32 dim = 10 + rand() % 5;
  Matrix<double> stats(2, dim + 1);
  Vector<BaseFloat> frame(dim);
  frame.SetRandn();
  double weight = 0.5;
  AddFrameToCmvnStats(weight, frame, &stats);
  AssertEqual(stats(0, dim), weight);
  for (int32 i = 0; i < dim; i++) {
    double x_stats = frame(i) * weight, x2_stats = frame(i) * x_stats;
    AssertEqual(x_stats, stats(0, i));
    AssertEqual(x2_stats, stats(1, i));
  }
}

// This function is used for testing.
void InitRandCmvnStats(int32 dim,
                       Matrix<double> *stats) {
  stats->Resize(2, dim + 1);
  int32 num_frames = 5 + rand() % 5;
  for (int32 i = 0; i < num_frames; i++) {
    Vector<BaseFloat> frame(dim);
    frame.SetRandn();
    AddFrameToCmvnStats(0.2 * (rand() % 5), frame, stats);
  }
}

// Also tests TransformStats, InvertTransform and ComposeTransforms.
void TestConvertStatsToTransform() {
  int32 dim = 10 + rand() % 5;
  Matrix<double> stats;
  InitRandCmvnStats(dim, &stats);

  Matrix<double> standard_target;
  InitStandardStats(dim, &standard_target);

  Matrix<double> transform;
  ConvertStatsToTransform(stats, &transform);

  Matrix<double> stats2;
  TransformStats(standard_target, transform, &stats2);
  stats2.Scale(stats(0, dim));
  AssertEqual(stats, stats2);

  // Now test InvertTransform.

  Matrix<double> inv_transform, standard_target_2;
  InvertTransform(transform, &inv_transform);
  TransformStats(stats, inv_transform, &standard_target_2);
  standard_target_2.Scale(1.0 / standard_target_2(0, dim));
  AssertEqual(standard_target, standard_target_2);

  Matrix<double> unit_transform;
  ComposeTransforms(transform, inv_transform, &unit_transform);
  KALDI_ASSERT(fabs(unit_transform.Row(0).Sum()) < 0.01); // shift == zeros.
  AssertEqual(dim, unit_transform.Row(1).Sum()); // scale = ones.
  
  // multiply the other way around too; should still produce the
  // unit transform.
  ComposeTransforms(inv_transform, transform, &unit_transform);
  KALDI_ASSERT(fabs(unit_transform.Row(0).Sum()) < 0.01); // shift == zeros.
  AssertEqual(dim, unit_transform.Row(1).Sum()); // scale = ones.
}


void TestGetTransform() {
  int32 dim = 10 + rand() % 5;
  Matrix<double> stats, stats2, transform, stats2check;
  InitRandCmvnStats(dim, &stats);
  InitRandCmvnStats(dim, &stats2);
  GetTransform(stats, stats2, &transform);
  TransformStats(stats, transform, &stats2check);
  stats2check.Scale(stats2(0, dim) / stats2check(0, dim));
  AssertEqual(stats2, stats2check);

  Matrix<double> stats3, stats4;
  ConvertInvTransformToStats(transform, &stats3);
  TransformStats(stats3, transform, &stats4);
  Matrix<double> unit_stats;
  InitStandardStats(dim, &unit_stats);
  AssertEqual(unit_stats, stats4);
}


} // end namespace cmvn_utils


} // end namespace kaldi


int main() {
  kaldi::cmvn_utils::TestInitStandardStats();
  kaldi::cmvn_utils::TestConvertStatsToTransform();
  kaldi::cmvn_utils::TestGetTransform();
}


