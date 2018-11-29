// cudamatrix/cu-compressed-matrix-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

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


#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix-lib.h"

using namespace kaldi;


namespace kaldi {

void CuCompressedMatrixTestSign() {
  int32 num_rows = RandInt(80, 100),
      num_cols = RandInt(80, 100);
  CuMatrix<BaseFloat> M(num_rows, num_cols);
  M.SetRandn();

  CuMatrix<BaseFloat> M2(num_rows, num_cols, kUndefined);

  CuCompressedMatrixBase *cm = NewCuCompressedMatrix(kCompressedMatrixUint8, 0.0);

  // this just stores (M(i, j) > 0 ? 1 : 0).
  cm->CopyFromMat(M);
  cm->CopyToMat(&M2);

  M.Heaviside(M);

  AssertEqual(M, M2);
  delete cm;
}

void CuCompressedMatrixTestNonnegative() {
  int32 num_rows = RandInt(80, 100),
      num_cols = RandInt(80, 100);
  CuMatrix<BaseFloat> M(num_rows, num_cols);
  M.SetRandUniform();

  BaseFloat range = 0.5 * RandInt(1, 5);
  M.Scale(range);

  CuCompressedMatrixType t = (RandInt(0, 1) == 0 ?
                              kCompressedMatrixUint8 :
                              kCompressedMatrixUint16);

  // since the input is in the correct range, truncating or not should make no
  // difference.
  bool truncate = (RandInt(0, 1) == 0);

  BaseFloat extra_error = 0.0;
  if (truncate && (RandInt(0, 1) == 0)) {
    // this tests that with truncate == true, adding a small offset, which would
    // take us outside the representable range, will not add too much extra
    // error.  (with truncate == false this would not be true because we wouldn't
    // round to the edges of the range, it would wrap around).
    extra_error = -0.01 * (RandInt(0, 1) == 0 ? 1.0 : -1.0);
    M.Add(extra_error);
  }

  CuCompressedMatrixBase *cm = NewCuCompressedMatrix(t, range, truncate);

  CuMatrix<BaseFloat> M2(num_rows, num_cols, kUndefined);

  cm->CopyFromMat(M);
  cm->CopyToMat(&M2);


  M2.AddMat(-1.0, M);

  BaseFloat diff_max = M2.Max(),
      diff_min = M2.Min();

  BaseFloat
      headroom = 1.1,
      max_expected_error = fabs(extra_error) + headroom * 0.5 *
         range / (t == kCompressedMatrixUint8 ? 255 : 65535);

  KALDI_ASSERT(diff_max < max_expected_error &&
               diff_min > -1.0 * max_expected_error);

  delete cm;
}

// this is like CuCompressedMatrixTestNonnegative but
// with signed integers, and input in the range [-range, +range].
void CuCompressedMatrixTestSymmetric() {
  int32 num_rows = RandInt(80, 100),
      num_cols = RandInt(80, 100);
  CuMatrix<BaseFloat> M(num_rows, num_cols);
  M.SetRandUniform();
  M.Scale(2.0);
  M.Add(-1.0);

  BaseFloat range = 0.5 * RandInt(1, 5);
  M.Scale(range);

  CuCompressedMatrixType t = (RandInt(0, 1) == 0 ?
                              kCompressedMatrixInt8 :
                              kCompressedMatrixInt16);

  // since the input is in the correct range, truncating or not should make no
  // difference.
  bool truncate = (RandInt(0, 1) == 0);

  BaseFloat extra_error = 0.0;
  if (truncate && (RandInt(0, 1) == 0)) {
    // this tests that with truncate == true, adding a small offset, which would
    // take us outside the representable range, will not add too much extra
    // error.  (with truncate == false this would not be true because we wouldn't
    // round to the edges of the range, it would wrap around).
    extra_error = -0.01 * (RandInt(0, 1) == 0 ? 1.0 : -1.0);
    M.Add(extra_error);
  }

  CuCompressedMatrixBase *cm = NewCuCompressedMatrix(t, range, truncate);

  CuMatrix<BaseFloat> M2(num_rows, num_cols, kUndefined);

  cm->CopyFromMat(M);
  cm->CopyToMat(&M2);


  M2.AddMat(-1.0, M);

  BaseFloat diff_max = M2.Max(),
      diff_min = M2.Min();

  BaseFloat
      headroom = 1.1,
      max_expected_error = fabs(extra_error) + headroom * 0.5 *
         range / (t == kCompressedMatrixInt8 ? 127 : 32767);

  KALDI_ASSERT(diff_max < max_expected_error &&
               diff_min > -1.0 * max_expected_error);

  delete cm;
}



} // namespace kaldi


int main() {
  SetVerboseLevel(1);
  // we don't run this test if CUDA is not compiled in, since
  // you can't instantiate class CuCompressedMatrix in that case.
#if HAVE_CUDA == 1
  CuDevice::Instantiate().SelectGpuId("yes");
  for (int32 i = 1; i < 10; i++) {
    CuCompressedMatrixTestSign();
    CuCompressedMatrixTestNonnegative();
    CuCompressedMatrixTestSymmetric();
  }

#endif
  return 0;
}
