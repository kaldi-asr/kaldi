// cudamatrix/cu-math-test.cc

// Copyright 2013 Johns Hopkins University (Author: David Snyder)

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
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-array.h"

#if defined(_MSC_VER)
#include <time.h>
#endif

using namespace kaldi;


namespace kaldi {


/*
 * Unit tests
 */
      
template<typename Real> 
static void UnitTestCuMathRandomize() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  CuMatrix<Real> src(M, N);
  CuMatrix<Real> tgt(M, N);
  CuArray<int32> copy_from_idx;

  src.SetRandn(); 
  int32 n_rows = src.NumRows();
  int32 n_columns = src.NumCols();
  std::vector<int32> copy_from_idx_vec;

  for (int32 i = 0; i < n_rows; i++) {
    copy_from_idx_vec.push_back(Rand() % n_rows);
  }
  copy_from_idx.CopyFromVec(copy_from_idx_vec);
  cu::Randomize(src, copy_from_idx, &tgt);
  
  for (int32 i = 0; i < n_rows; i++) {
    for (int32 j = 0; j < n_columns; j++) {
      Real src_val = src(copy_from_idx_vec.at(i), j);
      Real tgt_val = tgt(i, j);
      AssertEqual(src_val, tgt_val);
    }
  }
}


template<typename Real> 
static void UnitTestCuMathCopy() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  CuMatrix<Real> src(M, N);
  CuMatrix<Real> tgt(M, N);
  CuArray<int32> copy_from_idx;

  src.SetRandn(); 
  int32 n_rows = src.NumRows();
  int32 n_columns = src.NumCols();
  std::vector<int32> copy_from_idx_vec;

  for (int32 i = 0; i < n_columns; i++) {
    copy_from_idx_vec.push_back(Rand() % n_columns);
  }
  copy_from_idx.CopyFromVec(copy_from_idx_vec);
  cu::Copy(src, copy_from_idx, &tgt);
  
  for (int32 i = 0; i < n_rows; i++) {
    for (int32 j = 0; j < n_columns; j++) {
      Real src_val = src(i, copy_from_idx_vec.at(j));
      Real tgt_val = tgt(i, j);
      AssertEqual(src_val, tgt_val);
    }
  }
}

template<typename Real> 
static void UnitTestCuMathSplice() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  CuMatrix<Real> src(M, N);
  CuArray<int32> frame_offsets;

  src.SetRandn(); 
  int32 n_rows = src.NumRows();
  int32 n_columns = src.NumCols();
  std::vector<int32> frame_offsets_vec;

  // The number of columns of tgt is rows(src) 
  // times n_frame_offsets, so we keep n_frame_offsets 
  // reasonably small (2 <= n <= 6).
  int32 n_frame_offsets = Rand() % 7 + 2;
  for (int32 i = 0; i < n_frame_offsets; i++) {
    frame_offsets_vec.push_back(Rand() % 2 * n_columns - n_columns);
  }

  CuMatrix<Real> tgt(M, N * n_frame_offsets);
  frame_offsets.CopyFromVec(frame_offsets_vec);
  cu::Splice(src, frame_offsets, &tgt);

  Matrix<Real> src_copy(src), tgt_copy(tgt);
  for (int32 i = 0; i < n_rows; i++) {
    for (int32 k = 0; k < n_frame_offsets; k++) {
      for (int32 j = 0; j < n_columns; j++) {
        Real src_val; 
        if (i + frame_offsets_vec.at(k) >= n_rows) {
          src_val = src_copy(n_rows-1, j);
        } else if (i + frame_offsets_vec.at(k) <= 0) {
          src_val = src_copy(0, j);
        } else {
          src_val = src_copy(i + frame_offsets_vec.at(k), j); 
        }
        Real tgt_val = tgt_copy(i, k * n_columns + j);
        AssertEqual(src_val, tgt_val);
      }
    }
  }
}


template<typename Real>
static void UnitTestCuMathNormalizePerRow() {

  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    Matrix<Real> Hi(row,col);
    Matrix<Real> Ho(row,col+1);
    Hi.SetRandn();
    Hi.Scale(5.0);

    CuMatrix<Real> Di(row, col);
    CuMatrix<Real> Do(row, col+1);
    Di.CopyFromMat(Hi);

    Real target_rms = 0.3456;
    bool add_log_stddev = true;
    const Real kSquaredNormFloor = 1.35525271560688e-20; // 2^-66

    //gpu
    cu::NormalizePerRow(Di, target_rms, add_log_stddev, &Do);

    //cpu
    {
      MatrixBase<Real>& in(Hi);
      MatrixBase<Real>& out(Ho);
      Real target_rms=0.3456;
      SubMatrix<Real> out_no_log(out, 0, out.NumRows(), 0, in.NumCols());
      if (in.Data() != out_no_log.Data())
        out_no_log.CopyFromMat(in);
      Vector<Real> in_norm(in.NumRows());
      Real d_scaled = in.NumCols() * target_rms * target_rms;
      in_norm.AddDiagMat2(1.0 / d_scaled, in, kNoTrans, 0.0);
      in_norm.ApplyFloor(kSquaredNormFloor);
      in_norm.ApplyPow(-0.5);
      out_no_log.MulRowsVec(in_norm);
      if (add_log_stddev) {
        in_norm.ApplyLog();
        in_norm.Scale(-1.0);
        in_norm.Add(log(target_rms));
        out.CopyColFromVec(in_norm, in.NumCols());
      }
    }

    Matrix<Real> Ho2(Do);
    AssertEqual(Ho,Ho2,0.00001);
  }

  for (int dim = 16; dim <= 1024; dim *= 2) {
    BaseFloat time_in_secs = 0.025;
    CuMatrix<Real> M(dim, dim), N(dim, dim + 1);
    M.SetRandn();
    N.SetRandn();
    Timer tim;
    int32 iter = 0;
    for (; tim.Elapsed() < time_in_secs; iter++) {
      cu::NormalizePerRow(M, Real(1), true, &N);
    }

    BaseFloat gflops = ((BaseFloat) dim * dim * iter)
        / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For CuMatrix::NormalizePerRow"
              << (sizeof(Real)==8?"<double>":"<float>") << ", for dim = "
              << dim << ", speed was " << gflops << " gigaflops.";
  }
}


template<typename Real> void CudaMathUnitTest() {
  #if HAVE_CUDA == 1  
    if (CuDevice::Instantiate().DoublePrecisionSupported())
  #endif
  UnitTestCuMathRandomize<Real>();
  UnitTestCuMathSplice<Real>();
  UnitTestCuMathCopy<Real>();
  UnitTestCuMathNormalizePerRow<Real>();
}


} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("yes"); // -2 .. automatic selection
#endif
    srand(time(NULL));
    kaldi::CudaMathUnitTest<float>();
    
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaMathUnitTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaMathUnitTest<float>();
#endif

    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

