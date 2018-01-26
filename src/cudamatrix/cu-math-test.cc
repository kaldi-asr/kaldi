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
static void UnitTestEnsureNonzero() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  Real epsilon = 0.1;
  CuMatrix<Real> x(M, N);
  x.SetRandn();
  CuMatrix<Real> y(M, N, kUndefined);
  cu::EnsureNonzero(x, epsilon, &y);
  Matrix<Real> x_cpu(x);
  Matrix<Real> y_cpu(y);
  for (int32 i = 0; i < 30; i++) {
    int32 r = RandInt(0, M-1), c = RandInt(0, N-1);
    Real src = x_cpu(r, c), dest = y_cpu(r, c);
    if (src <= -epsilon || src >= epsilon) {
      KALDI_ASSERT(src == dest);
    } else if (src >= 0) {
      KALDI_ASSERT(dest == epsilon);
    } else {
      KALDI_ASSERT(dest == -epsilon);
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
static void UnitTestCuMathComputeLstmNonlinearity() {
  for (int i = 0; i < 3; i++) {
    int32 num_rows = 1 + Rand() % 100;
    int32 cell_dim = 1 + Rand() % 2000;
    int32 dropout_dim = (RandInt(0, 1) == 0 ? 0 : 3);
    Matrix<Real> Hinput(num_rows, 5 * cell_dim + dropout_dim);
    Matrix<Real> Hparams(3, cell_dim);
    Matrix<Real> Houtput(num_rows, 2 * cell_dim);
    Hinput.SetRandn();
    Hparams.SetRandn();

    CuMatrix<Real> Dinput(Hinput);
    CuMatrix<Real> Dparams(Hparams);
    CuMatrix<Real> Doutput(Houtput);

    cu::CpuComputeLstmNonlinearity(Hinput, Hparams, &Houtput);
    cu::ComputeLstmNonlinearity(Dinput, Dparams, &Doutput);

    Matrix<Real> HDoutput(Doutput);
    AssertEqual(Houtput, HDoutput);
  }

  for (int i = 16; i <= 1024; i *= 2) {
    BaseFloat time_in_secs = 0.025;
    int32 num_rows = i;
    int32 cell_dim = i;
    int32 dropout_dim = (RandInt(0, 1) == 0 ? 0 : 3);
    CuMatrix<Real> input(num_rows, 5 * cell_dim + dropout_dim);
    CuMatrix<Real> params(3, cell_dim);
    CuMatrix<Real> output(num_rows, 2 * cell_dim);
    input.SetRandn();
    params.SetRandn();

    Timer tim;
    int32 iter = 0;
    for (; tim.Elapsed() < time_in_secs; iter++)
      cu::ComputeLstmNonlinearity(input, params, &output);

    BaseFloat gflops = ((BaseFloat) i * i * iter) / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For ComputeLstmNonlinearity"
              << (sizeof(Real)==8 ? "<double>" : "<float>") << ", for dim = "
              << i << ", speed was " << gflops << " gigaflops";
    if (tim.Elapsed() > 0.05)
      break;
  }
}

void UnitTestLstmNonlinearity() {
  for (int32 loop = 0; loop < 10; loop++) {

    // problem dimensions.
    int32 num_rows = RandInt(5, 20),
          cell_dim = RandInt(2, 200),
        dropout_dim = (RandInt(0, 1) == 0 ? 0 : 3);

    // Pick the (input or params block), and output block, for which we'll
    // spot-check the derivative values.  This will give us test failures
    // that are fine-grained enough to assist debugging.
    int32 test_input = RandInt(0, 4),
        test_params = RandInt(0, 2),
        test_output = RandInt(0, 1);

    // set one of test_input or test_params to -1, meaning we're not testing that
    // thing.  only test one at a time.
    if (RandInt(0, 1) == 0)
      test_input = -1;
    else
      test_params = -1;


    CuMatrix<BaseFloat> input(num_rows, cell_dim * 5 + dropout_dim),
        params(3, cell_dim),
        output_deriv(num_rows, cell_dim * 2);
    input.SetRandn();
    params.SetRandn();
    // set just one block of the output deriv to a random value.
    output_deriv.ColRange(test_output * cell_dim, cell_dim).SetRandn();



    CuMatrix<BaseFloat> output(num_rows, cell_dim * 2);

    cu::ComputeLstmNonlinearity(input, params, &output);

    BaseFloat baseline_objf = TraceMatMat(output, output_deriv, kTrans);

    // not really testing self repair here... will debug it when we actually run
    // it, by looking at the diagnostics.
    CuMatrix<double> deriv_sum(5, cell_dim),
        value_sum(5, cell_dim);
    CuVector<BaseFloat> self_repair_config(10.0); // leave at zero... we don't really test this here.
    CuMatrix<BaseFloat>
        self_repair_sum(5, cell_dim),
        input_deriv(num_rows, 5 * cell_dim + dropout_dim),
        params_deriv(3, cell_dim);

    double count_in = 0.0;

    // get derivative w.r.t. input and params, which we are testing.
    cu::BackpropLstmNonlinearity(input, params, output_deriv, deriv_sum,
                                 self_repair_config, count_in,
                                 &input_deriv, &params_deriv,
                                 &value_sum, &deriv_sum, &self_repair_sum);


    int32 test_dim = 5;  // number of separate offsets we add while testing the
    // derivatives... reduces randomness in test.
    BaseFloat delta = 1.0e-03;
    Vector<BaseFloat> predicted_objf_change(test_dim),
        measured_objf_change(test_dim);

    for (int32 i = 0; i < test_dim; i++) {
      CuMatrix<BaseFloat> delta_input(num_rows, 5 * cell_dim + dropout_dim),
          delta_params(3, cell_dim);
      if (test_input >= 0) {
        delta_input.ColRange(test_input * cell_dim, cell_dim).SetRandn();
        delta_input.Scale(delta);
      }
      if (test_params >= 0) {
        delta_params.Row(test_params).SetRandn();
        delta_params.Scale(delta);
      }

      predicted_objf_change(i) = TraceMatMat(delta_input, input_deriv, kTrans) +
          TraceMatMat(delta_params, params_deriv, kTrans);

      CuMatrix<BaseFloat> perturbed_input(input);
      perturbed_input.AddMat(1.0, delta_input);

      CuMatrix<BaseFloat> perturbed_params(params);
      perturbed_params.AddMat(1.0, delta_params);

      CuMatrix<BaseFloat> perturbed_output(num_rows, 2 * cell_dim);
      cu::ComputeLstmNonlinearity(perturbed_input, perturbed_params,
                                  &perturbed_output);
      BaseFloat new_objf = TraceMatMat(perturbed_output, output_deriv, kTrans),
          objf_change = new_objf - baseline_objf;
      measured_objf_change(i) = objf_change;
    }
    KALDI_LOG << "LSTM nonlinearity test: num_rows=" << num_rows
              << ", cell_dim=" << cell_dim
              << ", dropout_dim=" << dropout_dim
              << ", test_input=" << test_input
              << ", test_params=" << test_params
              << ", test_output=" << test_output
              << ", predicted_objf_change=" << predicted_objf_change
              << ", measured_objf_change=" << measured_objf_change;

    if (!ApproxEqual(predicted_objf_change, measured_objf_change, BaseFloat(0.1F))) {
      KALDI_ERR << "LSTM nonlinearity test failed.";
    }
  }
}

template<typename Real>
static void UnitTestBackpropLstmNonlinearity() {
  for (int i = 0; i < 3; i++) {
    int32 num_rows = 1 + Rand() % 200;
    int32 cell_dim = 1 + Rand() % 2000,
       dropout_dim = (RandInt(0, 1) == 0 ? 0 : 3);
//    KALDI_LOG << num_rows << ", " << cell_dim;

    Matrix<Real> hinput(num_rows, 5 * cell_dim + dropout_dim);
    Matrix<Real> hparams(3, cell_dim);
    Matrix<Real> houtput_deriv(num_rows, 2 * cell_dim);
    Matrix<double> hderiv_sum_in(5, cell_dim);
    Vector<Real> hself_repair_config(10);
    double count_in;
    Matrix<Real> hinput_deriv(num_rows, 5 * cell_dim + dropout_dim);
    Matrix<Real> hparams_deriv(3, cell_dim);
    Matrix<double> hvalue_sum_out(5, cell_dim);
    Matrix<double> hderiv_sum_out(5, cell_dim);
    Matrix<Real> hself_repair_sum_out(5, cell_dim);

    hinput.SetRandn();
    hparams.SetRandn();
    houtput_deriv.SetRandn();
    hderiv_sum_in.SetRandn();
    hself_repair_config.SetRandn();
    count_in = Rand() % num_rows;

    hinput_deriv.SetRandn();
    hparams_deriv.SetRandn();
    hvalue_sum_out.SetRandn();
    hderiv_sum_out.SetRandn();
    hself_repair_sum_out.SetRandn();

    CuMatrix<Real> dinput(hinput);
    CuMatrix<Real> dparams(hparams);
    CuMatrix<Real> doutput_deriv(houtput_deriv);
    CuMatrix<double> dderiv_sum_in(hderiv_sum_in);
    CuVector<Real> dself_repair_config(hself_repair_config);

    CuMatrix<Real> dinput_deriv(hinput_deriv);
    CuMatrix<Real> dparams_deriv(hparams_deriv);
    CuMatrix<double> dvalue_sum_out(hvalue_sum_out);
    CuMatrix<double> dderiv_sum_out(hderiv_sum_out);
    CuMatrix<Real> dself_repair_sum_out(hself_repair_sum_out);

    cu::CpuBackpropLstmNonlinearity(hinput, hparams, houtput_deriv,
                                    hderiv_sum_in, hself_repair_config,
                                    count_in, (MatrixBase<Real>*) NULL,
                                    (MatrixBase<Real>*) NULL,
                                    (MatrixBase<double>*) NULL,
                                    (MatrixBase<double>*) NULL,
                                    (MatrixBase<Real>*) NULL);
    cu::BackpropLstmNonlinearity(dinput, dparams, doutput_deriv, dderiv_sum_in,
                                 dself_repair_config, count_in,
                                 (CuMatrixBase<Real>*) NULL,
                                 (CuMatrixBase<Real>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<Real>*) NULL);

    cu::CpuBackpropLstmNonlinearity(hinput, hparams, houtput_deriv,
                                    hderiv_sum_in, hself_repair_config,
                                    count_in, (MatrixBase<Real>*) NULL,
                                    &hparams_deriv, &hvalue_sum_out,
                                    &hderiv_sum_out, &hself_repair_sum_out);
    cu::BackpropLstmNonlinearity(dinput, dparams, doutput_deriv, dderiv_sum_in,
                                 dself_repair_config, count_in,
                                 (CuMatrixBase<Real>*) NULL, &dparams_deriv,
                                 &dvalue_sum_out, &dderiv_sum_out,
                                 &dself_repair_sum_out);

    cu::CpuBackpropLstmNonlinearity(hinput, hparams, houtput_deriv,
                                    hderiv_sum_in, hself_repair_config,
                                    count_in, &hinput_deriv,
                                    (MatrixBase<Real>*) NULL,
                                    (MatrixBase<double>*) NULL,
                                    (MatrixBase<double>*) NULL,
                                    (MatrixBase<Real>*) NULL);
    cu::BackpropLstmNonlinearity(dinput, dparams, doutput_deriv, dderiv_sum_in,
                                 dself_repair_config, count_in, &dinput_deriv,
                                 (CuMatrixBase<Real>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<Real>*) NULL);

    cu::CpuBackpropLstmNonlinearity(hinput, hparams, houtput_deriv,
                                    hderiv_sum_in, hself_repair_config,
                                    count_in, &hinput_deriv, &hparams_deriv,
                                    &hvalue_sum_out, &hderiv_sum_out,
                                    &hself_repair_sum_out);
    cu::BackpropLstmNonlinearity(dinput, dparams, doutput_deriv, dderiv_sum_in,
                                 dself_repair_config, count_in, &dinput_deriv,
                                 &dparams_deriv, &dvalue_sum_out,
                                 &dderiv_sum_out, &dself_repair_sum_out);

    Matrix<Real> hdinput_deriv(dinput_deriv);
    Matrix<Real> hdparams_deriv(dparams_deriv);
    Matrix<double> hdvalue_sum_out(dvalue_sum_out);
    Matrix<double> hdderiv_sum_out(dderiv_sum_out);
    Matrix<Real> hdself_repair_sum_out(dself_repair_sum_out);

//    KALDI_LOG<< "input_deriv" << hinput_deriv << "d" << hdinput_deriv;
//    KALDI_LOG<< "hparams_deriv" << hparams_deriv << "d" << hdparams_deriv;
//    KALDI_LOG<< "hvalue_sum_out" << hvalue_sum_out << "d" << hdvalue_sum_out;
//    KALDI_LOG<< "hderiv_sum_out" << hderiv_sum_out << "d" << hdderiv_sum_out;
//    KALDI_LOG<< "hself_repair_sum_out" << hself_repair_sum_out << "d" << hdself_repair_sum_out;

    AssertEqual(hinput_deriv, hdinput_deriv);
    AssertEqual(hparams_deriv, hdparams_deriv);
    AssertEqual(hvalue_sum_out, hdvalue_sum_out);
    AssertEqual(hderiv_sum_out, hdderiv_sum_out);
    AssertEqual(hself_repair_sum_out, hdself_repair_sum_out);
  }

  for (int i = 16; i <= 2048; i *= 2) {
    BaseFloat time_in_secs = 0.025;
    int32 num_rows = i;
    int32 cell_dim = i;
    int32 dropout_dim = (RandInt(0, 1) == 0 ? 0 : 3);

    CuMatrix<Real> input(num_rows, 5 * cell_dim + dropout_dim);
    CuMatrix<Real> params(3, cell_dim);
    CuMatrix<Real> output_deriv(num_rows, 2 * cell_dim);
    CuMatrix<double> deriv_sum_in(5, cell_dim);
    CuVector<Real> self_repair_config(10);
    double count_in;

    CuMatrix<Real> input_deriv(num_rows, 5 * cell_dim + dropout_dim);
    CuMatrix<Real> params_deriv(3, cell_dim);
    CuMatrix<double> value_sum_out(5, cell_dim);
    CuMatrix<double> deriv_sum_out(5, cell_dim);
    CuMatrix<Real> self_repair_sum_out(5, cell_dim);

    input.SetRandn();
    params.SetRandn();
    output_deriv.SetRandn();
    deriv_sum_in.SetRandn();
    self_repair_config.SetRandn();
    count_in = Rand() % num_rows;

    Timer tim;
    int32 iter = 0;
    for (; tim.Elapsed() < time_in_secs; iter++)
      cu::BackpropLstmNonlinearity(input, params, output_deriv, deriv_sum_in,
                                   self_repair_config, count_in, &input_deriv,
                                   &params_deriv, &value_sum_out,
                                   &deriv_sum_out, &self_repair_sum_out);


    BaseFloat gflops = ((BaseFloat) i * i * iter) / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For BackpropLstmNonlinearity"
              << (sizeof(Real) == 8 ? "<double>" : "<float>") << ", for dim = "
              << i << ", speed was " << gflops << " gigaflops";
    if (tim.Elapsed() > 0.05)
      break;
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
    KALDI_LOG << "For CuMath::NormalizePerRow"
              << (sizeof(Real)==8?"<double>":"<float>") << ", for dim = "
              << dim << ", speed was " << gflops << " gigaflops.";
    if (tim.Elapsed() > 0.05)
      break;
  }
}

template<typename Real>
static void UnitTestCuDiffNormalizePerRow() {
  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    Matrix<Real> Hi(row, col);
    Matrix<Real> Ho(row, col + 1);
    Matrix<Real> Hid(row, col);
    Matrix<Real> Hod(row, col + 1);
    Hi.SetRandn();
    Hod.SetRandn();
    Hi.Scale(5.0);

    CuMatrix<Real> Di(row, col);
    CuMatrix<Real> Do(row, col + 1);
    CuMatrix<Real> Did(row, col);
    CuMatrix<Real> Dod(row, col + 1);
    Di.CopyFromMat(Hi);
    Dod.CopyFromMat(Hod);

    Real target_rms = 0.3456;
    bool add_log_stddev = true;
    const Real kSquaredNormFloor = 1.3552527156068805425e-20; // 2^-66

    //gpu
    cu::DiffNormalizePerRow(Di, Dod, target_rms, add_log_stddev, &Did);

    //cpu
    {
      MatrixBase<Real>* in_deriv = &Hid;
      MatrixBase<Real>& out_deriv(Hod);
      MatrixBase<Real>& in_value(Hi);

      const SubMatrix<Real> out_deriv_no_log(out_deriv, 0, out_deriv.NumRows(),
                                             0, in_value.NumCols());
      Vector<Real> dot_products(out_deriv.NumRows());
      dot_products.AddDiagMatMat(1.0, out_deriv_no_log, kNoTrans, in_value,
                                 kTrans, 0.0);
      Vector<Real> in_norm(in_value.NumRows());
      Real d_scaled = (in_value.NumCols() * target_rms * target_rms);
      in_norm.AddDiagMat2(1.0, in_value, kNoTrans, 0.0);
      if (add_log_stddev) {
        Vector<Real> log_stddev_deriv(in_norm), // log_stddev deriv as dF/dy .* (x^T x)^-1
        out_deriv_for_stddev(out_deriv.NumRows(), kUndefined);
        // f = log(sqrt(max(epsi, x^T x / D)))
        // df/dx = epsi^2 * D < x^T x ? (1/(x^T x)) * x  : 0.
        // we don't compute this exactly below for the case when x^2 x is very
        // small, but we do make sure that the deriv isn't infinity when the input
        // is zero.
        log_stddev_deriv.ApplyFloor(in_value.NumCols() * kSquaredNormFloor);
        log_stddev_deriv.ApplyPow(-1.0);
        out_deriv_for_stddev.CopyColFromMat(out_deriv,
                                            (out_deriv.NumCols() - 1));
        log_stddev_deriv.MulElements(out_deriv_for_stddev);
        if (in_deriv)
          in_deriv->AddDiagVecMat(1.0, log_stddev_deriv, in_value, kNoTrans,
                                  1.0);
      }
      in_norm.Scale(1.0 / d_scaled);
      in_norm.ApplyFloor(kSquaredNormFloor);
      in_norm.ApplyPow(-0.5);
      if (in_deriv) {
        if (in_deriv->Data() != out_deriv_no_log.Data())
          in_deriv->AddDiagVecMat(1.0, in_norm, out_deriv_no_log, kNoTrans,
                                  1.0);
        else
          in_deriv->MulRowsVec(in_norm);
        in_norm.ReplaceValue(1.0 / sqrt(kSquaredNormFloor), 0.0);
        in_norm.ApplyPow(3.0);
        dot_products.MulElements(in_norm);

        in_deriv->AddDiagVecMat(-1.0 / d_scaled, dot_products, in_value,
                                kNoTrans, 1.0);
      }

      Matrix<Real> Hid2(Did);
      AssertEqual(Hid, Hid2, 0.00001);
    }
  }

  for (int dim = 16; dim <= 1024; dim *= 2) {
    BaseFloat time_in_secs = 0.025;
    CuMatrix<Real> id(dim, dim), iv(dim, dim), od(dim, dim + 1);
    iv.SetRandn();
    od.SetRandn();
    Timer tim;
    int32 iter = 0;
    for (; tim.Elapsed() < time_in_secs; iter++) {
      cu::DiffNormalizePerRow(iv, od, Real(0.456), true, &id);
    }
    BaseFloat fdim = dim;
    BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For CuMath::DiffNormalizePerRow"
              << (sizeof(Real)==8?"<double>":"<float>")
              << ", for dim = " << dim << ", speed was " << gflops
              << " gigaflops.";
  }
}



template<typename Real> void CudaMathUnitTest() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported())
#endif

  UnitTestCuMathComputeLstmNonlinearity<Real>();
  UnitTestCuMathRandomize<Real>();
  UnitTestCuMathSplice<Real>();
  UnitTestCuMathCopy<Real>();
  UnitTestLstmNonlinearity();
  UnitTestEnsureNonzero<Real>();
  UnitTestBackpropLstmNonlinearity<Real>();
  UnitTestCuMathNormalizePerRow<Real>();
  UnitTestCuDiffNormalizePerRow<Real>();
}

} // namespace kaldi


int main() {
  SetVerboseLevel(1);
  int32 loop = 0;
#if HAVE_CUDA == 1
  for (; loop < 2; loop++) {
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
#if HAVE_CUDA == 1
  } // No for loop if 'HAVE_CUDA != 1',
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}
