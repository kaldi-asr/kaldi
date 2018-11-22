// matrix/matrix-functions-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)
//           2018  Institute of Acoustics, CAS (Gaofeng Cheng)

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

#include "matrix/matrix-functions.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

void SvdRescalerTestIdentity() {
  // this tests the case where f() is the identity function.
  int32 dim = 10;
  Matrix<BaseFloat> mat(dim, dim);
  if (RandInt(0, 1) == 0)
    mat.SetRandn();
  // else zero.

  SvdRescaler sc;
  sc.Init(&mat, false);

  BaseFloat *lambda = sc.InputSingularValues(),
      *f_lambda= sc.OutputSingularValues(),
      *fprime_lambda = sc.OutputSingularValueDerivs();
  for (int32 i = 0; i < dim; i++) {
    f_lambda[i] = lambda[i];
    fprime_lambda[i] = 1.0;
  }
  Matrix<BaseFloat> output(dim, dim, kUndefined);
  sc.GetOutput(&output);
  AssertEqual(mat, output, 0.001);
  Matrix<BaseFloat> output_deriv(dim, dim, kUndefined),
      input_deriv(dim, dim);
  output_deriv.SetRandn();
  sc.ComputeInputDeriv(output_deriv, &input_deriv);
  KALDI_LOG << output_deriv << input_deriv;
  AssertEqual(output_deriv, input_deriv);
}

void SvdRescalerTestPowerDiag() {
  // this tests the case where f() is a power function with random exponent,
  // and the matrix is diagonal.
  int32 dim = 10;
  BaseFloat power = 0.25 * RandInt(0, 4);
  Matrix<BaseFloat> mat(dim, dim);
  for (int32 i = 0; i < dim; i++) {
    mat(i, i) = 0.25 * RandInt(0, 10);
    // if power < 1.0, we can't allow zero diagonal
    // elements, or the derivatives would be undefined.
    if (power < 1.0 && mat(i, i) == 0.0)
      mat(i, i) = 0.333;
  }

  SvdRescaler sc;
  sc.Init(&mat, false);

  BaseFloat *lambda = sc.InputSingularValues(),
      *f_lambda= sc.OutputSingularValues(),
      *fprime_lambda = sc.OutputSingularValueDerivs();
  for (int32 i = 0; i < dim; i++) {
    f_lambda[i] = pow(lambda[i], power);
    fprime_lambda[i] = power * pow(lambda[i], power - 1.0);
  }
  Matrix<BaseFloat> output(dim, dim, kUndefined);
  sc.GetOutput(&output);
  KALDI_ASSERT(mat.IsDiagonal(0.001));
  Matrix<BaseFloat> output_deriv(dim, dim, kUndefined),
      input_deriv(dim, dim);
  output_deriv.SetRandn();
  sc.ComputeInputDeriv(output_deriv, &input_deriv);

  for (int32 i = 0; i < dim; i++) {
    BaseFloat oderiv = output_deriv(i, i),
        ideriv = input_deriv(i, i),
        x = mat(i, i),
        df = power * pow(x, power - 1.0);
    AssertEqual(ideriv, oderiv * df);
  }
}


} // namespace kaldi

int main() {
  for (int32 i = 0; i < 10; i++) {
    kaldi::SvdRescalerTestIdentity();
    kaldi::SvdRescalerTestPowerDiag();
  }
  std::cout << "Test OK.\n";
}
