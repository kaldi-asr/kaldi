// ivector/xvector-test.cc

// Copyright 2016  Daniel Povey
//                 David Snyder

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

#include "ivector/xvector.h"
#include "util/kaldi-io.h"
#include "cudamatrix/cu-matrix-lib.h"

namespace kaldi {

bool TestXvectorExtractorDerivative(BaseFloat perturb_delta) {
  int32 xvector_dim = RandInt(4, 30),
        num_rows = 2 * RandInt(2, 10); // The number of rows must be even
                                       // and greater than 2.
  CuSpMatrix<BaseFloat> S(xvector_dim);
  S.SetRandn();
  // Necessary to keep the similarity scores from getting too large or small.
  S.Scale(1.0e-01);
  BaseFloat b = RandInt(-100, 100) / 10.0,
            tot_weight,
            tot_objf;
  int32 S_dim = S.NumCols() * (S.NumCols() + 1) / 2;
  CuMatrix<BaseFloat> xvector_pairs(num_rows, xvector_dim, kSetZero),
                      deriv_xvector(num_rows, xvector_dim, kSetZero);
  CuVector<BaseFloat> deriv_S_and_b(S_dim + 1, kSetZero);
  xvector_pairs.SetRandn();
  ComputeXvectorObjfAndDeriv(xvector_pairs, S, b, &deriv_xvector,
    &deriv_S_and_b, &tot_objf, &tot_weight);
  CuVector<BaseFloat> deriv_xvector_vec(xvector_dim);

  // Sum over the derivatives for xvector input.
  deriv_xvector_vec.AddRowSumMat(1.0, deriv_xvector, 0.0);
  BaseFloat l2_xvector = 0,
            l2_S = 0,
            l2_b = 0;

  // Compare the xvector derivatives calculated above with a numerical
  // approximation.
  for (int32 i = 0; i < xvector_dim; i++) {
    CuMatrix<BaseFloat> xvector_pairs_p(xvector_pairs);
    CuMatrix<BaseFloat> xvector_pairs_n(xvector_pairs);
    for (int32 j = 0; j < num_rows; j++) {
      xvector_pairs_p(j, i) += perturb_delta;
      xvector_pairs_n(j, i) += -perturb_delta;
    }
    CuMatrix<BaseFloat> deriv_xvector_tmp(num_rows, xvector_dim, kSetZero);
    CuVector<BaseFloat> deriv_S_and_b_tmp(S_dim + 1, kSetZero);
    BaseFloat tot_objf_p;
    BaseFloat tot_objf_n;
    ComputeXvectorObjfAndDeriv(xvector_pairs_p, S, b, &deriv_xvector_tmp,
      &deriv_S_and_b_tmp, &tot_objf_p, &tot_weight);
    ComputeXvectorObjfAndDeriv(xvector_pairs_n, S, b, &deriv_xvector_tmp,
      &deriv_S_and_b_tmp, &tot_objf_n, &tot_weight);
    BaseFloat delta = (tot_objf_p  - tot_objf_n)
      * 1.0 / (2.0 * perturb_delta);
    l2_xvector += pow(deriv_xvector_vec(i) - delta, 2);
  }

  // Compare the S derivative calculated above with a numerical
  // approximation.
  for (int32 i = 0; i < S_dim; i++) {
    CuSpMatrix<BaseFloat> S_p(S);
    CuSpMatrix<BaseFloat> S_n(S);
    S_p.Data()[i] += perturb_delta;
    S_n.Data()[i] -= perturb_delta;
    CuMatrix<BaseFloat> deriv_xvector_tmp(num_rows, xvector_dim, kSetZero);
    CuVector<BaseFloat> deriv_S_and_b_tmp(S_dim + 1, kSetZero);
    BaseFloat tot_objf_p;
    BaseFloat tot_objf_n;
    ComputeXvectorObjfAndDeriv(xvector_pairs, S_p, b, &deriv_xvector_tmp,
      &deriv_S_and_b_tmp, &tot_objf_p, &tot_weight);
    ComputeXvectorObjfAndDeriv(xvector_pairs, S_n, b, &deriv_xvector_tmp,
      &deriv_S_and_b_tmp, &tot_objf_n, &tot_weight);
    BaseFloat delta = (tot_objf_p  - tot_objf_n)
      * 1.0 / (2.0 * perturb_delta);
    l2_S += pow(deriv_S_and_b(i) - delta, 2);
  }

  // Compare the b derivative calculated above with a numerical
  // approximation.
  BaseFloat b_p = b + perturb_delta;
  BaseFloat b_n = b - perturb_delta;
  CuMatrix<BaseFloat> deriv_xvector_tmp(num_rows, xvector_dim, kSetZero);
  CuVector<BaseFloat> deriv_S_and_b_tmp(S_dim + 1, kSetZero);
  BaseFloat tot_objf_p;
  BaseFloat tot_objf_n;
  ComputeXvectorObjfAndDeriv(xvector_pairs, S, b_p, &deriv_xvector_tmp,
    &deriv_S_and_b_tmp, &tot_objf_p, &tot_weight);
  ComputeXvectorObjfAndDeriv(xvector_pairs, S, b_n, &deriv_xvector_tmp,
    &deriv_S_and_b_tmp, &tot_objf_n, &tot_weight);
  BaseFloat delta = (tot_objf_p  - tot_objf_n) * 1.0 / (2.0 * perturb_delta);
  l2_b = pow(deriv_S_and_b(S_dim) - delta, 2);
  KALDI_ASSERT(l2_xvector < 1.0e-03);
  KALDI_ASSERT(l2_S <  1.0e-03);
  KALDI_ASSERT(l2_b < 1.0e-03);
  return true;
}

void UnitTestXvectorExtractor() {
  if (!TestXvectorExtractorDerivative(1.0e-02) &&
     !TestXvectorExtractorDerivative(1.0e-03) &&
     !TestXvectorExtractorDerivative(1.0e-04) &&
     !TestXvectorExtractorDerivative(1.0e-05))
  KALDI_ERR << "Xvector derivative test failed";
}

} // namespace kaldi

int main() {
  using namespace kaldi;
  for (int32 i = 0; i < 3; i++)
    UnitTestXvectorExtractor();
  std::cout << "Xvector derivative tests succeeded.\n";
  return 0;
}
