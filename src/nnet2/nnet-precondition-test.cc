// nnet2/nnet-precondition-test.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet2/nnet-precondition.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet2 {

void UnitTestPreconditionDirections() {
  MatrixIndexT N = 2 + rand() % 30,
               D = 1 + rand() % 20;
  BaseFloat lambda = 0.1;
  CuMatrix<BaseFloat> R(N, D), P(N, D);
  R.SetRandn();
  P.SetRandn(); // contents should be overwritten.

  PreconditionDirections(R, lambda, &P);
  // The rest of this function will do the computation the function is doing in
  // a different, less efficient way and compare with the function call.
  
  CuSpMatrix<BaseFloat> G(D);
  G.SetUnit();
  G.ScaleDiag(lambda);
  // G += R^T R.
  G.AddMat2(1.0/(N-1), R, kTrans, 1.0);
  
  for (int32 n = 0; n < N; n++) {
    CuSubVector<BaseFloat> rn(R, n);
    CuSpMatrix<BaseFloat> Gn(G);
    Gn.AddVec2(-1.0/(N-1), rn); // subtract the
    // outer product of "this" vector.
    Gn.Invert();
    CuSubVector<BaseFloat> pn(P, n);
    CuVector<BaseFloat> pn_compare(D);
    pn_compare.AddSpVec(1.0, Gn, rn, 0.0);
    KALDI_ASSERT(pn.ApproxEqual(pn_compare, 0.1));
  }
}


} // namespace nnet2
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;
  for (int32 i = 0; i < 10; i++)
    UnitTestPreconditionDirections();
}
