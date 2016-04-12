// nnet3/nnet-utils-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {


void UnitTestNnetContext() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationOptions gen_config;
    
    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    std::istringstream is(configs[0]);
    nnet.ReadConfig(is);

    // this test doesn't really test anything except that it runs;
    // we manually inspect the output.
    int32 left_context, right_context;
    ComputeSimpleNnetContext(nnet, &left_context, &right_context);
    KALDI_LOG << "Left,right-context= " << left_context << ","
              << right_context << " for config: " << configs[0];

    KALDI_LOG << "Info for nnet is: " << NnetInfo(nnet);
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId("yes");
#endif


  Matrix<BaseFloat> mat1(1000, 500, kUndefined);
  mat1.SetRandn();
  Matrix<BaseFloat> mat2(1000, 800, kUndefined);
  mat2.SetRandn();
  Matrix<BaseFloat> mat3(800, 500, kUndefined);
  mat3.SetRandn();

  CuMatrix<BaseFloat> cmat1(mat1);
  CuMatrix<BaseFloat> cmat2(mat2);
  CuMatrix<BaseFloat> cmat3(mat3);

  mat1.AddMatMat(1.5, mat2, kNoTrans, mat3, kNoTrans, 1.0);
  cmat1.AddMatMat(1.5, cmat2, kNoTrans, cmat3, kNoTrans, 1.0);

  KALDI_LOG << mat1.Sum();
  KALDI_LOG << cmat1.Sum();

  int m = 50, n = 100;
  Matrix<BaseFloat> a(m, n);
  Matrix<BaseFloat> b(n ,m);
  a.SetRandn();
  b.SetRandUniform();
  // multiply 2 small matrices in CPU:
  Matrix<BaseFloat> c(m, m);
  c.AddMatMat(1.0, a, kNoTrans, b, kNoTrans, 0.0);
  // multiply same matrices in GPU:
  CuMatrix<BaseFloat> c1(m, m);
  c1.AddMatMat(1.0, CuMatrix<BaseFloat>(a), kNoTrans, CuMatrix<BaseFloat>(b), kNoTrans, 0.0);
  // check that relative differnence is <1%
  KALDI_LOG << "Diffing...";
  Matrix<BaseFloat> ok(c1);
  ok.AddMat(-1.0, c);
  KALDI_LOG << ok.Sum();
  KALDI_LOG << ok.Min();
  KALDI_LOG << ok.Max();


#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

//  SetVerboseLevel(2);

//  UnitTestNnetContext();
//  UnitTestConvertRepeatedToBlockAffine();
//  UnitTestConvertRepeatedToBlockAffineComposite();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}
