// nnet/nnet-component-test.cc
// Copyright 2014-2015  Brno University of Technology (author: Karel Vesely),
//                      The Johns Hopkins University (author: Sri Harish Mallidi)

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

#include <sstream>
#include <fstream>
#include <algorithm>

#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-convolutional-component.h"
#include "nnet/nnet-convolutional-2d-component.h"
#include "nnet/nnet-max-pooling-component.h"
#include "nnet/nnet-max-pooling-2d-component.h"
#include "nnet/nnet-average-pooling-2d-component.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet1 {

  /*
   * Helper functions
   */
  template<typename Real>
  void ReadCuMatrixFromString(const std::string& s, CuMatrix<Real>* m) {
    std::istringstream is(s + "\n");
    m->Read(is, false);  // false for ascii
  }

  Component* ReadComponentFromString(const std::string& s) {
    std::istringstream is(s + "\n");
    return Component::Read(is, false);  // false for ascii
  }


  /*
   * Unit tests,
   */
  void UnitTestLengthNorm() {
    // make L2-length normalization component,
    Component* c = ReadComponentFromString("<LengthNormComponent> 5 5");
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 \n 2 3 5 6 8 ] ", &mat_in);
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    // check the length,
    mat_out.MulElements(mat_out);  // ^2,
    CuVector<BaseFloat> check_length_is_one(2);
    check_length_is_one.AddColSumMat(1.0, mat_out, 0.0);  // sum_of_cols(x^2),
    check_length_is_one.ApplyPow(0.5);  // L2norm = sqrt(sum_of_cols(x^2)),
    CuVector<BaseFloat> ones(2);
    ones.Set(1.0);
    AssertEqual(check_length_is_one, ones);
  }

  void UnitTestSimpleSentenceAveragingComponent() {
    // make SimpleSentenceAveraging component,
    Component* c = ReadComponentFromString(
      "<SimpleSentenceAveragingComponent> 2 2 <GradientBoost> 10.0"
    );
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0.5 \n 1 1 \n 2 1.5 ] ", &mat_in);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    // check the output,
    CuVector<BaseFloat> ones(2);
    ones.Set(1.0);
    for (int32 i = 0; i < mat_out.NumRows(); i++) {
      AssertEqual(mat_out.Row(i), ones);
    }

    // backpropagate,
    CuMatrix<BaseFloat> dummy1(3, 2), dummy2(3, 2), diff_out(mat_in), diff_in;
    // the average 1.0 in 'diff_in' will be boosted by 10.0,
    c->Backpropagate(dummy1, dummy2, diff_out, &diff_in);
    // check the output,
    CuVector<BaseFloat> tens(2); tens.Set(10);
    for (int32 i = 0; i < diff_in.NumRows(); i++) {
      AssertEqual(diff_in.Row(i), tens);
    }
  }

  void UnitTestConvolutionalComponentUnity() {
    // make 'identity' convolutional component,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 5 5 \
      <PatchDim> 1 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ 1 \
      ] <Bias> [ 0 ]"
    );

    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 ] ", &mat_in);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_in, mat_out);

    // backpropagate,
    CuMatrix<BaseFloat> mat_out_diff(mat_in), mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_out_diff " << mat_out_diff
              << " mat_in_diff " << mat_in_diff;
    AssertEqual(mat_out_diff, mat_in_diff);

    // clean,
    delete c;
  }

  void UnitTestConvolutionalComponent3x3() {
    // make 3x3 convolutional component,
    // design such weights and input so output is zero,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 9 15 \
      <PatchDim> 3 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ -1 -2 -7   0 0 0   1 2 7 ; \
                  -1  0  1  -3 0 3  -2 2 0 ; \
                  -4  0  0  -3 0 3   4 0 0 ] \
      <Bias> [ -20 -20 -20 ]"
    );

    // prepare input, reference output,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 3 5 7 9  2 4 6 8 10  3 5 7 9 11 ]", &mat_in);
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 0 0 0  0 0 0  0 0 0 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_out, mat_out_ref);

    // prepare mat_out_diff, mat_in_diff_ref,
    CuMatrix<BaseFloat> mat_out_diff;
    ReadCuMatrixFromString("[ 1 0 0  1 1 0  1 1 1 ]", &mat_out_diff);
    // hand-computed back-propagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref;
    ReadCuMatrixFromString("[ -1 -4 -15 -8 -6   0 -3 -6 3 6   1 1 14 11 7 ]",
                           &mat_in_diff_ref);

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    // clean,
    delete c;
  }


  void UnitTestMaxPoolingComponent() {
    // make max-pooling component, assuming 4 conv. neurons,
    // non-overlapping pool of size 3,
    Component* c = Component::Init(
        "<MaxPoolingComponent> <InputDim> 24 <OutputDim> 8 \
         <PoolSize> 3 <PoolStep> 3 <PoolStride> 4"
    );

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 3 8 2 9 \
                              8 3 9 3 \
                              2 4 9 6 \
                              \
                              2 4 2 0 \
                              6 4 9 4 \
                              7 3 0 3;\
                              \
                              5 4 7 8 \
                              3 9 5 6 \
                              3 4 8 9 \
                              \
                              5 4 5 6 \
                              3 1 4 5 \
                              8 2 1 7 ]", &mat_in);

    // expected output (max values in columns),
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 8 8 9 9 \
                              7 4 9 4;\
                              5 9 8 9 \
                              8 4 5 7 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // locations of max values will be shown,
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    mat_out_diff.Set(1);
    // expected backpropagated values (hand-computed),
    CuMatrix<BaseFloat> mat_in_diff_ref;
    ReadCuMatrixFromString("[ 0 1 0 1 \
                              1 0 1 0 \
                              0 0 1 0 \
                              \
                              0 1 0 0 \
                              0 1 1 1 \
                              1 0 0 0;\
                              \
                              1 0 0 0 \
                              0 1 0 0 \
                              0 0 1 1 \
                              \
                              0 1 1 0 \
                              0 0 0 0 \
                              1 0 0 1 ]", &mat_in_diff_ref);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestMaxPooling2DComponent() { /* Implemented by Harish Mallidi */
    // make max-pooling2d component
    Component* c = Component::Init(
      "<MaxPooling2DComponent> <InputDim> 56 <OutputDim> 18 \
       <FmapXLen> 4 <FmapYLen> 7 <PoolXLen> 2 <PoolYLen> 3 \
       <PoolXStep> 1 <PoolYStep> 2"
    );

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 \
      11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 \
      22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);

    // expected output (max values in the patch)
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 9 9 11 11 13 13 16 16 18 18 \
      20 20 23 23 25 25 27 27 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);


    // locations of max values will be shown
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString(
      "[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]", &mat_out_diff
    );

    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref;  // hand-computed back-propagated values,
    ReadCuMatrixFromString("[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
      0.25 0.25 0 0 1 1 0 0 0 0 0.75 0.75 0 0 1 1 0 0 2.5 2.5 \
      0 0 0 0 3 3 0 0 3.5 3.5 0 0 8 8 ]", &mat_in_diff_ref
    );

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestAveragePooling2DComponent() { /* Implemented by Harish Mallidi */
    // make average-pooling2d component
    Component* c = Component::Init(
      "<AveragePooling2DComponent> <InputDim> 56 <OutputDim> 18 \
       <FmapXLen> 4 <FmapYLen> 7 <PoolXLen> 2 <PoolYLen> 3 \
       <PoolXStep> 1 <PoolYStep> 2"
    );

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 \
      11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 \
      21 21 22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);

    // expected output (max values in the patch)
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 4.5 4.5 6.5 6.5 8.5 8.5 11.5 11.5 13.5 13.5 \
      15.5 15.5 18.5 18.5 20.5 20.5 22.5 22.5 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);


    // locations of max values will be shown
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]", &mat_out_diff);

    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref;  // hand-computed back-propagated values,
    ReadCuMatrixFromString("[  0 0 0 0 0.0833333 0.0833333 0.166667 0.166667 \
      0.25 0.25 0.333333 0.333333 0.333333 0.333333 0.25 0.25 0.25 0.25 \
      0.333333 0.333333 0.416667 0.416667 0.5 0.5 0.583333 0.583333 0.583333 \
      0.583333 0.75 0.75 0.75 0.75 0.833333 0.833333 0.916667 0.916667 1 1 \
      1.08333 1.08333 1.08333 1.08333 1 1 1 1 1.08333 1.08333 1.16667 1.16667 \
      1.25 1.25 1.33333 1.33333 1.33333 1.33333 ]", &mat_in_diff_ref
    );

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }


  void UnitTestConvolutional2DComponent() { /* Implemented by Harish Mallidi */
    // Convolutional2D component
    Component* c = ReadComponentFromString("<Convolutional2DComponent> 18 56 \
      <LearnRateCoef> 0 <BiasLearnRateCoef> 0 <FmapXLen> 4 <FmapYLen> 7 \
      <FiltXLen> 2 <FiltYLen> 3 <FiltXStep> 1 <FiltYStep> 2 <ConnectFmap> 1 \
      <Filters> [ 0 0 1 1 2 2 3 3 4 4 5 5 ; 0 0 1 1 2 2 3 3 4 4 5 5 ] \
      <Bias> [ 0 0 ]"
    );

    // input matrix
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 \
      11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 \
      21 21 22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);

    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 206 206 266 266 326 326 416 416 476 476 536 536 \
      626 626 686 686 746 746 ]", &mat_out_ref);

    // propagate
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // prepare mat_out_diff, mat_in_diff_ref,
    CuMatrix<BaseFloat> mat_out_diff;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]",
                           &mat_out_diff);

    CuMatrix<BaseFloat> mat_in_diff_ref;
    ReadCuMatrixFromString("[ 0 0 0 0 0 0 2 2 2 2 4 4 8 8 0 0 3 3 4.5 4.5 8 8 \
      9.5 9.5 13 13 20 20 9 9 18 18 19.5 19.5 23 23 24.5 24.5 28 28 41 41 \
      36 36 48 48 51 51 56 56 59 59 64 64 80 80 ]", &mat_in_diff_ref);

    // backpropagate
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestDropoutComponent() {
    Component* c = ReadComponentFromString("<Dropout> 100 100 <DropoutRetention> 0.7");
    // buffers,
    CuMatrix<BaseFloat> in(777, 100),
                        out,
                        out_diff,
                        in_diff;
    // init,
    in.Set(2.0);

    // propagate,
    c->Propagate(in, &out);
    AssertEqual(in.Sum(), out.Sum(), 0.01);

    // backprop,
    out_diff = in;
    c->Backpropagate(in, out, out_diff, &in_diff);
    AssertEqual(in_diff, out);

    delete c;
  }

}  // namespace nnet1
}  // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet1;

  for (kaldi::int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      // use no GPU,
      CuDevice::Instantiate().SelectGpuId("no");
    else
      // use GPU when available,
      CuDevice::Instantiate().SelectGpuId("optional");
#endif
    // unit-tests :
    UnitTestLengthNorm();
    UnitTestSimpleSentenceAveragingComponent();
    UnitTestConvolutionalComponentUnity();
    UnitTestConvolutionalComponent3x3();
    UnitTestMaxPoolingComponent();
    UnitTestConvolutional2DComponent();
    UnitTestMaxPooling2DComponent();
    UnitTestAveragePooling2DComponent();
    UnitTestDropoutComponent();
    // end of unit-tests,
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
