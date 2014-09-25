// nnet/nnet-component-test.cc
// Copyright 2014  Brno University of Technology (author: Karel Vesely),
//                 The Johns Hopkins University (author: Sri Harish Mallidi)

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


#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-convolutional-component.h"
#include "nnet/nnet-convolutional-2d-component.h"
#include "nnet/nnet-max-pooling-component.h"
#include "nnet/nnet-max-pooling-2d-component.h"
#include "nnet/nnet-average-pooling-2d-component.h"
#include "util/common-utils.h"

#include <sstream>
#include <fstream>
#include <algorithm>

namespace kaldi {
namespace nnet1 {

  /*
   * Helper functions
   */  
  template<typename Real>
  void ReadCuMatrixFromString(const std::string& s, CuMatrix<Real>* m) {
    std::istringstream is(s + "\n");
    m->Read(is, false); // false for ascii
  }

  Component* ReadComponentFromString(const std::string& s) {
    std::istringstream is(s + "\n");
    return Component::Read(is, false); // false for ascii
  }
  /*
   */

  void UnitTestConvolutionalComponentUnity() {
    // make 'identity' convolutional component,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 5 5 \
      <PatchDim> 1 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <Filters> [ 1 \
      ] <Bias> [ 0 ]"
    );
    
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 ] ", &mat_in);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_in,mat_out);

    // backpropagate,
    CuMatrix<BaseFloat> mat_out_diff(mat_in), mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_out_diff " << mat_out_diff << " mat_in_diff " << mat_in_diff;
    AssertEqual(mat_out_diff,mat_in_diff);
    
    // clean,
    delete c;
  }

  void UnitTestConvolutionalComponent3x3() {
    // make 3x3 convolutional component, design such weights and input so output is zero,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 9 15 \
      <PatchDim> 3 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
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
    CuMatrix<BaseFloat> mat_in_diff_ref; // hand-computed back-propagated values,
    ReadCuMatrixFromString("[ -1 -4 -15 -8 -6   0 -3 -6 3 6   1 1 14 11 7 ]", &mat_in_diff_ref);

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);
    
    // clean,
    delete c;
  }



  void UnitTestMaxPoolingComponent() {
    // make max-pooling component, assuming 4 conv. neurons, non-overlapping pool of size 3,
    Component* c = Component::Init("<MaxPoolingComponent> <InputDim> 24 <OutputDim> 8 \
                     <PoolSize> 3 <PoolStep> 3 <PoolStride> 4");

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
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // locations of max values will be shown,
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    mat_out_diff.Set(1);
    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref; // hand-computed back-propagated values,
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
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }


  /* TODO for Harish!
  void UnitTestMaxPooling2DComponent(){
    std::string dim_str;

    std::ifstream infile("/home/harish/kaldi_cnn_testfiles/avgpool1.txt");
    std::getline(infile, dim_str);
    
    std::stringstream stream(dim_str);    
    
    std::vector<int> dims;
    int n;
    while(stream >> n){
      dims.push_back(n);
    }

    std::string comp_data_str, matrix_str;
    std::getline(infile, comp_data_str);
    std::getline(infile, matrix_str);

    MaxPooling2DComponent* c = new MaxPooling2DComponent(dims[0], dims[1]);
    
    std::istringstream is_comp_data(comp_data_str);
    c->ReadData(is_comp_data, false);

    std::istringstream is_matrix(matrix_str);
    CuMatrix<BaseFloat> mat_in;
    mat_in.Read(is_matrix, false);

    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_out " << mat_out;
    
    std::string mat_out_diff_str;
    std::getline(infile, mat_out_diff_str);
    std::istringstream is_mat_out_diff(mat_out_diff_str);
    CuMatrix<BaseFloat> out_diff, in_diff;
    out_diff.Read(is_mat_out_diff, false);
    
    c->Backpropagate(mat_in, mat_out, out_diff, &in_diff);
    KALDI_LOG << "out_diff" << out_diff;
    KALDI_LOG << "in_diff " << in_diff;

    delete c;

  }
  */


  /* TODO for Harish:
  void UnitTestAveragePooling2DComponent(){
    std::string dim_str;

    std::ifstream infile("/home/harish/kaldi_cnn_testfiles/avgpool1.txt");
    std::getline(infile, dim_str);
    
    std::stringstream stream(dim_str);    
    
    std::vector<int> dims;
    int n;
    while(stream >> n){
      dims.push_back(n);
    }

    std::string comp_data_str, matrix_str;
    std::getline(infile, comp_data_str);
    std::getline(infile, matrix_str);

    AveragePooling2DComponent* c = new AveragePooling2DComponent(dims[0], dims[1]);
    
    std::istringstream is_comp_data(comp_data_str);
    c->ReadData(is_comp_data, false);

    std::istringstream is_matrix(matrix_str);
    CuMatrix<BaseFloat> mat_in;
    mat_in.Read(is_matrix, false);

    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_out " << mat_out;
    
    std::string mat_out_diff_str;
    std::getline(infile, mat_out_diff_str);
    std::istringstream is_mat_out_diff(mat_out_diff_str);
    CuMatrix<BaseFloat> out_diff, in_diff;
    out_diff.Read(is_mat_out_diff, false);
    
    c->Backpropagate(mat_in, mat_out, out_diff, &in_diff);
    KALDI_LOG << "out_diff" << out_diff;
    KALDI_LOG << "in_diff " << in_diff;

    delete c;

  }
  */


  /* TODO for Harish:
  void UnitTestConvolutional2DComponent() {

    std::string dim_str;

    std::ifstream infile("/home/harish/kaldi_cnn_testfiles/filt6.txt");
    std::getline(infile, dim_str);
    
    std::stringstream stream(dim_str);    
    
    std::vector<int> dims;
    int n;
    while(stream >> n){
      dims.push_back(n);
    }

    std::string comp_data_str, matrix_str;
    std::getline(infile, comp_data_str);
    std::getline(infile, matrix_str);

    Convolutional2DComponent* c = new Convolutional2DComponent(dims[0], dims[1]);
    
    std::istringstream is_comp_data(comp_data_str);
    c->ReadData(is_comp_data, false);

    std::istringstream is_matrix(matrix_str);
    CuMatrix<BaseFloat> mat_in;
    mat_in.Read(is_matrix, false);

    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_out " << mat_out;

    std::string mat_out_diff_str;
    std::getline(infile, mat_out_diff_str);
    std::istringstream is_mat_out_diff(mat_out_diff_str);
    CuMatrix<BaseFloat> out_diff, in_diff;
    out_diff.Read(is_mat_out_diff, false);
    // CuMatrix<BaseFloat> out_diff(mat_out), in_diff;
    
    c->Backpropagate(mat_in, mat_out, out_diff, &in_diff);
    KALDI_LOG << "out_diff" << out_diff;
    KALDI_LOG << "in_diff " << in_diff;

    c->Update(mat_in, out_diff);

    delete c;
    
  }
  */

} // namespace nnet1
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet1;

  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // use no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // use GPU when available
#endif
    // unit-tests :
    UnitTestConvolutionalComponentUnity();
    UnitTestConvolutionalComponent3x3();
    UnitTestMaxPoolingComponent();
    // UnitTestConvolutional2DComponent();
    // UnitTestMaxPooling2DComponent();
    // UnitTestAveragePooling2DComponent();
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
