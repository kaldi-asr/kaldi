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

  void UnitTestConvolutionalComponent() {

    ConvolutionalComponent* c = new ConvolutionalComponent(5,5);

    std::string comp_data_str = "<PatchDim> 1 <PatchStep> 1 <PatchStride> 5 <Filters> [ 1 \n] <Bias> [ 0 ]\n";
    std::istringstream is_comp_data(comp_data_str);
    c->ReadData(is_comp_data, false);

    std::string matrix_str = "[ 1 2 3 4 5 ] ";
    std::istringstream is_matrix(matrix_str);
    CuMatrix<BaseFloat> mat_in;
    mat_in.Read(is_matrix, false);
    
    // propagate 
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_in,mat_out);

    // backpropagate
    CuMatrix<BaseFloat> mat_out_diff(mat_in), mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_out_diff " << mat_out_diff << " mat_in_diff " << mat_in_diff;
    AssertEqual(mat_out_diff,mat_in_diff);
    
    // once again
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_out_diff " << mat_out_diff << " mat_in_diff " << mat_in_diff;
    AssertEqual(mat_out_diff,mat_in_diff);

    delete c;
  }

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

  void UnitTestMaxPoolingComponent() {
    MaxPoolingComponent* m = new MaxPoolingComponent(9,7);
    
    std::string comp_data_str = "<PoolSize> 3 <PoolStep> 1 <PoolStride> 1 \n";
    std::istringstream is_comp_data(comp_data_str);
    m->ReadData(is_comp_data, false);

    std::string matrix_str = "[ 1 2 1 1 2 1 1 2 1 ; 2 3 2 2 3 2 2 3 2 ; 2 2 2 1 2 1 1 2 1 ; 1 2 3 1 4 1 1 2 1 ] ";
    std::istringstream is_matrix(matrix_str);

    // expected output
    std::string exp_out_str = "[ 2 2 2 ; 3 3 3 ] ";
    std::istringstream is_exp_out_str(exp_out_str);
    CuMatrix<BaseFloat> mat_exp;
    mat_exp.Read(is_exp_out_str, false);


    CuMatrix<BaseFloat> mat_in;
    CuMatrix<BaseFloat> mat_out;
    CuMatrix<BaseFloat> inp_diff;
    mat_in.Read(is_matrix, false);
    
    KALDI_LOG << mat_in.ColRange(0, 2);
    m->Propagate(mat_in,&mat_out);
 
   KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out << "mat_exp" << mat_exp;
   m->Backpropagate(mat_in, mat_out, mat_out, &inp_diff);
   KALDI_LOG << inp_diff;
    
    // KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out << "mat_exp" << mat_exp;
    // AssertEqual(mat_out, mat_exp);

    delete m;
    
  }

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

  void UnitTestMatOperations(){
    //    CuMatrix<BaseFloat> A;

    Vector<BaseFloat> v(10), w(9);
    CuVector<BaseFloat> b(9);
    CuArray<int32> id;

    for(int i=0; i < 9; i++) {
      v(i) = i;
      w(i) = i+1;
    }
    Matrix<BaseFloat> M(10,9);
    Matrix<BaseFloat> W(10,9);
    CuMatrix<BaseFloat> A, B(10,9);

    M.AddVecVec(1.0, v, w);
    A = M;
    B.Set(-1e20);
    B.Max(A);
    A.FindRowMaxId(&id);
    CuMatrix<BaseFloat> C(A);
    C.Set(2);
    KALDI_LOG << "C=" << C;
    KALDI_LOG << "A=" << A;
    // KALDI_LOG << "id=" << id;
    
    

    // KALDI_LOG << "A " << B.Max(A);
    // b.AddRowSumMat(1.0, A, 0.0);
    // KALDI_LOG << b;
    // b.AddRowSumMat(1.0, A, 0.0);
    // KALDI_LOG << b;
     // CuSubMatrix<BaseFloat> As(A.ColRange(0,1));    
     // KALDI_LOG << "As " << As;

     // std::vector<MatrixIndexT> id(2,4);
     // CuMatrix<BaseFloat> B;
     // B.Resize(A.NumRows(), 2, kSetZero);
     // B.CopyCols(A, id);
     // KALDI_LOG << "B " << B ;
     // KALDI_LOG << "Sum="<< B.Sum();

    // Matrix<BaseFloat> C(2,2), D(2,2), E(2,2);
    // Vector<BaseFloat> c(2);
    // c(0)=1;c(1)=2;
    // C.AddVecVec(1.0,c,c);
    // KALDI_LOG << "C " << C;

    // D(1,1)=1;

    // // KALDI_LOG << "D " <<D;

    // // C.MulElements(D);
    
    // // KALDI_LOG << "C " << C;

    // CuMatrix<BaseFloat> CuC, CuD;
    // CuC = C;
    // CuD = D;
    
    // KALDI_LOG << "CuC " << CuC;
    // CuC.MulElements(CuD);
    // KALDI_LOG << "CuC " << CuC;
    // KALDI_LOG << "Sum=" << CuC.Sum();

    
    
  }

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
    UnitTestConvolutionalComponent();
    UnitTestMaxPoolingComponent();
    // UnitTestConvolutional2DComponent();
    // UnitTestMatOperations();
    // UnitTestMaxPooling2DComponent();
    // UnitTestAveragePooling2DComponent();
    // end of unit-tests,
    if (loop == 0)
        KALDI_LOG << "Tests without GPU use succeeded.\n";
      else
        KALDI_LOG << "Tests with GPU use (if available) succeeded.\n";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0; 
}
