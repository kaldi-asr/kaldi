// nnet/nnet-affine-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_AFFINE_TRANSFORM_NOBIAS_H_
#define KALDI_NNET_NNET_AFFINE_TRANSFORM_NOBIAS_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class AffineTransformNobias : public UpdatableComponent {
 public:
  AffineTransformNobias(int32 dim_in, int32 dim_out) 
    : UpdatableComponent(dim_in, dim_out), 
      linearity_(dim_out, dim_in), linearity_corr_(dim_out, dim_in), learn_rate_coef_(1.0)
  { }
  ~AffineTransformNobias()
  { }

  Component* Copy() const { return new AffineTransformNobias(*this); }
  ComponentType GetType() const { return kAffineTransformNobias; }
  
  void InitData(std::istream &is) {
    // define options
    float param_stddev = 0.1;
    float learn_rate_coef = 1.0;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev)";
      is >> std::ws; // eat-up whitespace
    }

    //
    // initialize
    //
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    for (int32 r=0; r<output_dim_; r++) {
      for (int32 c=0; c<input_dim_; c++) {
        mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    linearity_ = mat;
    //
    learn_rate_coef_ = learn_rate_coef;
    //
  }

  void ReadData(std::istream &is, bool binary) {
    // learning-rate coefficien
    ExpectToken(is, binary, "<LearnRateCoef>");
    ReadBasicType(is, binary, &learn_rate_coef_);
    // weights
    linearity_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    linearity_.Write(os, binary);
  }

  int32 NumParams() const { return linearity_.NumRows()*linearity_.NumCols(); }
  
  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 linearity_num_elem = linearity_.NumRows() * linearity_.NumCols(); 
    wei_copy->Range(0,linearity_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(linearity_));
  }
  
  std::string Info() const {
    return std::string("\n  linearity") + MomentStatistics(linearity_);
  }
  std::string InfoGradient() const {
    return std::string("\n  linearity_grad") + MomentStatistics(linearity_corr_) +
           ", lr-coef " + ToString(learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 0.0);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    // multiply error derivative by weights
    in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
  }


  void Update(const CuMatrix<BaseFloat> &input, const CuMatrix<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    const int32 num_frames = input.NumRows();
    // compute gradient (incl. momentum)
    linearity_corr_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, mmt);
    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr*l2*num_frames, linearity_);
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
    }
    // update
    linearity_.AddMat(-lr*learn_rate_coef_, linearity_corr_);
  }

  /// Accessors to the component parameters
  const CuMatrix<BaseFloat>& GetLinearity() {
    return linearity_;
  }

  void SetLinearity(const CuMatrix<BaseFloat>& linearity) {
    KALDI_ASSERT(linearity.NumRows() == linearity_.NumRows());
    KALDI_ASSERT(linearity.NumCols() == linearity_.NumCols());
    linearity_.CopyFromMat(linearity);
  }

  const CuMatrix<BaseFloat>& GetLinearityCorr() {
    return linearity_corr_;
  }


 private:
  CuMatrix<BaseFloat> linearity_;
  CuMatrix<BaseFloat> linearity_corr_;
  
  BaseFloat learn_rate_coef_;
};

} // namespace nnet1
} // namespace kaldi

#endif
