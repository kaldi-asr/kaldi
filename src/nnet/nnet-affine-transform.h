// nnet/nnet-affine-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)
//           2018 Alibaba.Inc (Author: ShiLiang Zhang) 
             	
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


#ifndef KALDI_NNET_NNET_AFFINE_TRANSFORM_H_
#define KALDI_NNET_NNET_AFFINE_TRANSFORM_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class AffineTransform : public UpdatableComponent {
 public:
  AffineTransform(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    linearity_(dim_out, dim_in), bias_(dim_out),
    linearity_corr_(dim_out, dim_in), bias_corr_(dim_out),
    max_norm_(0.0)
  { }
  ~AffineTransform()
  { }

  Component* Copy() const { return new AffineTransform(*this); }
  ComponentType GetType() const { return kAffineTransform; }

  void InitData(std::istream &is) {
    // define options
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1;
    int xavier_flag = 0;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm_);
      else if (token == "<Xavier>") ReadBasicType(is, false, &xavier_flag);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
    }

    //
    // Initialize trainable parameters,
    // if Xavier_flag=1, use the “Xavier” initialization
    if(xavier_flag){
      float range = sqrt(6)/sqrt(OutputDim() + InputDim());
      linearity_.Resize(OutputDim(), InputDim(), kSetZero);
      RandUniform(0.0, range, &linearity_);

      bias_.Resize(OutputDim(),kSetZero);
    }
    else{
      // Gaussian with given std_dev (mean = 0),
      linearity_.Resize(OutputDim(), InputDim());
      RandGauss(0.0, param_stddev, &linearity_);
      // Uniform,
      bias_.Resize(OutputDim());
      RandUniform(bias_mean, bias_range, &bias_);
    }
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          break;
        case 'M': ExpectToken(is, binary, "<MaxNorm>");
          ReadBasicType(is, binary, &max_norm_);
          break;
        default:
          std::string token;
          ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    // Read the data (data follow the tokens),

    // weight matrix,
    linearity_.Read(is, binary);
    // bias vector,
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);
    if (!binary) os << "\n";
    // weights
    linearity_.Write(os, binary);
    bias_.Write(os, binary);
  }

  int32 NumParams() const {
    return linearity_.NumRows()*linearity_.NumCols() + bias_.Dim();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 linearity_num_elem = linearity_.NumRows() * linearity_.NumCols();
    gradient->Range(0, linearity_num_elem).CopyRowsFromMat(linearity_corr_);
    gradient->Range(linearity_num_elem, bias_.Dim()).CopyFromVec(bias_corr_);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 linearity_num_elem = linearity_.NumRows() * linearity_.NumCols();
    params->Range(0, linearity_num_elem).CopyRowsFromMat(linearity_);
    params->Range(linearity_num_elem, bias_.Dim()).CopyFromVec(bias_);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 linearity_num_elem = linearity_.NumRows() * linearity_.NumCols();
    linearity_.CopyRowsFromVec(params.Range(0, linearity_num_elem));
    bias_.CopyFromVec(params.Range(linearity_num_elem, bias_.Dim()));
  }

  std::string Info() const {
    return std::string("\n  linearity") +
      MomentStatistics(linearity_) +
      ", lr-coef " + ToString(learn_rate_coef_) +
      ", max-norm " + ToString(max_norm_) +
      "\n  bias" + MomentStatistics(bias_) +
      ", lr-coef " + ToString(bias_learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  linearity_grad") +
      MomentStatistics(linearity_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_) +
      ", max-norm " + ToString(max_norm_) +
      "\n  bias_grad" + MomentStatistics(bias_corr_) +
      ", lr-coef " + ToString(bias_learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by weights
    in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    const int32 num_frames = input.NumRows();
    // compute gradient (incl. momentum)
    linearity_corr_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, mmt);
    bias_corr_.AddRowSumMat(1.0, diff, mmt);
    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr*l2*num_frames, linearity_);
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
    }
    // update
    linearity_.AddMat(-lr, linearity_corr_);
    bias_.AddVec(-lr_bias, bias_corr_);
    // max-norm
    if (max_norm_ > 0.0) {
      CuMatrix<BaseFloat> lin_sqr(linearity_);
      lin_sqr.MulElements(linearity_);
      CuVector<BaseFloat> l2(OutputDim());
      l2.AddColSumMat(1.0, lin_sqr, 0.0);
      l2.ApplyPow(0.5);  // we have per-neuron L2 norms,
      CuVector<BaseFloat> scl(l2);
      scl.Scale(1.0/max_norm_);
      scl.ApplyFloor(1.0);
      scl.InvertElements();
      linearity_.MulRowsVec(scl);  // shink to sphere!
    }
  }

  /// Accessors to the component parameters,
  const CuVectorBase<BaseFloat>& GetBias() const { return bias_; }

  void SetBias(const CuVectorBase<BaseFloat>& bias) {
    KALDI_ASSERT(bias.Dim() == bias_.Dim());
    bias_.CopyFromVec(bias);
  }

  const CuMatrixBase<BaseFloat>& GetLinearity() const { return linearity_; }

  void SetLinearity(const CuMatrixBase<BaseFloat>& linearity) {
    KALDI_ASSERT(linearity.NumRows() == linearity_.NumRows());
    KALDI_ASSERT(linearity.NumCols() == linearity_.NumCols());
    linearity_.CopyFromMat(linearity);
  }

 private:
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  BaseFloat max_norm_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_AFFINE_TRANSFORM_H_
