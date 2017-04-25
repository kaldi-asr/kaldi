// nnet/nnet-parametric-relu.h

// Copyright 2016 Brno University of Technology (author: Murali Karthick B)
//           2011-2014  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_PARAMETRIC_RELU_H_
#define KALDI_NNET_NNET_PARAMETRIC_RELU_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class ParametricRelu : public UpdatableComponent {
 public:
  ParametricRelu(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    alpha_(dim_out),
    beta_(dim_out),
    alpha_corr_(dim_out),
    beta_corr_(dim_out),
    alpha_learn_rate_coef_(0.0),
    beta_learn_rate_coef_(0.0)
  { }

  ~ParametricRelu()
  { }

  Component* Copy() const { return new ParametricRelu(*this); }
  ComponentType GetType() const { return kParametricRelu; }

  void InitData(std::istream &is) {
    // define options
    BaseFloat alpha = 1.0, beta = 0.0;

    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<Alpha>") ReadBasicType(is, false, &alpha);
      else if (token == "<Beta>") ReadBasicType(is, false, &beta);
      else if (token == "<AlphaLearnRateCoef>") ReadBasicType(is, false, &alpha_learn_rate_coef_);
      else if (token == "<BetaLearnRateCoef>") ReadBasicType(is, false, &beta_learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                  << " (Alpha|Beta|AlphaLearnRateCoef|BetaLearnRateCoef)";
    }

    // Initialize trainable parameters,
    alpha_.Set(alpha);
    beta_.Set(beta);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'A': ExpectToken(is, binary, "<AlphaLearnRateCoef>");
          ReadBasicType(is, binary, &alpha_learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BetaLearnRateCoef>");
          ReadBasicType(is, binary, &beta_learn_rate_coef_);
          break;
        default:
          std::string token;
          ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    // ParametricRelu scaling parameters
    alpha_.Read(is, binary);
    beta_.Read(is, binary);
    KALDI_ASSERT(alpha_.Dim() == output_dim_);
    KALDI_ASSERT(beta_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<AlphaLearnRateCoef>");
    WriteBasicType(os, binary, alpha_learn_rate_coef_);
    WriteToken(os, binary, "<BetaLearnRateCoef>");
    WriteBasicType(os, binary, beta_learn_rate_coef_);

    // ParametricRelu scales for each neuron,
    if (!binary) os << "\n";
    alpha_.Write(os, binary);
    beta_.Write(os, binary);
  }

  int32 NumParams() const {
    return alpha_.Dim() + beta_.Dim();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 alpha_num_elem = alpha_.Dim();
    int32 beta_num_elem = beta_.Dim();
    gradient->Range(0, alpha_num_elem).CopyFromVec(Vector<BaseFloat>(alpha_corr_));
    gradient->Range(alpha_num_elem, beta_num_elem).CopyFromVec(Vector<BaseFloat>(beta_corr_));
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 alpha_num_elem = alpha_.Dim();
    int32 beta_num_elem = beta_.Dim();
    params->Range(0, alpha_num_elem).CopyFromVec(Vector<BaseFloat>(alpha_));
    params->Range(alpha_num_elem, beta_num_elem).CopyFromVec(Vector<BaseFloat>(beta_));
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 alpha_num_elem = alpha_.Dim();
    int32 beta_num_elem = beta_.Dim();
    alpha_.CopyFromVec(params.Range(0, alpha_num_elem));
    beta_.CopyFromVec(params.Range(alpha_num_elem, beta_num_elem));
  }

  std::string Info() const {
    return std::string("\n  alpha") +
      MomentStatistics(alpha_) +
      ", alpha-lr-coef " + ToString(alpha_learn_rate_coef_) +
      "\n  beta" + MomentStatistics(beta_) +
      ", beta-lr-coef " + ToString(beta_learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  alpha_grad") +
      MomentStatistics(alpha_corr_) +
      ", alpha-lr-coef " + ToString(alpha_learn_rate_coef_) +
      "\n  beta_grad" + MomentStatistics(beta_corr_) +
      ", beta-lr-coef " + ToString(beta_learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // out = (in < 0.0 ? aplha*in : beta*in)
    out->ParametricRelu(in, alpha_, beta_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // in_diff = (in > 0 ? alpha * out_diff : beta * out_diff)
    in_diff->DiffParametricRelu(in, out_diff, alpha_, beta_);
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // we use these hyperparameters,
    const BaseFloat alpha_lr = opts_.learn_rate * alpha_learn_rate_coef_;
    const BaseFloat beta_lr = opts_.learn_rate * beta_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;

    if (alpha_learn_rate_coef_ > 0.0) {
       // get gradient,
       alpha_aux_ = input;
       alpha_aux_.ApplyFloor(0.0); // masking positive Relu inputs,
       alpha_aux_.MulElements(diff);
       alpha_corr_.AddRowSumMat(1.0, alpha_aux_, mmt);
       // update,
       alpha_.AddVec(-alpha_lr, alpha_corr_);
    }
    if (beta_learn_rate_coef_ > 0.0) {
       // get gradient,
       beta_aux_ = input;
       beta_aux_.ApplyCeiling(0.0); // masking positive Relu inputs,
       beta_aux_.MulElements(diff);
       beta_corr_.AddRowSumMat(1.0, beta_aux_, mmt);
       beta_.AddVec(-beta_lr, beta_corr_);
    }
  }

 private:
  CuVector<BaseFloat> alpha_;  ///< Vector of 'alphas', one value per neuron.
  CuVector<BaseFloat> beta_;  ///< Vector of 'betas', one value per neuron.

  CuVector<BaseFloat> alpha_corr_;  ///< Vector of 'alpha' updates.
  CuVector<BaseFloat> beta_corr_;  ///< Vector of 'beta' updates.

  /// Auxiliary matrix for getting 'alpha' updates,
  CuMatrix<BaseFloat> alpha_aux_;
  /// Auxiliary matrix for getting 'beta' updates,
  CuMatrix<BaseFloat> beta_aux_;

  /// Controls learning rate for alpha (0.0 disables learning),
  BaseFloat alpha_learn_rate_coef_;
  /// Controls learning rate for beta (0.0 disables learning),
  BaseFloat beta_learn_rate_coef_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_PARAMETRIC_RELU_H_
