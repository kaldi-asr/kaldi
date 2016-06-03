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
    clip_gradient_(0.0)
  { }
  ~ParametricRelu()
  { }

  Component* Copy() const { return new ParametricRelu(*this); }
  ComponentType GetType() const { return kParametricRelu; }

  void InitData(std::istream &is) {
    // define options
    float init_alpha = 1, init_beta = 0.25;
    float fix_alpha = 0.0, fix_beta = 0.0;

    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<InitAlpha>")  ReadBasicType(is, false, &init_alpha);
      else if (token == "<InitBeta>")  ReadBasicType(is, false, &init_beta);
      else if (token == "<BiasLearnRateCoef>")  ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<LearnRateCoef>")  ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<FixAlpha>")  ReadBasicType(is, false, &fix_alpha);
      else if (token == "<FixBeta>")   ReadBasicType(is, false, &fix_beta);
      else if (token == "<ClipGradient>")  ReadBasicType(is, false, &clip_gradient_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                  << " (InitAlpha|InitBeta|BiasLearnRateCoef|LearnRateCoef|FixAlpha|FixBeta|ClipGradient)";
    }

    //
    // Initialize trainable parameters,
    //

    Vector<BaseFloat> veca(output_dim_);
    for (int32 i = 0; i < output_dim_; i++) {
      // elements of vector is initialized with input alpha
      veca(i) = init_alpha;
    }
    alpha_ = veca;
    //
    Vector<BaseFloat> vecb(output_dim_);
    for (int32 i = 0; i < output_dim_; i++) {
      // elements of vector is initialized with input beta
      vecb(i) = init_beta;
    }
    beta_ = vecb;
    //
    fix_alpha_ = fix_alpha;
    fix_beta_ = fix_beta;
    KALDI_ASSERT(clip_gradient_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'F': ExpectToken(is, binary, "<FixAlpha>");
          ReadBasicType(is, binary, &fix_alpha_);
          ExpectToken(is, binary, "<FixBeta>");
          ReadBasicType(is, binary, &fix_beta_);
          break;
         case 'C': ExpectToken(is, binary, "<ClipGradient>");
          ReadBasicType(is, binary, &clip_gradient_);
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
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<FixAlpha>");
    WriteBasicType(os, binary, fix_alpha_);
    WriteToken(os, binary, "<FixBeta>");
    WriteBasicType(os, binary, fix_beta_);
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);

    // ParametricRelu scaling parameters
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
      ", bias-lr-coef " + ToString(bias_learn_rate_coef_) +
      "\n  beta" + MomentStatistics(beta_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  alpha_grad") +
      MomentStatistics(alpha_corr_) +
      ", bias-lr-coef " + ToString(bias_learn_rate_coef_) +
      "\n  beta_grad" + MomentStatistics(beta_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // Multiply activations by ReLU scalars (max_alpha, min_beta):
    // // out = in * (in >= 0.0 ? in * max_alpha : in * min_beta)
    out->ParametricRelu(in, alpha_, beta_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by activations (alpha, beta)
    // ((dE/da)*w) === out_diff, f(y) == out,
    //  (out > 0 ) ? out_diff * alpha : out_diff * beta
    in_diff->DiffParametricRelu(out, out_diff, alpha_, beta_);
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat alr = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat blr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    if (clip_gradient_ > 0.0) {  // gradient clipping
      alpha_corr_.ApplyFloor(-clip_gradient_);
      beta_corr_.ApplyFloor(-clip_gradient_);
      alpha_corr_.ApplyCeiling(clip_gradient_);
      beta_corr_.ApplyCeiling(clip_gradient_);
    }
    // compute gradient (incl. momentum)
    if (!fix_alpha_) {  // the alpha parameter is learnable
       in_alpha_.Resize(input.NumRows(), input.NumCols());
       in_alpha_.CopyFromMat(input);
       in_alpha_.ApplyFloor(0.0);
       in_alpha_.MulElements(diff);
       alpha_corr_.AddRowSumMat(1.0, in_alpha_, mmt);
       alpha_.AddVec(-alr, alpha_corr_);
    }
    if (!fix_beta_) {  // the beta parameter is learnable
       in_beta_.Resize(input.NumRows(), input.NumCols());
       in_beta_.CopyFromMat(input);
       in_beta_.ApplyCeiling(0.0);
       in_beta_.MulElements(diff);
       beta_corr_.AddRowSumMat(1.0, in_beta_, mmt);
       beta_.AddVec(-blr, beta_corr_);
    }
  }

  /// Accessors to the component parameters,
  const CuVectorBase<BaseFloat>& GetAlpha() const { return alpha_; }

  void SetAlpha(const CuVectorBase<BaseFloat>& alpha) {
    KALDI_ASSERT(alpha.Dim() == alpha_.Dim());
    alpha_.CopyFromVec(alpha);
  }
  const CuVectorBase<BaseFloat>& GetBeta() const { return beta_; }

  void SetBeta(const CuVectorBase<BaseFloat>& beta) {
    KALDI_ASSERT(beta.Dim() == beta_.Dim());
    beta_.CopyFromVec(beta);
  }

 private:
  CuVector<BaseFloat> alpha_; /// < Vector of 'alphas', one value per neuron.
  CuVector<BaseFloat> beta_; /// < Vector of 'betas', one value per neuron.

  CuVector<BaseFloat> alpha_corr_; /// < Vector of 'alpha' updates.
  CuVector<BaseFloat> beta_corr_; /// < Vector of 'beta' updates.

  CuMatrix<BaseFloat> in_alpha_;
  CuMatrix<BaseFloat> in_beta_;

  BaseFloat clip_gradient_;
  BaseFloat fix_alpha_;
  BaseFloat fix_beta_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_PARAMETRIC_RELU_H_
