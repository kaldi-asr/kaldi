// nnet/nnet-biasedlinearity.h

// Copyright 2011  Karel Vesely

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


#ifndef KALDI_NNET_BIASEDLINEARITY_H
#define KALDI_NNET_BIASEDLINEARITY_H


#include "nnet/nnet-component.h"

namespace kaldi {

class BiasedLinearity : public UpdatableComponent {
 public:
  BiasedLinearity(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet) 
    : UpdatableComponent(dim_in, dim_out, nnet), 
      linearity_(dim_out, dim_in), bias_(dim_out),
      linearity_corr_(dim_out, dim_in), bias_corr_(dim_out) 
  { }
  ~BiasedLinearity()
  { }

  ComponentType GetType() const {
    return kBiasedLinearity;
  }

  void ReadData(std::istream &is, bool binary) {
    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    linearity_.Write(os, binary);
    bias_.Write(os, binary);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddScaledRow(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err, CuMatrix<BaseFloat> *out_err) {
    // multiply error by weights
    out_err->AddMatMat(1.0, in_err, kNoTrans, linearity_, kNoTrans, 0.0);
  }


  void Update(const CuMatrix<BaseFloat> &input, const CuMatrix<BaseFloat> &err) {
    // compute gradient
    linearity_corr_.AddMatMat(1.0, err, kTrans, input, kNoTrans, momentum_);
    bias_corr_.AddRowSumMat(1.0, err, momentum_);
    // l2 regularization
    if (l2_penalty_ != 0.0) {
      BaseFloat l2 = learn_rate_*l2_penalty_*input.NumRows();
      linearity_.AddMat(-l2, linearity_);
    }
    // l1 regularization
    if (l1_penalty_ != 0.0) {
      BaseFloat l1 = learn_rate_*input.NumRows()*l1_penalty_;
      cu::RegularizeL1(&linearity_, &linearity_corr_, l1, learn_rate_);
    }
    // update
    linearity_.AddMat(-learn_rate_, linearity_corr_);
    bias_.AddVec(-learn_rate_, bias_corr_);
  }

 private:
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;
};

} // namespace

#endif
