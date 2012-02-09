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


#include "nnet_cpu/nnet-component.h"

namespace kaldi {

class BiasedLinearity : public UpdatableComponent {
 public:
  BiasedLinearity(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet* nnet) 
    : UpdatableComponent(dim_in, dim_out, nnet), 
      linearity_(dim_out,dim_in), bias_(dim_out),
      linearity_corr_(dim_out,dim_in), bias_corr_(dim_out) 
  { }
  ~BiasedLinearity()
  { }

  ComponentType GetType() const {
    return kBiasedLinearity;
  }

  void ReadData(std::istream& is, bool binary) {
    linearity_.Read(is,binary);
    bias_.Read(is,binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }

  void WriteData(std::ostream& os, bool binary) const {
    linearity_.Write(os,binary);
    bias_.Write(os,binary);
  }

  void PropagateFnc(const Matrix<BaseFloat>& in, Matrix<BaseFloat>* out) {
    //precopy bias
    for (MatrixIndexT i=0; i<out->NumRows(); i++) {
      out->CopyRowFromVec(bias_,i);
    }
    //multiply by weights^t
    out->AddMatMat(1.0,in,kNoTrans,linearity_,kTrans,1.0);
  }

  void BackpropagateFnc(const Matrix<BaseFloat>& in_err, Matrix<BaseFloat>* out_err) {
    //multiply error by weights
    out_err->AddMatMat(1.0,in_err,kNoTrans,linearity_,kNoTrans,0.0);
  }


  void Update(const Matrix<BaseFloat>& input, const Matrix<BaseFloat>& err) {
    
    //compute gradient
    linearity_corr_.AddMatMat(1.0,err,kTrans,input,kNoTrans,momentum_);
    bias_corr_.Scale(momentum_);
    bias_corr_.AddRowSumMat(err);
    //l2 regularization
    if(l2_penalty_ != 0.0) {
      linearity_.AddMat(-learn_rate_*l2_penalty_*input.NumRows(),linearity_);
    }
    //l1 regularization
    if(l1_penalty_ != 0.0) {
      BaseFloat l1 = learn_rate_*input.NumRows()*l1_penalty_;
      for(MatrixIndexT r=0; r<linearity_.NumRows(); r++) {
        for(MatrixIndexT c=0; c<linearity_.NumCols(); c++) {
          if(linearity_(r,c)==0.0) continue; //skip L1 if zero weight!
          BaseFloat l1sign = l1;
          if(linearity_(r,c) < 0.0) 
            l1sign = -l1;
          BaseFloat before = linearity_(r,c);
          BaseFloat after = linearity_(r,c)-learn_rate_*linearity_corr_(r,c)-l1sign;
          if((after > 0.0) ^ (before > 0.0)) {
            linearity_(r,c) = 0.0;
            linearity_corr_(r,c) = 0.0;
          } else {
            linearity_(r,c) -= l1sign;
          }
        }
      }
    }
    //update
    linearity_.AddMat(-learn_rate_,linearity_corr_);
    bias_.AddVec(-learn_rate_,bias_corr_);

    /*
    std::cout <<"I"<< input.Row(0);
    std::cout <<"E"<< err.Row(0);
    std::cout <<"CORL"<< linearity_corr_.Row(0);
    std::cout <<"CORB"<< bias_corr_;
    std::cout <<"L"<< linearity_.Row(0);
    std::cout <<"B"<< bias_;
    std::cout << "\n";
    */

    //std::cout << l1_penalty_ << l2_penalty_ << momentum_ << learn_rate_ << "\n";
  }

 private:
  Matrix<BaseFloat> linearity_;
  Vector<BaseFloat> bias_;

  Matrix<BaseFloat> linearity_corr_;
  Vector<BaseFloat> bias_corr_;
};

} //namespace

#endif
