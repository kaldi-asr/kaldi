// nnet/nnet-activation.h

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


#ifndef KALDI_NNET_ACTIVATION_H
#define KALDI_NNET_ACTIVATION_H

#include "nnet_cpu/nnet-component.h"
namespace kaldi {

class Sigmoid : public Component {
 public:
  Sigmoid(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet* nnet) 
    : Component(dim_in, dim_out, nnet)
  { }
  ~Sigmoid()
  { }

  ComponentType GetType() const {
    return kSigmoid;
  }

  void PropagateFnc(const Matrix<BaseFloat>& in, Matrix<BaseFloat>* out) {
    // y = 1/(1+e^-x)
    for(MatrixIndexT r=0; r<out->NumRows(); r++) {
      for(MatrixIndexT c=0; c<out->NumCols(); c++) {
        (*out)(r, c) = 1.0/(1.0+exp(-in(r, c)));
      }
    }
  }

  void BackpropagateFnc(const Matrix<BaseFloat>& in_err, Matrix<BaseFloat>* out_err) {
    // ey = y(1-y)ex
    const Matrix<BaseFloat>& y = nnet_->PropagateBuffer()[nnet_->IndexOfLayer(*this)+1];

    for(MatrixIndexT r=0; r<out_err->NumRows(); r++) {
      for(MatrixIndexT c=0; c<out_err->NumCols(); c++) {
        (*out_err)(r, c) = y(r, c)*(1.0-y(r, c))*in_err(r, c);
      }
    }
  }
};


class Softmax : public Component {
 public:
  Softmax(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet* nnet) 
    : Component(dim_in, dim_out, nnet)
  { }
  ~Softmax()
  { }

  ComponentType GetType() const {
    return kSoftmax;
  }

  void PropagateFnc(const Matrix<BaseFloat>& in, Matrix<BaseFloat>* out) {
    // y = e^x_j/sum_j(e^x_j)
    out->CopyFromMat(in);
    for(MatrixIndexT r=0; r<out->NumRows(); r++) {
      out->Row(r).ApplySoftMax();
    }
  }

  void BackpropagateFnc(const Matrix<BaseFloat>& in_err, Matrix<BaseFloat>* out_err) {
    // simply copy the error
    // (ie. assume crossentropy error function, 
    // while in_err contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    out_err->CopyFromMat(in_err);
  }
};



} // namespace

#endif

