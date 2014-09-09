// nnet/nnet-activation.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_ACTIVATION_H_
#define KALDI_NNET_NNET_ACTIVATION_H_

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {
namespace nnet1 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    out->ApplySoftMaxPerRow(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }
};



class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
  }
};



class Tanh : public Component {
 public:
  Tanh(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Tanh()
  { }

  Component* Copy() const { return new Tanh(*this); }
  ComponentType GetType() const { return kTanh; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (e^x - e^(-x)) / (e^x + e^(-x))
    out->Tanh(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (1 - y^2)ex
    in_diff->DiffTanh(out, out_diff);
  }
};



class Dropout : public Component {
 public:
  Dropout(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out), dropout_retention_(0.5)
  { }
  ~Dropout()
  { }

  Component* Copy() const { return new Dropout(*this); }
  ComponentType GetType() const { return kDropout; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<DropoutRetention>") ReadBasicType(is, false, &dropout_retention_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (DropoutRetention)";
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<DropoutRetention>");
      ReadBasicType(is, binary, &dropout_retention_);
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<DropoutRetention>");
    WriteBasicType(os, binary, dropout_retention_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    // switch off 50% of the inputs...
    dropout_mask_.Resize(out->NumRows(),out->NumCols());
    dropout_mask_.Set(dropout_retention_);
    rand_.BinarizeProbs(dropout_mask_,&dropout_mask_);
    out->MulElements(dropout_mask_);
    // rescale to keep same dynamic range as w/o dropout
    out->Scale(1.0/dropout_retention_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(dropout_mask_);
    // enlarge output to fit dynamic range w/o dropout
    in_diff->Scale(1.0/dropout_retention_);
  }
  
  BaseFloat GetDropoutRetention() {
    return dropout_retention_;
  }

  void SetDropoutRetention(BaseFloat dr) {
    dropout_retention_ = dr;
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

 private:
  CuRand<BaseFloat> rand_;
  CuMatrix<BaseFloat> dropout_mask_;
  BaseFloat dropout_retention_;
};



} // namespace nnet1
} // namespace kaldi

#endif

