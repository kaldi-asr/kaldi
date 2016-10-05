// nnet/nnet-activation.h

// Copyright 2011-2016  Brno University of Technology (author: Karel Vesely)

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

#include <string>
#include <vector>
#include <cmath>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet1 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    out->ApplySoftMaxPerRow(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function,
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }
};


class HiddenSoftmax : public Component {
 public:
  HiddenSoftmax(int32 dim_in, int32 dim_out) :
    Component(dim_in, dim_out)
  { }

  ~HiddenSoftmax()
  { }

  Component* Copy() const { return new HiddenSoftmax(*this); }
  ComponentType GetType() const { return kHiddenSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    out->ApplySoftMaxPerRow(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // This Softmax should be used for a hidden layer, it calculates
    // the true Jacobian of Softmax: J = diag(out) - out*out^T

    // The backpropagation formual is:
    // in_diff = out_diff \odot out - out(out_diff^T * out)
    // (where \odot is Hadamard product)

    // 1st term, out_diff \odot out,
    in_diff->CopyFromMat(out_diff);
    in_diff->MulElements(out);

    // 2nd term, -out(out_diff^T * out),
    diag_out_diff_out_.Resize(out.NumRows());
    diag_out_diff_out_.AddDiagMatMat(1.0, out_diff, kNoTrans, out, kTrans, 0.0);
    in_diff->AddDiagVecMat(-1.0, diag_out_diff_out_, out, kNoTrans, 1.0);
  }

 private:
  /// buffer for dot-products in BackpropagateFnc,
  CuVector<BaseFloat> diag_out_diff_out_;
};

class BlockSoftmax : public Component {
 public:
  BlockSoftmax(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  ~BlockSoftmax()
  { }

  Component* Copy() const { return new BlockSoftmax(*this); }
  ComponentType GetType() const { return kBlockSoftmax; }

  void InitData(std::istream &is) {
    // parse config
    std::string token,
      dims_str;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<BlockDims>") is >> dims_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockDims)";
    }
    // parse dims,
    if (!kaldi::SplitStringToIntegers(dims_str, ",:", false, &block_dims))
      KALDI_ERR << "Invalid block-dims " << dims_str;
    // sanity check
    int32 sum = 0;
    for (int32 i = 0; i < block_dims.size(); i++) {
      sum += block_dims[i];
    }
    KALDI_ASSERT(sum == OutputDim());
  }

  void ReadData(std::istream &is, bool binary) {
    ReadIntegerVector(is, binary, &block_dims);
    block_offset.resize(block_dims.size()+1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }
    // check
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_dims);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // perform softmax per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      // get the blocks,
      CuSubMatrix<BaseFloat> in_bl =
        in.ColRange(block_offset[bl], block_dims[bl]);
      CuSubMatrix<BaseFloat> out_bl =
        out->ColRange(block_offset[bl], block_dims[bl]);
      // y = e^x_j/sum_j(e^x_j),
      out_bl.ApplySoftMaxPerRow(in_bl);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    // (assuming we already got softmax-cross-entropy derivative in out_diff)
    in_diff->CopyFromMat(out_diff);

    // Set the derivatives to zero for the matrix-lines in which
    // the sum of 'derivatives' was 1.0 (i.e. there was no target):
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      // get the block,
      CuSubMatrix<BaseFloat> diff_bl =
        in_diff->ColRange(block_offset[bl], block_dims[bl]);
      // get the sum of each row,
      CuVector<BaseFloat> row_sum(diff_bl.NumRows());
      row_sum.AddColSumMat(1.0, diff_bl, 0.0);  // 0: keep as-is, 1: zero-out
      // we'll scale rows by 0/1 masks,
      CuVector<BaseFloat> row_diff_mask(row_sum);
      row_diff_mask.Scale(-1.0);  // 0: keep as-is, -1: zero-out
      row_diff_mask.Add(1.0);  // 1: keep as-is, 0: zero-out
      // here we should have only 0's and 1's,
      diff_bl.MulRowsVec(row_diff_mask);
    }
  }

  std::string Info() const {
    return "\n  softmax-dims " + ToString(block_dims);
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
};




class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex,
    in_diff->DiffSigmoid(out, out_diff);
  }
};



class Tanh : public Component {
 public:
  Tanh(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  ~Tanh()
  { }

  Component* Copy() const { return new Tanh(*this); }
  ComponentType GetType() const { return kTanh; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // y = (e^x - e^(-x)) / (e^x + e^(-x)),
    out->Tanh(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (1 - y^2)ex
    in_diff->DiffTanh(out, out_diff);
  }
};



class Dropout : public Component {
 public:
  Dropout(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out),
      dropout_rate_(0.5)
  { }

  ~Dropout()
  { }

  Component* Copy() const { return new Dropout(*this); }
  ComponentType GetType() const { return kDropout; }

  void InitData(std::istream &is) {
    is >> std::ws;  // eat-up whitespace
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<DropoutRate>") ReadBasicType(is, false, &dropout_rate_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (DropoutRate)";
    }
    KALDI_ASSERT(dropout_rate_ >= 0.0 && dropout_rate_ < 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    bool finished = false;
    while ('<' == Peek(is, binary) && !finished) {
      std::string token;
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'D': ReadToken(is, false, &token);
          /**/ if (token == "<DropoutRate>") ReadBasicType(is, binary, &dropout_rate_);
          else if (token == "<DropoutRetention>") { /* compatibility */
            BaseFloat dropout_retention;
            ReadBasicType(is, binary, &dropout_retention);
            dropout_rate_ = 1.0 - dropout_retention;
          } else KALDI_ERR << "Unknown token: " << token;
          break;
        case '!': ExpectToken(is, binary, "<!EndOfComponent>");
          finished = true;
          break;
        default: ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    KALDI_ASSERT(dropout_rate_ >= 0.0 && dropout_rate_ < 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<DropoutRate>");
    WriteBasicType(os, binary, dropout_rate_);
  }

  std::string Info() const {
    return std::string("<DropoutRate> ") + ToString(dropout_rate_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    // set N inputs to zero, according to the 'dropout_rate_' ...
    dropout_mask_.Resize(out->NumRows(), out->NumCols());
    rand_.RandUniform(&dropout_mask_);  // [0..1]
    dropout_mask_.Add(-dropout_rate_);  // [(-rate)..(1-rate)]
    dropout_mask_.Heaviside(dropout_mask_); // (x > 0.0 ? 1 : 0)
    out->MulElements(dropout_mask_);
    // rescale to keep the same dynamic range as w/o dropout,
    out->Scale(1.0 / (1.0 - dropout_rate_));
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(dropout_mask_);
    // enlarge the output to fit same dynamic range as w/o dropout
    in_diff->Scale(1.0 / (1.0 - dropout_rate_));
  }

  BaseFloat GetDropoutRate() { return dropout_rate_; }

  void SetDropoutRate(BaseFloat dr) {
    dropout_rate_ = dr;
    KALDI_ASSERT(dropout_rate_ >= 0.0 && dropout_rate_ < 1.0);
  }

 private:
  BaseFloat dropout_rate_;  ///< probability that a neuron is dropped,

  CuRand<BaseFloat> rand_;  ///< generator of random numbers,

  CuMatrix<BaseFloat> dropout_mask_;  // random binary mask,
                                      // 1 = keep neuron, 0 = drop neuron,
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_ACTIVATION_H_

