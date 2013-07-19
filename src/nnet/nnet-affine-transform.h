// nnet/nnet-affine-transform.h

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


#ifndef KALDI_NNET_NNET_AFFINE_TRANSFORM_H_
#define KALDI_NNET_NNET_AFFINE_TRANSFORM_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class AffineTransform : public UpdatableComponent {
 public:
  AffineTransform(int32 dim_in, int32 dim_out, Nnet *nnet) 
    : UpdatableComponent(dim_in, dim_out, nnet), 
      linearity_(dim_out, dim_in), bias_(dim_out),
      linearity_corr_(dim_out, dim_in), bias_corr_(dim_out) 
  { }
  ~AffineTransform()
  { }

  ComponentType GetType() const {
    return kAffineTransform;
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

  std::string Info() const {
    return std::string("\n  linearity") + MomentStatistics(linearity_) +
           "\n  bias" + MomentStatistics(bias_);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
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
    bias_.AddVec(-lr, bias_corr_);
  }

  /// Accessors to the component parameters
  const CuVector<BaseFloat>& GetBias() {
    return bias_;
  }

  void SetBias(const CuVector<BaseFloat>& bias) {
    KALDI_ASSERT(bias.Dim() == bias_.Dim());
    bias_.CopyFromVec(bias);
  }

  const CuMatrix<BaseFloat>& GetLinearity() {
    return linearity_;
  }

  void SetLinearity(const CuMatrix<BaseFloat>& linearity) {
    KALDI_ASSERT(linearity.NumRows() == linearity_.NumRows());
    KALDI_ASSERT(linearity.NumCols() == linearity_.NumCols());
    linearity_.CopyFromMat(linearity);
  }

  const CuVector<BaseFloat>& GetBiasCorr() {
    return bias_corr_;
  }

  const CuMatrix<BaseFloat>& GetLinearityCorr() {
    return linearity_corr_;
  }


 private:
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;
};

} // namespace nnet1
} // namespace kaldi

#endif
