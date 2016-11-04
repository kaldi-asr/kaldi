// nnet/nnet-linear-transform.h

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


#ifndef KALDI_NNET_NNET_LINEAR_TRANSFORM_H_
#define KALDI_NNET_NNET_LINEAR_TRANSFORM_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class LinearTransform : public UpdatableComponent {
 public:
  LinearTransform(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    linearity_(dim_out, dim_in),
    linearity_corr_(dim_out, dim_in)
  { }

  ~LinearTransform()
  { }

  Component* Copy() const { return new LinearTransform(*this); }
  ComponentType GetType() const { return kLinearTransform; }

  void InitData(std::istream &is) {
    // define options
    float param_stddev = 0.1;
    std::string read_matrix_file;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<ReadMatrix>") ReadToken(is, false, &read_matrix_file);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|ReadMatrix|LearnRateCoef)";
    }

    if (read_matrix_file != "") {  // load from file,
      bool binary;
      Input in(read_matrix_file, &binary);
      linearity_.Read(in.Stream(), binary);
      in.Close();
      // check dims,
      if (OutputDim() != linearity_.NumRows() ||
          InputDim() != linearity_.NumCols()) {
        KALDI_ERR << "Dimensionality mismatch! Expected matrix"
                  << " r=" << OutputDim() << " c=" << InputDim()
                  << ", loaded matrix " << read_matrix_file
                  << " with r=" << linearity_.NumRows()
                  << " c=" << linearity_.NumCols();
      }
      KALDI_LOG << "Loaded <LinearTransform> matrix from file : "
                << read_matrix_file;
      return;
    }

    //
    // Initialize trainable parameters,
    //
    // Gaussian with given std_dev (mean = 0),
    linearity_.Resize(OutputDim(), InputDim());
    RandGauss(0.0, param_stddev, &linearity_);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        default:
          std::string token;
          ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    // Read the data (data follow the tokens),

    // weights
    linearity_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    if (!binary) os << "\n";
    linearity_.Write(os, binary);
  }

  int32 NumParams() const {
    return linearity_.NumRows()*linearity_.NumCols();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    gradient->CopyRowsFromMat(linearity_corr_);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    params->CopyRowsFromMat(linearity_);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    linearity_.CopyRowsFromVec(params);
  }

  void SetLinearity(const MatrixBase<BaseFloat>& l) {
    KALDI_ASSERT(l.NumCols() == linearity_.NumCols());
    KALDI_ASSERT(l.NumRows() == linearity_.NumRows());
    linearity_.CopyFromMat(l);
  }

  std::string Info() const {
    return std::string("\n  linearity") +
      MomentStatistics(linearity_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  linearity_grad") +
      MomentStatistics(linearity_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 0.0);
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
  const CuMatrixBase<BaseFloat>& GetLinearity() { return linearity_; }

  void SetLinearity(const CuMatrixBase<BaseFloat>& linearity) {
    KALDI_ASSERT(linearity.NumRows() == linearity_.NumRows());
    KALDI_ASSERT(linearity.NumCols() == linearity_.NumCols());
    linearity_.CopyFromMat(linearity);
  }

  const CuMatrixBase<BaseFloat>& GetLinearityCorr() { return linearity_corr_; }

 private:
  CuMatrix<BaseFloat> linearity_;
  CuMatrix<BaseFloat> linearity_corr_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_LINEAR_TRANSFORM_H_
