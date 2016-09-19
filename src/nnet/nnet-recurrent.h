// nnet/nnet-lstm-projected-streams.h

// Copyright 2016  Brno University of Technology (author: Karel Vesely)

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



#ifndef KALDI_NNET_NNET_RECURRENT_STREAMS_H_
#define KALDI_NNET_NNET_RECURRENT_STREAMS_H_

#include <string>
#include <vector>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {
namespace nnet1 {


/**
 * Component with recurrent connections, 'tanh' non-linearity.
 * No internal state preserved, starting each sequence from zero vector.
 *
 * Can be used in 'per-sentence' training and multi-stream training.
 */
class RecurrentComponent : public MultistreamComponent {
 public:
  RecurrentComponent(int32 input_dim, int32 output_dim):
    MultistreamComponent(input_dim, output_dim)
  { }

  ~RecurrentComponent()
  { }

  Component* Copy() const { return new RecurrentComponent(*this); }
  ComponentType GetType() const { return kRecurrentComponent; }

  void InitData(std::istream &is) {
    // define options,
    float param_scale = 0.02;
    // parse the line from prototype,
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<GradClip>") ReadBasicType(is, false, &grad_clip_);
      else if (token == "<DiffClip>") ReadBasicType(is, false, &diff_clip_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<ParamScale>") ReadBasicType(is, false, &param_scale);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (GradClip|DiffClip|LearnRateCoef|BiasLearnRateCoef|ParamScale)";
    }

    // init the weights and biases (from uniform dist.),
    w_forward_.Resize(output_dim_, input_dim_);
    w_recurrent_.Resize(output_dim_, output_dim_);
    bias_.Resize(output_dim_);

    RandUniform(0.0, 2.0 * param_scale, &w_forward_);
    RandUniform(0.0, 2.0 * param_scale, &w_recurrent_);
    RandUniform(0.0, 2.0 * param_scale, &bias_);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      std::string token;
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'G': ExpectToken(is, binary, "<GradClip>");
          ReadBasicType(is, binary, &grad_clip_);
          break;
        case 'D': ExpectToken(is, binary, "<DiffClip>");
          ReadBasicType(is, binary, &diff_clip_);
          break;
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          break;
        default: ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }

    // Read the data (data follow the tokens),
    w_forward_.Read(is, binary);
    w_recurrent_.Read(is, binary);
    bias_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<GradClip>");
    WriteBasicType(os, binary, grad_clip_);
    WriteToken(os, binary, "<DiffClip>");
    WriteBasicType(os, binary, diff_clip_);

    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);

    if (!binary) os << "\n";
    w_forward_.Write(os, binary);
    w_recurrent_.Write(os, binary);
    bias_.Write(os, binary);
  }

  int32 NumParams() const {
    return w_forward_.NumRows() * w_forward_.NumCols() +
      w_recurrent_.NumRows() * w_recurrent_.NumCols() +
      bias_.Dim();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_forward_corr_.NumRows() * w_forward_corr_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_forward_corr_);

    offset += len; len = w_recurrent_corr_.NumRows() * w_recurrent_corr_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_recurrent_corr_);

    offset += len; len = bias_corr_.Dim();
    gradient->Range(offset, len).CopyFromVec(bias_corr_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_forward_.NumRows() * w_forward_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_forward_);

    offset += len; len = w_recurrent_.NumRows() * w_recurrent_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_recurrent_);

    offset += len; len = bias_.Dim();
    params->Range(offset, len).CopyFromVec(bias_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_forward_.NumRows() * w_forward_.NumCols();
    w_forward_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = w_recurrent_.NumRows() * w_recurrent_.NumCols();
    w_recurrent_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = bias_.Dim();
    bias_.CopyFromVec(params.Range(offset, len));

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  std::string Info() const {
    return std::string("  ") +
      "\n  w_forward_  "   + MomentStatistics(w_forward_) +
      "\n  w_recurrent_  " + MomentStatistics(w_recurrent_) +
      "\n  bias_  "        + MomentStatistics(bias_);
  }

  std::string InfoGradient() const {
    return std::string("") +
      "( learn_rate_coef " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef " + ToString(bias_learn_rate_coef_) +
      ", grad-clip " + ToString(grad_clip_) +
      ", diff-clip " + ToString(diff_clip_) + " )" +
      "\n  Gradients:" +
      "\n  w_forward_corr_  "   + MomentStatistics(w_forward_corr_) +
      "\n  w_recurrent_corr_  "   + MomentStatistics(w_recurrent_corr_) +
      "\n  bias_corr_  "     + MomentStatistics(bias_corr_) +
      "\n  Forward-pass:" +
      "\n  out_  " + MomentStatistics(out_) +
      "\n  Backward-pass:" +
      "\n  out_diff_bptt_  " + MomentStatistics(out_diff_bptt_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {


    KALDI_ASSERT(in.NumRows() % NumStreams() == 0);
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();

    // Precopy bias,
    out->AddVecToRows(1.0, bias_, 0.0);
    // Apply 'forward' connections,
    out->AddMatMat(1.0, in, kNoTrans, w_forward_, kTrans, 1.0);

    // First line of 'out' w/o recurrent signal, apply 'tanh' directly,
    out->RowRange(0, S).Tanh(out->RowRange(0, S));

    // Apply 'recurrent' connections,
    for (int32 t = 1; t < T; t++) {
      out->RowRange(t*S, S).AddMatMat(1.0, out->RowRange((t-1)*S, S), kNoTrans, w_recurrent_, kTrans, 1.0);
      out->RowRange(t*S, S).Tanh(out->RowRange(t*S, S));
      // Zero output for padded frames,
      if (sequence_lengths_.size() == S) {
        for (int32 s = 0; s < S; s++) {
          if (t >= sequence_lengths_[s]) {
            out->Row(t*S + s).SetZero();
          }
        }
      }
      //
    }

    out_ = (*out);  // We'll need a copy for updating the recurrent weights!

    // We are DONE ;)
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {

    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();

    // Apply BPTT on 'out_diff',
    out_diff_bptt_ = out_diff;
    for (int32 t = T-1; t >= 1; t--) {
      // buffers,
      CuSubMatrix<BaseFloat> d_t = out_diff_bptt_.RowRange(t*S, S);
      CuSubMatrix<BaseFloat> d_t1 = out_diff_bptt_.RowRange((t-1)*S, S);
      const CuSubMatrix<BaseFloat> y_t = out.RowRange(t*S, S);

      // BPTT,
      d_t.DiffTanh(y_t, d_t);
      d_t1.AddMatMat(1.0, d_t, kNoTrans, w_recurrent_, kNoTrans, 1.0);

      // clipping,
      if (diff_clip_ > 0.0) {
        d_t1.ApplyFloor(-diff_clip_);
        d_t1.ApplyCeiling(diff_clip_);
      }

      // Zero diff for padded frames,
      if (sequence_lengths_.size() == S) {
        for (int32 s = 0; s < S; s++) {
          if (t >= sequence_lengths_[s]) {
            out_diff_bptt_.Row(t*S + s).SetZero();
          }
        }
      }
    }

    // Apply 'DiffTanh' on first block,
    CuSubMatrix<BaseFloat> d_t = out_diff_bptt_.RowRange(0, S);
    const CuSubMatrix<BaseFloat> y_t = out.RowRange(0, S);
    d_t.DiffTanh(y_t, d_t);

    // Transform diffs to 'in_diff',
    in_diff->AddMatMat(1.0, out_diff_bptt_, kNoTrans, w_forward_, kNoTrans, 0.0);

    // We are DONE ;)
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    int32 T = input.NumRows() / NumStreams();
    int32 S = NumStreams();

    // getting the learning rate,
    const BaseFloat lr  = opts_.learn_rate;
    const BaseFloat mmt = opts_.momentum;

    if (bias_corr_.Dim() != OutputDim()) {
      w_forward_corr_.Resize(w_forward_.NumRows(), w_forward_.NumCols(), kSetZero);
      w_recurrent_corr_.Resize(w_recurrent_.NumRows(), w_recurrent_.NumCols(), kSetZero);
      bias_corr_.Resize(OutputDim(), kSetZero);
    }

    // getting the gradients,
    w_forward_corr_.AddMatMat(1.0, out_diff_bptt_, kTrans, input, kNoTrans, mmt);


    w_recurrent_corr_.AddMatMat(1.0, out_diff_bptt_.RowRange(S, (T-1)*S), kTrans,
                                               out_.RowRange(0, (T-1)*S), kNoTrans, mmt);

    bias_corr_.AddRowSumMat(1.0, out_diff_bptt_, mmt);

    // updating,
    w_forward_.AddMat(-lr * learn_rate_coef_, w_forward_corr_);
    w_recurrent_.AddMat(-lr * learn_rate_coef_, w_recurrent_corr_);
    bias_.AddVec(-lr * bias_learn_rate_coef_, bias_corr_);
  }

 private:

  BaseFloat grad_clip_;  ///< Clipping of the update,
  BaseFloat diff_clip_;  ///< Clipping in the BPTT loop,

  // trainable parameters,
  CuMatrix<BaseFloat> w_forward_;
  CuMatrix<BaseFloat> w_recurrent_;
  CuVector<BaseFloat> bias_;

  // udpate buffers,
  CuMatrix<BaseFloat> w_forward_corr_;
  CuMatrix<BaseFloat> w_recurrent_corr_;
  CuVector<BaseFloat> bias_corr_;

  // forward propagation buffer,
  CuMatrix<BaseFloat> out_;

  // back-propagate buffer,
  CuMatrix<BaseFloat> out_diff_bptt_;

};  // class RecurrentComponent

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_RECURRENT_STREAMS_H_
