// nnet/nnet-lstm-projected-streams.h

// Copyright 2015-2016  Brno University of Technology (author: Karel Vesely)
// Copyright 2014  Jiayu DU (Jerry), Wei Li

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


#ifndef KALDI_NNET_NNET_LSTM_PROJECTED_H_
#define KALDI_NNET_NNET_LSTM_PROJECTED_H_

#include <string>
#include <vector>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * g: squashing neuron near input
 * i: Input gate
 * f: Forget gate
 * o: Output gate
 * c: memory Cell (CEC)
 * h: squashing neuron near output
 * m: output neuron of Memory block
 * r: recurrent projection neuron
 * y: output neuron of LSTMP
 *************************************/

namespace kaldi {
namespace nnet1 {

class LstmProjected : public MultistreamComponent {
 public:
  LstmProjected(int32 input_dim, int32 output_dim):
    MultistreamComponent(input_dim, output_dim),
    cell_dim_(0),
    proj_dim_(output_dim),
    cell_clip_(50.0),
    diff_clip_(1.0),
    cell_diff_clip_(0.0),
    grad_clip_(250.0)
  { }

  ~LstmProjected()
  { }

  Component* Copy() const { return new LstmProjected(*this); }
  ComponentType GetType() const { return kLstmProjected; }

  void InitData(std::istream &is) {
    // define options,
    float param_range = 0.1;
    // parse the line from prototype,
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ParamRange>") ReadBasicType(is, false, &param_range);
      else if (token == "<CellDim>") ReadBasicType(is, false, &cell_dim_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<CellClip>") ReadBasicType(is, false, &cell_clip_);
      else if (token == "<DiffClip>") ReadBasicType(is, false, &diff_clip_);
      else if (token == "<CellDiffClip>") ReadBasicType(is, false, &cell_diff_clip_);
      else if (token == "<GradClip>") ReadBasicType(is, false, &grad_clip_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamRange|CellDim|LearnRateCoef|BiasLearnRateCoef|CellClip|DiffClip|GradClip)";
    }

    // init the weights and biases (from uniform dist.),
    w_gifo_x_.Resize(4*cell_dim_, input_dim_, kUndefined);
    w_gifo_r_.Resize(4*cell_dim_, proj_dim_, kUndefined);
    bias_.Resize(4*cell_dim_, kUndefined);
    peephole_i_c_.Resize(cell_dim_, kUndefined);
    peephole_f_c_.Resize(cell_dim_, kUndefined);
    peephole_o_c_.Resize(cell_dim_, kUndefined);
    w_r_m_.Resize(proj_dim_, cell_dim_, kUndefined);
    //       (mean), (range)
    RandUniform(0.0, 2.0 * param_range, &w_gifo_x_);
    RandUniform(0.0, 2.0 * param_range, &w_gifo_r_);
    RandUniform(0.0, 2.0 * param_range, &bias_);
    RandUniform(0.0, 2.0 * param_range, &peephole_i_c_);
    RandUniform(0.0, 2.0 * param_range, &peephole_f_c_);
    RandUniform(0.0, 2.0 * param_range, &peephole_o_c_);
    RandUniform(0.0, 2.0 * param_range, &w_r_m_);

    KALDI_ASSERT(cell_dim_ > 0);
    KALDI_ASSERT(learn_rate_coef_ >= 0.0);
    KALDI_ASSERT(bias_learn_rate_coef_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      std::string token;
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'C': ReadToken(is, false, &token);
          /**/ if (token == "<CellDim>") ReadBasicType(is, binary, &cell_dim_);
          else if (token == "<CellClip>") ReadBasicType(is, binary, &cell_clip_);
          else if (token == "<CellDiffClip>") ReadBasicType(is, binary, &cell_diff_clip_);
          else if (token == "<ClipGradient>") ReadBasicType(is, binary, &grad_clip_); // bwd-compat.
          else KALDI_ERR << "Unknown token: " << token;
          break;
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          break;
        case 'D': ExpectToken(is, binary, "<DiffClip>");
          ReadBasicType(is, binary, &diff_clip_);
          break;
        case 'G': ExpectToken(is, binary, "<GradClip>");
          ReadBasicType(is, binary, &grad_clip_);
          break;
        default: ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    KALDI_ASSERT(cell_dim_ != 0);

    // Read the model parameters,
    w_gifo_x_.Read(is, binary);
    w_gifo_r_.Read(is, binary);
    bias_.Read(is, binary);

    peephole_i_c_.Read(is, binary);
    peephole_f_c_.Read(is, binary);
    peephole_o_c_.Read(is, binary);

    w_r_m_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<CellDim>");
    WriteBasicType(os, binary, cell_dim_);

    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);

    WriteToken(os, binary, "<CellClip>");
    WriteBasicType(os, binary, cell_clip_);
    WriteToken(os, binary, "<DiffClip>");
    WriteBasicType(os, binary, diff_clip_);
    WriteToken(os, binary, "<CellDiffClip>");
    WriteBasicType(os, binary, cell_diff_clip_);
    WriteToken(os, binary, "<GradClip>");
    WriteBasicType(os, binary, grad_clip_);

    // write model parameters,
    if (!binary) os << "\n";
    w_gifo_x_.Write(os, binary);
    w_gifo_r_.Write(os, binary);
    bias_.Write(os, binary);

    peephole_i_c_.Write(os, binary);
    peephole_f_c_.Write(os, binary);
    peephole_o_c_.Write(os, binary);

    w_r_m_.Write(os, binary);
  }

  int32 NumParams() const {
    return ( w_gifo_x_.NumRows() * w_gifo_x_.NumCols() +
         w_gifo_r_.NumRows() * w_gifo_r_.NumCols() +
         bias_.Dim() +
         peephole_i_c_.Dim() +
         peephole_f_c_.Dim() +
         peephole_o_c_.Dim() +
         w_r_m_.NumRows() * w_r_m_.NumCols() );
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_gifo_x_corr_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_gifo_r_corr_);

    offset += len; len = bias_.Dim();
    gradient->Range(offset, len).CopyFromVec(bias_corr_);

    offset += len; len = peephole_i_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_i_c_corr_);

    offset += len; len = peephole_f_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_f_c_corr_);

    offset += len; len = peephole_o_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_o_c_corr_);

    offset += len; len = w_r_m_.NumRows() * w_r_m_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_r_m_corr_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_gifo_x_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_gifo_r_);

    offset += len; len = bias_.Dim();
    params->Range(offset, len).CopyFromVec(bias_);

    offset += len; len = peephole_i_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_i_c_);

    offset += len; len = peephole_f_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_f_c_);

    offset += len; len = peephole_o_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_o_c_);

    offset += len; len = w_r_m_.NumRows() * w_r_m_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_r_m_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    w_gifo_x_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    w_gifo_r_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = bias_.Dim();
    bias_.CopyFromVec(params.Range(offset, len));

    offset += len; len = peephole_i_c_.Dim();
    peephole_i_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = peephole_f_c_.Dim();
    peephole_f_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = peephole_o_c_.Dim();
    peephole_o_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = w_r_m_.NumRows() * w_r_m_.NumCols();
    w_r_m_.CopyRowsFromVec(params.Range(offset, len));

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  std::string Info() const {
    return std::string("cell-dim ") + ToString(cell_dim_) + " " +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      ", cell_clip_ " + ToString(cell_clip_) +
      ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
      "\n  w_gifo_x_  "   + MomentStatistics(w_gifo_x_) +
      "\n  w_gifo_r_  "   + MomentStatistics(w_gifo_r_) +
      "\n  bias_  "     + MomentStatistics(bias_) +
      "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
      "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
      "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_) +
      "\n  w_r_m_  "    + MomentStatistics(w_r_m_);
  }

  std::string InfoGradient() const {
    // disassemble forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // disassemble backpropagate buffer into different neurons,
    const CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    return std::string("") +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      ", cell_clip_ " + ToString(cell_clip_) +
      ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
      "\n  ### Gradients " +
      "\n  w_gifo_x_corr_  "   + MomentStatistics(w_gifo_x_corr_) +
      "\n  w_gifo_r_corr_  "   + MomentStatistics(w_gifo_r_corr_) +
      "\n  bias_corr_  "     + MomentStatistics(bias_corr_) +
      "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
      "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
      "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_) +
      "\n  w_r_m_corr_  "    + MomentStatistics(w_r_m_corr_) +
      "\n  ### Activations (mostly after non-linearities)" +
      "\n  YI(0..1)^  " + MomentStatistics(YI) +
      "\n  YF(0..1)^  " + MomentStatistics(YF) +
      "\n  YO(0..1)^  " + MomentStatistics(YO) +
      "\n  YG(-1..1)  " + MomentStatistics(YG) +
      "\n  YC(-R..R)* " + MomentStatistics(YC) +
      "\n  YH(-1..1)  " + MomentStatistics(YH) +
      "\n  YM(-1..1)  " + MomentStatistics(YM) +
      "\n  YR(-R..R)  " + MomentStatistics(YR) +
      "\n  ### Derivatives (w.r.t. inputs of non-linearities)" +
      "\n  DI^ " + MomentStatistics(DI) +
      "\n  DF^ " + MomentStatistics(DF) +
      "\n  DO^ " + MomentStatistics(DO) +
      "\n  DG  " + MomentStatistics(DG) +
      "\n  DC* " + MomentStatistics(DC) +
      "\n  DH  " + MomentStatistics(DH) +
      "\n  DM  " + MomentStatistics(DM) +
      "\n  DR  " + MomentStatistics(DR);
  }

  /**
   * TODO: Do we really need this?
   */
  void ResetStreams(const std::vector<int32>& stream_reset_flag) {
    KALDI_ASSERT(NumStreams() == stream_reset_flag.size());
    if (prev_nnet_state_.NumRows() != stream_reset_flag.size()) {
      prev_nnet_state_.Resize(NumStreams(), 7*cell_dim_ + 1*proj_dim_, kSetZero);
    } else {
      for (int s = 0; s < NumStreams(); s++) {
        if (stream_reset_flag[s] == 1) {
          prev_nnet_state_.Row(s).SetZero();
        }
      }
    }
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {

    // reset context on each sentence if 'sequence_lengths_' not set
    // (happens in 'nnet-forward' or 'single-stream' training),
    if (sequence_lengths_.size() == 0) {
      ResetStreams(std::vector<int32>(1, 1));
    }

    KALDI_ASSERT(in.NumRows() % NumStreams() == 0);
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();

    // buffers,
    propagate_buf_.Resize((T+2)*S, 7 * cell_dim_ + proj_dim_, kSetZero);
    if (prev_nnet_state_.NumRows() != NumStreams()) {
      prev_nnet_state_.Resize(NumStreams(), 7*cell_dim_ + 1*proj_dim_, kSetZero); // lazy init,
    } else {
      propagate_buf_.RowRange(0, S).CopyFromMat(prev_nnet_state_); // use the 'previous-state',
    }

    // split activations by neuron types,
    CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    CuSubMatrix<BaseFloat> YGIFO(propagate_buf_.ColRange(0, 4*cell_dim_));

    // x -> g, i, f, o, not recurrent, do it all in once
    YGIFO.RowRange(1*S, T*S).AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);

    // bias -> g, i, f, o
    YGIFO.RowRange(1*S, T*S).AddVecToRows(1.0, bias_);

    // BufferPadding [T0]:dummy, [1, T]:current sequence, [T+1]:dummy
    for (int t = 1; t <= T; t++) {
      // multistream buffers for current time-step,
      CuSubMatrix<BaseFloat> y_all(propagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));
       CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_gifo(YGIFO.RowRange(t*S, S));

      // r(t-1) -> g, i, f, o
      y_gifo.AddMatMat(1.0, YR.RowRange((t-1)*S, S), kNoTrans, w_gifo_r_, kTrans,  1.0);

      // c(t-1) -> i(t) via peephole
      y_i.AddMatDiagVec(1.0, YC.RowRange((t-1)*S, S), kNoTrans, peephole_i_c_, 1.0);

      // c(t-1) -> f(t) via peephole
      y_f.AddMatDiagVec(1.0, YC.RowRange((t-1)*S, S), kNoTrans, peephole_f_c_, 1.0);

      // i, f sigmoid squashing
      y_i.Sigmoid(y_i);
      y_f.Sigmoid(y_f);

      // g tanh squashing
      y_g.Tanh(y_g);

      // g * i -> c
      y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);
      // c(t-1) * f -> c(t) via forget-gate
      y_c.AddMatMatElements(1.0, YC.RowRange((t-1)*S, S), y_f, 1.0);

      if (cell_clip_ > 0.0) {
        y_c.ApplyFloor(-cell_clip_);   // optional clipping of cell activation,
        y_c.ApplyCeiling(cell_clip_);  // google paper Interspeech2014: LSTM for LVCSR
      }

      // c(t) -> o(t) via peephole (non-recurrent, using c(t))
      y_o.AddMatDiagVec(1.0, y_c, kNoTrans, peephole_o_c_, 1.0);

      // o sigmoid squashing,
      y_o.Sigmoid(y_o);

      // h tanh squashing,
      y_h.Tanh(y_c);

      // h * o -> m via output gate,
      y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);

      // m -> r
      y_r.AddMatMat(1.0, y_m, kNoTrans, w_r_m_, kTrans, 0.0);

      // set zeros to padded frames,
      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            y_all.Row(s).SetZero();
          }
        }
      }
    }

    // set the 'projection layer' output as the LSTM output,
    out->CopyFromMat(YR.RowRange(1*S, T*S));

    // the state in the last 'frame' is transferred (can be zero vector)
    prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S, S));
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {

    // the number of sequences to be processed in parallel
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();

    // buffer,
    backpropagate_buf_.Resize((T+2)*S, 7 * cell_dim_ + proj_dim_, kSetZero);

    // split activations by neuron types,
    CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // split derivatives by neuron types,
    CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_.ColRange(0, 4*cell_dim_));

    // pre-copy partial derivatives from the LSTM output,
    DR.RowRange(1*S, T*S).CopyFromMat(out_diff);

    // BufferPadding [T0]:dummy, [1,T]:current sequence, [T+1]: dummy,
    for (int t = T; t >= 1; t--) {
      CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
      // CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
      // CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));
      // CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S, S));

      CuSubMatrix<BaseFloat> d_all(backpropagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_i(DI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_f(DF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_o(DO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_r(DR.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_gifo(DGIFO.RowRange(t*S, S));

      // r
      //   Version 1 (precise gradients):
      //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
      d_r.AddMatMat(1.0, DGIFO.RowRange((t+1)*S, S), kNoTrans, w_gifo_r_, kNoTrans, 1.0);

      /*
      //   Version 2 (Alex Graves' PhD dissertation):
      //   only backprop g(t+1) to r(t)
      CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, cell_dim_));
      d_r.AddMatMat(1.0, DG.RowRange((t+1)*S,S), kNoTrans, w_g_r_, kNoTrans, 1.0);
      */

      /*
      //   Version 3 (Felix Gers' PhD dissertation):
      //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
      //   CEC(with forget connection) is the only "error-bridge" through time
      */

      // r -> m
      d_m.AddMatMat(1.0, d_r, kNoTrans, w_r_m_, kNoTrans, 0.0);

      // m -> h via output gate
      d_h.AddMatMatElements(1.0, d_m, y_o, 0.0);
      d_h.DiffTanh(y_h, d_h);

      // o
      d_o.AddMatMatElements(1.0, d_m, y_h, 0.0);
      d_o.DiffSigmoid(y_o, d_o);

      // c
      // 1. diff from h(t)
      // 2. diff from c(t+1) (via forget-gate between CEC)
      // 3. diff from i(t+1) (via peephole)
      // 4. diff from f(t+1) (via peephole)
      // 5. diff from o(t)   (via peephole, not recurrent)
      d_c.AddMat(1.0, d_h);
      d_c.AddMatMatElements(1.0, DC.RowRange((t+1)*S, S), YF.RowRange((t+1)*S,S), 1.0);
      d_c.AddMatDiagVec(1.0, DI.RowRange((t+1)*S, S), kNoTrans, peephole_i_c_, 1.0);
      d_c.AddMatDiagVec(1.0, DF.RowRange((t+1)*S, S), kNoTrans, peephole_f_c_, 1.0);
      d_c.AddMatDiagVec(1.0, d_o                    , kNoTrans, peephole_o_c_, 1.0);
      // optionally clip the cell_derivative,
      if (cell_diff_clip_ > 0.0) {
        d_c.ApplyFloor(-cell_diff_clip_);
        d_c.ApplyCeiling(cell_diff_clip_);
      }

      // f
      d_f.AddMatMatElements(1.0, d_c, YC.RowRange((t-1)*S,S), 0.0);
      d_f.DiffSigmoid(y_f, d_f);

      // i
      d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
      d_i.DiffSigmoid(y_i, d_i);

      // c -> g via input gate
      d_g.AddMatMatElements(1.0, d_c, y_i, 0.0);
      d_g.DiffTanh(y_g, d_g);

      // Clipping per-frame derivatives for the next `t'.
      // Clipping applied to gates and input gate (as done in Google).
      // [ICASSP2015, Sak, Learning acoustic frame labelling...],
      //
      // The path from 'out_diff' to 'd_c' via 'd_h' is unclipped,
      // which is probably important for the 'Constant Error Carousel'
      // to work well.
      //
      if (diff_clip_ > 0.0) {
        d_gifo.ApplyFloor(-diff_clip_);
        d_gifo.ApplyCeiling(diff_clip_);
      }

      // set zeros to padded frames,
      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            d_all.Row(s).SetZero();
          }
        }
      }
    }

    // g,i,f,o -> x, calculating input derivatives,
    in_diff->AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kNoTrans, w_gifo_x_, kNoTrans, 0.0);

    // lazy initialization of udpate buffers,
    if (w_gifo_x_corr_.NumRows() == 0) {
      w_gifo_x_corr_.Resize(4*cell_dim_, input_dim_, kSetZero);
      w_gifo_r_corr_.Resize(4*cell_dim_, proj_dim_, kSetZero);
      bias_corr_.Resize(4*cell_dim_, kSetZero);
      peephole_i_c_corr_.Resize(cell_dim_, kSetZero);
      peephole_f_c_corr_.Resize(cell_dim_, kSetZero);
      peephole_o_c_corr_.Resize(cell_dim_, kSetZero);
      w_r_m_corr_.Resize(proj_dim_, cell_dim_, kSetZero);
    }

    // calculate delta
    const BaseFloat mmt = opts_.momentum;

    // weight x -> g, i, f, o
    w_gifo_x_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S, T*S), kTrans,
                                  in                      , kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    w_gifo_r_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S, T*S), kTrans,
                                  YR.RowRange(0*S, T*S)   , kNoTrans, mmt);
    // bias of g, i, f, o
    bias_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1*S, T*S), mmt);

    // recurrent peephole c -> i
    peephole_i_c_corr_.AddDiagMatMat(1.0, DI.RowRange(1*S, T*S), kTrans,
                                          YC.RowRange(0*S, T*S), kNoTrans, mmt);
    // recurrent peephole c -> f
    peephole_f_c_corr_.AddDiagMatMat(1.0, DF.RowRange(1*S, T*S), kTrans,
                                          YC.RowRange(0*S, T*S), kNoTrans, mmt);
    // peephole c -> o
    peephole_o_c_corr_.AddDiagMatMat(1.0, DO.RowRange(1*S, T*S), kTrans,
                                          YC.RowRange(1*S, T*S), kNoTrans, mmt);

    w_r_m_corr_.AddMatMat(1.0, DR.RowRange(1*S, T*S), kTrans,
                               YM.RowRange(1*S, T*S), kNoTrans, mmt);
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {

    // apply the gradient clipping,
    if (grad_clip_ > 0.0) {
      w_gifo_x_corr_.ApplyFloor(-grad_clip_);
      w_gifo_x_corr_.ApplyCeiling(grad_clip_);
      w_gifo_r_corr_.ApplyFloor(-grad_clip_);
      w_gifo_r_corr_.ApplyCeiling(grad_clip_);
      bias_corr_.ApplyFloor(-grad_clip_);
      bias_corr_.ApplyCeiling(grad_clip_);
      w_r_m_corr_.ApplyFloor(-grad_clip_);
      w_r_m_corr_.ApplyCeiling(grad_clip_);
      peephole_i_c_corr_.ApplyFloor(-grad_clip_);
      peephole_i_c_corr_.ApplyCeiling(grad_clip_);
      peephole_f_c_corr_.ApplyFloor(-grad_clip_);
      peephole_f_c_corr_.ApplyCeiling(grad_clip_);
      peephole_o_c_corr_.ApplyFloor(-grad_clip_);
      peephole_o_c_corr_.ApplyCeiling(grad_clip_);
    }

    const BaseFloat lr  = opts_.learn_rate;

    w_gifo_x_.AddMat(-lr * learn_rate_coef_, w_gifo_x_corr_);
    w_gifo_r_.AddMat(-lr * learn_rate_coef_, w_gifo_r_corr_);
    bias_.AddVec(-lr * bias_learn_rate_coef_, bias_corr_, 1.0);

    peephole_i_c_.AddVec(-lr * bias_learn_rate_coef_, peephole_i_c_corr_, 1.0);
    peephole_f_c_.AddVec(-lr * bias_learn_rate_coef_, peephole_f_c_corr_, 1.0);
    peephole_o_c_.AddVec(-lr * bias_learn_rate_coef_, peephole_o_c_corr_, 1.0);

    w_r_m_.AddMat(-lr * learn_rate_coef_, w_r_m_corr_);
  }

 private:
  // dims
  int32 cell_dim_;
  int32 proj_dim_;  ///< recurrent projection layer dim

  BaseFloat cell_clip_;  ///< Clipping of 'cell-values' in forward pass (per-frame),
  BaseFloat diff_clip_;  ///< Clipping of 'derivatives' in backprop (per-frame),
  BaseFloat cell_diff_clip_; ///< Clipping of 'cell-derivatives' accumulated over CEC (per-frame),
  BaseFloat grad_clip_;  ///< Clipping of the updates,

  // buffer for transfering state across batches,
  CuMatrix<BaseFloat> prev_nnet_state_;

  // feed-forward connections: from x to [g, i, f, o]
  CuMatrix<BaseFloat> w_gifo_x_;
  CuMatrix<BaseFloat> w_gifo_x_corr_;

  // recurrent projection connections: from r to [g, i, f, o]
  CuMatrix<BaseFloat> w_gifo_r_;
  CuMatrix<BaseFloat> w_gifo_r_corr_;

  // biases of [g, i, f, o]
  CuVector<BaseFloat> bias_;
  CuVector<BaseFloat> bias_corr_;

  // peephole from c to i, f, g
  // peephole connections are block-internal, so we use vector form
  CuVector<BaseFloat> peephole_i_c_;
  CuVector<BaseFloat> peephole_f_c_;
  CuVector<BaseFloat> peephole_o_c_;

  CuVector<BaseFloat> peephole_i_c_corr_;
  CuVector<BaseFloat> peephole_f_c_corr_;
  CuVector<BaseFloat> peephole_o_c_corr_;

  // projection layer r: from m to r
  CuMatrix<BaseFloat> w_r_m_;
  CuMatrix<BaseFloat> w_r_m_corr_;

  // propagate buffer: output of [g, i, f, o, c, h, m, r]
  CuMatrix<BaseFloat> propagate_buf_;

  // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
  CuMatrix<BaseFloat> backpropagate_buf_;
};  // class LstmProjected

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_LSTM_PROJECTED_H_
