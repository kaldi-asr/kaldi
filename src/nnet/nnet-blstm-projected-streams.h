// nnet/nnet-lstm-projected-streams.h

// Copyright 2014  Jiayu DU (Jerry), Wei Li
// Copyright 2015  Chongjia Ni
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



#ifndef KALDI_NNET_BLSTM_PROJECTED_STREAMS_H_
#define KALDI_NNET_BLSTM_PROJECTED_STREAMS_H_

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
 * f-*: forward direction
 * b-*: backward direction
 *************************************/

namespace kaldi {
namespace nnet1 {

class BLstmProjectedStreams : public UpdatableComponent {
 public:
  BLstmProjectedStreams(int32 input_dim, int32 output_dim) :
    UpdatableComponent(input_dim, output_dim),
    ncell_(0),
    nrecur_(output_dim),
    nstream_(0),
    clip_gradient_(0.0)
    //, dropout_rate_(0.0)
  { }

  ~BLstmProjectedStreams()
  { }

  Component* Copy() const { return new BLstmProjectedStreams(*this); }
  ComponentType GetType() const { return kBLstmProjectedStreams; }

  static void InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
    m.SetRandUniform();  // uniform in [0, 1]
    m.Add(-0.5);         // uniform in [-0.5, 0.5]
    m.Scale(2 * scale);  // uniform in [-scale, +scale]
  }

  static void InitVecParam(CuVector<BaseFloat> &v, float scale) {
    Vector<BaseFloat> tmp(v.Dim());
    for (int i=0; i < tmp.Dim(); i++) {
      tmp(i) = (RandUniform() - 0.5) * 2 * scale;
    }
    v = tmp;
  }

  void InitData(std::istream &is) {
    // define options
    float param_scale = 0.02;
    // parse config
    std::string token;
    while (!is.eof()) {
      ReadToken(is, false, &token);
      if (token == "<CellDim>")
        ReadBasicType(is, false, &ncell_);
      else if (token == "<ClipGradient>")
        ReadBasicType(is, false, &clip_gradient_);
      //else if (token == "<DropoutRate>")
      //  ReadBasicType(is, false, &dropout_rate_);
      else if (token == "<ParamScale>")
        ReadBasicType(is, false, &param_scale);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
               << " (CellDim|NumStream|ParamScale)";
               //<< " (CellDim|NumStream|DropoutRate|ParamScale)";
      is >> std::ws;
    }

    // init weight and bias (Uniform)
    // forward direction
    f_w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);
    f_w_gifo_r_.Resize(4*ncell_, nrecur_, kUndefined);
    f_w_r_m_.Resize(nrecur_, ncell_, kUndefined);

    InitMatParam(f_w_gifo_x_, param_scale);
    InitMatParam(f_w_gifo_r_, param_scale);
    InitMatParam(f_w_r_m_, param_scale);
    // backward direction
    b_w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);
    b_w_gifo_r_.Resize(4*ncell_, nrecur_, kUndefined);
    b_w_r_m_.Resize(nrecur_, ncell_, kUndefined);

    InitMatParam(b_w_gifo_x_, param_scale);
    InitMatParam(b_w_gifo_r_, param_scale);
    InitMatParam(b_w_r_m_, param_scale);

    // forward direction
    f_bias_.Resize(4*ncell_, kUndefined);
    // backward direction
    b_bias_.Resize(4*ncell_, kUndefined);
    InitVecParam(f_bias_, param_scale);
    InitVecParam(b_bias_, param_scale);

    // This is for input gate, forgot gate and output gate connected with the previous cell
    // forward direction
    f_peephole_i_c_.Resize(ncell_, kUndefined);
    f_peephole_f_c_.Resize(ncell_, kUndefined);
    f_peephole_o_c_.Resize(ncell_, kUndefined);
    // backward direction
    b_peephole_i_c_.Resize(ncell_, kUndefined);
    b_peephole_f_c_.Resize(ncell_, kUndefined);
    b_peephole_o_c_.Resize(ncell_, kUndefined);

    InitVecParam(f_peephole_i_c_, param_scale);
    InitVecParam(f_peephole_f_c_, param_scale);
    InitVecParam(f_peephole_o_c_, param_scale);

    InitVecParam(b_peephole_i_c_, param_scale);
    InitVecParam(b_peephole_f_c_, param_scale);
    InitVecParam(b_peephole_o_c_, param_scale);

    // init delta buffers
    // forward direction
    f_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    f_w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);
    f_bias_corr_.Resize(4*ncell_, kSetZero);

    // backward direction
    b_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    b_w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);
    b_bias_corr_.Resize(4*ncell_, kSetZero);

    // peep hole connect
    // forward direction
    f_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_o_c_corr_.Resize(ncell_, kSetZero);
    // backward direction
    b_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_o_c_corr_.Resize(ncell_, kSetZero);

    // forward direction
    f_w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero);
    // backward direction
    b_w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero);

    KALDI_ASSERT(clip_gradient_ >= 0.0);
  }


  void ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<CellDim>");
    ReadBasicType(is, binary, &ncell_);
    ExpectToken(is, binary, "<ClipGradient>");
    ReadBasicType(is, binary, &clip_gradient_);
    //ExpectToken(is, binary, "<DropoutRate>");
    //ReadBasicType(is, binary, &dropout_rate_);

    // reading parameters corresponding to forward direction
    f_w_gifo_x_.Read(is, binary);
    f_w_gifo_r_.Read(is, binary);
    f_bias_.Read(is, binary);

    f_peephole_i_c_.Read(is, binary);
    f_peephole_f_c_.Read(is, binary);
    f_peephole_o_c_.Read(is, binary);

    f_w_r_m_.Read(is, binary);

    // init delta buffers
    f_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    f_w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);
    f_bias_corr_.Resize(4*ncell_, kSetZero);

    f_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_o_c_corr_.Resize(ncell_, kSetZero);

    f_w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero);


    // reading parameters corresponding to backward direction
    b_w_gifo_x_.Read(is, binary);
    b_w_gifo_r_.Read(is, binary);
    b_bias_.Read(is, binary);

    b_peephole_i_c_.Read(is, binary);
    b_peephole_f_c_.Read(is, binary);
    b_peephole_o_c_.Read(is, binary);

    b_w_r_m_.Read(is, binary);

    // init delta buffers
    b_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    b_w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);
    b_bias_corr_.Resize(4*ncell_, kSetZero);

    b_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_o_c_corr_.Resize(ncell_, kSetZero);

    b_w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero);
  }


  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<CellDim>");
    WriteBasicType(os, binary, ncell_);
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);
    // WriteToken(os, binary, "<DropoutRate>");
    // WriteBasicType(os, binary, dropout_rate_);

    // writing parameters corresponding to forward direction
    f_w_gifo_x_.Write(os, binary);
    f_w_gifo_r_.Write(os, binary);
    f_bias_.Write(os, binary);

    f_peephole_i_c_.Write(os, binary);
    f_peephole_f_c_.Write(os, binary);
    f_peephole_o_c_.Write(os, binary);

    f_w_r_m_.Write(os, binary);

    // writing parameters corresponding to backward direction
    b_w_gifo_x_.Write(os, binary);
    b_w_gifo_r_.Write(os, binary);
    b_bias_.Write(os, binary);

    b_peephole_i_c_.Write(os, binary);
    b_peephole_f_c_.Write(os, binary);
    b_peephole_o_c_.Write(os, binary);

    b_w_r_m_.Write(os, binary);
  }


  int32 NumParams() const {
    return 2*( f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols() +
         f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols() +
         f_bias_.Dim() +
         f_peephole_i_c_.Dim() +
         f_peephole_f_c_.Dim() +
         f_peephole_o_c_.Dim() +
         f_w_r_m_.NumRows() * f_w_r_m_.NumCols() );
  }


  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 offset, len;

    // Copying parameters corresponding to forward direction
    offset = 0;  len = f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(f_w_gifo_x_);

    offset += len; len =f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(f_w_gifo_r_);

    offset += len; len = f_bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_bias_);

    offset += len; len = f_peephole_i_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_peephole_i_c_);

    offset += len; len = f_peephole_f_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_peephole_f_c_);

    offset += len; len = f_peephole_o_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_peephole_o_c_);

    offset += len; len = f_w_r_m_.NumRows() * f_w_r_m_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(f_w_r_m_);

    // Copying parameters corresponding to backward direction
    offset += len; len = b_w_gifo_x_.NumRows() * b_w_gifo_x_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(b_w_gifo_x_);

    offset += len; len = b_w_gifo_r_.NumRows() * b_w_gifo_r_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(b_w_gifo_r_);

    offset += len; len = b_bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_bias_);

    offset += len; len = b_peephole_i_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_peephole_i_c_);

    offset += len; len = b_peephole_f_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_peephole_f_c_);

    offset += len; len = b_peephole_o_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_peephole_o_c_);

    offset += len; len = b_w_r_m_.NumRows() * b_w_r_m_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(b_w_r_m_);

    return;
  }


  std::string Info() const {
    return std::string("  ")  +
      "\n  Forward Direction weights:" +
      "\n  f_w_gifo_x_  "     + MomentStatistics(f_w_gifo_x_) +
      "\n  f_w_gifo_r_  "     + MomentStatistics(f_w_gifo_r_) +
      "\n  f_bias_  "         + MomentStatistics(f_bias_) +
      "\n  f_peephole_i_c_  " + MomentStatistics(f_peephole_i_c_) +
      "\n  f_peephole_f_c_  " + MomentStatistics(f_peephole_f_c_) +
      "\n  f_peephole_o_c_  " + MomentStatistics(f_peephole_o_c_) +
      "\n  f_w_r_m_  "        + MomentStatistics(f_w_r_m_) +
      "\n  Backward Direction weights:" +
      "\n  b_w_gifo_x_  "     + MomentStatistics(b_w_gifo_x_) +
      "\n  b_w_gifo_r_  "     + MomentStatistics(b_w_gifo_r_) +
      "\n  b_bias_  "         + MomentStatistics(b_bias_) +
      "\n  b_peephole_i_c_  " + MomentStatistics(b_peephole_i_c_) +
      "\n  b_peephole_f_c_  " + MomentStatistics(b_peephole_f_c_) +
      "\n  b_peephole_o_c_  " + MomentStatistics(b_peephole_o_c_) +
      "\n  b_w_r_m_  "        + MomentStatistics(b_w_r_m_);
  }


  std::string InfoGradient() const {
    // disassembling forward-pass forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YR(f_propagate_buf_.ColRange(7*ncell_, nrecur_));

    // disassembling forward-pass back-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> F_DG(f_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DI(f_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DF(f_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DO(f_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DC(f_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DH(f_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DM(f_backpropagate_buf_.ColRange(6*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DR(f_backpropagate_buf_.ColRange(7*ncell_, nrecur_));

    // disassembling backward-pass forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YR(b_propagate_buf_.ColRange(7*ncell_, nrecur_));

    // disassembling backward-pass back-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> B_DG(b_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DI(b_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DF(b_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DO(b_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DC(b_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DH(b_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DM(b_backpropagate_buf_.ColRange(6*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DR(b_backpropagate_buf_.ColRange(7*ncell_, nrecur_));

    return std::string("  ") +
      "\n The Gradients:" +
      "\n  Forward Direction:" +
      "\n  f_w_gifo_x_corr_  "     + MomentStatistics(f_w_gifo_x_corr_) +
      "\n  f_w_gifo_r_corr_  "     + MomentStatistics(f_w_gifo_r_corr_) +
      "\n  f_bias_corr_  "         + MomentStatistics(f_bias_corr_) +
      "\n  f_peephole_i_c_corr_  " + MomentStatistics(f_peephole_i_c_corr_) +
      "\n  f_peephole_f_c_corr_  " + MomentStatistics(f_peephole_f_c_corr_) +
      "\n  f_peephole_o_c_corr_  " + MomentStatistics(f_peephole_o_c_corr_) +
      "\n  f_w_r_m_corr_  "        + MomentStatistics(f_w_r_m_corr_) +
      "\n  Backward Direction:" +
      "\n  b_w_gifo_x_corr_  "   + MomentStatistics(b_w_gifo_x_corr_) +
      "\n  b_w_gifo_r_corr_  "   + MomentStatistics(b_w_gifo_r_corr_) +
      "\n  b_bias_corr_  "     + MomentStatistics(b_bias_corr_) +
      "\n  b_peephole_i_c_corr_  " + MomentStatistics(b_peephole_i_c_corr_) +
      "\n  b_peephole_f_c_corr_  " + MomentStatistics(b_peephole_f_c_corr_) +
      "\n  b_peephole_o_c_corr_  " + MomentStatistics(b_peephole_o_c_corr_) +
      "\n  b_w_r_m_corr_  "    + MomentStatistics(b_w_r_m_corr_) +
      "\n The Activations:" +
      "\n  Forward Direction:" +
      "\n  F_YG  " + MomentStatistics(F_YG) +
      "\n  F_YI  " + MomentStatistics(F_YI) +
      "\n  F_YF  " + MomentStatistics(F_YF) +
      "\n  F_YC  " + MomentStatistics(F_YC) +
      "\n  F_YH  " + MomentStatistics(F_YH) +
      "\n  F_YO  " + MomentStatistics(F_YO) +
      "\n  F_YM  " + MomentStatistics(F_YM) +
      "\n  F_YR  " + MomentStatistics(F_YR) +
      "\n  Backward Direction:" +
      "\n  B_YG  " + MomentStatistics(B_YG) +
      "\n  B_YI  " + MomentStatistics(B_YI) +
      "\n  B_YF  " + MomentStatistics(B_YF) +
      "\n  B_YC  " + MomentStatistics(B_YC) +
      "\n  B_YH  " + MomentStatistics(B_YH) +
      "\n  B_YO  " + MomentStatistics(B_YO) +
      "\n  B_YM  " + MomentStatistics(B_YM) +
      "\n  B_YR  " + MomentStatistics(B_YR) +
      "\n The Derivatives:" +
      "\n  Forward Direction:" +
      "\n  F_DG  " + MomentStatistics(F_DG) +
      "\n  F_DI  " + MomentStatistics(F_DI) +
      "\n  F_DF  " + MomentStatistics(F_DF) +
      "\n  F_DC  " + MomentStatistics(F_DC) +
      "\n  F_DH  " + MomentStatistics(F_DH) +
      "\n  F_DO  " + MomentStatistics(F_DO) +
      "\n  F_DM  " + MomentStatistics(F_DM) +
      "\n  F_DR  " + MomentStatistics(F_DR) +
      "\n  Backward Direction:" +
      "\n  B_DG  " + MomentStatistics(B_DG) +
      "\n  B_DI  " + MomentStatistics(B_DI) +
      "\n  B_DF  " + MomentStatistics(B_DF) +
      "\n  B_DC  " + MomentStatistics(B_DC) +
      "\n  B_DH  " + MomentStatistics(B_DH) +
      "\n  B_DO  " + MomentStatistics(B_DO) +
      "\n  B_DM  " + MomentStatistics(B_DM) +
      "\n  B_DR  " + MomentStatistics(B_DR);
  }


  void ResetLstmStreams(const std::vector<int32> &stream_reset_flag) {
    // allocate f_prev_nnet_state_, b_prev_nnet_state_ if not done yet,
    if (nstream_ == 0) {
      // Karel: we just got number of streams! (before the 1st batch comes)
      nstream_ = stream_reset_flag.size();
      // forward direction
      f_prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);
      // backward direction
      b_prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);
      KALDI_LOG << "Running training with " << nstream_ << " streams.";
    }
    // reset flag: 1 - reset stream network state
    KALDI_ASSERT(f_prev_nnet_state_.NumRows() == stream_reset_flag.size());
    KALDI_ASSERT(b_prev_nnet_state_.NumRows() == stream_reset_flag.size());
    for (int s = 0; s < stream_reset_flag.size(); s++) {
      if (stream_reset_flag[s] == 1) {
        // forward direction
        f_prev_nnet_state_.Row(s).SetZero();
        // backward direction
        b_prev_nnet_state_.Row(s).SetZero();
      }
    }
  }


  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    int DEBUG = 0;

    static bool do_stream_reset = false;
    if (nstream_ == 0) {
      do_stream_reset = true;
      nstream_ = 1; // Karel: we are in nnet-forward, so we will use 1 stream,
      // forward direction
      f_prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);
      // backward direction
      b_prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);
      KALDI_LOG << "Running nnet-forward with per-utterance BLSTM-state reset";
    }
    if (do_stream_reset) {
      // resetting the forward and backward streams
      f_prev_nnet_state_.SetZero();
      b_prev_nnet_state_.SetZero();
    }
    KALDI_ASSERT(nstream_ > 0);

    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    // 0:forward pass history, [1, T]:current sequence, T+1:dummy
    // forward direction
    f_propagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);
    f_propagate_buf_.RowRange(0*S,S).CopyFromMat(f_prev_nnet_state_);

    // backward direction
    b_propagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);
    // for the backward direction, we initialize it at (T+1) frame
    b_propagate_buf_.RowRange((T+1)*S,S).CopyFromMat(b_prev_nnet_state_);

    // disassembling forward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YR(f_propagate_buf_.ColRange(7*ncell_, nrecur_));

    CuSubMatrix<BaseFloat> F_YGIFO(f_propagate_buf_.ColRange(0, 4*ncell_));

    // disassembling backward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YR(b_propagate_buf_.ColRange(7*ncell_, nrecur_));

    CuSubMatrix<BaseFloat> B_YGIFO(b_propagate_buf_.ColRange(0, 4*ncell_));

    // forward direction
    // x -> g, i, f, o, not recurrent, do it all in once
    F_YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, f_w_gifo_x_, kTrans, 0.0);

    // bias -> g, i, f, o
    F_YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, f_bias_);

    for (int t = 1; t <= T; t++) {
      // multistream buffers for current time-step
      CuSubMatrix<BaseFloat> y_g(F_YG.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_i(F_YI.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_f(F_YF.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_o(F_YO.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_c(F_YC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_h(F_YH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_m(F_YM.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_r(F_YR.RowRange(t*S,S));

      CuSubMatrix<BaseFloat> y_gifo(F_YGIFO.RowRange(t*S,S));

      // r(t-1) -> g, i, f, o
      y_gifo.AddMatMat(1.0, F_YR.RowRange((t-1)*S,S), kNoTrans, f_w_gifo_r_, kTrans,  1.0);

      // c(t-1) -> i(t) via peephole
      y_i.AddMatDiagVec(1.0, F_YC.RowRange((t-1)*S,S), kNoTrans, f_peephole_i_c_, 1.0);

      // c(t-1) -> f(t) via peephole
      y_f.AddMatDiagVec(1.0, F_YC.RowRange((t-1)*S,S), kNoTrans, f_peephole_f_c_, 1.0);

      // i, f sigmoid squashing
      y_i.Sigmoid(y_i);
      y_f.Sigmoid(y_f);

      // g tanh squashing
      y_g.Tanh(y_g);

      // g -> c
      y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);

      // c(t-1) -> c(t) via forget-gate
      y_c.AddMatMatElements(1.0, F_YC.RowRange((t-1)*S,S), y_f, 1.0);

      y_c.ApplyFloor(-50);   // optional clipping of cell activation
      y_c.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR

      // h tanh squashing
      y_h.Tanh(y_c);

      // c(t) -> o(t) via peephole (non-recurrent) & o squashing
      y_o.AddMatDiagVec(1.0, y_c, kNoTrans, f_peephole_o_c_, 1.0);

      // o sigmoid squashing
      y_o.Sigmoid(y_o);

      // h -> m via output gate
      y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);

      // m -> r
      y_r.AddMatMat(1.0, y_m, kNoTrans, f_w_r_m_, kTrans, 0.0);

      if (DEBUG) {
        std::cerr << "forward direction forward-pass frame " << t << "\n";
        std::cerr << "activation of g: " << y_g;
        std::cerr << "activation of i: " << y_i;
        std::cerr << "activation of f: " << y_f;
        std::cerr << "activation of o: " << y_o;
        std::cerr << "activation of c: " << y_c;
        std::cerr << "activation of h: " << y_h;
        std::cerr << "activation of m: " << y_m;
        std::cerr << "activation of r: " << y_r;
      }
    }

    // backward direction
    B_YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, b_w_gifo_x_, kTrans, 0.0);
    //// LSTM forward dropout
    //// Google paper 2014: Recurrent Neural Network Regularization
    //// by Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals
    //if (dropout_rate_ != 0.0) {
    //  dropout_mask_.Resize(in.NumRows(), 4*ncell_, kUndefined);
    //  dropout_mask_.SetRandUniform();   // [0,1]
    //  dropout_mask_.Add(-dropout_rate_);  // [-dropout_rate, 1-dropout_rate_],
    //  dropout_mask_.ApplyHeaviside();   // -tive -> 0.0, +tive -> 1.0
    //  YGIFO.RowRange(1*S,T*S).MulElements(dropout_mask_);
    //}

    // bias -> g, i, f, o
    B_YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, b_bias_);

    // backward direction, from T to 1, t--
    for (int t = T; t >= 1; t--) {
      // multistream buffers for current time-step
      CuSubMatrix<BaseFloat> y_g(B_YG.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_i(B_YI.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_f(B_YF.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_o(B_YO.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_c(B_YC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_h(B_YH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_m(B_YM.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_r(B_YR.RowRange(t*S,S));

      CuSubMatrix<BaseFloat> y_gifo(B_YGIFO.RowRange(t*S,S));

      // r(t+1) -> g, i, f, o
      y_gifo.AddMatMat(1.0, B_YR.RowRange((t+1)*S,S), kNoTrans, b_w_gifo_r_, kTrans,  1.0);

      // c(t+1) -> i(t) via peephole
      y_i.AddMatDiagVec(1.0, B_YC.RowRange((t+1)*S,S), kNoTrans, b_peephole_i_c_, 1.0);

      // c(t+1) -> f(t) via peephole
      y_f.AddMatDiagVec(1.0, B_YC.RowRange((t+1)*S,S), kNoTrans, b_peephole_f_c_, 1.0);

      // i, f sigmoid squashing
      y_i.Sigmoid(y_i);
      y_f.Sigmoid(y_f);

      // g tanh squashing
      y_g.Tanh(y_g);

      // g -> c
      y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);

      // c(t+1) -> c(t) via forget-gate
      y_c.AddMatMatElements(1.0, B_YC.RowRange((t+1)*S,S), y_f, 1.0);

      y_c.ApplyFloor(-50);   // optional clipping of cell activation
      y_c.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR

      // h tanh squashing
      y_h.Tanh(y_c);

      // c(t) -> o(t) via peephole (non-recurrent) & o squashing
      y_o.AddMatDiagVec(1.0, y_c, kNoTrans, b_peephole_o_c_, 1.0);

      // o sigmoid squashing
      y_o.Sigmoid(y_o);

      // h -> m via output gate
      y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);

      // m -> r
      y_r.AddMatMat(1.0, y_m, kNoTrans, b_w_r_m_, kTrans, 0.0);

      if (DEBUG) {
        std::cerr << "backward direction forward-pass frame " << t << "\n";
        std::cerr << "activation of g: " << y_g;
        std::cerr << "activation of i: " << y_i;
        std::cerr << "activation of f: " << y_f;
        std::cerr << "activation of o: " << y_o;
        std::cerr << "activation of c: " << y_c;
        std::cerr << "activation of h: " << y_h;
        std::cerr << "activation of m: " << y_m;
        std::cerr << "activation of r: " << y_r;
      }
    }

    // According to definition of BLSTM, for output YR of BLSTM, YR should be F_YR + B_YR
    CuSubMatrix<BaseFloat> YR(F_YR.RowRange(1*S,T*S));
    YR.AddMat(1.0,B_YR.RowRange(1*S,T*S));

    // recurrent projection layer is also feed-forward as BLSTM output
    out->CopyFromMat(YR);

    // now the last frame state becomes previous network state for next batch
    f_prev_nnet_state_.CopyFromMat(f_propagate_buf_.RowRange(T*S,S));

    // now the last frame (,that is the first frame) becomes previous netwok state for next batch
    b_prev_nnet_state_.CopyFromMat(b_propagate_buf_.RowRange(1*S,S));
  }


  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
              const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

    int DEBUG = 0;

    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;
    // disassembling forward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YR(f_propagate_buf_.ColRange(7*ncell_, nrecur_));

    // 0:dummy, [1,T] frames, T+1 backward pass history
    f_backpropagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);

    // disassembling forward-pass back-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> F_DG(f_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DI(f_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DF(f_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DO(f_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DC(f_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DH(f_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DM(f_backpropagate_buf_.ColRange(6*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DR(f_backpropagate_buf_.ColRange(7*ncell_, nrecur_));

    CuSubMatrix<BaseFloat> F_DGIFO(f_backpropagate_buf_.ColRange(0, 4*ncell_));

    // projection layer to BLSTM output is not recurrent, so backprop it all in once
    F_DR.RowRange(1*S,T*S).CopyFromMat(out_diff);

    for (int t = T; t >= 1; t--) {
      CuSubMatrix<BaseFloat> y_g(F_YG.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_i(F_YI.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_f(F_YF.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_o(F_YO.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_c(F_YC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_h(F_YH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_m(F_YM.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_r(F_YR.RowRange(t*S,S));

      CuSubMatrix<BaseFloat> d_g(F_DG.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_i(F_DI.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_f(F_DF.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_o(F_DO.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_c(F_DC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_h(F_DH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_m(F_DM.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_r(F_DR.RowRange(t*S,S));

      // r
      //   Version 1 (precise gradients):
      //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
      d_r.AddMatMat(1.0, F_DGIFO.RowRange((t+1)*S,S), kNoTrans, f_w_gifo_r_, kNoTrans, 1.0);

      /*
      //   Version 2 (Alex Graves' PhD dissertation):
      //   only backprop g(t+1) to r(t)
      CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
      d_r.AddMatMat(1.0, DG.RowRange((t+1)*S,S), kNoTrans, w_g_r_, kNoTrans, 1.0);
      */

      /*
      //   Version 3 (Felix Gers' PhD dissertation):
      //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
      //   CEC(with forget connection) is the only "error-bridge" through time
      ;
      */

      // r -> m
      d_m.AddMatMat(1.0, d_r, kNoTrans, f_w_r_m_, kNoTrans, 0.0);

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
      d_c.AddMatMatElements(1.0, F_DC.RowRange((t+1)*S,S), F_YF.RowRange((t+1)*S,S), 1.0);
      d_c.AddMatDiagVec(1.0, F_DI.RowRange((t+1)*S,S), kNoTrans, f_peephole_i_c_, 1.0);
      d_c.AddMatDiagVec(1.0, F_DF.RowRange((t+1)*S,S), kNoTrans, f_peephole_f_c_, 1.0);
      d_c.AddMatDiagVec(1.0, d_o           , kNoTrans, f_peephole_o_c_, 1.0);

      // f
      d_f.AddMatMatElements(1.0, d_c, F_YC.RowRange((t-1)*S,S), 0.0);
      d_f.DiffSigmoid(y_f, d_f);

      // i
      d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
      d_i.DiffSigmoid(y_i, d_i);

      // c -> g via input gate
      d_g.AddMatMatElements(1.0, d_c, y_i, 0.0);
      d_g.DiffTanh(y_g, d_g);

      // debug info
      if (DEBUG) {
        std::cerr << "backward-pass frame " << t << "\n";
        std::cerr << "derivative wrt input r " << d_r;
        std::cerr << "derivative wrt input m " << d_m;
        std::cerr << "derivative wrt input h " << d_h;
        std::cerr << "derivative wrt input o " << d_o;
        std::cerr << "derivative wrt input c " << d_c;
        std::cerr << "derivative wrt input f " << d_f;
        std::cerr << "derivative wrt input i " << d_i;
        std::cerr << "derivative wrt input g " << d_g;
      }
    }

    // disassembling backward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YR(b_propagate_buf_.ColRange(7*ncell_, nrecur_));

    // 0:dummy, [1,T] frames, T+1 backward pass history
    b_backpropagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);

    // disassembling backward-pass back-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> B_DG(b_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DI(b_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DF(b_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DO(b_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DC(b_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DH(b_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DM(b_backpropagate_buf_.ColRange(6*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DR(b_backpropagate_buf_.ColRange(7*ncell_, nrecur_));

    CuSubMatrix<BaseFloat> B_DGIFO(b_backpropagate_buf_.ColRange(0, 4*ncell_));

    // projection layer to BLSTM output is not recurrent, so backprop it all in once
    B_DR.RowRange(1*S,T*S).CopyFromMat(out_diff);

    for (int t = 1; t <= T; t++) {
      CuSubMatrix<BaseFloat> y_g(B_YG.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_i(B_YI.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_f(B_YF.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_o(B_YO.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_c(B_YC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_h(B_YH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_m(B_YM.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> y_r(B_YR.RowRange(t*S,S));

      CuSubMatrix<BaseFloat> d_g(B_DG.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_i(B_DI.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_f(B_DF.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_o(B_DO.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_c(B_DC.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_h(B_DH.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_m(B_DM.RowRange(t*S,S));
      CuSubMatrix<BaseFloat> d_r(B_DR.RowRange(t*S,S));

      // r
      //   Version 1 (precise gradients):
      //   backprop error from g(t-1), i(t-1), f(t-1), o(t-1) to r(t)
      d_r.AddMatMat(1.0, B_DGIFO.RowRange((t-1)*S,S), kNoTrans, b_w_gifo_r_, kNoTrans, 1.0);

      /*
      //   Version 2 (Alex Graves' PhD dissertation):
      //   only backprop g(t+1) to r(t)
      CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
      d_r.AddMatMat(1.0, DG.RowRange((t+1)*S,S), kNoTrans, w_g_r_, kNoTrans, 1.0);
      */

      /*
      //   Version 3 (Felix Gers' PhD dissertation):
      //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
      //   CEC(with forget connection) is the only "error-bridge" through time
      */

      // r -> m
      d_m.AddMatMat(1.0, d_r, kNoTrans, b_w_r_m_, kNoTrans, 0.0);

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
      d_c.AddMatMatElements(1.0, B_DC.RowRange((t-1)*S,S), B_YF.RowRange((t-1)*S,S), 1.0);
      d_c.AddMatDiagVec(1.0, B_DI.RowRange((t-1)*S,S), kNoTrans, b_peephole_i_c_, 1.0);
      d_c.AddMatDiagVec(1.0, B_DF.RowRange((t-1)*S,S), kNoTrans, b_peephole_f_c_, 1.0);
      d_c.AddMatDiagVec(1.0, d_o                     , kNoTrans, b_peephole_o_c_, 1.0);

      // f
      d_f.AddMatMatElements(1.0, d_c, B_YC.RowRange((t-1)*S,S), 0.0);
      d_f.DiffSigmoid(y_f, d_f);

      // i
      d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
      d_i.DiffSigmoid(y_i, d_i);

      // c -> g via input gate
      d_g.AddMatMatElements(1.0, d_c, y_i, 0.0);
      d_g.DiffTanh(y_g, d_g);

      // debug info
      if (DEBUG) {
        std::cerr << "backward-pass frame " << t << "\n";
        std::cerr << "derivative wrt input r " << d_r;
        std::cerr << "derivative wrt input m " << d_m;
        std::cerr << "derivative wrt input h " << d_h;
        std::cerr << "derivative wrt input o " << d_o;
        std::cerr << "derivative wrt input c " << d_c;
        std::cerr << "derivative wrt input f " << d_f;
        std::cerr << "derivative wrt input i " << d_i;
        std::cerr << "derivative wrt input g " << d_g;
      }
    }

    // g,i,f,o -> x, do it all in once
    // forward direction difference
    in_diff->AddMatMat(1.0, F_DGIFO.RowRange(1*S,T*S), kNoTrans, f_w_gifo_x_, kNoTrans, 0.0);
    // backward direction difference
    in_diff->AddMatMat(1.0, B_DGIFO.RowRange(1*S,T*S), kNoTrans, b_w_gifo_x_, kNoTrans, 1.0);


    // backward pass dropout
    //if (dropout_rate_ != 0.0) {
    //  in_diff->MulElements(dropout_mask_);
    //}

    // calculate delta
    const BaseFloat mmt = opts_.momentum;

    // forward direction
    // weight x -> g, i, f, o
    f_w_gifo_x_corr_.AddMatMat(1.0, F_DGIFO.RowRange(1*S,T*S), kTrans, 
                                    in,                        kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    f_w_gifo_r_corr_.AddMatMat(1.0, F_DGIFO.RowRange(1*S,T*S), kTrans, 
                                    F_YR.RowRange(0*S,T*S),    kNoTrans, mmt);
    // bias of g, i, f, o
    f_bias_corr_.AddRowSumMat(1.0, F_DGIFO.RowRange(1*S,T*S), mmt);

    // recurrent peephole c -> i
    f_peephole_i_c_corr_.AddDiagMatMat(1.0, F_DI.RowRange(1*S,T*S), kTrans,
                                            F_YC.RowRange(0*S,T*S), kNoTrans, mmt);
    // recurrent peephole c -> f
    f_peephole_f_c_corr_.AddDiagMatMat(1.0, F_DF.RowRange(1*S,T*S), kTrans,
                                            F_YC.RowRange(0*S,T*S), kNoTrans, mmt);
    // peephole c -> o
    f_peephole_o_c_corr_.AddDiagMatMat(1.0, F_DO.RowRange(1*S,T*S), kTrans,
                                            F_YC.RowRange(1*S,T*S), kNoTrans, mmt);

    f_w_r_m_corr_.AddMatMat(1.0, F_DR.RowRange(1*S,T*S), kTrans,
                                 F_YM.RowRange(1*S,T*S), kNoTrans, mmt);

    // apply the gradient clipping for forwardpass gradients
    if (clip_gradient_ > 0.0) {
      f_w_gifo_x_corr_.ApplyFloor(-clip_gradient_);
      f_w_gifo_x_corr_.ApplyCeiling(clip_gradient_);
      f_w_gifo_r_corr_.ApplyFloor(-clip_gradient_);
      f_w_gifo_r_corr_.ApplyCeiling(clip_gradient_);
      f_bias_corr_.ApplyFloor(-clip_gradient_);
      f_bias_corr_.ApplyCeiling(clip_gradient_);
      f_w_r_m_corr_.ApplyFloor(-clip_gradient_);
      f_w_r_m_corr_.ApplyCeiling(clip_gradient_);
      f_peephole_i_c_corr_.ApplyFloor(-clip_gradient_);
      f_peephole_i_c_corr_.ApplyCeiling(clip_gradient_);
      f_peephole_f_c_corr_.ApplyFloor(-clip_gradient_);
      f_peephole_f_c_corr_.ApplyCeiling(clip_gradient_);
      f_peephole_o_c_corr_.ApplyFloor(-clip_gradient_);
      f_peephole_o_c_corr_.ApplyCeiling(clip_gradient_);
    }

    // backward direction backpropagate
    // weight x -> g, i, f, o
    b_w_gifo_x_corr_.AddMatMat(1.0, B_DGIFO.RowRange(1*S,T*S), kTrans, in, kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    b_w_gifo_r_corr_.AddMatMat(1.0, B_DGIFO.RowRange(1*S,T*S), kTrans,
                                    B_YR.RowRange(0*S,T*S)   , kNoTrans, mmt);
    // bias of g, i, f, o
    b_bias_corr_.AddRowSumMat(1.0, B_DGIFO.RowRange(1*S,T*S), mmt);

    // recurrent peephole c -> i, c(t+1) --> i ##commented by chongjia
    b_peephole_i_c_corr_.AddDiagMatMat(1.0, B_DI.RowRange(1*S,T*S), kTrans,
                                            B_YC.RowRange(2*S,T*S), kNoTrans, mmt);
    // recurrent peephole c -> f, c(t+1) --> f ###commented by chongjia
    b_peephole_f_c_corr_.AddDiagMatMat(1.0, B_DF.RowRange(1*S,T*S), kTrans,
                                            B_YC.RowRange(2*S,T*S), kNoTrans, mmt);
    // peephole c -> o
    b_peephole_o_c_corr_.AddDiagMatMat(1.0, B_DO.RowRange(1*S,T*S), kTrans,
                                            B_YC.RowRange(1*S,T*S), kNoTrans, mmt);

    b_w_r_m_corr_.AddMatMat(1.0, B_DR.RowRange(1*S,T*S), kTrans,
                                 B_YM.RowRange(1*S,T*S), kNoTrans, mmt);

    // apply the gradient clipping for backwardpass gradients
    if (clip_gradient_ > 0.0) {
      b_w_gifo_x_corr_.ApplyFloor(-clip_gradient_);
      b_w_gifo_x_corr_.ApplyCeiling(clip_gradient_);
      b_w_gifo_r_corr_.ApplyFloor(-clip_gradient_);
      b_w_gifo_r_corr_.ApplyCeiling(clip_gradient_);
      b_bias_corr_.ApplyFloor(-clip_gradient_);
      b_bias_corr_.ApplyCeiling(clip_gradient_);
      b_w_r_m_corr_.ApplyFloor(-clip_gradient_);
      b_w_r_m_corr_.ApplyCeiling(clip_gradient_);
      b_peephole_i_c_corr_.ApplyFloor(-clip_gradient_);
      b_peephole_i_c_corr_.ApplyCeiling(clip_gradient_);
      b_peephole_f_c_corr_.ApplyFloor(-clip_gradient_);
      b_peephole_f_c_corr_.ApplyCeiling(clip_gradient_);
      b_peephole_o_c_corr_.ApplyFloor(-clip_gradient_);
      b_peephole_o_c_corr_.ApplyCeiling(clip_gradient_);
    }

    // forward direction
    if (DEBUG) {
      std::cerr << "gradients(with optional momentum): \n";
      std::cerr << "w_gifo_x_corr_ " << f_w_gifo_x_corr_;
      std::cerr << "w_gifo_r_corr_ " << f_w_gifo_r_corr_;
      std::cerr << "bias_corr_ "     << f_bias_corr_;
      std::cerr << "w_r_m_corr_ "    << f_w_r_m_corr_;
      std::cerr << "peephole_i_c_corr_ " << f_peephole_i_c_corr_;
      std::cerr << "peephole_f_c_corr_ " << f_peephole_f_c_corr_;
      std::cerr << "peephole_o_c_corr_ " << f_peephole_o_c_corr_;
    }
    // backward direction
    if (DEBUG) {
      std::cerr << "gradients(with optional momentum): \n";
      std::cerr << "w_gifo_x_corr_ " << b_w_gifo_x_corr_;
      std::cerr << "w_gifo_r_corr_ " << b_w_gifo_r_corr_;
      std::cerr << "bias_corr_ "     << b_bias_corr_;
      std::cerr << "w_r_m_corr_ "    << b_w_r_m_corr_;
      std::cerr << "peephole_i_c_corr_ " << b_peephole_i_c_corr_;
      std::cerr << "peephole_f_c_corr_ " << b_peephole_f_c_corr_;
      std::cerr << "peephole_o_c_corr_ " << b_peephole_o_c_corr_;
    }
  }


  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr  = opts_.learn_rate;
    // forward direction update
    f_w_gifo_x_.AddMat(-lr, f_w_gifo_x_corr_);
    f_w_gifo_r_.AddMat(-lr, f_w_gifo_r_corr_);
    f_bias_.AddVec(-lr, f_bias_corr_, 1.0);

    f_peephole_i_c_.AddVec(-lr, f_peephole_i_c_corr_, 1.0);
    f_peephole_f_c_.AddVec(-lr, f_peephole_f_c_corr_, 1.0);
    f_peephole_o_c_.AddVec(-lr, f_peephole_o_c_corr_, 1.0);

    f_w_r_m_.AddMat(-lr, f_w_r_m_corr_);

    // backward direction update
    b_w_gifo_x_.AddMat(-lr, b_w_gifo_x_corr_);
    b_w_gifo_r_.AddMat(-lr, b_w_gifo_r_corr_);
    b_bias_.AddVec(-lr, b_bias_corr_, 1.0);

    b_peephole_i_c_.AddVec(-lr, b_peephole_i_c_corr_, 1.0);
    b_peephole_f_c_.AddVec(-lr, b_peephole_f_c_corr_, 1.0);
    b_peephole_o_c_.AddVec(-lr, b_peephole_o_c_corr_, 1.0);

    b_w_r_m_.AddMat(-lr, b_w_r_m_corr_);

    /* For L2 regularization see "vanishing & exploding difficulties" in nnet-lstm-projected-streams.h */
  }

 private:
  // dims
  int32 ncell_;   ///< the number of cell blocks
  int32 nrecur_;  ///< recurrent projection layer dim
  int32 nstream_;

  CuMatrix<BaseFloat> f_prev_nnet_state_;
  CuMatrix<BaseFloat> b_prev_nnet_state_;

  // gradient-clipping value,
  BaseFloat clip_gradient_;

  // non-recurrent dropout
  //BaseFloat dropout_rate_;
  //CuMatrix<BaseFloat> dropout_mask_;

  // feed-forward connections: from x to [g, i, f, o]
  // forward direction
  CuMatrix<BaseFloat> f_w_gifo_x_;
  CuMatrix<BaseFloat> f_w_gifo_x_corr_;
  // backward direction
  CuMatrix<BaseFloat> b_w_gifo_x_;
  CuMatrix<BaseFloat> b_w_gifo_x_corr_;

  // recurrent projection connections: from r to [g, i, f, o]
  // forward direction
  CuMatrix<BaseFloat> f_w_gifo_r_;
  CuMatrix<BaseFloat> f_w_gifo_r_corr_;
  // backward direction
  CuMatrix<BaseFloat> b_w_gifo_r_;
  CuMatrix<BaseFloat> b_w_gifo_r_corr_;

  // biases of [g, i, f, o]
  // forward direction
  CuVector<BaseFloat> f_bias_;
  CuVector<BaseFloat> f_bias_corr_;
  // backward direction
  CuVector<BaseFloat> b_bias_;
  CuVector<BaseFloat> b_bias_corr_;

  // peephole from c to i, f, g
  // peephole connections are block-internal, so we use vector form
  // forward direction
  CuVector<BaseFloat> f_peephole_i_c_;
  CuVector<BaseFloat> f_peephole_f_c_;
  CuVector<BaseFloat> f_peephole_o_c_;
  // backward direction
  CuVector<BaseFloat> b_peephole_i_c_;
  CuVector<BaseFloat> b_peephole_f_c_;
  CuVector<BaseFloat> b_peephole_o_c_;

  // forward direction
  CuVector<BaseFloat> f_peephole_i_c_corr_;
  CuVector<BaseFloat> f_peephole_f_c_corr_;
  CuVector<BaseFloat> f_peephole_o_c_corr_;
  // backward direction
  CuVector<BaseFloat> b_peephole_i_c_corr_;
  CuVector<BaseFloat> b_peephole_f_c_corr_;
  CuVector<BaseFloat> b_peephole_o_c_corr_;

  // projection layer r: from m to r
  // forward direction
  CuMatrix<BaseFloat> f_w_r_m_;
  CuMatrix<BaseFloat> f_w_r_m_corr_;
  // backward direction
  CuMatrix<BaseFloat> b_w_r_m_;
  CuMatrix<BaseFloat> b_w_r_m_corr_;

  // propagate buffer: output of [g, i, f, o, c, h, m, r]
  // forward direction
  CuMatrix<BaseFloat> f_propagate_buf_;
  // backward direction
  CuMatrix<BaseFloat> b_propagate_buf_;


  // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
  // forward direction
  CuMatrix<BaseFloat> f_backpropagate_buf_;
  // backward direction
  CuMatrix<BaseFloat> b_backpropagate_buf_;

};
} // namespace nnet1
} // namespace kaldi

#endif
