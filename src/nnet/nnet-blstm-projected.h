// nnet/nnet-blstm-projected-streams.h

// Copyright 2016  Brno University of Techology (author: Karel Vesely)
// Copyright 2015  Chongjia Ni
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


#ifndef KALDI_NNET_NNET_BLSTM_PROJECTED_H_
#define KALDI_NNET_NNET_BLSTM_PROJECTED_H_

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
 * f-*: forward direction
 * b-*: backward direction
 *************************************/

namespace kaldi {
namespace nnet1 {

class BlstmProjected : public MultistreamComponent {
 public:
  BlstmProjected(int32 input_dim, int32 output_dim):
    MultistreamComponent(input_dim, output_dim),
    cell_dim_(0),
    proj_dim_(static_cast<int32>(output_dim/2)),
    cell_clip_(50.0),
    diff_clip_(1.0),
    cell_diff_clip_(0.0),
    grad_clip_(250.0)
  { }

  ~BlstmProjected()
  { }

  Component* Copy() const { return new BlstmProjected(*this); }
  ComponentType GetType() const { return kBlstmProjected; }

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
    // forward direction,
    f_w_gifo_x_.Resize(4*cell_dim_, input_dim_, kUndefined);
    f_w_gifo_r_.Resize(4*cell_dim_, proj_dim_, kUndefined);
    f_bias_.Resize(4*cell_dim_, kUndefined);
    f_peephole_i_c_.Resize(cell_dim_, kUndefined);
    f_peephole_f_c_.Resize(cell_dim_, kUndefined);
    f_peephole_o_c_.Resize(cell_dim_, kUndefined);
    f_w_r_m_.Resize(proj_dim_, cell_dim_, kUndefined);
    //       (mean), (range)
    RandUniform(0.0, 2.0 * param_range, &f_w_gifo_x_);
    RandUniform(0.0, 2.0 * param_range, &f_w_gifo_r_);
    RandUniform(0.0, 2.0 * param_range, &f_bias_);
    RandUniform(0.0, 2.0 * param_range, &f_peephole_i_c_);
    RandUniform(0.0, 2.0 * param_range, &f_peephole_f_c_);
    RandUniform(0.0, 2.0 * param_range, &f_peephole_o_c_);
    RandUniform(0.0, 2.0 * param_range, &f_w_r_m_);

    // Add 1.0 to forget-gate bias
    // [Miao IS16: AN EMPIRICAL EXPLORATION...]
    f_bias_.Range(2*cell_dim_, cell_dim_).Add(1.0);

    // backward direction,
    b_w_gifo_x_.Resize(4*cell_dim_, input_dim_, kUndefined);
    b_w_gifo_r_.Resize(4*cell_dim_, proj_dim_, kUndefined);
    b_bias_.Resize(4*cell_dim_, kUndefined);
    b_peephole_i_c_.Resize(cell_dim_, kUndefined);
    b_peephole_f_c_.Resize(cell_dim_, kUndefined);
    b_peephole_o_c_.Resize(cell_dim_, kUndefined);
    b_w_r_m_.Resize(proj_dim_, cell_dim_, kUndefined);

    RandUniform(0.0, 2.0 * param_range, &b_w_gifo_x_);
    RandUniform(0.0, 2.0 * param_range, &b_w_gifo_r_);
    RandUniform(0.0, 2.0 * param_range, &b_bias_);
    RandUniform(0.0, 2.0 * param_range, &b_peephole_i_c_);
    RandUniform(0.0, 2.0 * param_range, &b_peephole_f_c_);
    RandUniform(0.0, 2.0 * param_range, &b_peephole_o_c_);
    RandUniform(0.0, 2.0 * param_range, &b_w_r_m_);

    // Add 1.0 to forget-gate bias,
    // [Miao IS16: AN EMPIRICAL EXPLORATION...]
    b_bias_.Range(2*cell_dim_, cell_dim_).Add(1.0);

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
    // Read the data (data follow the tokens),

    // reading parameters corresponding to forward direction
    f_w_gifo_x_.Read(is, binary);
    f_w_gifo_r_.Read(is, binary);
    f_bias_.Read(is, binary);

    f_peephole_i_c_.Read(is, binary);
    f_peephole_f_c_.Read(is, binary);
    f_peephole_o_c_.Read(is, binary);

    f_w_r_m_.Read(is, binary);

    // reading parameters corresponding to backward direction
    b_w_gifo_x_.Read(is, binary);
    b_w_gifo_r_.Read(is, binary);
    b_bias_.Read(is, binary);

    b_peephole_i_c_.Read(is, binary);
    b_peephole_f_c_.Read(is, binary);
    b_peephole_o_c_.Read(is, binary);

    b_w_r_m_.Read(is, binary);
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

    if (!binary) os << "\n";
    // writing parameters, forward direction,
    f_w_gifo_x_.Write(os, binary);
    f_w_gifo_r_.Write(os, binary);
    f_bias_.Write(os, binary);

    f_peephole_i_c_.Write(os, binary);
    f_peephole_f_c_.Write(os, binary);
    f_peephole_o_c_.Write(os, binary);

    f_w_r_m_.Write(os, binary);

    if (!binary) os << "\n";
    // writing parameters, backward direction,
    b_w_gifo_x_.Write(os, binary);
    b_w_gifo_r_.Write(os, binary);
    b_bias_.Write(os, binary);

    b_peephole_i_c_.Write(os, binary);
    b_peephole_f_c_.Write(os, binary);
    b_peephole_o_c_.Write(os, binary);

    b_w_r_m_.Write(os, binary);
  }

  int32 NumParams() const {
    return 2 * ( f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols() +
      f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols() +
      f_bias_.Dim() +
      f_peephole_i_c_.Dim() +
      f_peephole_f_c_.Dim() +
      f_peephole_o_c_.Dim() +
      f_w_r_m_.NumRows() * f_w_r_m_.NumCols() );
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 offset, len;

    // Copying parameters corresponding to forward direction
    offset = 0;    len = f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(f_w_gifo_x_corr_);

    offset += len; len = f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(f_w_gifo_r_corr_);

    offset += len; len = f_bias_.Dim();
    gradient->Range(offset, len).CopyFromVec(f_bias_corr_);

    offset += len; len = f_peephole_i_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(f_peephole_i_c_corr_);

    offset += len; len = f_peephole_f_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(f_peephole_f_c_corr_);

    offset += len; len = f_peephole_o_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(f_peephole_o_c_corr_);

    offset += len; len = f_w_r_m_.NumRows() * f_w_r_m_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(f_w_r_m_corr_);

    // Copying parameters corresponding to backward direction
    offset += len; len = b_w_gifo_x_.NumRows() * b_w_gifo_x_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(b_w_gifo_x_corr_);

    offset += len; len = b_w_gifo_r_.NumRows() * b_w_gifo_r_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(b_w_gifo_r_corr_);

    offset += len; len = b_bias_.Dim();
    gradient->Range(offset, len).CopyFromVec(b_bias_corr_);

    offset += len; len = b_peephole_i_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(b_peephole_i_c_corr_);

    offset += len; len = b_peephole_f_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(b_peephole_f_c_corr_);

    offset += len; len = b_peephole_o_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(b_peephole_o_c_corr_);

    offset += len; len = b_w_r_m_.NumRows() * b_w_r_m_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(b_w_r_m_corr_);

    // check the dim,
    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 offset, len;

    // Copying parameters corresponding to forward direction
    offset = 0;    len = f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(f_w_gifo_x_);

    offset += len; len = f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(f_w_gifo_r_);

    offset += len; len = f_bias_.Dim();
    params->Range(offset, len).CopyFromVec(f_bias_);

    offset += len; len = f_peephole_i_c_.Dim();
    params->Range(offset, len).CopyFromVec(f_peephole_i_c_);

    offset += len; len = f_peephole_f_c_.Dim();
    params->Range(offset, len).CopyFromVec(f_peephole_f_c_);

    offset += len; len = f_peephole_o_c_.Dim();
    params->Range(offset, len).CopyFromVec(f_peephole_o_c_);

    offset += len; len = f_w_r_m_.NumRows() * f_w_r_m_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(f_w_r_m_);

    // Copying parameters corresponding to backward direction
    offset += len; len = b_w_gifo_x_.NumRows() * b_w_gifo_x_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(b_w_gifo_x_);

    offset += len; len = b_w_gifo_r_.NumRows() * b_w_gifo_r_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(b_w_gifo_r_);

    offset += len; len = b_bias_.Dim();
    params->Range(offset, len).CopyFromVec(b_bias_);

    offset += len; len = b_peephole_i_c_.Dim();
    params->Range(offset, len).CopyFromVec(b_peephole_i_c_);

    offset += len; len = b_peephole_f_c_.Dim();
    params->Range(offset, len).CopyFromVec(b_peephole_f_c_);

    offset += len; len = b_peephole_o_c_.Dim();
    params->Range(offset, len).CopyFromVec(b_peephole_o_c_);

    offset += len; len = b_w_r_m_.NumRows() * b_w_r_m_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(b_w_r_m_);

    // check the dim,
    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 offset, len;

    // Copying parameters corresponding to forward direction
    offset = 0;    len = f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols();
    f_w_gifo_x_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols();
    f_w_gifo_r_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = f_bias_.Dim();
    f_bias_.CopyFromVec(params.Range(offset, len));

    offset += len; len = f_peephole_i_c_.Dim();
    f_peephole_i_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = f_peephole_f_c_.Dim();
    f_peephole_f_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = f_peephole_o_c_.Dim();
    f_peephole_o_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = f_w_r_m_.NumRows() * f_w_r_m_.NumCols();
    f_w_r_m_.CopyRowsFromVec(params.Range(offset, len));

    // Copying parameters corresponding to backward direction
    offset += len; len = b_w_gifo_x_.NumRows() * b_w_gifo_x_.NumCols();
    b_w_gifo_x_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = b_w_gifo_r_.NumRows() * b_w_gifo_r_.NumCols();
    b_w_gifo_r_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = b_bias_.Dim();
    b_bias_.CopyFromVec(params.Range(offset, len));

    offset += len; len = b_peephole_i_c_.Dim();
    b_peephole_i_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = b_peephole_f_c_.Dim();
    b_peephole_f_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = b_peephole_o_c_.Dim();
    b_peephole_o_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = b_w_r_m_.NumRows() * b_w_r_m_.NumCols();
    b_w_r_m_.CopyRowsFromVec(params.Range(offset, len));

    // check the dim,
    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }


  std::string Info() const {
    return std::string("cell-dim 2x") + ToString(cell_dim_) + " " +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      ", cell_clip_ " + ToString(cell_clip_) +
      ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
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
    // forward-direction activations,
    const CuSubMatrix<BaseFloat> YG_FW(f_propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YI_FW(f_propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YF_FW(f_propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YO_FW(f_propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YC_FW(f_propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YH_FW(f_propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YM_FW(f_propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YR_FW(f_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // forward-direction derivatives,
    const CuSubMatrix<BaseFloat> DG_FW(f_backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DI_FW(f_backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DF_FW(f_backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DO_FW(f_backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DC_FW(f_backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DH_FW(f_backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DM_FW(f_backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DR_FW(f_backpropagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // backward-direction activations,
    const CuSubMatrix<BaseFloat> YG_BW(b_propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YI_BW(b_propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YF_BW(b_propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YO_BW(b_propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YC_BW(b_propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YH_BW(b_propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YM_BW(b_propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> YR_BW(b_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // backward-direction derivatives,
    const CuSubMatrix<BaseFloat> DG_BW(b_backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DI_BW(b_backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DF_BW(b_backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DO_BW(b_backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DC_BW(b_backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DH_BW(b_backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DM_BW(b_backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DR_BW(b_backpropagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    return std::string("") +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      ", cell_clip_ " + ToString(cell_clip_) +
      ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
      "\n  ### Gradients " +
      "\n  f_w_gifo_x_corr_  "     + MomentStatistics(f_w_gifo_x_corr_) +
      "\n  f_w_gifo_r_corr_  "     + MomentStatistics(f_w_gifo_r_corr_) +
      "\n  f_bias_corr_  "         + MomentStatistics(f_bias_corr_) +
      "\n  f_peephole_i_c_corr_  " + MomentStatistics(f_peephole_i_c_corr_) +
      "\n  f_peephole_f_c_corr_  " + MomentStatistics(f_peephole_f_c_corr_) +
      "\n  f_peephole_o_c_corr_  " + MomentStatistics(f_peephole_o_c_corr_) +
      "\n  f_w_r_m_corr_  "        + MomentStatistics(f_w_r_m_corr_) +
      "\n  ---" +
      "\n  b_w_gifo_x_corr_  "     + MomentStatistics(b_w_gifo_x_corr_) +
      "\n  b_w_gifo_r_corr_  "     + MomentStatistics(b_w_gifo_r_corr_) +
      "\n  b_bias_corr_  "         + MomentStatistics(b_bias_corr_) +
      "\n  b_peephole_i_c_corr_  " + MomentStatistics(b_peephole_i_c_corr_) +
      "\n  b_peephole_f_c_corr_  " + MomentStatistics(b_peephole_f_c_corr_) +
      "\n  b_peephole_o_c_corr_  " + MomentStatistics(b_peephole_o_c_corr_) +
      "\n  b_w_r_m_corr_  "        + MomentStatistics(b_w_r_m_corr_) +
      "\n" +
      "\n  ### Activations (mostly after non-linearities)" +
      "\n  YI_FW(0..1)^  " + MomentStatistics(YI_FW) +
      "\n  YF_FW(0..1)^  " + MomentStatistics(YF_FW) +
      "\n  YO_FW(0..1)^  " + MomentStatistics(YO_FW) +
      "\n  YG_FW(-1..1)  " + MomentStatistics(YG_FW) +
      "\n  YC_FW(-R..R)* " + MomentStatistics(YC_FW) +
      "\n  YH_FW(-1..1)  " + MomentStatistics(YH_FW) +
      "\n  YM_FW(-1..1)  " + MomentStatistics(YM_FW) +
      "\n  YR_FW(-R..R)  " + MomentStatistics(YR_FW) +
      "\n  ---" +
      "\n  YI_BW(0..1)^  " + MomentStatistics(YI_BW) +
      "\n  YF_BW(0..1)^  " + MomentStatistics(YF_BW) +
      "\n  YO_BW(0..1)^  " + MomentStatistics(YO_BW) +
      "\n  YG_BW(-1..1)  " + MomentStatistics(YG_BW) +
      "\n  YC_BW(-R..R)* " + MomentStatistics(YC_BW) +
      "\n  YH_BW(-1..1)  " + MomentStatistics(YH_BW) +
      "\n  YM_BW(-1..1)  " + MomentStatistics(YM_BW) +
      "\n  YR_BW(-R..R)  " + MomentStatistics(YR_BW) +
      "\n" +
      "\n  ### Derivatives (w.r.t. inputs of non-linearities)" +
      "\n  DI_FW^ " + MomentStatistics(DI_FW) +
      "\n  DF_FW^ " + MomentStatistics(DF_FW) +
      "\n  DO_FW^ " + MomentStatistics(DO_FW) +
      "\n  DG_FW  " + MomentStatistics(DG_FW) +
      "\n  DC_FW* " + MomentStatistics(DC_FW) +
      "\n  DH_FW  " + MomentStatistics(DH_FW) +
      "\n  DM_FW  " + MomentStatistics(DM_FW) +
      "\n  DR_FW  " + MomentStatistics(DR_FW) +
      "\n  ---" +
      "\n  DI_BW^ " + MomentStatistics(DI_BW) +
      "\n  DF_BW^ " + MomentStatistics(DF_BW) +
      "\n  DO_BW^ " + MomentStatistics(DO_BW) +
      "\n  DG_BW  " + MomentStatistics(DG_BW) +
      "\n  DC_BW* " + MomentStatistics(DC_BW) +
      "\n  DH_BW  " + MomentStatistics(DH_BW) +
      "\n  DM_BW  " + MomentStatistics(DM_BW) +
      "\n  DR_BW  " + MomentStatistics(DR_BW);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {

    KALDI_ASSERT(in.NumRows() % NumStreams() == 0);
    int32 S = NumStreams();
    int32 T = in.NumRows() / NumStreams();

    // buffers,
    f_propagate_buf_.Resize((T+2)*S, 7 * cell_dim_ + proj_dim_, kSetZero);
    b_propagate_buf_.Resize((T+2)*S, 7 * cell_dim_ + proj_dim_, kSetZero);

    // forward-direction activations,
    CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YR(f_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    CuSubMatrix<BaseFloat> F_YGIFO(f_propagate_buf_.ColRange(0, 4*cell_dim_));

    // backward-direction activations,
    CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YR(b_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    CuSubMatrix<BaseFloat> B_YGIFO(b_propagate_buf_.ColRange(0, 4*cell_dim_));

    // FORWARD DIRECTION,
    // x -> g, i, f, o, not recurrent, do it all in once
    F_YGIFO.RowRange(1*S, T*S).AddMatMat(1.0, in, kNoTrans, f_w_gifo_x_, kTrans, 0.0);

    // bias -> g, i, f, o
    F_YGIFO.RowRange(1*S, T*S).AddVecToRows(1.0, f_bias_);

    // BufferPadding [T0]:dummy, [1, T]:current sequence, [T+1]:dummy
    for (int t = 1; t <= T; t++) {
      // multistream buffers for current time-step,
      CuSubMatrix<BaseFloat> y_all(f_propagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_g(F_YG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_i(F_YI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_f(F_YF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_o(F_YO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_c(F_YC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_h(F_YH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_m(F_YM.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_r(F_YR.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_gifo(F_YGIFO.RowRange(t*S, S));

      // r(t-1) -> g, i, f, o
      y_gifo.AddMatMat(1.0, F_YR.RowRange((t-1)*S, S), kNoTrans, f_w_gifo_r_, kTrans, 1.0);

      // c(t-1) -> i(t) via peephole
      y_i.AddMatDiagVec(1.0, F_YC.RowRange((t-1)*S, S), kNoTrans, f_peephole_i_c_, 1.0);

      // c(t-1) -> f(t) via peephole
      y_f.AddMatDiagVec(1.0, F_YC.RowRange((t-1)*S, S), kNoTrans, f_peephole_f_c_, 1.0);

      // i, f sigmoid squashing
      y_i.Sigmoid(y_i);
      y_f.Sigmoid(y_f);

      // g tanh squashing
      y_g.Tanh(y_g);

      // g * i -> c
      y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);
      // c(t-1) * f -> c(t) via forget-gate
      y_c.AddMatMatElements(1.0, F_YC.RowRange((t-1)*S, S), y_f, 1.0);

      if (cell_clip_ > 0.0) {
        y_c.ApplyFloor(-cell_clip_);   // Optional clipping of cell activation,
        y_c.ApplyCeiling(cell_clip_);  // Google paper Interspeech2014: LSTM for LVCSR
      }

      // c(t) -> o(t) via peephole (not recurrent, using c(t))
      y_o.AddMatDiagVec(1.0, y_c, kNoTrans, f_peephole_o_c_, 1.0);

      // o sigmoid squashing,
      y_o.Sigmoid(y_o);

      // c -> h, tanh squashing,
      y_h.Tanh(y_c);

      // h * o -> m via output gate,
      y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);

      // m -> r
      y_r.AddMatMat(1.0, y_m, kNoTrans, f_w_r_m_, kTrans, 0.0);

      // set zeros to padded frames,
      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            y_all.Row(s).SetZero();
          }
        }
      }
    }

    // BACKWARD DIRECTION,
    // x -> g, i, f, o, not recurrent, do it all in once
    B_YGIFO.RowRange(1*S, T*S).AddMatMat(1.0, in, kNoTrans, b_w_gifo_x_, kTrans, 0.0);

    // bias -> g, i, f, o
    B_YGIFO.RowRange(1*S, T*S).AddVecToRows(1.0, b_bias_);

    // BufferPadding [T0]:dummy, [1, T]:current sequence, [T+1]:dummy
    for (int t = T; t >= 1; t--) {
      // multistream buffers for current time-step,
      CuSubMatrix<BaseFloat> y_all(b_propagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_g(B_YG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_i(B_YI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_f(B_YF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_o(B_YO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_c(B_YC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_h(B_YH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_m(B_YM.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_r(B_YR.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_gifo(B_YGIFO.RowRange(t*S, S));

      // r(t+1) -> g, i, f, o
      y_gifo.AddMatMat(1.0, B_YR.RowRange((t+1)*S, S), kNoTrans, b_w_gifo_r_, kTrans, 1.0);

      // c(t+1) -> i(t) via peephole
      y_i.AddMatDiagVec(1.0, B_YC.RowRange((t+1)*S, S), kNoTrans, b_peephole_i_c_, 1.0);

      // c(t+1) -> f(t) via peephole
      y_f.AddMatDiagVec(1.0, B_YC.RowRange((t+1)*S, S), kNoTrans, b_peephole_f_c_, 1.0);

      // i, f sigmoid squashing
      y_i.Sigmoid(y_i);
      y_f.Sigmoid(y_f);

      // g tanh squashing
      y_g.Tanh(y_g);

      // g * i -> c
      y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);
      // c(t+1) * f -> c(t) via forget-gate
      y_c.AddMatMatElements(1.0, B_YC.RowRange((t+1)*S, S), y_f, 1.0);

      if (cell_clip_ > 0.0) {
        y_c.ApplyFloor(-cell_clip_);   // optional clipping of cell activation,
        y_c.ApplyCeiling(cell_clip_);  // google paper Interspeech2014: LSTM for LVCSR
      }

      // c(t) -> o(t) via peephole (not recurrent, using c(t))
      y_o.AddMatDiagVec(1.0, y_c, kNoTrans, b_peephole_o_c_, 1.0);

      // o sigmoid squashing,
      y_o.Sigmoid(y_o);

      // h tanh squashing,
      y_h.Tanh(y_c);

      // h * o -> m via output gate,
      y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);

      // m -> r
      y_r.AddMatMat(1.0, y_m, kNoTrans, b_w_r_m_, kTrans, 0.0);

      // set zeros to padded frames,
      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            y_all.Row(s).SetZero();
          }
        }
      }
    }

    CuMatrix<BaseFloat> YR_FB;
    YR_FB.Resize((T+2)*S, 2 * proj_dim_, kSetZero);
    // forward part
    YR_FB.ColRange(0, proj_dim_).CopyFromMat(f_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    // backward part
    YR_FB.ColRange(proj_dim_, proj_dim_).CopyFromMat(b_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    // recurrent projection layer is also feed-forward as BLSTM output
    out->CopyFromMat(YR_FB.RowRange(1*S, T*S));
  }


  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {

    // the number of sequences to be processed in parallel
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();

    // buffers,
    f_backpropagate_buf_.Resize((T+2)*S, 7 * cell_dim_ + proj_dim_, kSetZero);
    b_backpropagate_buf_.Resize((T+2)*S, 7 * cell_dim_ + proj_dim_, kSetZero);

    // FORWARD DIRECTION,
    // forward-direction activations,
    CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_YR(f_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // forward-direction derivatives,
    CuSubMatrix<BaseFloat> F_DG(f_backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_DI(f_backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_DF(f_backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_DO(f_backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_DC(f_backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_DH(f_backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_DM(f_backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> F_DR(f_backpropagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    CuSubMatrix<BaseFloat> F_DGIFO(f_backpropagate_buf_.ColRange(0, 4*cell_dim_));

    // pre-copy partial derivatives from the BLSTM output,
    F_DR.RowRange(1*S, T*S).CopyFromMat(out_diff.ColRange(0, proj_dim_));

    // BufferPadding [T0]:dummy, [1,T]:current sequence, [T+1]: dummy,
    for (int t = T; t >= 1; t--) {
      CuSubMatrix<BaseFloat> y_g(F_YG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_i(F_YI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_f(F_YF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_o(F_YO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_c(F_YC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_h(F_YH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_m(F_YM.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_r(F_YR.RowRange(t*S, S));

      CuSubMatrix<BaseFloat> d_all(f_backpropagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_g(F_DG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_i(F_DI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_f(F_DF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_o(F_DO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_c(F_DC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_h(F_DH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_m(F_DM.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_r(F_DR.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_gifo(F_DGIFO.RowRange(t*S, S));

      // r
      //   Version 1 (precise gradients):
      //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
      d_r.AddMatMat(1.0, F_DGIFO.RowRange((t+1)*S, S), kNoTrans, f_w_gifo_r_, kNoTrans, 1.0);

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
      ;
      */

      // r -> m
      d_m.AddMatMat(1.0, d_r, kNoTrans, f_w_r_m_, kNoTrans, 0.0);

      // m -> h, via output gate
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
      d_c.AddMatMatElements(1.0, F_DC.RowRange((t+1)*S, S), F_YF.RowRange((t+1)*S, S), 1.0);
      d_c.AddMatDiagVec(1.0, F_DI.RowRange((t+1)*S, S), kNoTrans, f_peephole_i_c_, 1.0);
      d_c.AddMatDiagVec(1.0, F_DF.RowRange((t+1)*S, S), kNoTrans, f_peephole_f_c_, 1.0);
      d_c.AddMatDiagVec(1.0, d_o                      , kNoTrans, f_peephole_o_c_, 1.0);
      // optionally clip the cell_derivative,
      if (cell_diff_clip_ > 0.0) {
        d_c.ApplyFloor(-cell_diff_clip_);
        d_c.ApplyCeiling(cell_diff_clip_);
      }

      // f
      d_f.AddMatMatElements(1.0, d_c, F_YC.RowRange((t-1)*S, S), 0.0);
      d_f.DiffSigmoid(y_f, d_f);

      // i
      d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
      d_i.DiffSigmoid(y_i, d_i);

      // c -> g, via input gate
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

    // BACKWARD DIRECTION,
    // backward-direction activations,
    CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_YR(b_propagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // backward-direction derivatives,
    CuSubMatrix<BaseFloat> B_DG(b_backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_DI(b_backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_DF(b_backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_DO(b_backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_DC(b_backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_DH(b_backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_DM(b_backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> B_DR(b_backpropagate_buf_.ColRange(7*cell_dim_, proj_dim_));
    CuSubMatrix<BaseFloat> B_DGIFO(b_backpropagate_buf_.ColRange(0, 4*cell_dim_));

    // pre-copy partial derivatives from the BLSTM output,
    B_DR.RowRange(1*S, T*S).CopyFromMat(out_diff.ColRange(proj_dim_, proj_dim_));

    // BufferPadding [T0]:dummy, [1,T]:current sequence, [T+1]: dummy,
    for (int t = 1; t <= T; t++) {
      CuSubMatrix<BaseFloat> y_g(B_YG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_i(B_YI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_f(B_YF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_o(B_YO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_c(B_YC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_h(B_YH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_m(B_YM.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> y_r(B_YR.RowRange(t*S, S));

      CuSubMatrix<BaseFloat> d_all(b_backpropagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_g(B_DG.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_i(B_DI.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_f(B_DF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_o(B_DO.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_c(B_DC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_h(B_DH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_m(B_DM.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_r(B_DR.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_gifo(B_DGIFO.RowRange(t*S, S));

      // r
      //   Version 1 (precise gradients):
      //   backprop error from g(t-1), i(t-1), f(t-1), o(t-1) to r(t)
      d_r.AddMatMat(1.0, B_DGIFO.RowRange((t-1)*S, S), kNoTrans, b_w_gifo_r_, kNoTrans, 1.0);

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
      d_c.AddMatMatElements(1.0, B_DC.RowRange((t-1)*S, S), B_YF.RowRange((t-1)*S, S), 1.0);
      d_c.AddMatDiagVec(1.0, B_DI.RowRange((t-1)*S, S), kNoTrans, b_peephole_i_c_, 1.0);
      d_c.AddMatDiagVec(1.0, B_DF.RowRange((t-1)*S, S), kNoTrans, b_peephole_f_c_, 1.0);
      d_c.AddMatDiagVec(1.0, d_o                      , kNoTrans, b_peephole_o_c_, 1.0);
      // optionally clip the cell_derivative,
      if (cell_diff_clip_ > 0.0) {
        d_c.ApplyFloor(-cell_diff_clip_);
        d_c.ApplyCeiling(cell_diff_clip_);
      }

      // f
      d_f.AddMatMatElements(1.0, d_c, B_YC.RowRange((t-1)*S, S), 0.0);
      d_f.DiffSigmoid(y_f, d_f);

      // i
      d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
      d_i.DiffSigmoid(y_i, d_i);

      // c -> g, via input gate,
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
    // forward direction difference
    in_diff->AddMatMat(1.0, F_DGIFO.RowRange(1*S, T*S), kNoTrans, f_w_gifo_x_, kNoTrans, 0.0);
    // backward direction difference
    in_diff->AddMatMat(1.0, B_DGIFO.RowRange(1*S, T*S), kNoTrans, b_w_gifo_x_, kNoTrans, 1.0);

    // lazy initialization of udpate buffers,
    if (f_w_gifo_x_corr_.NumRows() == 0) {
      // init delta buffers,
      // forward direction,
      f_w_gifo_x_corr_.Resize(4*cell_dim_, input_dim_, kSetZero);
      f_w_gifo_r_corr_.Resize(4*cell_dim_, proj_dim_, kSetZero);
      f_bias_corr_.Resize(4*cell_dim_, kSetZero);
      f_peephole_i_c_corr_.Resize(cell_dim_, kSetZero);
      f_peephole_f_c_corr_.Resize(cell_dim_, kSetZero);
      f_peephole_o_c_corr_.Resize(cell_dim_, kSetZero);
      f_w_r_m_corr_.Resize(proj_dim_, cell_dim_, kSetZero);

      // backward direction,
      b_w_gifo_x_corr_.Resize(4*cell_dim_, input_dim_, kSetZero);
      b_w_gifo_r_corr_.Resize(4*cell_dim_, proj_dim_, kSetZero);
      b_bias_corr_.Resize(4*cell_dim_, kSetZero);
      b_peephole_i_c_corr_.Resize(cell_dim_, kSetZero);
      b_peephole_f_c_corr_.Resize(cell_dim_, kSetZero);
      b_peephole_o_c_corr_.Resize(cell_dim_, kSetZero);
      b_w_r_m_corr_.Resize(proj_dim_, cell_dim_, kSetZero);
    }

    // calculate delta
    const BaseFloat mmt = opts_.momentum;

    // forward direction
    // weight x -> g, i, f, o
    f_w_gifo_x_corr_.AddMatMat(1.0, F_DGIFO.RowRange(1*S, T*S), kTrans,
                                    in,                        kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    f_w_gifo_r_corr_.AddMatMat(1.0, F_DGIFO.RowRange(1*S, T*S), kTrans,
                                    F_YR.RowRange(0*S, T*S),    kNoTrans, mmt);
    // bias of g, i, f, o
    f_bias_corr_.AddRowSumMat(1.0, F_DGIFO.RowRange(1*S, T*S), mmt);

    // recurrent peephole c -> i
    f_peephole_i_c_corr_.AddDiagMatMat(1.0, F_DI.RowRange(1*S, T*S), kTrans,
                                            F_YC.RowRange(0*S, T*S), kNoTrans, mmt);
    // recurrent peephole c -> f
    f_peephole_f_c_corr_.AddDiagMatMat(1.0, F_DF.RowRange(1*S, T*S), kTrans,
                                            F_YC.RowRange(0*S, T*S), kNoTrans, mmt);
    // peephole c -> o
    f_peephole_o_c_corr_.AddDiagMatMat(1.0, F_DO.RowRange(1*S, T*S), kTrans,
                                            F_YC.RowRange(1*S, T*S), kNoTrans, mmt);

    f_w_r_m_corr_.AddMatMat(1.0, F_DR.RowRange(1*S, T*S), kTrans,
                                 F_YM.RowRange(1*S, T*S), kNoTrans, mmt);

    // backward direction backpropagate
    // weight x -> g, i, f, o
    b_w_gifo_x_corr_.AddMatMat(1.0, B_DGIFO.RowRange(1*S, T*S), kTrans, in, kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    b_w_gifo_r_corr_.AddMatMat(1.0, B_DGIFO.RowRange(1*S, T*S), kTrans,
                                    B_YR.RowRange(0*S, T*S)   , kNoTrans, mmt);
    // bias of g, i, f, o
    b_bias_corr_.AddRowSumMat(1.0, B_DGIFO.RowRange(1*S, T*S), mmt);

    // recurrent peephole c -> i, c(t+1) --> i
    b_peephole_i_c_corr_.AddDiagMatMat(1.0, B_DI.RowRange(1*S, T*S), kTrans,
                                            B_YC.RowRange(2*S, T*S), kNoTrans, mmt);
    // recurrent peephole c -> f, c(t+1) --> f
    b_peephole_f_c_corr_.AddDiagMatMat(1.0, B_DF.RowRange(1*S, T*S), kTrans,
                                            B_YC.RowRange(2*S, T*S), kNoTrans, mmt);
    // peephole c -> o
    b_peephole_o_c_corr_.AddDiagMatMat(1.0, B_DO.RowRange(1*S, T*S), kTrans,
                                            B_YC.RowRange(1*S, T*S), kNoTrans, mmt);

    b_w_r_m_corr_.AddMatMat(1.0, B_DR.RowRange(1*S, T*S), kTrans,
                                 B_YM.RowRange(1*S, T*S), kNoTrans, mmt);
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {

    // apply the gradient clipping,
    if (grad_clip_ > 0.0) {
      f_w_gifo_x_corr_.ApplyFloor(-grad_clip_);
      f_w_gifo_x_corr_.ApplyCeiling(grad_clip_);
      f_w_gifo_r_corr_.ApplyFloor(-grad_clip_);
      f_w_gifo_r_corr_.ApplyCeiling(grad_clip_);
      f_bias_corr_.ApplyFloor(-grad_clip_);
      f_bias_corr_.ApplyCeiling(grad_clip_);
      f_w_r_m_corr_.ApplyFloor(-grad_clip_);
      f_w_r_m_corr_.ApplyCeiling(grad_clip_);
      f_peephole_i_c_corr_.ApplyFloor(-grad_clip_);
      f_peephole_i_c_corr_.ApplyCeiling(grad_clip_);
      f_peephole_f_c_corr_.ApplyFloor(-grad_clip_);
      f_peephole_f_c_corr_.ApplyCeiling(grad_clip_);
      f_peephole_o_c_corr_.ApplyFloor(-grad_clip_);
      f_peephole_o_c_corr_.ApplyCeiling(grad_clip_);

      b_w_gifo_x_corr_.ApplyFloor(-grad_clip_);
      b_w_gifo_x_corr_.ApplyCeiling(grad_clip_);
      b_w_gifo_r_corr_.ApplyFloor(-grad_clip_);
      b_w_gifo_r_corr_.ApplyCeiling(grad_clip_);
      b_bias_corr_.ApplyFloor(-grad_clip_);
      b_bias_corr_.ApplyCeiling(grad_clip_);
      b_w_r_m_corr_.ApplyFloor(-grad_clip_);
      b_w_r_m_corr_.ApplyCeiling(grad_clip_);
      b_peephole_i_c_corr_.ApplyFloor(-grad_clip_);
      b_peephole_i_c_corr_.ApplyCeiling(grad_clip_);
      b_peephole_f_c_corr_.ApplyFloor(-grad_clip_);
      b_peephole_f_c_corr_.ApplyCeiling(grad_clip_);
      b_peephole_o_c_corr_.ApplyFloor(-grad_clip_);
      b_peephole_o_c_corr_.ApplyCeiling(grad_clip_);
    }

    const BaseFloat lr = opts_.learn_rate;

    // forward direction update
    f_w_gifo_x_.AddMat(-lr * learn_rate_coef_, f_w_gifo_x_corr_);
    f_w_gifo_r_.AddMat(-lr * learn_rate_coef_, f_w_gifo_r_corr_);
    f_bias_.AddVec(-lr * bias_learn_rate_coef_, f_bias_corr_, 1.0);

    f_peephole_i_c_.AddVec(-lr * bias_learn_rate_coef_, f_peephole_i_c_corr_, 1.0);
    f_peephole_f_c_.AddVec(-lr * bias_learn_rate_coef_, f_peephole_f_c_corr_, 1.0);
    f_peephole_o_c_.AddVec(-lr * bias_learn_rate_coef_, f_peephole_o_c_corr_, 1.0);

    f_w_r_m_.AddMat(-lr * learn_rate_coef_, f_w_r_m_corr_);

    // backward direction update
    b_w_gifo_x_.AddMat(-lr * learn_rate_coef_, b_w_gifo_x_corr_);
    b_w_gifo_r_.AddMat(-lr * learn_rate_coef_, b_w_gifo_r_corr_);
    b_bias_.AddVec(-lr * bias_learn_rate_coef_, b_bias_corr_, 1.0);

    b_peephole_i_c_.AddVec(-lr * bias_learn_rate_coef_, b_peephole_i_c_corr_, 1.0);
    b_peephole_f_c_.AddVec(-lr * bias_learn_rate_coef_, b_peephole_f_c_corr_, 1.0);
    b_peephole_o_c_.AddVec(-lr * bias_learn_rate_coef_, b_peephole_o_c_corr_, 1.0);

    b_w_r_m_.AddMat(-lr * learn_rate_coef_, b_w_r_m_corr_);
  }

 private:
  // dims
  int32 cell_dim_;  ///< the number of memory-cell blocks,
  int32 proj_dim_;  ///< recurrent projection layer dim,

  BaseFloat cell_clip_;  ///< Clipping of 'cell-values' in forward pass (per-frame),
  BaseFloat diff_clip_;  ///< Clipping of 'derivatives' in backprop (per-frame),
  BaseFloat cell_diff_clip_; ///< Clipping of 'cell-derivatives' accumulated over CEC (per-frame),
  BaseFloat grad_clip_;  ///< Clipping of the updates,

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
  // peephole connections are diagonal, so we use vector form,
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
};  // class BlstmProjected

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_BLSTM_PROJECTED_H_
