// nnet/nnet-sentence-averaging-component.h

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_SENTENCE_AVERAGING_COMPONENT_H_
#define KALDI_NNET_NNET_SENTENCE_AVERAGING_COMPONENT_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {


/**
 * SimpleSentenceAveragingComponent does not have nested network,
 * it is intended to be used inside of a <ParallelComponent>.
 * For training use 'nnet-train-perutt'.
 *
 * The sentence-averaging typically leads to small gradients, so we boost it 100x 
 * by default (boost = multiply, it's equivalent to applying learning-rate factor).
 */
class SimpleSentenceAveragingComponent : public Component {
 public:
  SimpleSentenceAveragingComponent(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out), gradient_boost_(100.0)
  { }
  ~SimpleSentenceAveragingComponent()
  { }

  Component* Copy() const { return new SimpleSentenceAveragingComponent(*this); }
  ComponentType GetType() const { return kSimpleSentenceAveragingComponent; }

  void InitData(std::istream &is) {
    is >> std::ws;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token);
      if (token == "<GradientBoost>") ReadBasicType(is, false, &gradient_boost_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config? (GradientBoost)";
      is >> std::ws; // eat-up whitespace
    }
  }

  void ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<GradientBoost>");
    ReadBasicType(is, binary, &gradient_boost_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<GradientBoost>");
    WriteBasicType(os, binary, gradient_boost_);
  }

  std::string Info() const {
    return std::string("\n  gradient-boost ") + ToString(gradient_boost_);
  }
  std::string InfoGradient() const {
    return Info();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // get the average row-vector,
    average_row_.Resize(InputDim());
    average_row_.AddRowSumMat(1.0/in.NumRows(), in, 0.0);
    // copy it on the output,
    out->AddVecToRows(1.0, average_row_, 0.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // While averaging, a single frame from input influenced all frames on the output,
    // hence the derivative w.r.t. single input frame is a sum of the output derivatives.
    // (while scaling by averaging constant 1/Nframes).
    //
    // In fact all the input frames influenced all the output frames,
    // so the derivarive is the same for all the input frames.
    //
    // getting the average output diff,
    average_diff_.Resize(OutputDim());
    average_diff_.AddRowSumMat(1.0/out_diff.NumRows(), out_diff, 0.0);
    // copy the derivative into the input diff, (applying gradient-boost!!)
    in_diff->AddVecToRows(gradient_boost_, average_diff_, 0.0);
  }

 private:
  CuVector<BaseFloat> average_row_; ///< auxiliary buffer for forward propagation,
  CuVector<BaseFloat> average_diff_; ///< auxiliary buffer for backpropagation,
  BaseFloat gradient_boost_; ///< increase of gradient applied in backpropagation,
};



/** Deprecated, keeping it as Katka Zmolikova used it in JSALT 2015 */
class SentenceAveragingComponent : public UpdatableComponent {
 public:
  SentenceAveragingComponent(int32 dim_in, int32 dim_out) 
    : UpdatableComponent(dim_in, dim_out), learn_rate_factor_(100.0)
  { }
  ~SentenceAveragingComponent()
  { }

  Component* Copy() const { return new SentenceAveragingComponent(*this); }
  ComponentType GetType() const { return kSentenceAveragingComponent; }

  void InitData(std::istream &is) {
    // define options
    std::string nested_nnet_filename;
    std::string nested_nnet_proto;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<NestedNnetFilename>") ReadToken(is, false, &nested_nnet_filename);
      else if (token == "<NestedNnetProto>") ReadToken(is, false, &nested_nnet_proto);
      else if (token == "<LearnRateFactor>") ReadBasicType(is, false, &learn_rate_factor_);
      else KALDI_ERR << "Unknown token " << token << " Typo in config?";
      is >> std::ws; // eat-up whitespace
    }
    // initialize (read already prepared nnet from file)
    KALDI_ASSERT((nested_nnet_proto != "") ^ (nested_nnet_filename != "")); //xor
    if (nested_nnet_filename != "") nnet_.Read(nested_nnet_filename);
    if (nested_nnet_proto != "") nnet_.Init(nested_nnet_proto);
    // check dims of nested nnet
    KALDI_ASSERT(InputDim() == nnet_.InputDim());
    KALDI_ASSERT(OutputDim() == nnet_.OutputDim() + InputDim());
  }

  void ReadData(std::istream &is, bool binary) {
    nnet_.Read(is, binary);
    KALDI_ASSERT(nnet_.InputDim() == InputDim());
    KALDI_ASSERT(nnet_.OutputDim() + InputDim() == OutputDim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    nnet_.Write(os, binary);
  }

  int32 NumParams() const { return nnet_.NumParams(); }
  void GetParams(Vector<BaseFloat>* wei_copy) const { wei_copy->Resize(NumParams()); nnet_.GetParams(wei_copy); }
  std::string Info() const { return std::string("nested_network {\n") + nnet_.Info() + "}\n"; }
  std::string InfoGradient() const { return std::string("nested_gradient {\n") + nnet_.InfoGradient() + "}\n"; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // Get NN output
    CuMatrix<BaseFloat> out_nnet;
    nnet_.Propagate(in, &out_nnet);
    // Get the average row (averaging over the time axis):
    // averaging corresponds to extraction of constant vector code for single sentence, 
    int32 num_inputs = in.NumCols(),
      nnet_outputs = nnet_.OutputDim(),
      num_frames = out_nnet.NumRows();
      
    CuVector<BaseFloat> average_row(nnet_outputs);
    average_row.AddRowSumMat(1.0/num_frames, out_nnet, 0.0);
    // Forwarding sentence codes along with input features
    out->ColRange(0,nnet_outputs).AddVecToRows(1.0, average_row, 0.0);
    out->ColRange(nnet_outputs,num_inputs).CopyFromMat(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    if (in_diff == NULL) return;
    int32 num_inputs = in.NumCols(),
      nnet_outputs = nnet_.OutputDim();
    in_diff->CopyFromMat(out_diff.ColRange(nnet_outputs,num_inputs));
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {

    int32 nnet_outputs = nnet_.OutputDim(),
      num_frames = diff.NumRows();
    // Passing the derivative into the nested network. The loss derivative is averaged:
    // single frame from nested network influenced all the frames in the main network,
    // so to get the derivative w.r.t. single frame from nested network we sum derivatives
    // of all frames from main network (and scale by 1/Nframes constant).
    //
    // In fact all the frames from nested network influenced all the input frames to main nnet,
    // so the loss derivarive w.r.t. nested network output is same for all frames in sentence.
    CuVector<BaseFloat> average_diff(nnet_outputs);
    average_diff.AddRowSumMat(1.0/num_frames, diff.ColRange(0,nnet_outputs), 0.0);
    CuMatrix<BaseFloat> nnet_out_diff(num_frames, nnet_outputs);
    nnet_out_diff.AddVecToRows(1.0, average_diff, 0.0);
    // 
    nnet_.Backpropagate(nnet_out_diff, NULL);
  }
 
  void SetTrainOptions(const NnetTrainOptions &opts) {
    UpdatableComponent::SetTrainOptions(opts_);
    // Pass the train options to the nnet
    NnetTrainOptions o(opts);
    //o.learn_rate *= 100; // GOOD
    //o.learn_rate *= 1000; // BAD
    o.learn_rate *= learn_rate_factor_;
    nnet_.SetTrainOptions(opts_);
  }

 private:
  Nnet nnet_;
  float learn_rate_factor_;
};
/* Deprecated */

} // namespace nnet1
} // namespace kaldi

#endif
