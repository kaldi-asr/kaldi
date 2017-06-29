// nnet3/nnet-attention-component.h

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_ATTENTION_COMPONENT_H_
#define KALDI_NNET3_NNET_ATTENTION_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "nnet3/attention.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-attention-component.h
///
/// Contains component(s) related to attention models.



/**
   RestrictedAttentionComponent implements an attention model with restricted
   temporal context.  I believe what is implemented here is termed
   self-attention, but I'm not 100% sure of the terminology used.  Note: this
   component is just a fixed nonlinearity, it's not updatable; all the
   parameters are expected to live in the previous component which is most
   likely going to be of type NaturalGradientAffineComponent.  For a more
   in-depth explanation, please see comments in the source of the file
   attention.h.  Also, look at the comments for InputDim() and OutputDim() which
   help to clarify the input and output formats.

   The following are the parameters accepted on the config line, with examples
   of their values.


     num-heads        E.g. num-heads=10.  Defaults to 1.  Having multiple heads
                      just means the same nonlinearity is repeated many times;
                      the input-dim and output-dim are multiples of num-heads.
     key-dim          E.g. key-dim=60.  Must be specified.  Dimension of input keys.
     value-dim        E.g. value-dim=60.  Must be specified.  Dimension of input
                      values (these are the things over which the component forms
                      a weighted sum, although first we add a positional encoding).
     time-stride      Stride for 't' value, e.g. 1 or 3.  For example, if time-stride=3,
                      to compute the output for t=10 we'd use the input for time
                      values like ... t=7, t=10, t=13, ... (depending on the
                      values of num-left-inputs and num-right-inputs).
     num-left-inputs  Number of frames to the left of the current frame, that we
                      use as input, e.g. 5.  (The frame used will be separated by 'time-stride').
                      Must be >= 0.
     num-right-inputs  Number of frames to the right of the current frame, that we
                      use as input, e.g. 2.  Must be >= 0.  You are not allowed to set
                      both num-left-inputs and num-right-inputs to zero.
     num-left-inputs-required  The number of frames to the left, that are
                      required in order to produce an output.  Defaults to the
                      same as num-left-inputs, but you can set it to a smaller
                      value if you want.  This can save computation at the start
                      of the file.  We'll use zero-padding for non-required
                      inputs.  Be careful with this because it interacts with
                      decoding settings; it would be advisable to increase the
                      extra-left-context parameter by the sum of the difference
                      between num-left-inputs-required and num-left-inputs,
                      although you could leave extra-left-context-initial at
                      zero.
     num-right-inputs-required  See num-left-inputs-required for explanation;
                      it's the mirror image.  Defaults to num-right-inputs.
                      However, be even more careful with the right-hand version;
                      if you set this, online (looped) decoding will not work
                      correctly.
     alpha            Scale on the inputs to the softmax function.
                      Defaults to 1.0 / sqrt(key-dim + 1 + num-left-inputs + num-right-inputs),
                      which is 1 / sqrt of the real key/query dimension after taking
                      into account the positional encoding.
     beta             Defaults to 1.0.  Scale on the positional encoding as it is appended
                      to the keys, and if output-context==true, to the output values.
     output-context  (Default: true).  If true, output the softmax that encodes which
                     positions we chose, in addition to the input values.
 */
class RestrictedAttentionComponent: public Component {
 public:

  // The use of this constructor should only precede InitFromConfig()
  RestrictedAttentionComponent() { }

  // Copy constructor
  RestrictedAttentionComponent(const RestrictedAttentionComponent &other)
    = default;

  virtual int32 InputDim() const {
    // the input is interpreted as being appended blocks one for each
    // head-index 0 <= h < num_heads_; each such block is
    // interpreted as (key, value, query).
    int32 query_dim = key_dim_ + context_dim_;
    return num_heads_ * (key_dim_ + value_dim_ + query_dim);
  }
  virtual int32 OutputDim() const {
    // the output consists of appended blocks, one for each head head-index
    // 0 <= h < num_heads_; each such block is is the averaged value-- plus the
    // softmax encoding of the positions we chose, if output_context_ == true.
    return num_heads_ * (value_dim_ + output_context_ ? context_dim_ : 0);
  }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "RestrictedAttentionComponent"; }
  virtual int32 Properties() const {
    return kReordersIndexes|kBackpropNeedsInput|kPropagateAdds|kStoresStats|kUsesMemo;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  // This ReorderIndexes function may insert 'blank' indexes (indexes with
  // t == kNoTime) as well as reordering the indexes.  This is allowed
  // behavior of ReorderIndexes functions.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;


  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new RestrictedAttentionComponent(*this);
  }


  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  // This function returns true if at least one of the input indexes used to
  // compute this output index is computable.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other):
        computation(other.computation) { }
    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "RestrictedAttentionComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }

    time_height_convolution::ConvolutionComputationIo io;
  };

  // This is what's returned as the 'memo' from the Propagate() function.
  struct Memo {
    // c is of dimension (num_heads_ * num-output-frames) by context_dim_,
    // where num-output-frames is the number of frames of output the
    // corresponding Propagate function produces.
    // Each block of 'num-output-frames' rows of c_t is the
    // post-softmax matrix of weights.
    CuMatrix<BaseFloat> c;
  };

 private:

  // Does the propagation for one head; this is called for each
  // head by the top-level Propagate function.  Later on we may
  // figure out a way to avoid doing this sequentially.
  // 'in' and 'out' are submatrices of the 'in' and 'out' passed
  // to the top-level Propagate function, and 'c' is a submatrix
  // of the 'c' matrix in the memo we're creating.
  //
  // Assumes 'c' has already been zerooed.
  void PropagateOneHead(
      const time_height_convolution::ConvolutionComputationIo &io,
      const CuMatrixBase<BaseFloat> &in,
      CuMatrixBase<BaseFloat> *c,
      CuMatrixBase<BaseFloat> *out) const;

  void Check();

  int32 num_heads_;
  int32 key_dim_;
  int32 value_dim_;
  int32 num_left_inputs_;
  int32 num_right_inputs_;
  int32 time_stride_;
  int32 context_dim_;  // This derived parameter equals 1 + num_left_inputs_ +
                       // num_right_inputs_.
  int32 num_left_inputs_required_;
  int32 num_right_inputs_required_;
  bool output_context_;


  double stats_count_;  // Count of frames corresponding to the stats.
  Vector<double> entropy_stats_;  // entropy stats, indexed per head.
                                  // (dimension is num_heads_).  Divide
                                  // by stats_count_ to normalize.
  CuMatrix<double> posterior_stats_;  // stats of posteriors of different
                                      // offsets, of dimension num_heads_ by
                                      // context_dim_.
};




} // namespace nnet3
} // namespace kaldi


#endif
