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
   temporal context.  What is implemented here is a case of self-attention,
   meaning that the set of indexes on the input is the same set as the indexes
   on the output (like an N->N mapping, ignoring edge effects, as opposed to an
   N->M mapping that you might see in a translation model).  "Restricted" means
   that the source indexes are constrained to be close to the destination
   indexes, i.e. when outputting something for time 't' we attend to a narrow
   range of source time indexes close to 't'.

   This component is just a fixed nonlinearity (albeit of a type that
   "knows about" time, i.e. the output at time 't' depends on inputs
   at a range of time values).  This component is not updatable; all the
   parameters are expected to live in the previous component which is most
   likely going to be of type NaturalGradientAffineComponent.  For a more
   in-depth explanation, please see comments in the source of the file
   attention.h.  Also, look at the comments for InputDim() and OutputDim() which
   help to clarify the input and output formats.

   The following are the parameters accepted on the config line, with examples
   of their values.


     num-heads        E.g. num-heads=10.  Defaults to 1.  Having multiple heads
                      just means the same nonlinearity is repeated many times.
                      InputDim() and OutputDim() are multiples of num-heads.
     key-dim          E.g. key-dim=60.  Must be specified.  Dimension of input keys.
     value-dim        E.g. value-dim=60.  Must be specified.  Dimension of input
                      values (these are the things over which the component forms
                      a weighted sum, although if output-context=true we append
                      to the output the weights of the weighted sum, as they might
                      also carry useful information.
     time-stride      Stride for 't' value, e.g. 1 or 3.  For example, if time-stride=3,
                      to compute the output for t=10 we'd use the input for time
                      values like ... t=7, t=10, t=13, ... (the ends of
                      this range depend on num-left-inputs and num-right-inputs).
     num-left-inputs  Number of frames to the left of the current frame, that we
                      use as input, e.g. 5.  (The 't' values used will be separated
                      by 'time-stride').  num-left-inputs must be >= 0.  Must be
                      specified.
     num-right-inputs  Number of frames to the right of the current frame, that we
                      use as input, e.g. 2.  Must be >= 0 and must be specified.
                      You are not allowed to set both num-left-inputs and
                      num-right-inputs to zero.
     num-left-inputs-required  The number of frames to the left, that are
                      required in order to produce an output.  Defaults to the
                      same as num-left-inputs, but you can set it to a smaller
                      value if you want.  We'll use zero-padding for
                      non-required inputs that are not present in the input.  Be
                      careful with this because it interacts with decoding
                      settings; for non-online decoding and for dumping of egs
                      it would be advisable to increase the extra-left-context
                      parameter by the sum of the difference between
                      num-left-inputs-required and num-left-inputs, although you
                      could leave extra-left-context-initial at zero.
     num-right-inputs-required  See num-left-inputs-required for explanation;
                      it's the mirror image.  Defaults to num-right-inputs.
                      However, be even more careful with the right-hand version;
                      if you set this, online (looped) decoding will not work
                      correctly.  It might be wiser just to reduce num-right-inputs
                      if you care about real-time decoding.
     key-scale        Scale on the keys (but not the added context).  Defaults to 1.0 /
                      sqrt(key-dim), like the 1/sqrt(d_k) value in the
                      "Attention is all you need" paper.  This helps prevent saturation
                      of the softmax.
     output-context  (Default: true).  If true, output the softmax that encodes which
                     positions we chose, in addition to the input values.
 */
class RestrictedAttentionComponent: public Component {
 public:

  // The use of this constructor should only precede InitFromConfig()
  RestrictedAttentionComponent() { }

  // Copy constructor
  RestrictedAttentionComponent(const RestrictedAttentionComponent &other);

  virtual int32 InputDim() const {
    // the input is interpreted as being appended blocks one for each head; each
    // such block is interpreted as (key, value, query).
    int32 query_dim = key_dim_ + context_dim_;
    return num_heads_ * (key_dim_ + value_dim_ + query_dim);
  }
  virtual int32 OutputDim() const {
    // the output consists of appended blocks, one for each head; each such
    // block is is the attention weighted average of the input values, to which
    // we append softmax encoding of the positions we chose, if output_context_
    // == true.
    return num_heads_ * (value_dim_ + (output_context_ ? context_dim_ : 0));
  }
  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "RestrictedAttentionComponent"; }
  virtual int32 Properties() const {
    return kReordersIndexes|kBackpropNeedsInput|kPropagateAdds|kBackpropAdds|
        kStoresStats|kUsesMemo;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void ZeroStats();

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new RestrictedAttentionComponent(*this);
  }
  virtual void DeleteMemo(void *memo) const { delete static_cast<Memo*>(memo); }

  // Some functions that are only to be reimplemented for GeneralComponents.

  // This ReorderIndexes function may insert 'blank' indexes (indexes with
  // t == kNoTime) as well as reordering the indexes.  This is allowed
  // behavior of ReorderIndexes functions.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

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
        io(other.io) { }
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


  // does the backprop for one head; called by Backprop().
  void BackpropOneHead(
      const time_height_convolution::ConvolutionComputationIo &io,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &c,
      const CuMatrixBase<BaseFloat> &out_deriv,
      CuMatrixBase<BaseFloat> *in_deriv) const;

  // This function, used in ReorderIndexes() and PrecomputedIndexes(),
  // works out what grid structure over time we will have at the input
  // and the output.
  // Note: it may produce a grid that encompasses more than what was
  // listed in 'input_indexes' and 'output_indexes'.  This is OK.
  // ReorderIndexes() will add placeholders with t == kNoTime for
  // padding, and at the input of this component those placeholders
  // will be zero; at the output the placeholders will be ignored.
  void GetComputationStructure(
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      time_height_convolution::ConvolutionComputationIo *io) const;

  // This function, used in ReorderIndexes(), obtains the indexes with the
  // correct structure and order (the structure is specified in the 'io' object.
  // This may involve not just reordering the provided indexes, but padding them
  // with indexes that have kNoTime as the time.
  //
  // Basically the indexes this function outputs form a grid where 't' has the
  // larger stride than the (n, x) pairs.   The number of distinct (n, x) pairs
  // should equal io.num_images.  Where 't' values need to appear in the
  // new indexes that were not present in the old indexes, they get replaced with
  // kNoTime.
  void GetIndexes(
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      time_height_convolution::ConvolutionComputationIo &io,
      std::vector<Index> *new_input_indexes,
      std::vector<Index> *new_output_indexes) const;

  /// Utility function used in GetIndexes().  Creates a grid of Indexes, where
  /// 't' has the larger stride, and within each block of Indexes for a given
  /// 't', we have the given list of (n, x) pairs.  For Indexes that we create
  /// where the 't' value was not present in 'index_set', we set the 't' value
  /// to kNoTime (indicating that it's only for padding, not a real input or an
  /// output that's ever used).
  static void CreateIndexesVector(
      const std::vector<std::pair<int32, int32> > &n_x_pairs,
      int32 t_start, int32 t_step, int32 num_t_values,
      const std::unordered_set<Index, IndexHasher> &index_set,
      std::vector<Index> *output_indexes);


  void Check() const;

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
  BaseFloat key_scale_;

  double stats_count_;  // Count of frames corresponding to the stats.
  Vector<double> entropy_stats_;  // entropy stats, indexed per head.
                                  // (dimension is num_heads_).  Divide
                                  // by stats_count_ to normalize.
  CuMatrix<double> posterior_stats_;  // stats of posteriors of different
                                      // offsets, of dimension num_heads_ *
                                      // context_dim_ (num-heads has the
                                      // larger stride).
};




} // namespace nnet3
} // namespace kaldi


#endif
