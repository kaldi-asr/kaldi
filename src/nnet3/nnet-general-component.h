// nnet3/nnet-general-component.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_GENERAL_COMPONENT_H_
#define KALDI_NNET3_NNET_GENERAL_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-general-component.h
/// This file contains declarations of components that are not "simple",
///   meaning they care about the indexes they are operating on, don't return
///   the kSimpleComponent flag in their Properties(), and may return a different
///   number of outputs than inputs.
///   Also see nnet-convolutional-component.h, which also contains
///   number of convolution-related 'general' components.



/**
   This Component takes a larger input-dim than output-dim, where the input-dim
   must be a multiple of the output-dim, and distributes different blocks of the
   input dimension to different 'x' values.  In the normal case where the input
   is only valid at x=0, the first block of output goes to x=0, the second block
   at x=1, and so on.  It also supports a more general usage, so in general a
   value 'x' at the output will map to block 'x % n_blocks' of the dimension
   blocks of the input, and to an x value 'x / n_blocks' of the input.  For negative
   x values the % and / operations are always rounded down, not towards zero.

   The config line is of the form
     input-dim=xx output-dim=xx
   where input-dim must be a multiple of the output-dim, and n_blocks is
   set to input-dim / output-dim.
   */
class DistributeComponent: public Component {
 public:
  DistributeComponent(int32 input_dim, int32 output_dim) {
    Init(input_dim, output_dim);
  }
  DistributeComponent(): input_dim_(0), output_dim_(0) { }
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }

  // use the default Info() function.
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "DistributeComponent"; }
  virtual int32 Properties() const { return 0; }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new DistributeComponent(input_dim_, output_dim_);
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

  // Some functions that are specific to this class.
  void Init(int32 input_dim, int32 output_dim);
 private:
  // computes the input index corresponding to a particular output index.
  // if block != NULL, also computes which block of the input this corresponds to.
  inline void ComputeInputIndexAndBlock(const Index &output_index,
                                        Index *input_index,
                                        int32 *block) const;
  inline void ComputeInputPointers(
      const ComponentPrecomputedIndexes *indexes,
      const CuMatrixBase<BaseFloat> &in,
      int32 num_output_rows,
      std::vector<const BaseFloat*> *input_pointers) const;
  // non-const version of the above.
  inline void ComputeInputPointers(
      const ComponentPrecomputedIndexes *indexes,
      int32 num_output_rows,
      CuMatrixBase<BaseFloat> *in,
      std::vector<BaseFloat*> *input_pointers) const;
  int32 input_dim_;
  int32 output_dim_;

};

class DistributeComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:

  // each pair is a pair (row, dim_offset), and by
  // computing (input.Data() + row * input.Stride() + dim_offset)
  // we get an address that points to the correct input location.
  std::vector<std::pair<int32, int32> > pairs;

  // this class has a virtual destructor so it can be deleted from a pointer
  // to ComponentPrecomputedIndexes.
  virtual ~DistributeComponentPrecomputedIndexes() { }

  virtual ComponentPrecomputedIndexes* Copy() const {
    return new DistributeComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &ostream, bool binary) const;

  virtual void Read(std::istream &istream, bool binary);

  virtual std::string Type() const { return "DistributeComponentPrecomputedIndexes"; }
};

/*
  Class StatisticsExtractionComponent is used together with
  StatisticsPoolingComponent to extract moving-average mean and
  standard-deviation statistics.

  StatisticsExtractionComponent is designed to extract statistics-- 0th-order,
  1st-order and optionally diagonal 2nd-order stats-- from small groups of
  frames, such as 10 frames.  The statistics will then be further processed by
  StatisticsPoolingComponent to compute moving-average means and (if configured)
  standard deviations.  The reason for the two-component way of doing this is
  efficiency, particularly in the graph-compilation phase.  (Otherwise there
  would be too many dependencies to process).  The StatisticsExtractionComponent
  is designed to let you extract statistics from fixed-size groups of frames
  (e.g. 10 frames), and in StatisticsPoolingComponent you are only expected to
  compute the averages at the same fixed period (e.g. 10 frames), so it's more
  efficient than if you were to compute a moving average at every single frame;
  and the computation of the intermediate stats means that most of the
  computation that goes into extracting the means and standard deviations for
  nearby frames is shared.

  The config line in a typical setup will be something like:

    input-dim=250 input-period=1 output-period=10 include-variance=true

  input-dim is self-explanatory.  The inputs will be obtained at multiples of
  input-period (e.g. it might be 3 for chain models).  output-period must be a
  multiple of input period, and the requested output indexes will be expected to
  be multiples of output-period (which you can ensure through use of the Round
  descriptor).  For instance, if you request the output on frame 80, it will
  consist of stats from input frames 80 through 89.

  An output of this component will be 'computable' any time at least one of
  the corresponding inputs is computable.

  In all cases the first dimension of the output will be a count (between 1 and
  10 inclusive in this example).  If include-variance=false, then the output
  dimension will be input-dim + 1.  and the output dimensions >0 will be
  1st-order statistics (sums of the input).  If include-variance=true, then the
  output dimension will be input-dim * 2 + 1, where the raw diagonal 2nd-order
  statistics are appended to the 0 and 1st order statistics.

  The default configuration values are:
     input-dim=-1 input-period=1 output-period=1 include-variance=true
 */
class StatisticsExtractionComponent: public Component {
 public:
  // Initializes to defaults which would not pass Check(); use InitFromConfig()
  // or Read() or copy constructor to really initialize.
  StatisticsExtractionComponent();
  // copy constructor, used in Copy().
  StatisticsExtractionComponent(const StatisticsExtractionComponent &other);

  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const {
    // count + sum stats [ + sum-squared stats].
    return 1 + input_dim_ + (include_variance_ ? input_dim_ : 0);
  }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "StatisticsExtractionComponent"; }
  virtual int32 Properties() const {
    return kPropagateAdds|kReordersIndexes|
        (include_variance_ ? kBackpropNeedsInput : 0);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new StatisticsExtractionComponent(*this);
  }

  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  // This function reorders the input and output indexes so that they
  // are sorted first on n and then x and then t.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

 private:
  // Checks that the parameters are valid.
  void Check() const;

  // Disallow assignment operator.
  StatisticsExtractionComponent &operator =(
      const StatisticsExtractionComponent &other);

  int32 input_dim_;
  int32 input_period_;
  int32 output_period_;
  bool include_variance_;
};

class StatisticsExtractionComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:
  // While creating the output we sum over row ranges of the input.
  // forward_indexes.Dim() equals the number of rows of the output, and each
  // element is a (start, end) range of inputs, that is summed over.
  CuArray<Int32Pair> forward_indexes;

  // This vector stores the number of inputs for each output.  Normally this will be
  // the same as the component's output_period_ / input_period_, but could be less
  // due to edge effects at the utterance boundary.
  CuVector<BaseFloat> counts;

  // Each input row participates in exactly one output element, and
  // 'backward_indexes' identifies which row of the output each row
  // of the input is part of.  It's used in backprop.
  CuArray<int32> backward_indexes;

  ComponentPrecomputedIndexes *Copy() const {
    return new StatisticsExtractionComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Read(std::istream &is, bool binary);

  virtual std::string Type() const { return "StatisticsExtractionComponentPrecomputedIndexes"; }
 private:
  virtual ~StatisticsExtractionComponentPrecomputedIndexes() { }
};

/*
  Class StatisticsPoolingComponent is used together with
  StatisticsExtractionComponent to extract moving-average mean and
  standard-deviation statistics.

  StatisticsPoolingComponent pools the stats over a specified window and
  computes means and possibly log-count and stddevs from them for you.

 # In StatisticsPoolingComponent, the first element of the input is interpreted
 # as a count, which we divide by.
 # Optionally the log of the count can be output, and you can allow it to be
 # repeated several times if you want (useful for systems using the jesus-layer).
 # The output dimension is equal to num-log-count-features plus (input-dim - 1).

 # If include-log-count==false, the output dimension is the input dimension minus one.
 # If output-stddevs=true, then it expects the input-dim to be of the form 2n+1 where n is
 #  presumably the original feature dim, and it interprets the last n dimensions of the feature
 #  as a variance; it outputs the square root of the variance instead of the actual variance.

 configs and their defaults:  input-dim=-1, input-period=1, left-context=-1, right-context=-1,
    num-log-count-features=0, output-stddevs=true, variance-floor=1.0e-10

 You'd access the output of the StatisticsPoolingComponent using rounding, e.g.
  Round(component-name, 10)
 or whatever, instead of just component-name, because its output is only defined at multiples
 of its input-period.

 The output of StatisticsPoolingComponent will only be defined if at least one input was defined.
 */
class StatisticsPoolingComponent: public Component {
 public:
  // Initializes to defaults which would not pass Check(); use InitFromConfig()
  // or Read() or copy constructor to really initialize.
  StatisticsPoolingComponent();
  // copy constructor, used in Copy()
  StatisticsPoolingComponent(const StatisticsPoolingComponent &other);

  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const {
    return input_dim_ + num_log_count_features_ - 1;
  }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "StatisticsPoolingComponent"; }
  virtual int32 Properties() const {
    return kReordersIndexes|kBackpropAdds|
        (output_stddevs_ || num_log_count_features_ > 0 ?
         kBackpropNeedsOutput : 0) |
        (num_log_count_features_ == 0 ? kBackpropNeedsInput : 0);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new StatisticsPoolingComponent(*this);
  }

  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  // returns true if at least one of its inputs is computable.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  // This function reorders the input and output indexes so that they
  // are sorted first on n and then x and then t.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

 private:
  // Checks that the parameters are valid.
  void Check() const;

  // Disallow assignment operator.
  StatisticsPoolingComponent &operator =(
      const StatisticsPoolingComponent &other);

  int32 input_dim_;
  int32 input_period_;
  int32 left_context_;
  int32 right_context_;
  int32 num_log_count_features_;
  bool output_stddevs_;
  BaseFloat variance_floor_;
};

class StatisticsPoolingComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:

  // in the first stage of creating the output we sum over row ranges of
  // the input.  forward_indexes.Dim() equals the number of rows of the
  // output, and each element is a (start, end) range of inputs, that is
  // summed over.
  CuArray<Int32Pair> forward_indexes;

  // backward_indexes contains the same information as forward_indexes, but in a
  // different format.  backward_indexes.Dim() is the same as the number of rows
  // of input, and each element contains the (start,end) of the range of outputs
  // for which this input index appears as an element of the sum for that
  // output.  This is possible because of the way the inputs and outputs are
  // ordered and because of how we select the elments to appear in the sum using
  // a window.  This quantity is used in backprop.
  CuArray<Int32Pair> backward_indexes;

  virtual ~StatisticsPoolingComponentPrecomputedIndexes() { }

  ComponentPrecomputedIndexes *Copy() const {
    return new StatisticsPoolingComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Read(std::istream &is, bool binary);

  virtual std::string Type() const { return "StatisticsPoolingComponentPrecomputedIndexes"; }
};

// BackpropTruncationComponent zeroes out the gradients every certain number
// of frames, as well as having gradient-clipping functionality as
// ClipGradientComponent.
// This component will be used to prevent gradient explosion problem in
// recurrent neural networks
class BackpropTruncationComponent: public Component {
 public:
  BackpropTruncationComponent(int32 dim,
                              BaseFloat scale,
                              BaseFloat clipping_threshold,
                              BaseFloat zeroing_threshold,
                              int32 zeroing_interval,
                              int32 recurrence_interval) {
    Init(dim, scale, clipping_threshold, zeroing_threshold,
        zeroing_interval, recurrence_interval);}

  BackpropTruncationComponent(): dim_(0), scale_(1.0), clipping_threshold_(-1),
    zeroing_threshold_(-1), zeroing_interval_(0), recurrence_interval_(0),
    num_clipped_(0), num_zeroed_(0), count_(0), count_zeroing_boundaries_(0) { }

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void InitFromConfig(ConfigLine *cfl);
  void Init(int32 dim, BaseFloat scale, BaseFloat clipping_threshold,
            BaseFloat zeroing_threshold, int32 zeroing_interval,
            int32 recurrence_interval);

  virtual std::string Type() const { return "BackpropTruncationComponent"; }

  virtual int32 Properties() const {
    return kPropagateInPlace|kBackpropInPlace;
  }

  virtual void ZeroStats();

  virtual Component* Copy() const;

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Info() const;
  virtual ~BackpropTruncationComponent() {
  }
 private:
  // input/output dimension
  int32 dim_;

  // Scale that is applied in the forward propagation (and of course in the
  // backprop to match.  Expected to normally be 1, but setting this to other
  // values (e.g.  slightly less than 1) can be used to produce variants of
  // LSTMs where the activations are bounded.
  BaseFloat scale_;

  // threshold (e.g., 30) to be used for clipping corresponds to max-row-norm
  BaseFloat clipping_threshold_;

  // threshold (e.g., 3) to be used for zeroing corresponds to max-row-norm
  BaseFloat zeroing_threshold_;

  // interval (e.g., 20, in number of frames) at which we would zero the
  // gradient if the norm of the gradient is above zeroing_threshold_
  int32 zeroing_interval_;

  // recurrence_interval_ should be the absolute recurrence offset used in RNNs
  // (e.g., 3). It is used to see whether the index the component is processing,
  // crosses a boundary that's a multiple of zeroing_interval_ frames.
  int32 recurrence_interval_;

  // component-node name, used in the destructor to print out stats of
  // self-repair
  std::string debug_info_;

  BackpropTruncationComponent &operator =
      (const BackpropTruncationComponent &other); // Disallow.

 protected:
  // variables to store stats
  // An element corresponds to rows of derivative matrix
  double num_clipped_;  // number of elements which were clipped
  double num_zeroed_;   // number of elements which were zeroed
  double count_;  // number of elements which were processed
  double count_zeroing_boundaries_; // number of zeroing boundaries where we had
                                    // the opportunity to perform zeroing
                                    // the gradient

};

class BackpropTruncationComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:

  // zeroing has the same dimension as the number of rows of out-deriv.
  // Each element in zeroing can take two possible values: -1.0, meaning its
  // corresponding frame is one that we need to consider zeroing the
  // gradient of, and 0.0 otherwise
  CuVector<BaseFloat> zeroing;

  // caches the negative sum of elements in zeroing for less CUDA calls
  // (the sum is computed by CPU). Note that this value would be positive.
  BaseFloat zeroing_sum;

  BackpropTruncationComponentPrecomputedIndexes(): zeroing_sum(0.0) {}

  // this class has a virtual destructor so it can be deleted from a pointer
  // to ComponentPrecomputedIndexes.
  virtual ~BackpropTruncationComponentPrecomputedIndexes() { }

  virtual ComponentPrecomputedIndexes* Copy() const {
    return new BackpropTruncationComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &ostream, bool binary) const;

  virtual void Read(std::istream &istream, bool binary);

  virtual std::string Type() const {
    return "BackpropTruncationComponentPrecomputedIndexes";
  }
};


/*
   ConstantComponent returns a constant value for all requested
   indexes, and it has no dependencies on any input.
   It's like a ConstantFunctionComponent, but done the "right"
   way without requiring an unnecessary input.
   It is optionally trainable, and optionally you can use natural
   gradient.

   Configuration values accepted by this component, with defaults if
   applicable:

      output-dim              Dimension that this component outputs.
      is-updatable=true       True if you want this to be updatable.
      use-natural-gradient=true  True if you want the update to use natural gradient.
      output-mean=0.0         Mean of the parameters at initialization (the parameters
                              are what it outputs).
      output-stddev=0.0       Standard deviation of the parameters at initialization.


  Values inherited from UpdatableComponent (see its declaration in
  nnet-component-itf for details):
     learning-rate
     learning-rate-factor
     max-change
*/
class ConstantComponent: public UpdatableComponent {
 public:
  // actually this component requires no inputs; this value
  // is really a don't-care.
  virtual int32 InputDim() const { return output_.Dim(); }

  virtual int32 OutputDim() const { return output_.Dim(); }

  virtual std::string Info() const;

  // possible parameter values with their defaults:
  // is-updatable=true use-natural-gradient=true output-dim=-1
  // output-mean=0 output-stddev=0
  virtual void InitFromConfig(ConfigLine *cfl);

  ConstantComponent();

  ConstantComponent(const ConstantComponent &other);

  virtual std::string Type() const { return "ConstantComponent"; }
  virtual int32 Properties() const {
    return
        (is_updatable_ ? kUpdatableComponent : 0);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const {
    desired_indexes->clear();  // requires no inputs.
  }

  // This function returns true if at least one of the input indexes used to
  // compute this output index is computable.
  // it's simple because this component requires no inputs.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const {
    if (used_inputs) used_inputs->clear();
    return true;
  }

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
 private:

  // the output value-- a vector.
  CuVector<BaseFloat> output_;

  bool is_updatable_;
  // if true, and if updatable, do natural-gradient update.
  bool use_natural_gradient_;
  OnlineNaturalGradient preconditioner_;

  const ConstantComponent &operator
  = (const ConstantComponent &other); // Disallow.
};



// DropoutMaskComponent outputs a random zero-or-one value for all dimensions of
// all requested indexes, and it has no dependencies on any input.  It's like a
// ConstantComponent, but with random output that has value zero
// a proportion (dropout_proportion) of the time, and otherwise one.
// This is not the normal way to implement dropout; you'd normally use a
// DropoutComponent (see nnet-simple-component.h).  This component is used while
// implementing per-frame dropout with the LstmNonlinearityComponent; we
// generate a two-dimensional output representing dropout
//
class DropoutMaskComponent: public RandomComponent {
 public:
  // actually this component requires no inputs; this value
  // is really a don't-care.
  virtual int32 InputDim() const { return output_dim_; }

  virtual int32 OutputDim() const { return output_dim_; }

  virtual std::string Info() const;

  // possible parameter values with their defaults:
  // dropout-proportion=0.5 output-dim=-1 continuous=false
  // With the 'continous=false' option (the default), it generates
  // 0 with probability 'dropout-proportion' and 1 otherwise.
  // With 'continuous=true' it outputs 1 plus dropout-proportion times
  //  a value uniformly distributed on [-2, 2].  (e.g. if dropout-proportion is
  // 0.5, this would amount to a value uniformly distributed on [0,2].)
  virtual void InitFromConfig(ConfigLine *cfl);

  DropoutMaskComponent();

  DropoutMaskComponent(const DropoutMaskComponent &other);

  virtual std::string Type() const { return "DropoutMaskComponent"; }
  virtual int32 Properties() const { return kRandomComponent; }
  // note: the matrix 'in' will be empty.
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  // backprop does nothing, there is nothing to backprop to and nothing
  // to update.
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const { }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const {
    desired_indexes->clear();  // requires no inputs.
  }

  // This function returns true if at least one of the input indexes used to
  // compute this output index is computable.
  // it's simple because this component requires no inputs.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const {
    if (used_inputs) used_inputs->clear();
    return true;
  }

  void SetDropoutProportion(BaseFloat p) { dropout_proportion_ = p; }

 private:

  // The output dimension
  int32 output_dim_;

  BaseFloat dropout_proportion_;

  bool continuous_;

  const DropoutMaskComponent &operator
  = (const DropoutMaskComponent &other); // Disallow.
};



/**
   GeneralDropoutComponent implements dropout, including a continuous
   variant where the thing we multiply is not just zero or one, but may
   be a continuous value.  It is intended for the case where you want to
   either share the dropout mask across all of time, or across groups
   of 't' values (e.g. the first block of 10 values gets one dropout
   mask, the second block of 10 gets another one, and so on).


   Configuration values accepted on the command line, with defaults:

       dim        Dimension of the input and output of this component,
                  e.g. 512

       block-dim  Block size if you want the dropout mask to repeat,
                  e.g. if dim=512 and you sent block-dim=128, there will
                  be a mask of dimension 128 repeated 4 times.  This can
                  be useful in convolutional setups.  If not specified,
                  block-dim defaults to 'dim'; if specified, it must be
                  a divisor of 'dim'.

       dropout-proportion=0.5   For conventional dropout, this is the proportion
                  of mask values that (in expectation) are zero; it would
                  normally be between 0 and 0.5.  The nonzero mask values
                  will be given values 1.0 / dropout_proportion, so that the
                  expected value is 1.0.  This behavior is different from
                  DropoutComponent and DropoutMaskComponent.

                  For continuous dropout (continuous==true), the dropout scales
                  will have values (1.0 + 2 * dropout-proportion *
                  Uniform[-1,1]).  This might seem like a strange choice, but it
                  means that dropout-proportion=0.5 gives us a kind of
                  'extremal' case where the dropout scales are distributed as
                  Uniform[0, 2] and we can pass in the dropout scale as if it
                  were a conventional dropout scale.

       time-period=0   This determines how the dropout mask interacts
                  with the time index (t).  In all cases, different sequences
                  (different 'n' values) get different dropout masks.
                  If time-period==0, then the dropout mask is shared across
                  all time values.  If you set time-period > 0, then the
                  dropout mask is shared across blocks of time values: for
                  instance if time-period==10, then we'll use one dropout
                  mask for t values 0 through 9, another for 10 through 19,
                  and so on.  In all cases, the dropout mask will be shared
                  across all 'x' values, although in most setups the x values
                  are just zero so this isn't very interesting.
                  If you set time-period==1 it would be similar to regular
                  dropout, and it would probably make more sense to just use the
                  normal DropoutComponent.

 */
class GeneralDropoutComponent: public RandomComponent {
 public:
  virtual int32 InputDim() const { return dim_; }

  virtual int32 OutputDim() const { return dim_; }

  virtual std::string Info() const;

  virtual void InitFromConfig(ConfigLine *cfl);

  GeneralDropoutComponent();

  GeneralDropoutComponent(const GeneralDropoutComponent &other);

  virtual std::string Type() const { return "GeneralDropoutComponent"; }
  virtual int32 Properties() const {
    return kRandomComponent|kPropagateInPlace|kBackpropInPlace|kUsesMemo|
        (block_dim_ != dim_ ? (kInputContiguous|kOutputContiguous) : 0);
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void DeleteMemo(void *memo) const {
    delete static_cast<CuMatrix<BaseFloat>*>(memo);
  }

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  void SetDropoutProportion(BaseFloat p) { dropout_proportion_ = p; }

 private:

  // Returns a random matrix of dimension 'num_mask_rows' by 'block_dim_'.  This
  // should not be called if test_mode_ is true or dropout_proportion_ is zero.
  CuMatrix<BaseFloat> *GetMemo(int32 num_mask_rows) const;


  // The input and output dimension
  int32 dim_;

  // block_dim_ must divide dim_.
  int32 block_dim_;

  // time_period_ can be zero if we want all 't' values to share the same
  // dropout mask, and a value more than zero if we want blocks of 't' values to
  // share the dropout mask.  For example, if time_period_ is 10, blocks of size
  // 10 frames will share the same dropout mask.
  int32 time_period_;

  BaseFloat dropout_proportion_;

  bool continuous_;

  bool test_mode_;

  const GeneralDropoutComponent &operator
  = (const GeneralDropoutComponent &other); // Disallow.
};

// This stores some precomputed indexes for GeneralDropoutComponent.
// This object is created for every instance of the Propagate()
// function in the compiled computation.
class GeneralDropoutComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:


  // num_mask_rows is the number of rows in the dropout-mask matrix;
  // it's num-cols is the block_dim_ of the component.
  int32 num_mask_rows;

  // 'indexes' is of dimension (the number of rows in the matrix we're doing
  // Propagate() or Backprop() on) times the (dim_ / block_dim_) of the
  // GeneralDropoutComponent.  Each value is in the range [0, num_mask_rows-1],
  // and each value is repeated (dim_ / block_dim_) times.  This array is used
  // to multiply the reshaped values or derivatives by the appropriate rows of
  // the dropout matrix.
  CuArray<int32> indexes;

  virtual ~GeneralDropoutComponentPrecomputedIndexes() { }

  ComponentPrecomputedIndexes *Copy() const {
    return new GeneralDropoutComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Read(std::istream &is, bool binary);

  virtual std::string Type() const {
    return "GeneralDropoutComponentPrecomputedIndexes";
  }
};






} // namespace nnet3
} // namespace kaldi


#endif
