// nnet3/nnet-convolutional-component.h

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

#ifndef KALDI_NNET3_NNET_CONVOLUTIONAL_COMPONENT_H_
#define KALDI_NNET3_NNET_CONVOLUTIONAL_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "nnet3/convolution.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-convolutional-component.h
///
/// This file can be viewed as 'overflow' from nnet-general-component.h.
/// It contains a number of components which implement some kind of
/// convolution.


/**
   TimeHeightConvolutionComponent implements 2-dimensional convolution where one
   of the dimensions of convolution (which traditionally would be called the
   width axis) is identified with time (i.e. the 't' component of Indexes).  For
   a deeper understanding of how this works, please see convolution.h.

   The following are the parameters accepted on the config line, with examples
   of their values.


   Parameters inherited from UpdatableComponent (see comment above declaration of
   UpdadableComponent in nnet-component-itf.h for details):
       learning-rate, learning-rate-factor, max-change

   Convolution-related parameters:

     num-filters-in   E.g. num-filters-in=32.  Number of input filters (the
                      number of separate versions of the input image).  The
                      filter-dim has stride 1 in the input and output vectors,
                      i.e. we order the input as (all-filters-for-height=0,
                      all-filters-for-height=1, etc.)
     num-filters-out  E.g. num-filters-out=64. The number of output filters (the
                      number of separate versions of the output image).  As with
                      the input, the filter-dim has stride 1.
     height-in        E.g. height-in=40.  The height of the input image.  The
                      width is not specified the the model level, as it's
                      identified with "t" and is called the time axis; the width
                      is determined by how many "t" values were available at the
                      input of the network, and how many were requested at the
                      output.
     height-out       E.g. height-out=40.  The height of the output image.  Will
                      normally be <= (the input height divided by
                      height-subsample-out).
     height-subsample-out E.g. height-subsample-out=2 (defaults to 1).
                      Subsampling factor on the height axis, e.g. you might set
                      this to 2 if you are doing subsampling on this layer,
                      which would involve discarding every other height
                      increment at the output.  There is no corresponding config
                      for the time dimension, as time subsampling is determined
                      by which 't' values you request at the output, together
                      with the values of 'time-offsets' at different layers of
                      the network.
     height-offsets   E.g. height-offsets=-1,0,1 The set of height offsets that
                      contribute to each output pixel: with the values -1,0,1,
                      height 10 at the output would see data from heights
                      9,10,11 at the input.  These values will normally be
                      consecutive.  Negative values imply zero-padding on the
                      bottom of the image, since output-height 0 is always
                      defined.  Zero-padding at the top of the image is
                      determined in a similar way (e.g. if height-in==height-out
                      and height-offsets=-1,0,1, then there is 1 pixel of
                      padding at the top and bottom of the image).
     time-offsets     E.g. time-offsets=-1,0,1 The time offsets that we require
                      at the input to produce a given output; these are
                      comparable to the offsets used in TDNNs.  Note that the
                      time axis is always numbered using an absolute scheme, so
                      that if there is subsampling on the time axis, then later
                      in the network you'll see time-offsets like "-2,0,2" or
                      "-4,0,4".  Subsampling on the time axis is not explicitly
                      specified but is implicit based on tracking dependencies.
     required-time-offsets E.g. required-time-offsets=0 (defaults to the same
                      value as time-offsets).  This is a set of time offsets,
                      which if specified must be a nonempty subset of
                      time-offsets; it determines whether zero-padding on the
                      time axis is allowed in cases where there is insufficient
                      input.  If not specified it defaults to the same as
                      'time-offsets', meaning there is no zero-padding on the
                      time axis.  Note: for speech tasks we tend to pad on the
                      time axis with repeats of the first or last frame, rather
                      than zero; and this is handled while preparing the data
                      and not by the core components of the nnet3 framework.  So
                      for speech tasks we wouldn't normally set this value.
     max-memory-mb    Maximum amount of temporary memory, in megabytes, that may
                      be used as temporary matrices in the convolution computation.
                      default=200.0.

   Initialization parameters:
      param-stddev    Standard deviation of the linear parameters of the
                      convolution.  Defaults to sqrt(1.0 / (num-filters-in *
                      num-height-offsets * num-time-offsets)), e.g.
                      sqrt(1.0/(64*3*3)) for a 3x3 kernel with 64 input
                      filters; this value will ensure that the output has
                      unit stddev if the input has unit stddev.
      bias-stddev     Standard deviation of bias terms.  default=0.0.
      init-unit       Defaults to false.  If true, it is required that
                      num-filters-in equal num-filters-out and there should
                      exist a (height, time) offset in the model equal to (0,
                      0).  We will initialize the parameter matrix to be
                      equivalent to the identity transform.  In this case,
                      param-stddev is ignored.


   Natural-gradient related options are below; you won't normally have to
   set these.

      use-natural-gradient e.g. use-natural-gradient=false (defaults to true).
                       You can set this to false to disable the natural gradient
                       updates (you won't normally want to do this).
      rank-out        Rank used in low-rank-plus-unit estimate of the Fisher-matrix
                      factor that has the dimension (num-rows of the parameter
                      space), which equals num-filters-out.  It
                      defaults to the minimum of 80, or half of the number of
                      output filters.
      rank-in         Rank used in low-rank-plus-unit estimate of the Fisher
                      matrix factor which has the dimension (num-cols of the
                      parameter matrix), which has the dimension
                      (num-input-filters * number of time-offsets * number of
                      height-offsets + 1), e.g. num-input-filters * 3 * 3 + 1
                      for a 3x3 kernel (the +1 is for the bias term).
                      It defaults to the minimum of 80, or half the
                      num-rows of the parameter matrix.  [note: I'm considering
                      decreasing this default to e.g. 40 or 20].
      num-minibatches-history
                      This is used setting the 'num_samples_history_in'
                      configuration value of the natural gradient object.
                      There is no concept of samples (frames) in the
                      application of natural gradient to the convnet, because
                      we do it all on the rows and columns of the derivative.
                      default=4.0.  A larger value means the Fisher matrix is
                      averaged over more minibatches (it's an exponential-decay
                      thing).
      alpha-out       Constant that determines how much we smooth the
                      Fisher-matrix factors with the unit matrix, for the
                      space of dimension num-filters-out.  default=4.0.
      alpha-in        Constant that determines how much we smooth the
                      Fisher-matrix factors with the unit matrix, for the
                      space of dimension (num-filters-in * num-time-offsets *
                      num-height-offsets + 1).  default=4.0.


   Example of a 3x3 kernel with no subsampling, and with zero-padding on both
   the the height and time axis, and where there has previously been no
   subsampling on the time axis:

     num-filters-in=32 num-filters-out=64 height-in=28 height-out=28 \
       height-subsample-out=1 height-offsets=-1,0,1 time-offsets=-1,0,1 \
       required-time-offsets=0

   Example of a 3x3 kernel with no subsampling, without zero-padding on
   either axis, and where there has *previously* been 2-fold subsampling
   on the time axis:

     num-filters-in=32 num-filters-out=64 height-in=20 height-out=18 \
       height-subsample-out=1 height-offsets=0,1,2 time-offsets=0,2,4

   [note: above, the choice to have the time-offsets start at zero rather than
   be centered is just a choice: it assumes that at the output of the network
   you would want to request indexes with t=0, while at the input the t values
   start from zero.]

   Example of a 3x3 kernel with subsampling on the height axis,
   without zero-padding on either axis, and where there has
   previously been 2-fold subsampling on the time axis:

     num-filters-in=32 num-filters-out=64 height-in=20 height-out=9 \
       height-subsample-out=2 height-offsets=0,1,2 time-offsets=0,2,4

  [note: subsampling on the time axis is not expressed in the layer itself:
  any time you increase the distance between time-offsets, like changing
  them from 0,1,2 to 0,2,4, you are effectively subsampling the previous
  layer-- assuming you only request the output at one time value or at
  multiples of the total subsampling factor.]

  Example of a 1x1 kernel:

     num-filters-in=64 num-filters-out=64 height-in=20 height-out=20 \
       height-subsample-out=1 height-offsets=0 time-offsets=0
 */
class TimeHeightConvolutionComponent: public UpdatableComponent {
 public:

  // The use of this constructor should only precede InitFromConfig()
  TimeHeightConvolutionComponent();

  // Copy constructor
  TimeHeightConvolutionComponent(const TimeHeightConvolutionComponent &other);

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "TimeHeightConvolutionComponent"; }
  virtual int32 Properties() const {
    return kUpdatableComponent|kLinearInParameters|
        kReordersIndexes|kBackpropAdds|kBackpropNeedsInput|
        kInputContiguous|kOutputContiguous;
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
    return new TimeHeightConvolutionComponent(*this);
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

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);


  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other):
        computation(other.computation) { }
    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "TimeHeightConvolutionComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }

    time_height_convolution::ConvolutionComputation computation;
  };

 private:

  void Check() const;

  // computes derived parameters required_time_offsets_ and all_time_offsets_.
  void ComputeDerived();

  // Function that updates linear_params_ and bias_params_, which
  // uses the natural gradient code.
  void UpdateNaturalGradient(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // Function that updates linear_params_ and bias_params_, which
  // does not use the natural gradient code.
  void UpdateSimple(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // Function called to initialize linear_params_ if init-unit=true in the config
  // line.
  void InitUnit();

  time_height_convolution::ConvolutionModel model_;

  // all_time_offsets_ is a copy of the corresponding variable in
  // model, stored as a vector instead of as a set for efficiency.
  std::vector<int32> all_time_offsets_;
  // time_offset_required_ is a vector with the same dimension as
  // 'all_time_offsets_', which is true if the corresponding time-offset
  // is a member of model_.required_time_offsets_.
  std::vector<bool> time_offset_required_;

  // the linear parameters of the convolution.
  // dimension is model_.ParamRows() by model.ParamCols(),
  // which equals num-filters-out by
  // (num-filters-in * patch-rows * patch-cols),
  // a.k.a.
  // (num-filters-in * num-time-offsets * num-height-offset).
  CuMatrix<BaseFloat> linear_params_;
  // the bias parameters of the convolution, dimension is
  // model_.num_filters_out.
  CuVector<BaseFloat> bias_params_;


  // Maximum amount of temporary memory in megabytes that is allowed to be used
  // in the convolution computation.  (this is per computation, but it's
  // released immediately after it's used, so it doesn't matter how many there
  // are).
  BaseFloat max_memory_mb_;

  // Controls whether or not the natural-gradient is used.
  // Note: even if this is true, if is_gradient_ (from the
  // UpdatableComponent base class) is true, we'll do the 'simple'
  // update that doesn't include natural gradient.
  bool use_natural_gradient_;

  // Apart from use_natural_gradient_, this is the only natural-gradient
  // config-line configuration variable that we store directly; the others are
  // stored inside the preconditioner_in_ and preconditioner_out_ objects.
  BaseFloat num_minibatches_history_;

  // Preconditioner for the input space, of dimension linear_params_.NumCols() +
  // 1 (the 1 is for the bias).  As with other natural-gradient objects, it's
  // not stored with the model on disk but is reinitialized each time we start
  // up.
  OnlineNaturalGradient preconditioner_in_;

  // Preconditioner for the output space, of dimension
  // linear_params_.NumRows().
  OnlineNaturalGradient preconditioner_out_;
};



/**
   TimeConvolutionComponent implements 1-dimensional convolution where the
   input vectors on successive "t" values are interpreted as successive,
   contiguous blocks of some kind of signal (e.g. a waveform), and we
   want to do convolution with a stride smaller than a single "t" value.
   For example, suppose we have a signal sampled as 8kHz and the input
   has dimension of 80, with 100 input "t" values per second.  And suppose
   we want to do convolution with a faster/smaller stride than 100 per second
   (e.g. 8 per frame, so a stride of 10 samples, meaning the output
   would consist of 8 blocks).

   To illustrate the terminology, we treat the fact that there 8 blocks in a
   single frame as "num-sub-frames=8", and the 10 samples in each sub-frame as
   "num-filters-in=10" (this is not explicitly specified, you specify the
   input-dim).


   The following are the parameters accepted on the config line, with examples
   of their values.

   Parameters inherited from UpdatableComponent (see comment above declaration of
   UpdadableComponent in nnet-component-itf.h for details):
       learning-rate, learning-rate-factor, max-change

   Convolution-related parameters:


      input-dim       The dimension of the input vectors; in the example we
                      gave above, this would be 80.
      sub-frames-per-frame  The number of sub-blocks in each frame-- in the example
                      we gave (8 blocks each with dimension 10), this would
                      be 8.  It must divide input-dim.   The num-filters-in
                      passed to the convolution component will be
                      input-dim/num-sub-frames (we don't expect you to think of
                      these as filters, they are just time samples, but
                      the convolution code thinks of them as filters).
      num-filters-out  E.g. num-filters-out=64. The number of output filters (the
                      number of separate versions of the output image).  The
                      output dimension of the component is num-sub-frames times
                      num-filters-out, consisting of 'num-sub-frames' blocks
                      each of dimension 'num-filters-out'.
      sub-frames-left-context  Number of sub-frames of left context that the
                      convolution sees as input.
      sub-frames-right-context  Number of sub-frames of right context that the
                      convolution sees as input.  Suppose we define
                      num-filters-in=input-dim/num-sub-frames, then
                      the number of parameters in the component will
                      will equal (num-sub-frames * num-filters-out *
                             (1 + sub-frames-left-context * sub-frames-right-context)).
      zero-pad        (default: true)  If true, this component will zero-pad
                      at the edges of available input, so that in order to compute
                      a single frame of output, we only need a single frame of input
                      with no context.  If false, the number of frames of context
                      like the input at t-1, t+1 and so on, will be determined
                      by sub-frames-left-context and sub-frames-right-context
                      (and, of course, num-sub-frames).


   Initialization parameters:
      param-stddev    Standard deviation of the linear parameters of the
                      convolution.  Defaults to sqrt(1.0 / (num-filters-in *
                      num-height-offsets * num-time-offsets)), e.g.
                      sqrt(1.0/(64*3*3)) for a 3x3 kernel with 64 input
                      filters; this value will ensure that the output has
                      unit stddev if the input has unit stddev.
      bias-stddev     Standard deviation of bias terms.  default=0.0.


   Natural-gradient related options are below; you won't normally have to
   set these.

      use-natural-gradient e.g. use-natural-gradient=false (defaults to true).
                       You can set this to false to disable the natural gradient
                       updates (you won't normally want to do this).
      rank-out        Rank used in low-rank-plus-unit estimate of the Fisher-matrix
                      factor that has the dimension (num-rows of the parameter
                      space), which equals num-filters-out.  It
                      defaults to the minimum of 80, or half of the number of
                      output filters.
      rank-in         Rank used in low-rank-plus-unit estimate of the Fisher
                      matrix factor which has the dimension (num-cols of the
                      parameter matrix), which has the dimension
                      (num-input-filters * number of time-offsets * number of
                      height-offsets + 1), e.g. num-input-filters * 3 * 3 + 1
                      for a 3x3 kernel (the +1 is for the bias term).
                      It defaults to the minimum of 80, or half the
                      num-rows of the parameter matrix.  [note: I'm considering
                      decreasing this default to e.g. 40 or 20].
      num-minibatches-history
                      This is used setting the 'num_samples_history_in'
                      configuration value of the natural gradient object.
                      There is no concept of samples (frames) in the
                      application of natural gradient to the convnet, because
                      we do it all on the rows and columns of the derivative.
                      default=4.0.  A larger value means the Fisher matrix is
                      averaged over more minibatches (it's an exponential-decay
                      thing).
      alpha-out       Constant that determines how much we smooth the
                      Fisher-matrix factors with the unit matrix, for the
                      space of dimension num-filters-out.  default=4.0.
      alpha-in        Constant that determines how much we smooth the
                      Fisher-matrix factors with the unit matrix, for the
                      space of dimension (num-filters-in * num-time-offsets *
                      num-height-offsets + 1).  default=4.0.


   Example  configuration:
      input-dim=80 num-sub-frames=8 num-filters-out=128 sub-frames-left-context=16 \
        sub-frames-right-context=15 zero-pad=false
 */
class TimeConvolutionComponent: public UpdatableComponent {
 public:

  // The use of this constructor should only precede InitFromConfig()
  TimeConvolutionComponent() { }

  // Copy constructor
  TimeConvolutionComponent(const TimeConvolutionComponent &other) = default;

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "TimeConvolutionComponent"; }
  virtual int32 Properties() const {
    return kUpdatableComponent|kLinearInParameters|
        kReordersIndexes|kBackpropAdds|kBackpropNeedsInput|
        kOutputContiguous;
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
    return new TimeConvolutionComponent(*this);
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

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);


  // This is public only because PrecomputedIndexes needs to be public.
  // Each Operation represents a single matrix-multiply.
  // Each sub-frame of the output will in the normal case have several
  // Operations to create it, each with a different frame-shift
  // on the input.  For each frame-shift of the input we
  // may see all of the sub-frames, or a subset of them, and
  // we'll usually be using a subset of the parameter matrix's
  // columns.
  // See the implementation of CreateOperations() for more details
  struct Operation {
    int32 output_start_col, output_num_cols;
    int32 input_start_row;  // input num-rows == output num-rows.
    int32 input_start_col, input_num_cols;
    int32 params_start_col;  // params num-cols == input_num_cols.

    void Write(std::ostream &os, bool binary) const;
    void Read(std::istream &is, bool binary);
  };

  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other) = default;
    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "TimeConvolutionComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }

    time_height_convolution::ConvolutionComputationIo io;
    std::vector<Operation> operations;
  };

 private:

  // Creates the structure of a computation.
  // There is a long comment by the implementation, with details
  // of how it works.
  void CreateOperations(
      const time_height_convolution::ConvolutionComputationIo &io,
      std::vector<Operation> *operations) const;

  void Check() const;

  // computes derived parameters frames_left_context_ and frames_right_context_.
  void ComputeDerived();

  // Function that updates linear_params_ and bias_params_, which
  // uses the natural gradient code.
  void UpdateNaturalGradient(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // Function that updates linear_params_ and bias_params_, which
  // does not use the natural gradient code.
  void UpdateSimple(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // This determines how many blocks ('sub-frames') we divide up each input
  // frame into.  The input sub-frames are thought of as being in a temporal
  // sequence that involves 't', so the next sub-frame after the
  // highest-numbered sub-frame at a particular 't', is sub-frame zero at 't+1'.
  int32 sub_frames_per_frame_;

  // The number of output filters.  The output dimension is sub_frames_per_frame_ *
  // num_filters_out_ (where the sub-frame index has the higher stride).
  int32 num_filters_out_;

  // The input dimension of each sub-frame.  InputDim() is
  // sub_frames_per_frame_ * samples_per_sub_frame_.
  int32 samples_per_sub_frame_;

  // the number of sub-frames of left context that the convolution sees (must be
  // >= 0).
  int32 sub_frames_left_context_;

  // the number of sub-frames of right context that the convolution sees (must be
  // >= 0).
  int32 sub_frames_right_context_;

  // this is derived from sub_frames_left_context_, it equals
  // (sub_frames_left_context_ + sub_frames_per_frame_ - 1) /
  //   sub_frames_per_frame_.
  int32 frames_left_context_;

  // this is derived from sub_frames_right_context_, it equals
  // (sub_frames_right_context_ + sub_frames_per_frame_ - 1) /
  //      sub_frames_per_frame_.
  int32 frames_right_context_;

  // If true, allow zero padding for all frames other than the central input
  // frame, so that the component does not require more frames than output
  // frames.  (However, it will still use them if they are present).
  bool zero_pad_;

  // the linear parameters of the convolution.
  // dimension is num_filters_out_ by
  // (total_context * samples_per_sub_frame_),
  // where total_context = (sub_frames_left_context_ + 1 + sub_frames_right_context_).
  // (and 'total_context' has the higher stride).
  CuMatrix<BaseFloat> linear_params_;
  // the bias parameters of the convolution, dimension is
  // model_.num_filters_out.
  CuVector<BaseFloat> bias_params_;

  // Controls whether or not the natural-gradient is used.
  // Note: even if this is true, if is_gradient_ (from the
  // UpdatableComponent base class) is true, we'll do the 'simple'
  // update that doesn't include natural gradient.
  bool use_natural_gradient_;

  // Apart from use_natural_gradient_, this is the only natural-gradient
  // config-line configuration variable that we store directly; the others are
  // stored inside the preconditioner_in_ and preconditioner_out_ objects.
  BaseFloat num_minibatches_history_;

  // Preconditioner for the input space, of dimension linear_params_.NumCols() +
  // 1 (the 1 is for the bias).  As with other natural-gradient objects, it's
  // not stored with the model on disk but is reinitialized each time we start
  // up.
  OnlineNaturalGradient preconditioner_in_;

  // Preconditioner for the output space, of dimension
  // linear_params_.NumRows().
  OnlineNaturalGradient preconditioner_out_;
};




} // namespace nnet3
} // namespace kaldi


#endif
