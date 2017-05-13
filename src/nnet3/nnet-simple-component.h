// nnet3/nnet-simple-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//           2014-2015  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen
//                2015  Daniel Galvez
//                2015  Tom Ko

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

#ifndef KALDI_NNET3_NNET_SIMPLE_COMPONENT_H_
#define KALDI_NNET3_NNET_SIMPLE_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-simple-component.h
///   This file contains declarations of components that are "simple", meaning
///   they don't care about the indexes they are operating on, produce one
///   output for one input, and return the kSimpleComponent flag in their
///   Properties(): for example, tanh and affine components.  In
///   nnet-general-component.h there are components that don't fit this pattern.

// This "nnet3" version of the p-norm component only supports the 2-norm.
class PnormComponent: public Component {
 public:
  void Init(int32 input_dim, int32 output_dim);
  explicit PnormComponent(int32 input_dim, int32 output_dim) {
    Init(input_dim, output_dim);
  }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kBackpropNeedsInput|kBackpropNeedsOutput;
  }
  PnormComponent(): input_dim_(0), output_dim_(0) { }
  virtual std::string Type() const { return "PnormComponent"; }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
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
  virtual Component* Copy() const { return new PnormComponent(input_dim_,
                                                              output_dim_); }

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  int32 input_dim_;
  int32 output_dim_;
};

// This component randomly zeros dropout_proportion of the input
// and the derivatives are backpropagated through the nonzero inputs.
// Typically this component used during training but not in test time.
// The idea is described under the name Dropout, in the paper
// "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".
class DropoutComponent : public RandomComponent {
 public:
  void Init(int32 dim, BaseFloat dropout_proportion = 0.0,
            bool dropout_per_frame = false);

  DropoutComponent(int32 dim, BaseFloat dropout = 0.0,
                   bool dropout_per_frame = false) {
    Init(dim, dropout, dropout_per_frame);
  }

  DropoutComponent(): dim_(0), dropout_proportion_(0.0),
                      dropout_per_frame_(false) { }

  virtual int32 Properties() const {
    return kLinearInInput|kBackpropInPlace|kSimpleComponent|kBackpropNeedsInput|
        kBackpropNeedsOutput|kRandomComponent;
  }
  virtual std::string Type() const { return "DropoutComponent"; }

  virtual void InitFromConfig(ConfigLine *cfl);

  virtual int32 InputDim() const { return dim_; }

  virtual int32 OutputDim() const { return dim_; }

  virtual void Read(std::istream &is, bool binary);

  // Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

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
  virtual Component* Copy() const { return new DropoutComponent(dim_,
                                               dropout_proportion_,
                                               dropout_per_frame_); }
  virtual std::string Info() const;

  void SetDropoutProportion(BaseFloat dropout_proportion) {
    dropout_proportion_ = dropout_proportion;
  }

 private:
  int32 dim_;
  /// dropout-proportion is the proportion that is dropped out,
  /// e.g. if 0.1, we set 10% to zero value.
  BaseFloat dropout_proportion_;
  bool dropout_per_frame_;
};

class ElementwiseProductComponent: public Component {
 public:
  void Init(int32 input_dim, int32 output_dim);
  explicit ElementwiseProductComponent(int32 input_dim, int32 output_dim) {
    Init(input_dim, output_dim);
  }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput;
  }
  ElementwiseProductComponent(): input_dim_(0), output_dim_(0) { }
  virtual std::string Type() const { return "ElementwiseProductComponent"; }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
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
  virtual Component* Copy() const { return new ElementwiseProductComponent(input_dim_,
                                                              output_dim_); }

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  int32 input_dim_;
  int32 output_dim_;
};

class NormalizeComponent: public Component {
 public:
 void Init(int32 input_dim, BaseFloat target_rms, bool add_log_stddev);
  explicit NormalizeComponent(int32 input_dim,
                              BaseFloat target_rms = 1.0,
                              bool add_log_stddev = false) {
    Init(input_dim, target_rms, add_log_stddev);
  }
  explicit NormalizeComponent(const NormalizeComponent &other);
  // note: there is some special code in NonlinerComponent::Info() that
  // specifically caters to this class.
  virtual int32 Properties() const {
    return (add_log_stddev_ ?
            kSimpleComponent|kBackpropNeedsInput|kBackpropAdds :
            kSimpleComponent|kBackpropNeedsInput|kPropagateInPlace|
            kBackpropAdds|kBackpropInPlace);
  }
  NormalizeComponent(): target_rms_(1.0), add_log_stddev_(false) { }
  virtual std::string Type() const { return "NormalizeComponent"; }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual Component* Copy() const { return new NormalizeComponent(*this); }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const {
    return (input_dim_ + (add_log_stddev_ ? 1 : 0));
  }
  virtual std::string Info() const;
 private:
  NormalizeComponent &operator = (const NormalizeComponent &other); // Disallow.
  enum { kExpSquaredNormFloor = -66 };
  static const BaseFloat kSquaredNormFloor;
  int32 input_dim_;
  BaseFloat target_rms_; // The target rms for outputs.
  // about 0.7e-20.  We need a value that's exactly representable in
  // float and whose inverse square root is also exactly representable
  // in float (hence, an even power of two).

  bool add_log_stddev_; // If true, log(max(epsi, sqrt(row_in^T row_in / D)))
                        // is an extra dimension of the output.
};


/*
   Implements the sigmoid nonlinearity, i.e. the function y = exp(-x).

   Configuration values accepted:
      dim              Dimension of this component, e.g. 1024

   Configuration values inherited from NonlinearComponent, and their
   local meanings:
      self-repair-lower-threshold e.g. self-repair-lower-threshold=0.05.  This
                    controls the self-repair mechanism, which for sigmoid units
                    consists of identifying units which are oversaturated (i.e.
                    usually close to -1 or +1) and nudging the inputs to be
                    closer to zero.  It gates on the average derivative of the
                    nonlinearity, which for sigmoid is a value between 0 and
                    0.25.  For units where the average function-derivative
                    accumulated during this iteration (job) of training is less
                    than this threshold, we activate self-repair, which consists
                    of adding (-self-repair-scale * (2*the output of the
                    nonlinearity - 1.0)) to the backpropagated derivatives.
                    This just happens to be a convenient-to-compute function
                    that's +1 for large negative inputs, and -1 for large positive
                    inputs, and smooth in between.
                    The default value of this is -1000, which the code internally
                    maps to 0.05 which is suitable for sigmoid units; if you do set it,
                    you can set it to a value like 0.025 or 0.075.
      self-repair-scale  Scale for the self-repair mechanism; see comments above.
                    default=0, but we usually set this to 1.0e-05 (or
                    occasionally 1.0e-04) in the scripts.

 */
class SigmoidComponent: public NonlinearComponent {
 public:
  explicit SigmoidComponent(const SigmoidComponent &other): NonlinearComponent(other) { }
  SigmoidComponent() { }
  virtual std::string Type() const { return "SigmoidComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kPropagateInPlace|kStoresStats;
  }
  virtual Component* Copy() const { return new SigmoidComponent(*this); }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
 private:
  // this function is called from Backprop code and only does something if the
  // self-repair-scale config value is set.
  void RepairGradients(const CuMatrixBase<BaseFloat> &out_value,
                       CuMatrixBase<BaseFloat> *in_deriv,
                       SigmoidComponent *to_update) const;

  SigmoidComponent &operator = (const SigmoidComponent &other); // Disallow.
};

/*
   Implements the tanh nonlinearity, i.e. the function y = tanh(x).

   Configuration values accepted:
      dim           Dimension of this component, e.g. 1024

   Configuration values inherited from NonlinearComponent, and their
   local meanings:
      self-repair-lower-threshold e.g. self-repair-lower-threshold=0.2.  This
                    controls the self-repair mechanism, which for tanh units
                    consists of identifying units which are oversaturated (i.e.
                    usually close to -1 or +1) and nudging the inputs to be
                    closer to zero.  It gates on the average derivative of
                    the nonlinearity, which for tanh is a value between 0 and 1.
                    For units where the average function-derivative accumulated
                    during this iteration (job) of training is less than
                    this threshold, we activate self-repair, which consists of
                    adding (-self-repair-scale * the output of the nonlinearity),
                    i.e. (-self-repair-scale * tanh(x)) to the backpropagated
                    derivatives.
                    The default value of this is -1000, which the code internally
                    maps to 0.2 which is suitable for tanh units; if you do set it,
                    you can set it to a value like 0.1 or 0.3.
      self-repair-scale  Scale for the self-repair mechanism; see comments above.
                    default=0, but we usually set this to 1.0e-05 (or
                    occasionally 1.0e-04) in the scripts.
 */
class TanhComponent: public NonlinearComponent {
 public:
  explicit TanhComponent(const TanhComponent &other): NonlinearComponent(other) { }
  TanhComponent() { }
  virtual std::string Type() const { return "TanhComponent"; }
  virtual Component* Copy() const { return new TanhComponent(*this); }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kPropagateInPlace|kStoresStats;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
 private:
  // this function is called from Backprop code and only does something if the
  // self-repair-scale config value is set.
  void RepairGradients(const CuMatrixBase<BaseFloat> &out_value,
                       CuMatrixBase<BaseFloat> *in_deriv,
                       TanhComponent *to_update) const;

  TanhComponent &operator = (const TanhComponent &other); // Disallow.
};


/*
   Implements the Rectified Linear Unit nonlinearity, a.k.a. ReLU.

   Configuration values accepted:
      dim              Dimension of this component, e.g. 1024

   Configuration values inherited from NonlinearComponent, and their
   local meanings:
      self-repair-lower-threshold e.g. self-repair-lower-threshold=0.05.  (Lower
                       threshold for self-repair, if set; in this case acts on
                       the average function-derivative, which is the proportion
                       of the time the output is > 0.  For any unit where the
                       average function-derivative is lower than this threshold,
                       we add 'self-repair-scale' to the backpropagated
                       derivatives in backprop.  There is no default
                       (default=-1000, which is interpreted specially).
      self-repair-upper-threshold e.g. self-repair-upper-threshold=0.95.
                       Like self-repair-lower-threshold, but controls self-repair
                       for units that are active *too* much of the time.  Units
                       whose average function-derivative exceeds this threshold
                       will have the negative of 'self-repair-scale' added to their
                       input derivatives in backprop.  There is no default
                       (default=-1000, which is interpreted specially).
      self-repair-scale  Scale for the self-repair mechanism; see comments for
                       self-repair-lower-threshold and self-repair-upper-threshold
                       for details.  default=0, but we usually set this to 1.0e-05
                       (or occasionally 1.0e-04) in the scripts.
 */
class RectifiedLinearComponent: public NonlinearComponent {
 public:
  explicit RectifiedLinearComponent(const RectifiedLinearComponent &other):
      NonlinearComponent(other) { }
  RectifiedLinearComponent() { }
  virtual std::string Type() const { return "RectifiedLinearComponent"; }
  virtual Component* Copy() const { return new RectifiedLinearComponent(*this); }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kBackpropNeedsOutput|kPropagateInPlace|
        kStoresStats;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
 private:
  // this function is called from Backprop code and only does something if the
  // self-repair-scale config value is set.
  void RepairGradients(CuMatrixBase<BaseFloat> *in_deriv,
                       RectifiedLinearComponent *to_update) const;

  RectifiedLinearComponent &operator = (const RectifiedLinearComponent &other); // Disallow.
};


class FixedAffineComponent;
class FixedScaleComponent;
class PerElementScaleComponent;
class PerElementOffsetComponent;

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// functions as a base-class for more specialized versions of
// AffineComponent.
class AffineComponent: public UpdatableComponent {
  friend class SoftmaxComponent; // Friend declaration relates to mixing up.
 public:

  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  AffineComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "AffineComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }


  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(const VectorBase<BaseFloat> &bias,
                         const MatrixBase<BaseFloat> &linear);
  const CuVector<BaseFloat> &BiasParams() const { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() const { return linear_params_; }
  explicit AffineComponent(const AffineComponent &other);
  // The next constructor is used in converting from nnet1.
  AffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                  const CuVectorBase<BaseFloat> &bias_params,
                  BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(std::string matrix_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.
  Component *CollapseWithNext(const AffineComponent &next) const ;
  Component *CollapseWithNext(const FixedAffineComponent &next) const;
  Component *CollapseWithNext(const FixedScaleComponent &next) const;
  Component *CollapseWithPrevious(const FixedAffineComponent &prev) const;

 protected:
  friend class NaturalGradientAffineComponent;
  // This function Update() is for extensibility; child classes may override
  // this, e.g. for natural gradient update.
  virtual void Update(
      const std::string &debug_info,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  // UpdateSimple is used when *this is a gradient.  Child classes may override
  // this if needed, but typically won't need to.
  virtual void UpdateSimple(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  const AffineComponent &operator = (const AffineComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
  CuVector<BaseFloat> bias_params_;
};

class RepeatedAffineComponent;

/// This class implements an affine transform using a block diagonal matrix
/// e.g., one whose weight matrix is all zeros except for blocks on the
/// diagonal. All these blocks have the same dimensions.
///  input-dim: num cols of block diagonal matrix.
///  output-dim: num rows of block diagonal matrix.
/// num-blocks: number of blocks in diagonal of the matrix.
/// num-blocks must divide both input-dim and output-dim
class BlockAffineComponent : public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return linear_params_.NumCols() * num_blocks_; }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  BlockAffineComponent() { }
  virtual std::string Type() const { return "BlockAffineComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
      kBackpropNeedsInput|kBackpropAdds;
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // BlockAffine-specific functions.
  void Init(int32 input_dim, int32 output_dim, int32 num_blocks,
            BaseFloat param_stddev, BaseFloat bias_mean,
            BaseFloat bias_stddev);
  explicit BlockAffineComponent(const BlockAffineComponent &other);
  explicit BlockAffineComponent(const RepeatedAffineComponent &rac);
 protected:
  // The matrix linear_params_ has a block structure, with num_blocks_ blocks of
  // equal size.  The blocks are stored in linear_params_ as
  // [ M
  //   N
  //   O ] but we actually treat it as the matrix:
  // [ M 0 0
  //   0 N 0
  //   0 0 O ]
  CuMatrix<BaseFloat> linear_params_;
  CuVector<BaseFloat> bias_params_;
  int32 num_blocks_;
 private:
  const BlockAffineComponent &operator = (const BlockAffineComponent &other); // Disallow.
};

class RepeatedAffineComponent: public UpdatableComponent {
 public:

  virtual int32 InputDim() const { return linear_params_.NumCols() * num_repeats_; }
  virtual int32 OutputDim() const { return linear_params_.NumRows() * num_repeats_; }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  RepeatedAffineComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "RepeatedAffineComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds|kInputContiguous|kOutputContiguous;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  const CuVector<BaseFloat> &BiasParams() const { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() const { return linear_params_; }
  explicit RepeatedAffineComponent(const RepeatedAffineComponent &other);

  void Init(int32 input_dim, int32 output_dim, int32 num_repeats,
            BaseFloat param_stddev, BaseFloat bias_mean,
            BaseFloat bias_stddev);
  friend BlockAffineComponent::BlockAffineComponent(const RepeatedAffineComponent &rac);
 protected:
  // This function Update(), called from backprop, is broken out for
  // extensibility to natural gradient update.
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // This function does nothing here but is redefined in child-class
  // NaturalGradientRepeatedAffineComponent.  This help avoid repeated code.
  virtual void SetNaturalGradientConfigs() { }

  const RepeatedAffineComponent &operator = (const RepeatedAffineComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
  CuVector<BaseFloat> bias_params_;
  int32 num_repeats_;
};

class NaturalGradientRepeatedAffineComponent: public RepeatedAffineComponent {
 public:
  // Use Init() to really initialize.
  NaturalGradientRepeatedAffineComponent() { }

  // Most of the public functions are inherited from RepeatedAffineComponent.
  virtual std::string Type() const {
    return "NaturalGradientRepeatedAffineComponent";
  }

  virtual Component* Copy() const;

  // Copy constructor
  explicit NaturalGradientRepeatedAffineComponent(
      const NaturalGradientRepeatedAffineComponent &other);
 private:
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  const NaturalGradientRepeatedAffineComponent &operator=(
      const NaturalGradientRepeatedAffineComponent &other); // Disallow.

  // Applies the default configuration to preconditioner_in_.
  virtual void SetNaturalGradientConfigs();

  // For efficiency reasons we only apply the natural gradient to the input
  // side, i.e. not to the space of output derivatives-- we believe the input
  // side is the more important side.  We don't make the natural-gradient
  // configurable; we just give it a reasonable configuration.
  // Instead of using the individual data-points, for efficiency reasons we use
  // the distribution of per-minibatch summed derivatives over each dimension of
  // the output space, as the source for the Fisher matrix.
  OnlineNaturalGradient preconditioner_in_;
};

class SoftmaxComponent: public NonlinearComponent {
 public:
  explicit SoftmaxComponent(const SoftmaxComponent &other):
      NonlinearComponent(other) { }
  SoftmaxComponent() { }
  virtual std::string Type() const { return "SoftmaxComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kStoresStats;
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
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
  virtual Component* Copy() const { return new SoftmaxComponent(*this); }
 private:
  SoftmaxComponent &operator = (const SoftmaxComponent &other); // Disallow.
};


/*
   Implements the log of a softmax nonlinearity, so it's the same
   as shifting each input vector by a constant offset so that, when
   exponentiated, it would sum to one.

   We usually use this in place of softmax because the log-scale
   output will not saturate.

   Configuration values accepted:
      dim            e.g. dim=8061.   Usually this is the last component
                     in a network, so 'dim' is the number of classes.
 */
class LogSoftmaxComponent: public NonlinearComponent {
 public:
  explicit LogSoftmaxComponent(const LogSoftmaxComponent &other):
      NonlinearComponent(other) { }
  LogSoftmaxComponent() { }
  virtual std::string Type() const { return "LogSoftmaxComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kStoresStats;
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

  virtual Component* Copy() const { return new LogSoftmaxComponent(*this); }
 private:
  LogSoftmaxComponent &operator = (const LogSoftmaxComponent &other); // Disallow.
};

/*
  Keywords: natural gradient descent, NG-SGD, naturalgradient.  For
  the top-level of the natural gradient code look here, and also in
  nnet-precondition-online.h.
  NaturalGradientAffineComponent is
  a version of AffineComponent that has a non-(multiple of unit) learning-rate
  matrix.  See nnet-precondition-online.h for a description of the technique.
  It is described, under the name Online NG-SGD, in the paper "Parallel
  training of DNNs with Natural Gradient and Parameter Averaging" (ICLR
  workshop, 2015) by Daniel Povey, Xiaohui Zhang and Sanjeev Khudanpur.

  Configuration values accepted by this component:

  Values inherited from UpdatableComponent (see its declaration in
  nnet-component-itf for details):
     learning-rate
     learning-rate-factor
     max-change

  Values used in initializing the component's parameters:
     input-dim             e.g. input-dim=1024.  The input dimension.
     output-dim            e.g. output-dim=1024.  The output dimension.
     param-stddev          e.g. param-stddev=0.025.  The standard deviation
                           used to randomly initialize the linear parameters
                           (as Gaussian random values * param-stddev).
                           Defaults to 1/sqrt(input-dim), which is Glorot
                           initialization.
     bias-stddev           e.g. bias-stddev=0.0.  The standard deviation
                           used to randomly initialize the bias parameters.
                           Defaults to 1.0 but we usually set it to 0.0
                           in the config.
     bias-mean             e.g. bias-mean=1.0.  Allows you to ininialize the
                           bias parameters with an offset.  Default is 0.0
                           which is normally suitable

     matrix                e.g. matrix=foo/bar/init.mat  May be used as an
                           alternative to (input-dim, output-dim, param-stddev,
                           bias-stddev, bias-mean) to initialize the parameters.
                           Dimension is output-dim by (input-dim + 1), last
                           column is interpreted as the bias.

   Options to the natural gradient (you won't normally have to set these,
   the defaults are suitable):

      num-samples-history   Number of frames used as the time-constant to
                            determine how 'up-to-date' the Fisher-matrix
                            estimates are.  Smaller -> more up-to-date, but more
                            noisy.  default=2000.
      alpha                 Constant that determines how much we smooth the
                            Fisher-matrix estimates with the unit matrix.
                            Larger means more smoothing. default=4.0
      rank-in               Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the input space.  default=20.
      rank-out              Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the output-derivative space.  default=80.
      update-period         Determines after with what frequency (in
                            minibatches) we update the Fisher-matrix estimates;
                            making this > 1 saves a little time in training.
                            default=4.
*/
class NaturalGradientAffineComponent: public AffineComponent {
 public:
  virtual std::string Type() const { return "NaturalGradientAffineComponent"; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev, BaseFloat bias_mean,
            int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha);
  void Init(int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history,
            BaseFloat alpha, std::string matrix_filename);
  // this constructor does not really initialize, use Init() or Read().
  NaturalGradientAffineComponent();
  void Resize(int32 input_dim, int32 output_dim);
  void InitFromConfig(ConfigLine *cfl);
  virtual std::string Info() const;
  virtual Component* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  // copy constructor
  explicit NaturalGradientAffineComponent(
      const NaturalGradientAffineComponent &other);
 private:
  // disallow assignment operator.
  NaturalGradientAffineComponent &operator= (
      const NaturalGradientAffineComponent&);

  // Configs for preconditioner.  The input side tends to be better conditioned ->
  // smaller rank needed, so make them separately configurable.
  int32 rank_in_;
  int32 rank_out_;
  int32 update_period_;
  BaseFloat num_samples_history_;
  BaseFloat alpha_;

  OnlineNaturalGradient preconditioner_in_;

  OnlineNaturalGradient preconditioner_out_;

  // Sets the configs rank, alpha and eta in the preconditioner objects,
  // from the class variables.
  void SetNaturalGradientConfigs();

  virtual void Update(
      const std::string &debug_info,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
};


/// FixedAffineComponent is an affine transform that is supplied
/// at network initialization time and is not trainable.
class FixedAffineComponent: public Component {
 public:
  FixedAffineComponent() { }
  virtual std::string Type() const { return "FixedAffineComponent"; }
  virtual std::string Info() const;

  // Copy constructor from AffineComponent-- can be used when we're done
  // training a particular part of the model and want to efficiently disable
  // further training.
  FixedAffineComponent(const AffineComponent &c);

  /// matrix should be of size input-dim+1 to output-dim, last col is offset
  void Init(const CuMatrixBase<BaseFloat> &matrix);

  // The ConfigLine cfl contains just the option matrix=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromConfig(ConfigLine *cfl);

  virtual int32 Properties() const { return kSimpleComponent|kBackpropAdds; }
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;


  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  // Function to provide access to linear_params_.
  const CuMatrix<BaseFloat> &LinearParams() const { return linear_params_; }
 protected:
  friend class AffineComponent;
  CuMatrix<BaseFloat> linear_params_;
  CuVector<BaseFloat> bias_params_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(FixedAffineComponent);
};

/// SumGroupComponent is used to sum up groups of posteriors.
/// It's used to introduce a kind of Gaussian-mixture-model-like
/// idea into neural nets.  This is basically a degenerate case of
/// MixtureProbComponent; we had to implement it separately to
/// be efficient for CUDA (we can use this one regardless whether
/// we have CUDA or not; it's the normal case we want anyway).
///
/// There are two forms of initialization in a config file: one
/// where the number of elements are specified for each group
/// individually as a vector, and one where only the total input
/// dimension and the output dimension (number of groups) is specified.
/// The second is used when all groups have the same size.
class SumGroupComponent: public Component {
public:
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  void Init(const std::vector<int32> &sizes); // the vector is of the input dim
                                              // (>= 1) for each output dim.
  void Init(int32 input_dim, int32 output_dim);
  void GetSizes(std::vector<int32> *sizes) const; // Get a vector saying, for
                                                  // each output-dim, how many
                                                  // inputs were summed over.
  virtual void InitFromConfig(ConfigLine *cfl);
  SumGroupComponent() { }
  virtual std::string Type() const { return "SumGroupComponent"; }
  virtual int32 Properties() const { return kSimpleComponent|kLinearInInput; }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(SumGroupComponent);
  // Note: Int32Pair is just struct{ int32 first; int32 second }; it's defined
  // in cu-matrixdim.h as extern "C" which is needed for the CUDA interface.
  CuArray<Int32Pair> indexes_; // for each output index, the (start, end) input
                               // index.
  CuArray<int32> reverse_indexes_; // for each input index, the output index.
  int32 input_dim_;
  int32 output_dim_;
};


/// FixedScaleComponent applies a fixed per-element scale; it's similar
/// to the Rescale component in the nnet1 setup (and only needed for nnet1
/// model conversion).
class FixedScaleComponent: public Component {
 public:
  FixedScaleComponent() { }
  virtual std::string Type() const { return "FixedScaleComponent"; }
  virtual std::string Info() const;
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kPropagateInPlace|kBackpropInPlace;
  }

  void Init(const CuVectorBase<BaseFloat> &scales);

  // The ConfigLine cfl contains only the option scales=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromConfig(ConfigLine *cfl);

  virtual int32 InputDim() const { return scales_.Dim(); }
  virtual int32 OutputDim() const { return scales_.Dim(); }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  friend class AffineComponent;  // necessary for collapse
  CuVector<BaseFloat> scales_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(FixedScaleComponent);
};


/// FixedBiasComponent applies a fixed per-element bias; it's similar
/// to the AddShift component in the nnet1 setup (and only needed for nnet1
/// model conversion.
class FixedBiasComponent: public Component {
 public:
  FixedBiasComponent() { }
  virtual std::string Type() const { return "FixedBiasComponent"; }
  virtual std::string Info() const;

  virtual int32 Properties() const {
    return kSimpleComponent|kPropagateInPlace|kBackpropInPlace;
  }

  void Init(const CuVectorBase<BaseFloat> &scales);

  // The ConfigLine cfl contains only the option bias=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual int32 InputDim() const { return bias_.Dim(); }
  virtual int32 OutputDim() const { return bias_.Dim(); }
  using Component::Propagate; // to avoid name hiding
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  CuVector<BaseFloat> bias_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(FixedBiasComponent);
};

/** NoOpComponent just duplicates its input.  We don't anticipate this being used
    very often, but it may sometimes make your life easier
    The only config parameter it accepts is 'dim', e.g. 'dim=400'.
*/
class NoOpComponent: public NonlinearComponent {
 public:
  explicit NoOpComponent(const NoOpComponent &other): NonlinearComponent(other) { }
  NoOpComponent() { }
  virtual std::string Type() const { return "NoOpComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kPropagateInPlace;
  }
  virtual Component* Copy() const { return new NoOpComponent(*this); }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
 private:
  NoOpComponent &operator = (const NoOpComponent &other); // Disallow.
};

/**  SumBlockComponent sums over blocks of its input: for instance, if
     you create one with the config "input-dim=400 output-dim=100",
     its output will be the sum over the 4 100-dimensional blocks of
     the input.

     The "scale" config parameter may be used if you want to do averaging
     instead of summing, e.g. "input-dim=400 output-dim=100 scale=0.25"
     will accomplish averaging.

     Accepted values on its config-file line are:
        input-dim  The input dimension.  Required.
        output-dim  The block dimension.  Required.  Must divide input-dim.
        scale      A scaling factor on the output.  Defaults to 1.0.
 */
class SumBlockComponent: public Component {
 public:
  explicit SumBlockComponent(const SumBlockComponent &other);
  SumBlockComponent() { }
  virtual std::string Type() const { return "SumBlockComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kPropagateAdds|kBackpropAdds;
  }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Info() const;
  virtual Component* Copy() const { return new SumBlockComponent(*this); }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
 private:
  int32 input_dim_;
  int32 output_dim_;
  BaseFloat scale_;
  SumBlockComponent &operator = (const SumBlockComponent &other); // Disallow.
};


// ClipGradientComponent just duplicates its input, but clips gradients
// during backpropagation if they cross a predetermined threshold.
// This component will be used to prevent gradient explosion problem in
// recurrent neural networks
class ClipGradientComponent: public Component {
 public:
  ClipGradientComponent(int32 dim, BaseFloat clipping_threshold,
                        bool norm_based_clipping,
                        BaseFloat self_repair_clipped_proportion_threshold,
                        BaseFloat self_repair_target,
                        BaseFloat self_repair_scale,
                        int32 num_clipped,
                        int32 count,
                        int32 num_self_repaired,
                        int32 num_backpropped) {
    Init(dim, clipping_threshold, norm_based_clipping,
         self_repair_clipped_proportion_threshold,
         self_repair_target,
         self_repair_scale,
         num_clipped, count,
         num_self_repaired, num_backpropped);}

  ClipGradientComponent(): dim_(0), clipping_threshold_(-1),
    norm_based_clipping_(false),
    self_repair_clipped_proportion_threshold_(1.0),
    self_repair_target_(0.0),
    self_repair_scale_(0.0),
    num_clipped_(0), count_(0),
    num_self_repaired_(0), num_backpropped_(0) { }

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void InitFromConfig(ConfigLine *cfl);
  void Init(int32 dim, BaseFloat clipping_threshold, bool norm_based_clipping,
            BaseFloat self_repair_clipped_proportion_threshold,
            BaseFloat self_repair_target,
            BaseFloat self_repair_scale,
            int32 num_clipped, int32 count,
            int32 num_self_repaired, int32 num_backpropped);

  virtual std::string Type() const { return "ClipGradientComponent"; }

  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kPropagateInPlace|kBackpropInPlace|
           kBackpropNeedsInput;
  }

  virtual void ZeroStats();

  virtual Component* Copy() const {
    return new ClipGradientComponent(dim_,
                                     clipping_threshold_,
                                     norm_based_clipping_,
                                     self_repair_clipped_proportion_threshold_,
                                     self_repair_target_,
                                     self_repair_scale_,
                                     num_clipped_,
                                     count_,
                                     num_self_repaired_,
                                     num_backpropped_);}

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Info() const;
  virtual ~ClipGradientComponent() {
    if (num_self_repaired_ > 0)
      KALDI_LOG << "ClipGradientComponent(node_name=" << debug_info_
                << ")'s self-repair was activated " << num_self_repaired_
                << " time(s) out of " << num_backpropped_
                << " times of calling Backprop() in this training job.";
  }
 private:
  int32 dim_;  // input/output dimension
  BaseFloat clipping_threshold_;  // threshold to be used for clipping
                                  // could correspond to max-row-norm (if
                                  // norm_based_clipping_ == true) or
                                  // max-absolute-value (otherwise)
  bool norm_based_clipping_;  // if true the max-row-norm will be clipped
                              // else element-wise absolute value clipping is
                              // done

  // some configuration values relating to self-repairing.
  BaseFloat self_repair_clipped_proportion_threshold_; // the threshold of
                                                       // clipped-proportion
                                                       // for self-repair to be
                                                       // activated
  BaseFloat self_repair_target_; // the target value towards which self-repair
                                 // is trying to set for in-deriv
  BaseFloat self_repair_scale_;  // constant scaling the self-repair vector
  std::string debug_info_;   // component-node name, used in the destructor to
                             // print out stats of self-repair

  // this function is called from Backprop code, and only does something if the
  // self-repair-scale config value is set and the current clipped proportion
  // exceeds the threshold. What it does is to add a term to in-deriv that
  // forces the input to the ClipGradientComponent to be close to some small
  // value (e.g., 0.0 or 0.5, depending on what the input is, e.g.,
  // Sigmoid or Tanh or Affine). The hope is that if the input is forced to be
  // small, the parameters on the path will also tend to be small, which may
  // help tamp down the divergence caused by gradient explosion.
  void RepairGradients(const std::string &debug_info,
                       const CuMatrixBase<BaseFloat> &in_value,
                       CuMatrixBase<BaseFloat> *in_deriv,
                       ClipGradientComponent *to_update) const;

  ClipGradientComponent &operator =
      (const ClipGradientComponent &other); // Disallow.

 protected:
  // variables to store stats
  // An element corresponds to rows of derivative matrix, when
  // norm_based_clipping_ is true,
  // else it corresponds to each element of the derivative matrix
  // Note: no stats are stored when norm_based_clipping_ is false
  int32 num_clipped_;  // number of elements which were clipped
  int32 count_;  // number of elements which were processed
  int32 num_self_repaired_; // number of times self-repair is activated
  int32 num_backpropped_; //number of times backprop is called

};

/** PermuteComponent changes the order of the columns (i.e. the feature or
    activation dimensions).  Output dimension i is mapped to input dimension
    column_map_[i], so it's like doing:
      for each row:
        for each feature/activation dimension i:
          output(row, i) = input(row, column_map_[i]).

*/
class PermuteComponent: public Component {
 public:
  PermuteComponent()  {}
  PermuteComponent(const std::vector<int32> &column_map) { Init(column_map); }

  virtual int32 InputDim() const { return column_map_.Dim(); }
  virtual int32 OutputDim() const { return column_map_.Dim(); }
  virtual void InitFromConfig(ConfigLine *cfl);
  void Init(const std::vector<int32> &column_map);

  virtual std::string Type() const { return "PermuteComponent"; }

  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput;
  }

  virtual void ZeroStats() {}

  virtual Component* Copy() const;

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Scale(BaseFloat scale) {}
  virtual void Add(BaseFloat alpha, const Component &other) {}
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Info() const;
 private:
  // computes the reverse column map.  Must not be called if column_map_.Dim()
  // == 0
  void ComputeReverseColumnMap();
  CuArray<int32> column_map_;
  // the following is a derived variable, not written to disk.
  // It is used in backprop.
  CuArray<int32> reverse_column_map_;
  PermuteComponent &operator =
      (const PermuteComponent &other); // Disallow.
};




// PerElementScaleComponent scales each dimension of its input with a separate
// trainable scale; it's like a linear component with a diagonal matrix.
class PerElementScaleComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return scales_.Dim(); }
  virtual int32 OutputDim() const { return scales_.Dim(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  PerElementScaleComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "PerElementScaleComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInInput|
        kLinearInParameters|kBackpropNeedsInput|kPropagateInPlace;
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  explicit PerElementScaleComponent(const PerElementScaleComponent &other);

  void Init(int32 dim, BaseFloat param_mean, BaseFloat param_stddev);
  void Init(std::string vector_filename);

 protected:
  friend class AffineComponent;  // necessary for collapse
  // This function Update() is for extensibility; child classes may override
  // this, e.g. for natural gradient update.
  virtual void Update(
      const std::string &debug_info,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  // UpdateSimple is used when *this is a gradient.  Child classes may override
  // this if needed, but typically won't need to.
  virtual void UpdateSimple(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  const PerElementScaleComponent &operator
      = (const PerElementScaleComponent &other); // Disallow.
  CuVector<BaseFloat> scales_;
};

/*
  PerElementOffsetComponent offsets each dimension of its input with a separate
  trainable bias; it's like an affine component with fixed weight matrix which
  is always equal to I.

  Accepted values on its config line, with defaults if applicable.

     vector           If specified, the offsets will be read from this file ('vector'
                      is interpreted as an rxfilename).

     dim              If 'vector' is not specified, you should specify the
                      dimension 'dim', and will be randomly initialized according
                      to 'param-mean' and 'param-stddev'.
     param-mean=0.0   Mean of randomly initialized offset parameters.
     param-stddev=0.0 Standard deviation of randomly initialized offset parameters.

*/
class PerElementOffsetComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return offsets_.Dim(); }
  virtual int32 OutputDim() const { return offsets_.Dim(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  PerElementOffsetComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "PerElementOffsetComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|
           kBackpropInPlace|kPropagateInPlace;
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


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  explicit PerElementOffsetComponent(const PerElementOffsetComponent &other);

  void Init(int32 dim, BaseFloat param_mean,
            BaseFloat param_stddev);
  void Init(std::string vector_filename);

 protected:
  const PerElementOffsetComponent &operator
      = (const PerElementOffsetComponent &other); // Disallow.
  CuVector<BaseFloat> offsets_;
};


// ConstantFunctionComponent returns constant function of its input,
// i.e. its output does not depend on its input.  It is the same as
// an affine component with the linear term fixed at zero.
// It is optionally trainable, and optionally you can use natural
// gradient.  The input is required only because it's more convenient
// to make SimpleComponents [but see ConstantComponent, which requires
// no inputs].
class ConstantFunctionComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_.Dim(); }

  virtual std::string Info() const;
  // possible parameter values with their defaults:
  // input-dim=-1 is-updatable=true use-natural-gradient=true output-dim=-1
  // output-mean=0 output-stddev=0
  virtual void InitFromConfig(ConfigLine *cfl);

  ConstantFunctionComponent();

  ConstantFunctionComponent(const ConstantFunctionComponent &other);

  virtual std::string Type() const { return "ConstantFunctionComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|
        (is_updatable_ ? kUpdatableComponent|kLinearInParameters : 0) |
        (InputDim() == OutputDim() ? kPropagateInPlace: 0) |
        kBackpropAdds;
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

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
 private:
  int32 input_dim_;
  // the output value-- a vector.
  CuVector<BaseFloat> output_;

  bool is_updatable_;
  // if true, and if updatable, do natural-gradient update.
  bool use_natural_gradient_;
  OnlineNaturalGradient preconditioner_;

  const ConstantFunctionComponent &operator
  = (const ConstantFunctionComponent &other); // Disallow.
};



// NaturalGradientPerElementScaleComponent is like PerElementScaleComponent but
// it uses a natural gradient update for the per-element scales, and enforces a
// maximum amount of change per minibatch, for stability.
class NaturalGradientPerElementScaleComponent: public PerElementScaleComponent {
 public:

  virtual std::string Info() const;

  virtual void InitFromConfig(ConfigLine *cfl);

  NaturalGradientPerElementScaleComponent() { } // use Init to really initialize.
  virtual std::string Type() const {
    return "NaturalGradientPerElementScaleComponent";
  }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions that are specific to this class:
  explicit NaturalGradientPerElementScaleComponent(
      const NaturalGradientPerElementScaleComponent &other);

  void Init(int32 dim, BaseFloat param_mean,
            BaseFloat param_stddev, int32 rank, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha);
  void Init(std::string vector_filename,
            int32 rank, int32 update_period, BaseFloat num_samples_history,
            BaseFloat alpha);

 private:
  // unlike the NaturalGradientAffineComponent, there is only one dimension to
  // consider as the parameters are a vector not a matrix, so we only need one
  // preconditioner.
  // The preconditioner stores its own configuration values; we write and read
  // these, but not the preconditioner object itself.
  OnlineNaturalGradient preconditioner_;

  // Override of the parent-class Update() function, called only
  // if this->is_gradient_ = false; this implements the natural
  // gradient update.
  virtual void Update(
      const std::string &debug_info,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  const NaturalGradientPerElementScaleComponent &operator
      = (const NaturalGradientPerElementScaleComponent &other); // Disallow.
};

/**
 * ConvolutionalComponent implements 2d-convolution.
 * It uses 3D filters on 3D inputs, but the 3D filters hop only over
 * 2 dimensions as it has same size as the input along the 3rd dimension.
 * Input : A matrix where each row is a  vectorized 3D-tensor.
 *        The 3D tensor has dimensions
 *        x: (e.g. time)
 *        y: (e.g. frequency)
 *        z: (e.g. channels like features/delta/delta-delta)
 *
 *        The component supports input vectorizations of type zyx and yzx.
 *        The default vectorization type is zyx.
 *        e.g. for input vectorization of type zyx the input is vectorized by
 *        spanning axes z, y and x of the tensor in that order.
 *        Given 3d tensor A with sizes (2, 2, 2) along the three dimensions
 *        the zyx vectorized input looks like
 *  A(0,0,0) A(0,0,1) A(0,1,0) A(0,1,1) A(1,0,0) A(1,0,1) A(1,1,0) A(1,1,1)
 *
 *
 * Output : The output is also a 3D tensor vectorized in the zyx format.
 *          The channel axis (z) in the output corresponds to the output of
 *          different filters. The first channel corresponds to the first filter
 *          i.e., first row of the filter_params_ matrix.
 *
 * Note: The component has to support yzx input vectorization as the binaries
 * like add-deltas generate yz vectorized output. These input vectors are
 * concatenated using the Append descriptor across time steps to form a yzx
 * vectorized 3D tensor input.
 * e.g. Append(Offset(input, -1), input, Offset(input, 1))
 *
 *
 * For information on the hyperparameters and parameters of this component see
 * the variable declarations.
 *
 * Propagation:
 * ------------
 * Convolution operation consists of a dot-products between the filter tensor
 * and input tensor patch, for various shifts of filter tensor along the x and y
 * axes input tensor. (Note: there is no shift along z-axis as the filter and
 * input tensor have same size along this axis).
 *
 * For a particular shift (i,j) of the filter tensor
 * along input tensor dimensions x and y, the elements of the input tensor which
 * overlap with the filter form the input tensor patch. This patch is vectorized
 * in zyx format. All the patches corresponding to various samples in the
 * mini-batch are stacked into a matrix, where each row corresponds to one
 * patch. Let this matrix be represented by X_{i,j}. The dot products with
 * various filters are computed simultaneously by computing the matrix product
 * with the filter_params_ matrix (W)
 * Y_{i,j} = X_{i,j}*W^T.
 * Each row of W corresponds to one filter 3D tensor vectorized in zyx format.
 *
 * All the matrix products corresponding to various shifts (i,j) of the
 * filter tensor are computed simultaneously using the AddMatMatBatched
 * call of CuMatrixBase class.
 *
 * BackPropagation:
 * ----------------
 *  Backpropagation to compute the input derivative (\nabla X_{i,j})
 *  consists of the a series of matrix products.
 *  \nablaX_{i,j} = \nablaY_{i,j}*W where \nablaY_{i,j} corresponds to the
 *   output derivative for a particular shift of the filter.
 *
 *   Once again these matrix products are computed simultaneously.
 *
 * Update:
 * -------
 *  The weight gradient is computed as
 *  \nablaW = \Sum_{i,j} (X_{i,j}^T *\nablaY_{i,j})
 *
 */
class ConvolutionComponent: public UpdatableComponent {
 public:
  enum TensorVectorizationType  {
    kYzx = 0,
    kZyx = 1
  };

  ConvolutionComponent();
  // constructor using another component
  ConvolutionComponent(const ConvolutionComponent &component);
  // constructor using parameters
  ConvolutionComponent(
    const CuMatrixBase<BaseFloat> &filter_params,
    const CuVectorBase<BaseFloat> &bias_params,
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step,
    TensorVectorizationType input_vectorization,
    BaseFloat learning_rate);

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "ConvolutionComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kBackpropNeedsInput|
           kBackpropAdds|kPropagateAdds;
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  void Update(const std::string &debug_info,
              const CuMatrixBase<BaseFloat> &in_value,
              const CuMatrixBase<BaseFloat> &out_deriv,
              const std::vector<CuSubMatrix<BaseFloat> *>& out_deriv_batch);


  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  void SetParams(const VectorBase<BaseFloat> &bias,
                 const MatrixBase<BaseFloat> &filter);
  const CuVector<BaseFloat> &BiasParams() const { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() const { return filter_params_; }
  void Init(int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
            int32 filt_x_dim, int32 filt_y_dim,
            int32 filt_x_step, int32 filt_y_step, int32 num_filters,
            TensorVectorizationType input_vectorization,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  // there is no filt_z_dim parameter as the length of the filter along
  // z-dimension is same as the input
  void Init(int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
            int32 filt_x_dim, int32 filt_y_dim,
            int32 filt_x_step, int32 filt_y_step,
            TensorVectorizationType input_vectorization,
            std::string matrix_filename);

  // resize the component, setting the parameters to zero, while
  // leaving any other configuration values the same
  void Resize(int32 input_dim, int32 output_dim);

  void Update(const std::string &debug_info,
              const CuMatrixBase<BaseFloat> &in_value,
              const CuMatrixBase<BaseFloat> &out_deriv);


 private:
  int32 input_x_dim_;   // size of the input along x-axis
                        // (e.g. number of time steps)

  int32 input_y_dim_;   // size of input along y-axis
                        // (e.g. number of mel-frequency bins)

  int32 input_z_dim_;   // size of input along z-axis
                        // (e.g. number of channels is 3 if the input has
                        // features + delta + delta-delta features

  int32 filt_x_dim_;    // size of the filter along x-axis

  int32 filt_y_dim_;    // size of the filter along y-axis

  // there is no filt_z_dim_ as it is always assumed to be
  // the same as input_z_dim_

  int32 filt_x_step_;   // the number of steps taken along x-axis of input
                        //  before computing the next dot-product
                        //  of filter and input

  int32 filt_y_step_;   // the number of steps taken along y-axis of input
                        // before computing the next dot-product of the filter
                        // and input

  // there is no filt_z_step_ as only dot product is possible along this axis

  TensorVectorizationType input_vectorization_; // type of vectorization of the
  // input 3D tensor. Accepts zyx and yzx formats

  CuMatrix<BaseFloat> filter_params_;
  // the filter (or kernel) matrix is a matrix of vectorized 3D filters
  // where each row in the matrix corresponds to one filter.
  // The 3D filter tensor is vectorizedin zyx format.
  // The first row of the matrix corresponds to the first filter and so on.
  // Keep in mind the vectorization type and order of filters when using file
  // based initialization.

  CuVector<BaseFloat> bias_params_;
  // the filter-specific bias vector (i.e., there is a seperate bias added
  // to the output of each filter).
  bool is_gradient_;

  void InputToInputPatches(const CuMatrixBase<BaseFloat>& in,
                           CuMatrix<BaseFloat> *patches) const;
  void InderivPatchesToInderiv(const CuMatrix<BaseFloat>& in_deriv_patches,
                               CuMatrixBase<BaseFloat> *in_deriv) const;
  const ConvolutionComponent &operator = (const ConvolutionComponent &other); // Disallow.
};


/*
  LstmNonlinearityComponent is a component that implements part of an LSTM, by
  combining together the sigmoids and tanh's, plus some diagonal terms, into
  a single block.
  We will refer to the LSTM formulation used in

  Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling"
  by H. Sak et al,
  http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf.

  Suppose the cell dimension is C.  Then outside this component, we compute
  the 4 * C-dimensional quantity consisting of 4 blocks as follows, by a single
  matrix multiplication:

  i_part = W_{ix} x_t + W_{im} m_{t-1} + b_i
  f_part = W_{fx} x_t + W_{fm} m_{t-1} + b_f
  c_part = W_{cx} x_t + W_{cm} m_{t-1} + b_c
  o_part = W_{cx} x_t + W_{om} m_{t-1} + b_o

  The part of the computation that takes place in this component is as follows.
  Its input is of dimension 5C [however, search for 'dropout' below],
  consisting of 5 blocks: (i_part, f_part, c_part, o_part, and c_{t-1}).  Its
  output is of dimension 2C, consisting of 2 blocks: c_t and m_t.

  To recap: the input is (i_part, f_part, c_part, o_part, c_{t-1}); the output is (c_t, m_t).

  This component has parameters, 3C of them in total: the diagonal matrices w_i, w_f
  and w_o.


  In the forward pass (Propagate), this component computes the following:

     i_t = Sigmoid(i_part + w_{ic}*c_{t-1})   (1)
     f_t = Sigmoid(f_part + w_{fc}*c_{t-1})   (2)
     c_t = f_t*c_{t-1} + i_t * Tanh(c_part)   (3)
     o_t = Sigmoid(o_part + w_{oc}*c_t)       (4)
     m_t = o_t * Tanh(c_t)                    (5)
    # note: the outputs are just c_t and m_t.

  [Note regarding dropout: optionally the input-dimension may be 5C + 3 instead
  of 5C in this case, the last three input dimensions will be interpreted as
  per-frame dropout masks on i_t, f_t and o_t respectively, so that on the RHS of
  (3), i_t is replaced by i_t * i_t_scale, and likewise for f_t and o_t.]

  The backprop is as you would think, but for the "self-repair" we need to pass
  in additional vectors (of the same dim as the parameters of the layer) that
  dictate whether or not we add an additional term to the backpropagated
  derivatives.  (This term helps force the input to the nonlinearities into the
  range where the derivatives are not too small).

  This component stores stats of the same form as are normally stored by the
  StoreStats() functions for the sigmoid and tanh units, i.e. averages of the
  activations and derivatives, but this is done inside the Backprop() functions.
  [the StoreStats() functions don't take the input data as an argument, so
  storing this data that way is impossible, and anyway it's more efficient to
  do it as part of backprop.]

  Configuration values accepted:
         cell-dim          e.g. cell-dim=1024  Cell dimension.  The input
                          dimension of this component is cell-dim * 5, and the
                          output dimension is cell-dim * 2.  Note: this
                          component implements only part of the LSTM layer,
                          see comments above.
         param-stddev     Standard deviation for random initialization of
                          the diagonal matrices (AKA peephole connections).
                          default=1.0, which is probably too high but
                          we couldn't see any reliable gain from decreasing it.
         tanh-self-repair-threshold   Equivalent to the self-repair-lower-threshold
                          in a TanhComponent; applies to both the tanh nonlinearities.
                          default=0.2, you probably won't want to changethis.
         sigmoid-self-repair-threshold   Equivalent to self-repair-lower-threshold
                          in a SigmoidComponent; applies to all three of the sigmoid
                          nonlinearities.  default=0.05, you probably won't want to
                          change this.
         self-repair-scale Equivalent to the self-repair-scale in a SigmoidComponent
                          or TanhComponent; applies to both the sigmoid and tanh
                          nonlinearities.  default=1.0e-05, which you probably won't
                          want to change unless dealing with an objective function
                          that has smaller or larger dynamic range than normal, in
                          which case you might want to make it smaller or larger.
*/
class LstmNonlinearityComponent: public UpdatableComponent {
 public:

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;
  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  LstmNonlinearityComponent(): use_dropout_(false) { }
  virtual std::string Type() const { return "LstmNonlinearityComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kBackpropNeedsInput;
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void ZeroStats();

  // Some functions that are specific to this class:
  explicit LstmNonlinearityComponent(
      const LstmNonlinearityComponent &other);

  void Init(int32 cell_dim, bool use_dropout,
            BaseFloat param_stddev,
            BaseFloat tanh_self_repair_threshold,
            BaseFloat sigmoid_self_repair_threshold,
            BaseFloat self_repair_scale);

 private:

  // Initializes the natural-gradient object with the configuration we
  // use for this object, which for now is hardcoded at the C++ level.
  void InitNaturalGradient();

  // Notation: C is the cell dimension; it equals params_.NumCols().

  // The dimension of the parameter matrix is (3 x C);
  // it contains the 3 diagonal parameter matrices w_i, w_f and w_o.
  CuMatrix<BaseFloat> params_;

  // If true, we expect an extra 2 dimensions on the input, for dropout masks
  // for i_t and f_t.
  bool use_dropout_;

  // Of dimension 5 * C, with a row for each of the Sigmoid/Tanh functions in
  // equations (1) through (5), this is the sum of the values of the nonliearities
  // (used for diagnostics only).  It is comparable to value_sum_ vector
  // in base-class NonlinearComponent.
  CuMatrix<double> value_sum_;

  // Of dimension 5 * C, with a row for each of the Sigmoid/Tanh functions in
  // equations (1) through (5), this is the sum of the derivatives of the
  // nonliearities (used for diagnostics and to control self-repair).  It is
  // comparable to the deriv_sum_ vector in base-class
  // NonlinearComponent.
  CuMatrix<double> deriv_sum_;

  // This matrix has dimension 10.  The contents are a block of 5 self-repair
  // thresholds (typically "0.05 0.05 0.2 0.05 0.2"), then a block of 5
  // self-repair scales (typically all 0.00001).  These are for each of the 5
  // nonlinearities in the LSTM component in turn (see comments in cu-math.h for
  // more info).
  CuVector<BaseFloat> self_repair_config_;

  // This matrix has dimension 5.  For each of the 5 nonlinearities in the LSTM
  // component (see comments in cu-math.h for more info), it contains the total,
  // over all frames represented in count_, of the number of dimensions that
  // were subject to self_repair.  To get the self-repair proportion you should
  // divide by (count_ times cell_dim_).
  CuVector<double> self_repair_total_;

  // The total count (number of frames) corresponding to the stats in value_sum_
  // and deriv_sum_.
  double count_;

  // Preconditioner for the parameters of this component [operates in the space
  // of dimension C].
  // The preconditioner stores its own configuration values; we write and read
  // these, but not the preconditioner object itself.
  OnlineNaturalGradient preconditioner_;

  const LstmNonlinearityComponent &operator
      = (const LstmNonlinearityComponent &other); // Disallow.
};




/*
 * MaxPoolingComponent :
 * Maxpooling component was firstly used in ConvNet for selecting an
 * representative activation in an area. It inspired Maxout nonlinearity.
 * Each output element of this component is the maximum of a block of
 * input elements where the block has a 3D dimension (pool_x_size_,
 * pool_y_size_, pool_z_size_).
 * Blocks could overlap if the shift value on any axis is smaller
 * than its corresponding pool size (e.g. pool_x_step_ < pool_x_size_).
 * If the shift values are euqal to their pool size, there is no
 * overlap; while if they all equal 1, the blocks overlap to
 * the greatest possible extent.
 *
 * This component is designed to be used after a ConvolutionComponent
 * so that the input matrix is propagated from a 2d-convolutional layer.
 * This component implements 3d-maxpooling which performs
 * max pooling along the three axes.
 * Input : A matrix where each row is a vectorized 3D-tensor.
 *        The 3D tensor has dimensions
 *        x: (e.g. time)
 *        y: (e.g. frequency)
 *        z: (e.g. channels like number of filters in the ConvolutionComponent)
 *
 *        The component assumes input vectorizations of type zyx
 *        which is the default output vectorization type of a ConvolutionComponent.
 *        e.g. for input vectorization of type zyx the input is vectorized by
 *        spanning axes z, y and x of the tensor in that order.
 *        Given 3d tensor A with sizes (2, 2, 2) along the three dimensions
 *        the zyx vectorized input looks like
 *  A(0,0,0) A(0,0,1) A(0,1,0) A(0,1,1) A(1,0,0) A(1,0,1) A(1,1,0) A(1,1,1)
 *
 * Output : The output is also a 3D tensor vectorized in the zyx format.
 *
 * For information on the hyperparameters and parameters of this component see
 * the variable declarations.
 *
 *
 */
class MaxpoolingComponent: public Component {
 public:

  MaxpoolingComponent(): input_x_dim_(0), input_y_dim_(0), input_z_dim_(0),
                           pool_x_size_(0), pool_y_size_(0), pool_z_size_(0),
                           pool_x_step_(0), pool_y_step_(0), pool_z_step_(0) { }
  // constructor using another component
  MaxpoolingComponent(const MaxpoolingComponent &component);

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "MaxpoolingComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput|kBackpropNeedsOutput|
           kBackpropAdds;
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
  virtual Component* Copy() const { return new MaxpoolingComponent(*this); }


 protected:
  void InputToInputPatches(const CuMatrixBase<BaseFloat>& in,
                           CuMatrix<BaseFloat> *patches) const;
  void InderivPatchesToInderiv(const CuMatrix<BaseFloat>& in_deriv_patches,
                               CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void Check() const;


  int32 input_x_dim_;   // size of the input along x-axis
  // (e.g. number of time steps)
  int32 input_y_dim_;   // size of input along y-axis
  // (e.g. number of mel-frequency bins)
  int32 input_z_dim_;   // size of input along z-axis
  // (e.g. number of filters in the ConvolutionComponent)

  int32 pool_x_size_;    // size of the pooling window along x-axis
  int32 pool_y_size_;    // size of the pooling window along y-axis
  int32 pool_z_size_;    // size of the pooling window along z-axis

  int32 pool_x_step_;   // the number of steps taken along x-axis of input
  //  before computing the next pool
  int32 pool_y_step_;   // the number of steps taken along y-axis of input
  // before computing the next pool
  int32 pool_z_step_;   // the number of steps taken along z-axis of input
  // before computing the next pool

};


/*
  BatchNormComponent

  This implements batch normalization; for each dimension of the
  input it normalizes the data to be zero-mean, unit-variance.  You
  can set the block-dim configuration value to implement spatial
  batch normalization, see the comment for the variable.

  It's a simple component (uses the kSimpleComponent flag), but it is unusual in
  that it will give different results if you call it on half the matrix at a
  time.  Most of the time this would be pretty harmless, so we still return the
  kSimpleComponent flag.  We may have to modify the test code a little to
  account for this, or possibly remove the kSimpleComponent flag.  In some sense
  each output Index depends on every input Index, but putting those dependencies
  explicitly into the dependency-tracking framework as a GeneralComponent
  would be very impractical and might lead to a lot of unnecessary things being
  computed.  You have to be a bit careful where you put this component, and understand
  what you're doing e.g. putting it in the path of a recurrence is a bit problematic
  if the minibatch size were small.
 */
class BatchNormComponent: public Component {
 public:

  BatchNormComponent(): dim_(0), block_dim_(0),
                        epsilon_(1.0e-03), target_rms_(1.0),
                        test_mode_(false), count_(0) { }

  // call this with 'true' to set 'test mode' where the batch normalization is
  // done with stored stats.  There won't normally be any need to specially
  // accumulate these stats; they are stored as a matter of course on each
  // iteration of training, as for NonlinearComponents, and we'll use the stats
  // from the most recent [script-level] iteration.
  void SetTestMode(bool test_mode);

  // constructor using another component
  BatchNormComponent(const BatchNormComponent &other);

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }

  virtual std::string Info() const;
  // supports the config variables dim, block-dim (which defaults to dim),
  // epsilon (which defaults to 1.0e-3), and target-rms (which defaults to 1.0,
  // and is a scaling on the output; it's comparable to the target-rms of
  // NormalizeComponent).  it also accepts a boolean 'test-mode' config which is
  // only intended for use in testing code, and not in real situations.  (note:
  // test-mode is a real thing that's used during 'inference' given a previously
  // computed model, and we do set test mode in real situations; we just don't
  // do so from the config, we use the function SetTestMode().
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "BatchNormComponent"; }
  virtual int32 Properties() const {
    // If the block-dim is less than the dim, we need the input and output
    // matrices to be contiguous (stride==num-cols), as we'll be reshaping
    // internally.  This is not much of a cost, because this will be used
    // in convnets where we have to do this anyway.
    return kSimpleComponent|kBackpropNeedsOutput|kPropagateInPlace|
        kBackpropInPlace|
        (block_dim_ < dim_ ? kInputContiguous|kOutputContiguous : 0)|
        (test_mode_ ? 0 : kUsesMemo|kStoresStats);
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
  virtual Component* Copy() const { return new BatchNormComponent(*this); }

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void ZeroStats();


  virtual void DeleteMemo(void *memo) const { delete static_cast<Memo*>(memo); }

  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);
 private:

  struct Memo {
    // number of frames (after any reshaping).
    int32 num_frames;
    // 'sum_sumsq_scale' is of dimension 4 by block_dim_:
    // Row 0 = mean = the mean of the rows of the input
    // Row 1 = uvar = the uncentered variance of the input (= sumsq / num_frames).
    // Row 2 = scale = the scale of the renormalization, which is
    // Row 3 is used as a temporary in Backprop.
    //    the inverse stddev of the input (modified by epsilon_,
    //    see the Propagate function.
    CuMatrix<BaseFloat> mean_uvar_scale;
  };

  void Check() const;

  // this function is used in a couple of places; it turns the raw stats into
  // the offset/scale term of a normalizing transform.
  static void ComputeOffsetAndScale(double count,
                                    BaseFloat epsilon,
                                    const Vector<double> &stats_sum,
                                    const Vector<double> &stats_sumsq,
                                    Vector<BaseFloat> *offset,
                                    Vector<BaseFloat> *scale);
  // computes derived parameters offset_ and scale_.
  void ComputeDerived();

  // Dimension of the input and output.
  int32 dim_;
  // This would normally be the same as dim_, but if it's less (and it must be >
  // 0 and must divide dim_), then each separate block of the input of dimension
  // 'block_dim_' is treated like a separate frame for the purposes of
  // normalization.  This can be used to implement spatial batch normalization
  // for convolutional setups-- assuming the filter-dim has stride 1, which it
  // always will in the new code in nnet-convolutional-component.h, when it's
  // finished.
  int32 block_dim_;

  // Used to avoid exact-zero variances, epsilon has the dimension of a
  // covariance; in this work it is applied as a floor, not as an additive term
  // (this is safer in the presence of numerical roundoff).
  BaseFloat epsilon_;

  // This value will normally be 1.0, which is the default, but you can set it
  // to other values as a way to control how fast the following layer learns
  // (smaller -> slower).  The same config exists in NormalizeComponent.
  BaseFloat target_rms_;

  // This is true if we want the batch normalization to operate in 'test mode'
  // meaning the data mean and stddev used for the normalziation are fixed
  // quantities based on previously accumulated stats.  Note: the stats we use
  // for this are based on the same 'StoreStats' mechanism as we use for
  // components like SigmoidComponent and ReluComponent; we'll be using
  // the stats from the most recent [script-level] iteration of training.
  bool test_mode_;


  // total count of stats stored by StoreStats().
  double count_;
  // sum-of-data component of stats of input data.
  CuVector<double> stats_sum_;
  // sum-of-squared component of stats of input data.
  CuVector<double> stats_sumsq_;

  // offset_ and scale_ are derived from stats_sum_ and stats_sumsq_; they
  // dictate the transform that is done in 'test mode'.  They are set only when
  // reading the model from disk and when calling SetTestMode(true); they are
  // resized to empty when the stats are updated, to ensure that out-of-date
  // values are not kept around.
  CuVector<BaseFloat> offset_;
  CuVector<BaseFloat> scale_;
};



/**
   CompositeComponent is a component representing a sequence of
   [simple] components.  The config line would be something like the following
   (imagine this is all on one line):

   component name=composite1 type=CompositeComponent max-rows-process=2048 num-components=3 \
      component1='type=BlockAffineComponent input-dim=1000 output-dim=10000 num-blocks=100' \
      component2='type=RectifiedLinearComponent dim=10000' \
      component3='type=BlockAffineComponent input-dim=10000 output-dim=1000 num-blocks=100'

   The reason you might want to use this component, instead of directly using
   the same sequence of components in the config file, is to save GPU memory (at
   the expense of more compute)-- because doing it like this means we have to
   re-do parts of the forward pass in the backprop phase, but we avoid using
   much memory for very long (and you can make the memory usage very small by
   making max-rows-process small).  We inherit from UpdatableComponent just in
   case one or more of the components in the sequence are updatable.

   It is an error to nest a CompositeComponent inside a CompositeComponent.
   The same effect can be accomplished by specifying a smaller max-rows-process
   in a single CompositeComponent.
 */
class CompositeComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;

  virtual void InitFromConfig(ConfigLine *cfl);

  virtual Component* Copy() const;

  CompositeComponent() { } // use Init() or InitFromConfig() to really initialize.

  // Initialize from this list of components; takes ownership of the pointers.
  void Init(const std::vector<Component*> &components,
            int32 max_rows_process);

  virtual std::string Type() const { return "CompositeComponent"; }

  // The properties depend on the properties of the constituent components.  As
  // a special case, we never return kStoresStats in the properties: by default
  // we store things like activation stats (e.g. for nonlinear components like
  // ReLU) as part of the backprop.  This means we may wastefully store stats
  // even when not requested, but it does save time as a separate StoreStats()
  // call would involve propagating the internals.
  virtual int32 Properties() const;

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  // note, we don't implement StoreStats() as it would be inefficient.  Instead,
  // by default we call StoreStats() on all members that have the flag set,
  // inside the Backprop.
  virtual void ZeroStats();

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  // Don't implement Copy() at this level: implement it in the child class.

  // Some functions from base-class UpdatableComponent.
  virtual void SetUnderlyingLearningRate(BaseFloat lrate);
  virtual void SetActualLearningRate(BaseFloat lrate);
  virtual void SetAsGradient();
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // note: we dont implement the StoreStats function as it would be quite
  // expensive; instead, by default we call StoreStats() for any components that
  // want to store stats, as part of the backprop pass.  This is not 100% ideal
  // but it will usually do what you want.  We can revisit this later if needed.

  // Functions to iterate over the internal components

  int32 NumComponents() const { return components_.size();}
  /// Gets the ith component in this component.
  /// The ordering is the same as in the config line. The caller
  /// does not own the received component.
  const Component* GetComponent(int32 i) const;
  /// Sets the ith component. After this call, CompositeComponent owns
  /// the reference to the argument component. Frees the previous
  /// ith component.
  void SetComponent(int32 i, Component *component);

  virtual ~CompositeComponent() { DeletePointers(&components_); }
 private:
  // returns the stride type, kDefaultStride or kStrideEqualNumCols,
  // at the output of the i'th component.
  inline MatrixStrideType GetStrideType(int32 i) const;

  // returns true if at least one of 'components_' returns the kUpdatable flag
  // in its flags.
  bool IsUpdatable() const;

  // the maximum number of
  int32 max_rows_process_;
  std::vector<Component*> components_;

};


} // namespace nnet3
} // namespace kaldi


#endif
