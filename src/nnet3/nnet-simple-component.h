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

/// @file  This file contains declarations of components that are "simple", meaning
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
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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

class NormalizeComponent: public NonlinearComponent {
  // note: although we inherit from NonlinearComponent, we don't actually bohter
  // accumulating the stats that NonlinearComponent is capable of accumulating.
 public:
 void Init(int32 dim, BaseFloat target_rms, bool add_log_stddev);
  explicit NormalizeComponent(int32 dim, BaseFloat target_rms = 1.0, 
    bool add_log_stddev = false) { Init(dim, target_rms, add_log_stddev); }
  explicit NormalizeComponent(const NormalizeComponent &other): NonlinearComponent(other),
    target_rms_(other.target_rms_), add_log_stddev_(other.add_log_stddev_) { }
  virtual int32 Properties() const {
    return (add_log_stddev_ ? kSimpleComponent|kBackpropNeedsInput :
            kSimpleComponent|kBackpropNeedsInput|kPropagateInPlace|
        kBackpropInPlace);
  }
  NormalizeComponent(): target_rms_(1.0), add_log_stddev_(false) { }
  virtual std::string Type() const { return "NormalizeComponent"; }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual Component* Copy() const { return new NormalizeComponent(*this); }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

  virtual int32 OutputDim() const { return (dim_ + (add_log_stddev_ ? 1 : 0)); } 

  virtual std::string Info() const;
 private:
  NormalizeComponent &operator = (const NormalizeComponent &other); // Disallow.
  static const BaseFloat kNormFloor;
  BaseFloat target_rms_; // The target rms for outputs.
  // about 0.7e-20.  We need a value that's exactly representable in
  // float and whose inverse square root is also exactly representable
  // in float (hence, an even power of two).

  bool add_log_stddev_; // If true, log(max(epsi, sqrt(row_in^T row_in / D)))  
                        // is an extra dimension of the output.
};


class SigmoidComponent: public NonlinearComponent {
 public:
  explicit SigmoidComponent(int32 dim): NonlinearComponent(dim) { }
  explicit SigmoidComponent(const SigmoidComponent &other): NonlinearComponent(other) { }
  SigmoidComponent() { }
  virtual std::string Type() const { return "SigmoidComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kPropagateInPlace|kStoresStats;
  }
  virtual Component* Copy() const { return new SigmoidComponent(*this); }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &out_value);
 private:
  SigmoidComponent &operator = (const SigmoidComponent &other); // Disallow.
};

class TanhComponent: public NonlinearComponent {
 public:
  explicit TanhComponent(int32 dim): NonlinearComponent(dim) { }
  explicit TanhComponent(const TanhComponent &other): NonlinearComponent(other) { }
  TanhComponent() { }
  virtual std::string Type() const { return "TanhComponent"; }
  virtual Component* Copy() const { return new TanhComponent(*this); }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kPropagateInPlace|kStoresStats;
  }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &out_value);
 private:
  TanhComponent &operator = (const TanhComponent &other); // Disallow.
};


class RectifiedLinearComponent: public NonlinearComponent {
 public:
  explicit RectifiedLinearComponent(int32 dim): NonlinearComponent(dim) { }
  explicit RectifiedLinearComponent(const RectifiedLinearComponent &other): NonlinearComponent(other) { }
  RectifiedLinearComponent() { }
  virtual std::string Type() const { return "RectifiedLinearComponent"; }
  virtual Component* Copy() const { return new RectifiedLinearComponent(*this); }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kBackpropNeedsOutput|kPropagateInPlace|
        kStoresStats;
  }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &out_value);
 private:
  RectifiedLinearComponent &operator = (const RectifiedLinearComponent &other); // Disallow.
};

/**
   This component is a fixed (non-trainable) nonlinearity that sums its inputs
   to produce outputs.  Currently the only supported configuration is that its
   input-dim is interpreted as consisting of n blocks, and the output is just a
   summation over the n blocks, where  n = input-dim / output-dim, so for instance
    output[n] = input[n] + input[block-size + n] + .... .
   Later if needed we can add a configuration variable that allows you to sum
   over 'interleaved' input.
 */
class SumReduceComponent: public Component {
 public:
  void Init(int32 input_dim, int32 output_dim);
  explicit SumReduceComponent(int32 input_dim, int32 output_dim) {
    Init(input_dim, output_dim);
  }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput;
  }
  SumReduceComponent(): input_dim_(0), output_dim_(0) { }
  virtual std::string Type() const { return "SumReduceComponent"; }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *, // to_update
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const { return new SumReduceComponent(input_dim_,
                                                                  output_dim_); }

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  int32 input_dim_;
  int32 output_dim_;
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


  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(const VectorBase<BaseFloat> &bias,
                         const MatrixBase<BaseFloat> &linear);
  const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }
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

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void SetZero(bool treat_as_gradient);
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
	     kBackpropNeedsInput|kBackpropAdds;
  }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }
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
  explicit SoftmaxComponent(int32 dim): NonlinearComponent(dim) { }
  explicit SoftmaxComponent(const SoftmaxComponent &other):
      NonlinearComponent(other) { }
  SoftmaxComponent() { }
  virtual std::string Type() const { return "SoftmaxComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kStoresStats;
  }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void StoreStats(const CuMatrixBase<BaseFloat> &out_value);

  virtual Component* Copy() const { return new SoftmaxComponent(*this); }
 private:
  SoftmaxComponent &operator = (const SoftmaxComponent &other); // Disallow.
};

class LogSoftmaxComponent: public NonlinearComponent {
 public:
  explicit LogSoftmaxComponent(int32 dim): NonlinearComponent(dim) { }
  explicit LogSoftmaxComponent(const LogSoftmaxComponent &other):
      NonlinearComponent(other) { }
  LogSoftmaxComponent() { }
  virtual std::string Type() const { return "LogSoftmaxComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kStoresStats;
  }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual Component* Copy() const { return new LogSoftmaxComponent(*this); }
 private:
  LogSoftmaxComponent &operator = (const LogSoftmaxComponent &other); // Disallow.
};

/// Keywords: natural gradient descent, NG-SGD, naturalgradient.  For
/// the top-level of the natural gradient code look here, and also in
/// nnet-precondition-online.h.
/// NaturalGradientAffineComponent is
/// a version of AffineComponent that has a non-(multiple of unit) learning-rate
/// matrix.  See nnet-precondition-online.h for a description of the technique.
/// It is described, under the name Online NG-SGD, in the paper "Parallel
/// training of DNNs with Natural Gradient and Parameter Averaging" (ICLR
/// workshop, 2015) by Daniel Povey, Xiaohui Zhang and Sanjeev Khudanpur.
class NaturalGradientAffineComponent: public AffineComponent {
 public:
  virtual std::string Type() const { return "NaturalGradientAffineComponent"; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev, BaseFloat bias_mean,
            int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_sample);
  void Init(int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history,
            BaseFloat alpha, BaseFloat max_change_per_sample,
            std::string matrix_filename);
  // this constructor does not really initialize, use Init() or Read().
  NaturalGradientAffineComponent();
  virtual void Resize(int32 input_dim, int32 output_dim);
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Info() const;
  virtual Component* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  // copy constructor
  explicit NaturalGradientAffineComponent(
      const NaturalGradientAffineComponent &other);
  virtual void ZeroStats();

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

  // If > 0, max_change_per_sample_ is the maximum amount of parameter
  // change (in L2 norm) that we allow per sample, averaged over the minibatch.
  // This was introduced in order to control instability.
  // Instead of the exact L2 parameter change, for
  // efficiency purposes we limit a bound on the exact
  // change.  The limit is applied via a constant <= 1.0
  // for each minibatch, A suitable value might be, for
  // example, 10 or so; larger if there are more
  // parameters.
  BaseFloat max_change_per_sample_;

  // update_count_ records how many updates we have done.
  double update_count_;

  // active_scaling_count_ records how many updates we have done,
  // where the scaling factor is active (not 1.0).
  double active_scaling_count_;

  // max_change_scale_stats_ records the sum of scaling factors
  // in each update, so we can compute the averaged scaling factor
  // in Info().
  double max_change_scale_stats_;

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

  /// matrix should be of size input-dim+1 to output-dim, last col is offset
  void Init(const CuMatrixBase<BaseFloat> &matrix);

  // The ConfigLine cfl contains just the option matrix=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromConfig(ConfigLine *cfl);

  virtual int32 Properties() const { return kSimpleComponent|kBackpropAdds; }
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
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

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *, // to_update
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  CuVector<BaseFloat> bias_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(FixedBiasComponent);
};

// NoOpComponent just duplicates its input.  We don't anticipate this being used
// very often, but it may sometimes make your life easier
class NoOpComponent: public NonlinearComponent {
 public:
  explicit NoOpComponent(int32 dim): NonlinearComponent(dim) { }
  explicit NoOpComponent(const NoOpComponent &other): NonlinearComponent(other) { }
  NoOpComponent() { }
  virtual std::string Type() const { return "NoOpComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kPropagateInPlace;
  }
  virtual Component* Copy() const { return new NoOpComponent(*this); }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
 private:
  NoOpComponent &operator = (const NoOpComponent &other); // Disallow.
};

// ClipGradientComponent just duplicates its input, but clips gradients
// during backpropagation if they cross a predetermined threshold.
// This component will be used to prevent gradient explosion problem in
// recurrent neural networks
class ClipGradientComponent: public Component {
 public:
  ClipGradientComponent(int32 dim, BaseFloat clipping_threshold,
                        bool norm_based_clipping, int32 num_clipped,
                        int32 count) {
    Init(dim, clipping_threshold, norm_based_clipping, num_clipped, count);}

  ClipGradientComponent(): dim_(0), clipping_threshold_(-1),
    norm_based_clipping_(false), num_clipped_(0), count_(0) { }

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void InitFromConfig(ConfigLine *cfl);
  void Init(int32 dim, BaseFloat clipping_threshold, bool norm_based_clipping,
            int32 num_clipped, int32 count);

  virtual std::string Type() const { return "ClipGradientComponent"; }

  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kPropagateInPlace|kBackpropInPlace;
  }

  virtual void ZeroStats();

  virtual Component* Copy() const {
    return new ClipGradientComponent(dim_,
                                     clipping_threshold_,
                                     norm_based_clipping_,
                                     num_clipped_,
                                     count_);}

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Info() const;
 private:
  int32 dim_;  // input/output dimension
  BaseFloat clipping_threshold_;  // threshold to be used for clipping
                                  // could correspond to max-row-norm (if
                                  // norm_based_clipping_ == true) or
                                  // max-absolute-value (otherwise)
  bool norm_based_clipping_;  // if true the max-row-norm will be clipped
                              // else element-wise absolute value clipping is
                              // done


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

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void SetZero(bool treat_as_gradient);
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


// PerElementOffsetComponent offsets each dimension of its input with a separate
// trainable bias; it's like an affine component with fixed weight matrix which is always equal to I.
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

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void SetZero(bool treat_as_gradient);
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
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_minibatch);
  void Init(std::string vector_filename,
            int32 rank, int32 update_period, BaseFloat num_samples_history,
            BaseFloat alpha, BaseFloat max_change_per_minibatch);

 private:
  // configuration value for imposing max-change...
  BaseFloat max_change_per_minibatch_;

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

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  void SetParams(const VectorBase<BaseFloat> &bias,
                 const MatrixBase<BaseFloat> &filter);
  const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return filter_params_; }
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

/**
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
  MaxpoolingComponent(const MaxpoolingComponent &component): 
             input_x_dim_(component.input_x_dim_),
             input_y_dim_(component.input_y_dim_),
             input_z_dim_(component.input_z_dim_),
             pool_x_size_(component.pool_x_size_), 
             pool_y_size_(component.pool_y_size_),
             pool_z_size_(component.pool_z_size_),
             pool_x_step_(component.pool_x_step_),
             pool_y_step_(component.pool_y_step_),
             pool_z_step_(component.pool_z_step_) { }

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;
  virtual void Check() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "MaxpoolingComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput|kBackpropNeedsOutput|
	    kBackpropAdds;
  }

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const { return new MaxpoolingComponent(*this); }

  void InputToInputPatches(const CuMatrixBase<BaseFloat>& in,
                           CuMatrix<BaseFloat> *patches) const;
  void InderivPatchesToInderiv(const CuMatrix<BaseFloat>& in_deriv_patches,
                               CuMatrixBase<BaseFloat> *in_deriv) const;

 protected:
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

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  virtual void SetLearningRate(BaseFloat lrate);
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void SetZero(bool treat_as_gradient);
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
 protected:
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
