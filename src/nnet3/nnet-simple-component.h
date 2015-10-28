// nnet3/nnet-simple-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//           2014-2015  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen

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

  virtual std::string Info() const;
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

  virtual std::string Info() const;
 protected:
  int32 input_dim_;
  int32 output_dim_;
};

class NormalizeComponent: public NonlinearComponent {
  // note: although we inherit from NonlinearComponent, we don't actually bohter
  // accumulating the stats that NonlinearComponent is capable of accumulating.
 public:
  explicit NormalizeComponent(int32 dim): NonlinearComponent(dim) { }
  explicit NormalizeComponent(const NormalizeComponent &other): NonlinearComponent(other) { }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput|kPropagateInPlace|
        kBackpropInPlace;
  }
  NormalizeComponent() { }
  virtual std::string Type() const { return "NormalizeComponent"; }
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
 private:
  NormalizeComponent &operator = (const NormalizeComponent &other); // Disallow.
  static const BaseFloat kNormFloor;
  // about 0.7e-20.  We need a value that's exactly representable in
  // float and whose inverse square root is also exactly representable
  // in float (hence, an even power of two).
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

class FixedAffineComponent;
class FixedScaleComponent;
class PerElementScaleComponent;

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
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(BaseFloat learning_rate,
            std::string matrix_filename);

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
  Component *CollapseWithNext(const PerElementScaleComponent &next) const;
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
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_init,
            BaseFloat bias_mean, BaseFloat bias_stddev,
            int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_sample);
  void Init(BaseFloat learning_rate, int32 rank_in,
            int32 rank_out, int32 update_period,
            BaseFloat num_samples_history,
            BaseFloat alpha, BaseFloat max_change_per_sample,
            std::string matrix_filename);

  virtual void Resize(int32 input_dim, int32 output_dim);
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Info() const;
  virtual Component* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  NaturalGradientAffineComponent();
  virtual void ZeroStats();

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NaturalGradientAffineComponent);

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

// SumGroupComponent is used to sum up groups of posteriors.
// It's used to introduce a kind of Gaussian-mixture-model-like
// idea into neural nets.  This is basically a degenerate case of
// MixtureProbComponent; we had to implement it separately to
// be efficient for CUDA (we can use this one regardless whether
// we have CUDA or not; it's the normal case we want anyway).
class SumGroupComponent: public Component {
public:
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  void Init(const std::vector<int32> &sizes); // the vector is of the input dim
                                              // (>= 1) for each output dim.
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

// PermuteComponent shuffles the columns in the input, according to the
// specification.
class PermuteComponent: public Component {
 public:
  PermuteComponent()  {}
  PermuteComponent(CuArray<int32> column_map): column_map_(column_map){}

  virtual int32 InputDim() const { return column_map_.Dim(); }
  virtual int32 OutputDim() const { return column_map_.Dim(); }
  virtual void InitFromConfig(ConfigLine *cfl);
  void Init(CuArray<int32> column_map) { column_map_ = column_map;}

  virtual std::string Type() const { return "PermuteComponent"; }

  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput;
  }

  virtual void ZeroStats() {}

  virtual Component* Copy() const {
    return new PermuteComponent(column_map_);}

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
  CuArray<int32> column_map_;
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

  void Init(BaseFloat learning_rate, int32 dim, BaseFloat param_mean,
            BaseFloat param_stddev);
  void Init(BaseFloat learning_rate, std::string vector_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 dim);

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

  void Init(BaseFloat learning_rate, int32 dim, BaseFloat param_mean,
            BaseFloat param_stddev, int32 rank, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_minibatch);
  void Init(BaseFloat learning_rate, std::string vector_filename,
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
    kYzx= 0,
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

  // Some functions that are specific to this class.
  void SetParams(const VectorBase<BaseFloat> &bias,
                 const MatrixBase<BaseFloat> &filter);
  const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return filter_params_; }
  void Init(BaseFloat learning_rate,
            int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
            int32 filt_x_dim, int32 filt_y_dim,
            int32 filt_x_step, int32 filt_y_step, int32 num_filters,
            TensorVectorizationType input_vectorization,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  // there is no filt_z_dim parameter as the length of the filter along
  // z-dimension is same as the input
  void Init(BaseFloat learning_rate,
            int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
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
 * Convolutional1dComponent implements convolution over frequency axis.
 * We assume the input featrues are spliced, i.e. each frame is in
 * fact a set of stacked frames, where we can form patches which span
 * over several frequency bands and whole time axis. A patch is the
 * instance of a filter on a group of frequency bands and whole time
 * axis. Shifts of the filter generate patches.
 *
 * The convolution is done over whole axis with same filter
 * coefficients, i.e. we don't use separate filters for different
 * 'regions' of frequency axis. Due to convolution, same weights are
 * used repeateadly, the final gradient is a sum of all
 * position-specific gradients (the sum was found better than
 * averaging).
 *
 * In order to have a fast implementations, the filters are
 * represented in vectorized form, where each rectangular filter
 * corresponds to a row in a matrix, where all the filters are
 * stored. The features are then re-shaped to a set of matrices, where
 * one matrix corresponds to single patch-position, where all the
 * filters get applied.
 *
 * The type of convolution is controled by hyperparameters:
 * patch_dim_     ... frequency axis size of the patch
 * patch_step_    ... size of shift in the convolution
 * patch_stride_  ... shift for 2nd dim of a patch
 *                    (i.e. frame length before splicing)
 * For instance, for a convolutional component after raw input,
 * if the input is 36-dim fbank feature with delta of order 2
 * and spliced using +/- 5 frames of contexts, the convolutional
 * component takes the input as a 36 x 33 image. The patch_stride_
 * should be configured 36. If patch_step_ and patch_dim_ are
 * configured 1 and 7, the Convolutional1dComponent creates a
 * 2D filter of 7 x 33, such that the convolution is actually done
 * only along the frequency axis. Specifically, the convolutional
 * output along the frequency axis is (36 - 7) / 1 + 1 = 30, and
 * the convolutional output along the temporal axis is 33 - 33 + 1 = 1,
 * resulting in an output image of 30 x 1, which is called a feature map
 * in ConvNet. Then if the output-dim is set 3840, the constructor
 * would know there should be 3840 / 30 = 128 distinct filters,
 * which will create 128 feature maps of 30 x 1 for one frame of
 * input. The feature maps are vectorized as a 3840-dim row vector
 * in the output matrix of this component. For details on progatation
 * of Convolutional1dComponent, check the function definition.
 *
 */
class Convolutional1dComponent: public UpdatableComponent {
 public:
  Convolutional1dComponent();
  // constructor using another component
  Convolutional1dComponent(const Convolutional1dComponent &component);
  // constructor using parameters
  Convolutional1dComponent(const CuMatrixBase<BaseFloat> &filter_params,
                           const CuVectorBase<BaseFloat> &bias_params,
                           BaseFloat learning_rate);

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "Convolutional1dComponent"; }
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

  // Some functions that are specific to this class.
  void SetParams(const VectorBase<BaseFloat> &bias,
                 const MatrixBase<BaseFloat> &filter);
  const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return filter_params_; }
  void Init(BaseFloat learning_rate, int32 input_dim, int32 output_dim,
            int32 patch_dim, int32 patch_step, int32 patch_stride,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(BaseFloat learning_rate,
            int32 patch_dim, int32 patch_step, int32 patch_stride,
            std::string matrix_filename);

  // resize the component, setting the parameters to zero, while
  // leaving any other configuration values the same
  void Resize(int32 input_dim, int32 output_dim);

  void Update(const std::string &debug_info,
	      const CuMatrixBase<BaseFloat> &in_value,
              const CuMatrixBase<BaseFloat> &out_deriv);

 private:
  int32 patch_dim_;
  int32 patch_step_;
  int32 patch_stride_;

  static void ReverseIndexes(const std::vector<int32> &forward_indexes,
                             int32 input_dim,
                             std::vector<std::vector<int32> > *backward_indexes);
  static void RearrangeIndexes(const std::vector<std::vector<int32> > &in,
                               std::vector<std::vector<int32> > *out);

  const Convolutional1dComponent &operator = (const Convolutional1dComponent &other); // Disallow.
  CuMatrix<BaseFloat> filter_params_;
  CuVector<BaseFloat> bias_params_;
  bool is_gradient_;
};

/**
 * MaxPoolingComponent :
 * Maxpooling component was firstly used in ConvNet for selecting an representative
 * activation in an area. It inspired Maxout nonlinearity.
 *
 * The input/output matrices are split to submatrices with width 'pool_stride_'.
 * For instance, a minibatch of 512 frames is propagated by a convolutional
 * layer, resulting in a 512 x 3840 input matrix for MaxpoolingComponent,
 * which is composed of 128 feature maps for each frame (128 x 30). If you want
 * a 3-to-1 maxpooling on each feature map, set 'pool_stride_' and 'pool_size_'
 * as 128 and 3 respectively. Maxpooling component would create an output
 * matrix of 512 x 1280. The 30 input neurons are grouped by a group size of 3, and
 * the maximum in a group is selected, creating a smaller feature map of 10.
 *
 * Our pooling does not supports overlaps, which simplifies the
 * implementation (and was not helpful for Ossama).
 */
class MaxpoolingComponent: public Component {
 public:
  explicit MaxpoolingComponent(int32 input_dim, int32 output_dim,
                               int32 pool_size, int32 pool_stride) {
    Init(input_dim, output_dim, pool_size, pool_stride);
  }
  MaxpoolingComponent(): input_dim_(0), output_dim_(0),
    pool_size_(0), pool_stride_(0) { }
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }

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
  virtual Component* Copy() const {
    return new MaxpoolingComponent(input_dim_, output_dim_,
		    pool_size_, pool_stride_); }

  // Some functions that are specific to this
  void Init(int32 input_dim, int32 output_dim,
            int32 pool_size, int32 pool_stride);

 protected:
  int32 input_dim_;
  int32 output_dim_;
  int32 pool_size_;
  int32 pool_stride_;
};


} // namespace nnet3
} // namespace kaldi


#endif
