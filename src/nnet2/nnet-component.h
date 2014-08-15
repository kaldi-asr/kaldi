// nnet2/nnet-component.h

// Copyright 2011-2013  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)
//	          2013  Xiaohui Zhang	

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

#ifndef KALDI_NNET2_NNET_COMPONENT_H_
#define KALDI_NNET2_NNET_COMPONENT_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "thread/kaldi-mutex.h"
#include "nnet2/nnet-precondition-online.h"

#include <iostream>

namespace kaldi {
namespace nnet2 {


/**
 * Abstract class, basic element of the network,
 * it is a box with defined inputs, outputs,
 * and tranformation functions interface.
 *
 * It is able to propagate and backpropagate
 * exact implementation is to be implemented in descendants.
 *
 */ 

class Component {
 public:
  Component(): index_(-1) { }
  
  virtual std::string Type() const = 0; // each type should return a string such as
  // "SigmoidComponent".

  /// Returns the index in the sequence of layers in the neural net; intended only
  /// to be used in debugging information.
  virtual int32 Index() const { return index_; }

  virtual void SetIndex(int32 index) { index_ = index; }

  /// Initialize, typically from a line of a config file.  The "args" will
  /// contain any parameters that need to be passed to the Component, e.g.
  /// dimensions.
  virtual void InitFromString(std::string args) = 0; 
  
  /// Get size of input vectors
  virtual int32 InputDim() const = 0;
  
  /// Get size of output vectors 
  virtual int32 OutputDim() const = 0;
  
  /// Number of left-context frames the component sees for each output frame;
  /// nonzero only for splicing layers.
  virtual int32 LeftContext() const { return 0; }

  /// Number of right-context frames the component sees for each output frame;
  /// nonzero only for splicing layers.
  virtual int32 RightContext() const { return 0; }

  /// Perform forward pass propagation Input->Output.  Each row is
  /// one frame or training example.  Interpreted as "num_chunks"
  /// equally sized chunks of frames; this only matters for layers
  /// that do things like context splicing.  Typically this variable
  /// will either be 1 (when we're processing a single contiguous
  /// chunk of data) or will be the same as in.NumFrames(), but
  /// other values are possible if some layers do splicing.
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const = 0; 
  
  /// Perform backward pass propagation of the derivative, and
  /// also either update the model (if to_update == this) or
  /// update another model or compute the model derivative (otherwise).
  /// Note: in_value and out_value are the values of the input and output
  /// of the component, and these may be dummy variables if respectively
  /// BackpropNeedsInput() or BackpropNeedsOutput() return false for
  /// that component (not all components need these).
  ///
  /// num_chunks lets us treat the input matrix as n contiguous-in-time
  /// chunks of equal size; it only matters if splicing is involved.
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,                        
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const = 0;
  
  virtual bool BackpropNeedsInput() const { return true; } // if this returns false,
  // the "in_value" to Backprop may be a dummy variable.
  virtual bool BackpropNeedsOutput() const { return true; } // if this returns false,
  // the "out_value" to Backprop may be a dummy variable.
  
  /// Read component from stream
  static Component* ReadNew(std::istream &is, bool binary);

  /// Copy component (deep copy).
  virtual Component* Copy() const = 0;

  /// Initialize the Component from one line that will contain
  /// first the type, e.g. SigmoidComponent, and then
  /// a number of tokens (typically integers or floats) that will
  /// be used to initialize the component.
  static Component *NewFromString(const std::string &initializer_line);

  /// Return a new Component of the given type e.g. "SoftmaxComponent",
  /// or NULL if no such type exists. 
  static Component *NewComponentOfType(const std::string &type);
  
  virtual void Read(std::istream &is, bool binary) = 0; // This Read function
  // requires that the Component has the correct type.
  
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const = 0;

  virtual std::string Info() const;

  virtual ~Component() { }

 private:
  int32 index_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Component);
};


/**
 * Class UpdatableComponent is a Component which has
 * trainable parameters and contains some global 
 * parameters for stochastic gradient descent
 * (learning rate, L2 regularization constant).
 * This is a base-class for Components with parameters.
 */
class UpdatableComponent: public Component {
 public:
  UpdatableComponent(const UpdatableComponent &other):
      learning_rate_(other.learning_rate_){ }
  
  void Init(BaseFloat learning_rate) {
    learning_rate_ = learning_rate;
  }
  UpdatableComponent(BaseFloat learning_rate) {
    Init(learning_rate);
  }

  /// Set parameters to zero, and if treat_as_gradient is true, we'll be
  /// treating this as a gradient so set the learning rate to 1 and make any
  /// other changes necessary (there's a variable we have to set for the
  /// MixtureProbComponent).
  virtual void SetZero(bool treat_as_gradient) = 0;
  
  UpdatableComponent(): learning_rate_(0.001) { }
  
  virtual ~UpdatableComponent() { }

  /// Here, "other" is a component of the same specific type.  This
  /// function computes the dot product in parameters, and is computed while
  /// automatically adjusting learning rates; typically, one of the two will
  /// actually contain the gradient.
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const = 0;
  
  /// We introduce a new virtual function that only applies to
  /// class UpdatableComponent.  This is used in testing.
  virtual void PerturbParams(BaseFloat stddev) = 0;
  
  /// This new virtual function scales the parameters
  /// by this amount.  
  virtual void Scale(BaseFloat scale) = 0;

  /// This new virtual function adds the parameters of another
  /// updatable component, times some constant, to the current
  /// parameters.
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other) = 0;
  
  /// Sets the learning rate of gradient descent
  void SetLearningRate(BaseFloat lrate) {  learning_rate_ = lrate; }
  /// Gets the learning rate of gradient descent
  BaseFloat LearningRate() const { return learning_rate_; }

  virtual std::string Info() const;
  
  // The next few functions are not implemented everywhere; they are
  // intended for use by L-BFGS code, and we won't implement them
  // for all child classes.
  
  /// The following new virtual function returns the total dimension of
  /// the parameters in this class.  E.g. used for L-BFGS update
  virtual int32 GetParameterDim() const { KALDI_ASSERT(0); return 0; }

  /// Turns the parameters into vector form.  We put the vector form on the CPU,
  /// because in the kinds of situations where we do this, we'll tend to use
  /// too much memory for the GPU.
  virtual void Vectorize(VectorBase<BaseFloat> *params) const { KALDI_ASSERT(0); }
  /// Converts the parameters from vector form.
  virtual void UnVectorize(const VectorBase<BaseFloat> &params) {
    KALDI_ASSERT(0);
  }
  
 protected: 
  BaseFloat learning_rate_; ///< learning rate (0.0..0.01)
 private:
  const UpdatableComponent &operator = (const UpdatableComponent &other); // Disallow.
};

/// Augments a scalar variable with powers of itself, e.g. x => {x, x^2}.
class PowerExpandComponent: public Component {
 public:
  void Init(int32 dim, int32 max_power = 2, BaseFloat higher_power_scale = 1.0);
  
  explicit PowerExpandComponent(int32 dim, int32 max_power = 2,
                                BaseFloat higher_power_scale = 1.0) {
    Init(dim, max_power, higher_power_scale);
  }
  PowerExpandComponent(): input_dim_(0), max_power_(2),
                          higher_power_scale_(1.0) { }
  virtual std::string Type() const { return "PowerExpandComponent"; }
  virtual void InitFromString(std::string args); 
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return max_power_ * input_dim_; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const { return new PowerExpandComponent(input_dim_,
                                                                    max_power_,
                                                                    higher_power_scale_); }
  
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

  virtual std::string Info() const;
 private:
  int32 input_dim_;
  int32 max_power_;
  BaseFloat higher_power_scale_; // Scale put on all powers
  // except the first one.
};


/// This kind of Component is a base-class for things like
/// sigmoid and softmax.
class NonlinearComponent: public Component {
 public:
  void Init(int32 dim) { dim_ = dim; count_ = 0.0; }
  explicit NonlinearComponent(int32 dim) { Init(dim); }
  NonlinearComponent(): dim_(0) { } // e.g. prior to Read().
  explicit NonlinearComponent(const NonlinearComponent &other);
  
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  
  /// We implement InitFromString at this level.
  virtual void InitFromString(std::string args);
  
  /// We implement Read at this level as it just needs the Type().
  virtual void Read(std::istream &is, bool binary);
  
  /// Write component to stream.
  virtual void Write(std::ostream &os, bool binary) const;
  
  void Scale(BaseFloat scale); // relates to scaling stats, not parameters.
  void Add(BaseFloat alpha, const NonlinearComponent &other); // relates to
                                                              // adding stats

  // The following functions are unique to NonlinearComponent.
  // They mostly relate to diagnostics.
  const CuVector<double> &ValueSum() const { return value_sum_; }
  const CuVector<double> &DerivSum() const { return deriv_sum_; }
  double Count() const { return count_; }

  // The following function is used when "widening" neural networks.
  void SetDim(int32 dim);
  
 protected:
  friend class NormalizationComponent;
  friend class SigmoidComponent;
  friend class TanhComponent;
  friend class SoftmaxComponent;
  friend class RectifiedLinearComponent;
  friend class SoftHingeComponent;
  
  // This function updates the stats "value_sum_", "deriv_sum_", and
  // count_. (If deriv == NULL, it won't update "deriv_sum_").
  // It will be called from the Backprop function of child classes.
  void UpdateStats(const CuMatrixBase<BaseFloat> &out_value,
                   const CuMatrixBase<BaseFloat> *deriv = NULL);
  
  const NonlinearComponent &operator = (const NonlinearComponent &other); // Disallow.
  int32 dim_;
  CuVector<double> value_sum_; // stats at the output.
  CuVector<double> deriv_sum_; // stats of the derivative of the nonlinearity (only
  // applicable to element-by-element nonlinearities, not Softmax.
  double count_;
};

class MaxoutComponent: public Component {
 public:
  void Init(int32 input_dim, int32 output_dim);
  explicit MaxoutComponent(int32 input_dim, int32 output_dim) {
    Init(input_dim, output_dim);
  }
  MaxoutComponent(): input_dim_(0), output_dim_(0) { }
  virtual std::string Type() const { return "MaxoutComponent"; }
  virtual void InitFromString(std::string args); 
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual Component* Copy() const { return new MaxoutComponent(input_dim_,
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

class PnormComponent: public Component {
 public:
  void Init(int32 input_dim, int32 output_dim, BaseFloat p);
  explicit PnormComponent(int32 input_dim, int32 output_dim, BaseFloat p) {
    Init(input_dim, output_dim, p);
  }
  PnormComponent(): input_dim_(0), output_dim_(0), p_(0) { }
  virtual std::string Type() const { return "PnormComponent"; }
  virtual void InitFromString(std::string args); 
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual Component* Copy() const { return new PnormComponent(input_dim_,
                                                              output_dim_, p_); }
  
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

  virtual std::string Info() const;
 protected:
  int32 input_dim_;
  int32 output_dim_;
  BaseFloat p_;
};

class NormalizeComponent: public NonlinearComponent {
 public:
  explicit NormalizeComponent(int32 dim): NonlinearComponent(dim) { }
  explicit NormalizeComponent(const NormalizeComponent &other): NonlinearComponent(other) { }
  NormalizeComponent() { }
  virtual std::string Type() const { return "NormalizeComponent"; }
  virtual Component* Copy() const { return new NormalizeComponent(*this); }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
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
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual Component* Copy() const { return new SigmoidComponent(*this); }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
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
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
 private:
  TanhComponent &operator = (const TanhComponent &other); // Disallow.
};

/// Take the absoute values of an input vector to a power.
/// The derivative for zero input will be treated as zero.
class PowerComponent: public NonlinearComponent {
 public:
  void Init(int32 dim, BaseFloat power = 2);
  explicit PowerComponent(int32 dim, BaseFloat power = 2) {
    Init(dim, power);
  }
  PowerComponent(): dim_(0), power_(2) { }
  virtual std::string Type() const { return "PowerComponent"; }
  virtual void InitFromString(std::string args); 
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual Component* Copy() const { return new PowerComponent(dim_, power_); }
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

  virtual std::string Info() const;

 private:
  int32 dim_;
  BaseFloat power_;
};

class RectifiedLinearComponent: public NonlinearComponent {
 public:
  explicit RectifiedLinearComponent(int32 dim): NonlinearComponent(dim) { }
  explicit RectifiedLinearComponent(const RectifiedLinearComponent &other): NonlinearComponent(other) { }
  RectifiedLinearComponent() { }
  virtual std::string Type() const { return "RectifiedLinearComponent"; }
  virtual Component* Copy() const { return new RectifiedLinearComponent(*this); }
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
 private:
  RectifiedLinearComponent &operator = (const RectifiedLinearComponent &other); // Disallow.
};

class SoftHingeComponent: public NonlinearComponent {
 public:
  explicit SoftHingeComponent(int32 dim): NonlinearComponent(dim) { }
  explicit SoftHingeComponent(const SoftHingeComponent &other): NonlinearComponent(other) { }
  SoftHingeComponent() { }
  virtual std::string Type() const { return "SoftHingeComponent"; }
  virtual Component* Copy() const { return new SoftHingeComponent(*this); }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
 private:
  SoftHingeComponent &operator = (const SoftHingeComponent &other); // Disallow.
};


// This class scales the input by a specified constant.  This is, of course,
// useless, but we use it when we want to change how fast the next layer learns.
// (e.g. a smaller scale will make the next layer learn slower.)
class ScaleComponent: public Component {
 public:
  explicit ScaleComponent(int32 dim, BaseFloat scale): dim_(dim), scale_(scale) { }
  explicit ScaleComponent(const ScaleComponent &other):
      dim_(other.dim_), scale_(other.scale_) { }
  ScaleComponent(): dim_(0), scale_(0.0) { }
  virtual std::string Type() const { return "ScaleComponent"; }
  virtual Component* Copy() const { return new ScaleComponent(*this); }
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update,
                        CuMatrix<BaseFloat> *in_deriv) const;

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void Read(std::istream &is, bool binary);
  
  virtual void Write(std::ostream &os, bool binary) const;

  void Init(int32 dim, BaseFloat scale);
  
  virtual void InitFromString(std::string args); 

  virtual std::string Info() const;
  
 private:
  int32 dim_;
  BaseFloat scale_;
  ScaleComponent &operator = (const ScaleComponent &other); // Disallow.
};



class SumGroupComponent; // Forward declaration.
class AffineComponent; // Forward declaration.

class SoftmaxComponent: public NonlinearComponent {
 public:
  explicit SoftmaxComponent(int32 dim): NonlinearComponent(dim) { }
  explicit SoftmaxComponent(const SoftmaxComponent &other): NonlinearComponent(other) { }  
  SoftmaxComponent() { }
  virtual std::string Type() const { return "SoftmaxComponent"; }  // Make it lower case
  // because each type of Component needs a different first letter.
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  
  void MixUp(int32 num_mixtures,
             BaseFloat power,
             BaseFloat min_count,
             BaseFloat perturb_stddev,
             AffineComponent *ac,
             SumGroupComponent *sc);
  
  virtual Component* Copy() const { return new SoftmaxComponent(*this); }
 private:
  SoftmaxComponent &operator = (const SoftmaxComponent &other); // Disallow.
};


class FixedAffineComponent;

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// function as a base-class for more specialized versions of
// AffineComponent.
class AffineComponent: public UpdatableComponent {
  friend class SoftmaxComponent; // Friend declaration relates to mixing up.
 public:
  explicit AffineComponent(const AffineComponent &other);
  // The next constructor is used in converting from nnet1.
  AffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                  const CuVectorBase<BaseFloat> &bias_params,
                  BaseFloat learning_rate);
  
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(BaseFloat learning_rate,
            std::string matrix_filename);

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.
  Component *CollapseWithNext(const AffineComponent &next) const ;
  Component *CollapseWithNext(const FixedAffineComponent &next) const;
  Component *CollapseWithPrevious(const FixedAffineComponent &prev) const;

  virtual std::string Info() const;
  virtual void InitFromString(std::string args);
  
  AffineComponent(): is_gradient_(false) { } // use Init to really initialize.
  virtual std::string Type() const { return "AffineComponent"; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value, // dummy
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual void SetZero(bool treat_as_gradient);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual Component* Copy() const;
  virtual void PerturbParams(BaseFloat stddev);
  // This new function is used when mixing up:
  virtual void SetParams(const VectorBase<BaseFloat> &bias,
                         const MatrixBase<BaseFloat> &linear);
  const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }

  virtual int32 GetParameterDim() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  /// This function is for getting a low-rank approximations of this
  /// AffineComponent by two AffineComponents.
  virtual void LimitRank(int32 dimension,
                         AffineComponent **a, AffineComponent **b) const;

  /// This function is implemented in widen-nnet.cc
  void Widen(int32 new_dimension,
             BaseFloat param_stddev,
             BaseFloat bias_stddev,
             std::vector<NonlinearComponent*> c2, // will usually have just one
                                                  // element.
             AffineComponent *c3);
 protected:
  friend class AffineComponentPreconditionedOnline;
  friend class AffineComponentA;
  // This function Update() is for extensibility; child classes may override this.
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  // UpdateSimple is used when *this is a gradient.  Child classes may
  // or may not override this.
  virtual void UpdateSimple(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);  

  const AffineComponent &operator = (const AffineComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
  CuVector<BaseFloat> bias_params_;

  bool is_gradient_; // If true, treat this as just a gradient.
};


/// PiecewiseLinearComponent is a kind of trainable version of the
/// RectifiedLinearComponent, in which each dimension of the nonlinearity has a
/// number of parameters that can be trained.  it's of the form 
/// alpha + beta x + gamma_1 |x - c_1| + gamma_2 |x - c_2| + ... + gamma_N |x - c_N|
/// where c_1 ... c_N on are constants (by default, equally
/// spaced between -1 and 1), and the alpha, beta and gamma quantities are trainable.
/// (Each dimension has separate alpha, beta and gamma quantities).
/// We require that N be odd so that the "middle" gamma quantity corresponds
/// to zero; this is for convenience of initialization so that it corresponds
/// to ReLus.
class PiecewiseLinearComponent: public UpdatableComponent {
 public:
  explicit PiecewiseLinearComponent(const PiecewiseLinearComponent &other);
  virtual int32 InputDim() const { return params_.NumRows(); }
  virtual int32 OutputDim() const { return params_.NumRows(); }

  void Init(int32 dim, int32 N,
            BaseFloat learning_rate,
            BaseFloat max_change);

  virtual std::string Info() const;
  
  virtual void InitFromString(std::string args);
  
  PiecewiseLinearComponent(): is_gradient_(false), max_change_(0.0) { } // use Init to really initialize.
  
  virtual std::string Type() const { return "PiecewiseLinearComponent"; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value, // dummy
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual void SetZero(bool treat_as_gradient);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual Component* Copy() const;
  virtual void PerturbParams(BaseFloat stddev);

  const CuMatrix<BaseFloat> &Params() { return params_; }
  
  virtual int32 GetParameterDim() const;

  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

 protected:
  const PiecewiseLinearComponent &operator = (const PiecewiseLinearComponent &other); // Disallow.
  CuMatrix<BaseFloat> params_;
  
  bool is_gradient_; // If true, treat this as just a gradient.
  BaseFloat max_change_; // If nonzero, maximum change allowed per individual
                         // parameter per minibatch.  
};


// This is an idea Dan is trying out, a little bit like
// preconditioning the update with the Fisher matrix, but the
// Fisher matrix has a special structure.
// [note: it is currently used in the standard receipe].
class AffineComponentPreconditioned: public AffineComponent {
 public:
  virtual std::string Type() const { return "AffineComponentPreconditioned"; }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev,
            BaseFloat alpha, BaseFloat max_change);
  void Init(BaseFloat learning_rate, BaseFloat alpha,
            BaseFloat max_change, std::string matrix_filename);
  
  virtual void InitFromString(std::string args);
  virtual std::string Info() const;
  virtual Component* Copy() const;
  AffineComponentPreconditioned(): alpha_(1.0), max_change_(0.0) { }
  void SetMaxChange(BaseFloat max_change) { max_change_ = max_change; }
 protected:
  KALDI_DISALLOW_COPY_AND_ASSIGN(AffineComponentPreconditioned);
  BaseFloat alpha_;
  BaseFloat max_change_; // If > 0, this is the maximum amount of parameter change (in L2 norm)
                         // that we allow per minibatch.  This was introduced in order to
                         // control instability.  Instead of the exact L2 parameter change,
                         // for efficiency purposes we limit a bound on the exact change.
                         // The limit is applied via a constant <= 1.0 for each minibatch,
                         // A suitable value might be, for example, 10 or so; larger if there are
                         // more parameters.

  /// The following function is only called if max_change_ > 0.  It returns the
  /// greatest value alpha <= 1.0 such that (alpha times the sum over the
  /// row-index of the two matrices of the product the l2 norms of the two rows
  /// times learning_rate_)
  /// is <= max_change.
  BaseFloat GetScalingFactor(const CuMatrix<BaseFloat> &in_value_precon,
                             const CuMatrix<BaseFloat> &out_deriv_precon);

  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
};


/// AffineComponentPreconditionedOnline is, like AffineComponentPreconditioned,
/// a version of AffineComponent that has a non-(multiple of unit) learning-rate
/// matrix.  See nnet-precondition-online.h for a description of the technique.
/// This method maintains an orthogonal matrix N with a small number of rows,
/// actually two (for input and output dims) which gets modified each time;
/// we maintain a mutex for access to this (we just use it to copy it when
/// we need it and write to it when we change it).  For multi-threaded use,
/// the parallelization method is to lock a mutex whenever we want to
/// read N or change it, but just quickly make a copy and release the mutex;
/// this is to ensure operations on N are atomic.
class AffineComponentPreconditionedOnline: public AffineComponent {
 public:
  virtual std::string Type() const {
    return "AffineComponentPreconditionedOnline";
  }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev,
            int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_sample);
  void Init(BaseFloat learning_rate, int32 rank_in,
            int32 rank_out, int32 update_period,
            BaseFloat num_samples_history,
            BaseFloat alpha, BaseFloat max_change_per_sample,
            std::string matrix_filename);

  // This constructor is used when converting neural networks partway through
  // training, from AffineComponent or AffineComponentPreconditioned to
  // AffineComponentPreconditionedOnline.
  AffineComponentPreconditionedOnline(const AffineComponent &orig,
                                      int32 rank_in, int32 rank_out,
                                      int32 update_period,
                                      BaseFloat eta, BaseFloat alpha);
  
  virtual void InitFromString(std::string args);
  virtual std::string Info() const;
  virtual Component* Copy() const;
  AffineComponentPreconditionedOnline(): max_change_per_sample_(0.0) { }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(AffineComponentPreconditionedOnline);


  // Configs for preconditioner.  The input side tends to be better conditioned ->
  // smaller rank needed, so make them separately configurable.
  int32 rank_in_;
  int32 rank_out_;
  int32 update_period_;
  BaseFloat num_samples_history_;
  BaseFloat alpha_;
  
  OnlinePreconditioner preconditioner_in_;

  OnlinePreconditioner preconditioner_out_;

  BaseFloat max_change_per_sample_;
  // If > 0, max_change_per_sample_ this is the maximum amount of parameter
  // change (in L2 norm) that we allow per sample, averaged over the minibatch.
  // This was introduced in order to control instability.
  // Instead of the exact L2 parameter change, for
  // efficiency purposes we limit a bound on the exact
  // change.  The limit is applied via a constant <= 1.0
  // for each minibatch, A suitable value might be, for
  // example, 10 or so; larger if there are more
  // parameters.

  /// The following function is only called if max_change_per_sample_ > 0, it returns a
  /// scaling factor alpha <= 1.0 (1.0 in the normal case) that enforces the
  /// "max-change" constraint.  "in_products" is the inner product with itself
  /// of each row of the matrix of preconditioned input features; "out_products"
  /// is the same for the output derivatives.  gamma_prod is a product of two
  /// scalars that are output by the preconditioning code (for the input and
  /// output), which we will need to multiply into the learning rate.
  /// out_products is a pointer because we modify it in-place.
  BaseFloat GetScalingFactor(const CuVectorBase<BaseFloat> &in_products,
                             BaseFloat gamma_prod,
                             CuVectorBase<BaseFloat> *out_products);

  // Sets the configs rank, alpha and eta in the preconditioner objects,
  // from the class variables.
  void SetPreconditionerConfigs();

  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
};


/// AffineComponentModified as as AffineComponent but we are careful about
/// the lengths of rows of the parameter matrix, when we do the update.
/// That means, for a given row, we first do an update along the direction of
/// the existing vector; we then take the update orthogonal to that direction,
/// but keep the length of the vector fixed.
class AffineComponentModified: public AffineComponent {
 public:
  virtual std::string Type() const { return "AffineComponentModified"; }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev,
            BaseFloat cutoff_length, BaseFloat max_change);
  void Init(BaseFloat learning_rate, BaseFloat cutoff_length,
            BaseFloat max_change, std::string matrix_filename);
  
  virtual void InitFromString(std::string args);
  virtual std::string Info() const;
  virtual Component* Copy() const;
  AffineComponentModified(): cutoff_length_(10.0), max_change_(0.1) { }
  
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(AffineComponentModified);

  BaseFloat cutoff_length_; /// If the length of the vector corresponding to
  /// this row of the parameter matrix is less than this, we just do a regular
  /// gradient descent update.  This would typically be less than
  /// sqrt(InputDim())-- a value smaller than the expected length of the
  /// parameter vector.
  
  BaseFloat max_change_; /// [if above the cutoff], this is the maximum
                         /// change allowed in the vector per minibatch,
                         /// as a proportion of the previous value.  We separately
                         /// apply this constraint to both the length and direction.  Should
                         /// be less than one, e.g. 0.1 or 0.01.

  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
};


class RandomComponent: public Component {
 public:
  // This function is required in testing code and in other places we need
  // consistency in the random number generation (e.g. when optimizing
  // validation-set performance), but check where else we call sRand().  You'll
  // need to call srand as well as making this call.  
  void ResetGenerator() { random_generator_.SeedGpu(0); }
 protected:
  CuRand<BaseFloat> random_generator_;
};



struct PreconditionConfig { // relates to AffineComponentA
  BaseFloat alpha;
  bool do_precondition;
  bool renormalize;
  
  PreconditionConfig(): alpha(0.1), do_precondition(true),
                        renormalize(true) { }
  void Register(OptionsItf *po) {
    po->Register("alpha", &alpha, "Smoothing constant used in "
                 "preconditioning of updates.");
    po->Register("do-precondition", &do_precondition, "Controls whether "
                 "or not preconditioning is applied in the L-BFGS update.");
    po->Register("renormalize", &renormalize, "If true, in the preconditioning "
                 "we renormalize with a scalar so the projected scatter has the "
                 "same trace as before preconditioning.");
  }
};


/**
   AffineComponentA is a special type of AffineComponent, that
   stores matrices for preconditioning similar to those used
   in the update function of AffineComponentPreconditioned.  This is
   intended for use as a preconditioner in L-BFGS updates.
   In this case we optionally store the preconditioning
   information with the gradient information, in a separate
   copy of the component.
*/
class AffineComponentA: public AffineComponent {
 public:
  AffineComponentA() { }
  
  virtual std::string Type() const { return "AffineComponentA"; }
  
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  // There is no Init function for now; we only have the
  // ability to initialize from another AffineComponent (or child
  // class).  This is because we imagine that the L-BFGS training
  // will be initialized from a system trained with SGD, for which
  // something like AffineComponentPreconditioned will be more
  // appropriate; we'll then convert the model.
  AffineComponentA(const AffineComponent &component);

  // We're not supporting initializing as this type.
  virtual void InitFromString(std::string args) { KALDI_ASSERT(0); }
  virtual Component* Copy() const;

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);

  
  // Some functions that are specific to this class:
  void InitializeScatter(); // Lets the class
  // know that it should accumulate the scatter matrix; sets
  // up input_scatter_ and output_scatter_.


  // This function uses the input_scatter_ and output_scatter_ variables of the
  // current class to transform the linear_params_ and bias_params_ variables of
  // "component".  If forward == true then we transform to the preconditioned
  // space; otherwise we transform back from the preconditioned to the canonical
  // space.  This is done differently depending if component->is_gradient_ ==
  // true, because gradients and parameters transform differently.  The alpha
  // value relates to smoothing with the unit matrix; it's not defined in quite
  // the same way as for AffineComponentPreconditioned.  See the code for
  // details.
  void Transform(const PreconditionConfig &config,
                 bool forward,
                 AffineComponent *component);

  // This function uses the input_scatter_ and output_scatter_ variables
  // current class to transform the linear_params_ and bias_params_ variables of
  // "component".  It is equivalent to multiplying by the inverse Fisher,
  // or approximate inverse Hessian.  It's the operation that you need
  // in optimization methods like L-BFGS, to transform from "gradient space"
  // into "model space".
  // Note: it's not const in this object, because we may cache stuff with the model.
  // See also the function "PreconditionNnet" in nnet-lbfgs.h, which
  // does this at the whole-neural-net level (by calling this function).
  void Precondition(const PreconditionConfig &config,
                    AffineComponent *component);
  
 private:

  // The following variables are not used for the actual neural net, but
  // only when is_gradient_ == true (when it's being used to store gradients),

  CuSpMatrix<double> input_scatter_; // scatter of (input vectors extended with 1.)
  // This is only set up if this->is_gradient = true, and InitializeScatter()
  // has been called.
  CuSpMatrix<double> output_scatter_;

  // The following four quantities may be cached by the function "Transform",
  // to avoid duplicating work.
  CuTpMatrix<double> in_C_;
  CuTpMatrix<double> in_C_inv_;
  CuTpMatrix<double> out_C_;
  CuTpMatrix<double> out_C_inv_;

  // The following two quantities may be cached by the function "Precondition",
  // to avoid duplicating work.
  CuSpMatrix<double> inv_fisher_in_;
  CuSpMatrix<double> inv_fisher_out_;
  
  // This function computes the matrix (and corresponding transpose-ness) that
  // we'd left-multiply a vector by when transforming the parameter/gradient
  // space.
  static void ComputeTransforms(const CuSpMatrix<double> &scatter,
                                const PreconditionConfig &config,
                                double tot_count,
                                CuTpMatrix<double> *C,
                                CuTpMatrix<double> *C_inv);

  // This function is called by "Precondition"; it pre-computes
  // certain quantities we'll need.
  static void ComputePreconditioner(const CuSpMatrix<double> &scatter,
                                    const PreconditionConfig &config,
                                    double tot_count,
                                    CuSpMatrix<double> *inv_fisher);

  void ClearPrecomputedQuantities();
  
  // The following update function is called when *this is
  // a gradient.  We only override this one.
  virtual void UpdateSimple(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
};


/// Splices a context window of frames together [over time]
class SpliceComponent: public Component {
 public:
  SpliceComponent() { }  // called only prior to Read() or Init().
  void Init(int32 input_dim,
            int32 left_context,
            int32 right_context,
            int32 const_component_dim=0);
  virtual std::string Type() const { return "SpliceComponent"; }
  virtual std::string Info() const;
  virtual void InitFromString(std::string args);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const;
  virtual int32 LeftContext() const { return left_context_; }
  virtual int32 RightContext() const { return right_context_; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(SpliceComponent);
  int32 input_dim_;
  int32 left_context_;
  int32 right_context_;
  int32 const_component_dim_;
};



/// This is as SpliceComponent but outputs the max of
/// any of the inputs (taking the max across time).
class SpliceMaxComponent: public Component {
 public:
  SpliceMaxComponent() { }  // called only prior to Read() or Init().
  void Init(int32 dim,
            int32 left_context,
            int32 right_context);
  virtual std::string Type() const { return "SpliceMaxComponent"; }
  virtual std::string Info() const;
  virtual void InitFromString(std::string args);
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual int32 LeftContext() const { return left_context_; }
  virtual int32 RightContext() const { return right_context_; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(SpliceMaxComponent);
  int32 dim_;
  int32 left_context_;
  int32 right_context_;
};


// Affine means a linear function plus an offset.  PreconInput means we
// precondition using the inverse of the variance of each dimension of the input
// data.  Note that this doesn't take into account any scaling of the samples,
// but this doesn't really matter.  This has some relation to AdaGrad, except
// it's being done not per input dimension, rather than per parameter, and also
// we multiply by a separately supplied and updated learning rate which will
// typically vary with time.  Note: avg_samples is the number of samples over
// which we average the variance of the input data.
class AffinePreconInputComponent: public AffineComponent {
 public:
  void Init(BaseFloat learning_rate,
                    int32 input_dim, int32 output_dim,
                    BaseFloat param_stddev,
                    BaseFloat bias_stddev,
                    BaseFloat avg_samples);
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value, // dummy
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  AffinePreconInputComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "AffinePreconInputComponent"; }
  virtual void InitFromString(std::string args);
  virtual void SetZero(bool treat_as_gradient);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(AffinePreconInputComponent);
  BaseFloat avg_samples_; // Config parameter; determines how many samples
  // we average the input feature variance over during training
  bool is_gradient_; // Set this to true if we consider this as a gradient.
  // In this case we don't do the input preconditioning.

  // Note: linear_params_ and bias_params_ are inherited from
  // AffineComponent.
  CuVector<BaseFloat> input_precision_; // Inverse variance of input features; used
  // to precondition the update.
};



// Affine means a linear function plus an offset.  "Block" means
// here that we support a number of equal-sized blocks of parameters,
// in the linear part, so e.g. 2 x 500 would mean 2 blocks of 500 each.
class BlockAffineComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return linear_params_.NumCols() * num_blocks_; }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }
  virtual int32 GetParameterDim() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Note: num_blocks must divide input_dim.
  void Init(BaseFloat learning_rate,
                    int32 input_dim, int32 output_dim,
                    BaseFloat param_stddev, BaseFloat bias_stddev,
                    int32 num_blocks);
  virtual void InitFromString(std::string args);
  
  BlockAffineComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "BlockAffineComponent"; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".                        
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual void SetZero(bool treat_as_gradient);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual Component* Copy() const;
  virtual void PerturbParams(BaseFloat stddev);
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
 protected:
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  // UpdateSimple is used when *this is a gradient.  Child classes may
  // override this.
  virtual void UpdateSimple(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
  
  // The matrix linear_parms_ has a block structure, with num_blocks_ blocks fo
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
  KALDI_DISALLOW_COPY_AND_ASSIGN(BlockAffineComponent);

};


// Affine means a linear function plus an offset.  "Block" means
// here that we support a number of equal-sized blocks of parameters,
// in the linear part, so e.g. 2 x 500 would mean 2 blocks of 500 each.
class BlockAffineComponentPreconditioned: public BlockAffineComponent {
 public:
  // Note: num_blocks must divide input_dim.
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev,
            int32 num_blocks, BaseFloat alpha);
  
  virtual void InitFromString(std::string args);
  
  BlockAffineComponentPreconditioned() { } // use Init to really initialize.
  virtual std::string Type() const { return "BlockAffineComponentPreconditioned"; }
  virtual void SetZero(bool treat_as_gradient);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BlockAffineComponentPreconditioned);
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  bool is_gradient_;
  BaseFloat alpha_;
};



// MixtureProbComponent is a linear transform, but it's kind of a special case.
// It's used to transform probabilities while retaining the sum-to-one
// constraint (after the softmax), so we require nonnegative
// elements that sum to one for each column.  In addition, this component
// implements a linear transformation that's a block matrix... not quite
// block diagonal, because the component matrices aren't necessarily square.
// They start off square, but as we mix up, they may get non-square.
//
// From its external interface, i.e. DotProduct(), Scale(), and Backprop(), if
// you use this class in the expected way (e.g. only calling DotProduct()
// between a gradient and the parameters), it behaves as if the parameters were
// stored as unnormalized log-prob and the gradients were taken w.r.t. that
// representation.  This is the only way for the Scale() function to make sense.
// In reality, the parameters are stored as probabilities (normalized to sum to
// one for each row).

class MixtureProbComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  void Init(BaseFloat learning_rate,
            BaseFloat diag_element,
            const std::vector<int32> &sizes);
  virtual void InitFromString(std::string args);  
  MixtureProbComponent() { }
  virtual void SetZero(bool treat_as_gradient);
  virtual std::string Type() const { return "MixtureProbComponent"; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  // Note: in_value and out_value are both dummy variables.
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const;
  
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void PerturbParams(BaseFloat stddev);

  virtual int32 GetParameterDim() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
 private:
  void Refresh(); // Refreshes params_ from log_params_.
  KALDI_DISALLOW_COPY_AND_ASSIGN(MixtureProbComponent);

  std::vector<CuMatrix<BaseFloat> > log_params_; // these are the
  // underlying parameters that are subject to gradient descent.
  std::vector<CuMatrix<BaseFloat> > params_; // these are derived from
  // log_params_.
  int32 input_dim_;
  int32 output_dim_;
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
  virtual void InitFromString(std::string args);
  SumGroupComponent() { }
  virtual std::string Type() const { return "SumGroupComponent"; }
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  // Note: in_value and out_value are both dummy variables.
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
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


/// PermuteComponent does a permutation of the dimensions (by default, a fixed
/// random permutation, but it may be specified).  Useful in conjunction with
/// block-diagonal transforms.
class PermuteComponent: public Component {
 public:
  void Init(int32 dim);
  void Init(const std::vector<int32> &reorder);
  PermuteComponent(int32 dim) { Init(dim); }
  PermuteComponent(const std::vector<int32> &reorder) { Init(reorder); }

  PermuteComponent() { } // e.g. prior to Read() or Init()
  
  virtual int32 InputDim() const { return reorder_.size(); }
  virtual int32 OutputDim() const { return reorder_.size(); }
  virtual Component *Copy() const;

  virtual void InitFromString(std::string args);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Type() const { return "PermuteComponent"; }
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &,
                        const CuMatrixBase<BaseFloat> &,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *,
                        CuMatrix<BaseFloat> *in_deriv) const;
  
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(PermuteComponent);
  std::vector<int32> reorder_; // This class sends input dimension i to
                               // output dimension reorder_[i].
};


/// Discrete cosine transform.
/// TODO: modify this Component so that it supports only keeping a subset 
class DctComponent: public Component {
 public:
  DctComponent() { dim_ = 0; } 
  virtual std::string Type() const { return "DctComponent"; }
  virtual std::string Info() const;
  //dim = dimension of vector being processed
  //dct_dim = effective lenght of DCT, i.e. how many compoments will be kept
  void Init(int32 dim, int32 dct_dim, bool reorder, int32 keep_dct_dim=0);
  // InitFromString takes numeric options
  // dim, dct-dim, and (optionally) reorder={true,false}, keep-dct-dim
  // Note: reorder defaults to false. keep-dct-dim defaults to dct-dim
  virtual void InitFromString(std::string args);
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dct_mat_.NumRows() * (dim_ / dct_mat_.NumCols()); }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
 private:
  void Reorder(CuMatrixBase<BaseFloat> *mat, bool reverse) const;
  int32 dim_; // The input dimension of the (sub)vector.

  bool reorder_; // If true, transformation matrix we use is not
  // block diagonal but is block diagonal after reordering-- so
  // effectively we transform with the Kronecker product D x I,
  // rather than a matrix with D's on the diagonal (i.e. I x D,
  // where x is the Kronecker product).  We'll set reorder_ to
  // true if we want to use this to transform in the time domain,
  // because the SpliceComponent splices blocks of e.g. MFCCs
  // together so each time is a dimension of the block.

  CuMatrix<BaseFloat> dct_mat_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DctComponent);
};


/// FixedLinearComponent is a linear transform that is supplied
/// at network initialization time and is not trainable.
class FixedLinearComponent: public Component {
 public:
  FixedLinearComponent() { } 
  virtual std::string Type() const { return "FixedLinearComponent"; }
  virtual std::string Info() const;
  
  void Init(const CuMatrixBase<BaseFloat> &matrix) { mat_ = matrix; }

  // InitFromString takes only the option matrix=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromString(std::string args);
  
  virtual int32 InputDim() const { return mat_.NumCols(); }
  virtual int32 OutputDim() const { return mat_.NumRows(); }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
 protected:
  friend class AffineComponent;
  CuMatrix<BaseFloat> mat_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(FixedLinearComponent);
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

  // InitFromString takes only the option matrix=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromString(std::string args);
  
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
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


/// FixedScaleComponent applies a fixed per-element scale; it's similar
/// to the Rescale component in the nnet1 setup (and only needed for nnet1
/// model conversion.
class FixedScaleComponent: public Component {
 public:
  FixedScaleComponent() { } 
  virtual std::string Type() const { return "FixedScaleComponent"; }
  virtual std::string Info() const;
  
  void Init(const CuVectorBase<BaseFloat> &scales); 
  
  // InitFromString takes only the option scales=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromString(std::string args);
  
  virtual int32 InputDim() const { return scales_.Dim(); }
  virtual int32 OutputDim() const { return scales_.Dim(); }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
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
  
  void Init(const CuVectorBase<BaseFloat> &scales); 
  
  // InitFromString takes only the option bias=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromString(std::string args);
  
  virtual int32 InputDim() const { return bias_.Dim(); }
  virtual int32 OutputDim() const { return bias_.Dim(); }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const;
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  CuVector<BaseFloat> bias_;  
  KALDI_DISALLOW_COPY_AND_ASSIGN(FixedBiasComponent);
};


/// This Component, if present, randomly zeroes half of
/// the inputs and multiplies the other half by two.
/// Typically you would use this in training but not in
/// test or when computing validation-set objective functions.
class DropoutComponent: public RandomComponent {
 public:
  /// dropout-proportion is the proportion that is dropped out,
  /// e.g. if 0.1, we set 10% to a low value.  [note, in
  /// some older code it was interpreted as the value not dropped
  /// out, so be careful.]  The low scale-value
  /// is equal to dropout_scale.  The high scale-value is chosen
  /// such that the expected scale-value is one.
  void Init(int32 dim,
            BaseFloat dropout_proportion = 0.5,
            BaseFloat dropout_scale = 0.0);
  DropoutComponent(int32 dim, BaseFloat dp = 0.5, BaseFloat sc = 0.0) {
    Init(dim, dp, sc);
  }
  DropoutComponent(): dim_(0), dropout_proportion_(0.5) { }
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void InitFromString(std::string args);

  virtual void Read(std::istream &is, bool binary);
  
  virtual void Write(std::ostream &os, bool binary) const;
      
  virtual std::string Type() const { return "DropoutComponent"; }

  void SetDropoutScale(BaseFloat scale) { dropout_scale_ = scale; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return true; }  
  virtual Component* Copy() const;
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual std::string Info() const;
 private:
  int32 dim_;  
  BaseFloat dropout_proportion_;
  BaseFloat dropout_scale_; // Set the scale that we scale "dropout_proportion_"
  // of the neurons by (default 0.0, but can be set arbitrarily close to 1.0).
};

/// This is a bit similar to dropout but adding (not multiplying) Gaussian
/// noise with a given standard deviation.
class AdditiveNoiseComponent: public RandomComponent {
 public:
  void Init(int32 dim, BaseFloat noise_stddev);
  AdditiveNoiseComponent(int32 dim, BaseFloat stddev) { Init(dim, stddev); }
  AdditiveNoiseComponent(): dim_(0), stddev_(1.0) { }
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void InitFromString(std::string args);

  virtual void Read(std::istream &is, bool binary);
  
  virtual void Write(std::ostream &os, bool binary) const;
      
  virtual std::string Type() const { return "AdditiveNoiseComponent"; }

  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }  
  virtual Component* Copy() const {
    return new AdditiveNoiseComponent(dim_, stddev_);
  }
  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         CuMatrix<BaseFloat> *out) const; 
  virtual void Backprop(const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        int32 num_chunks,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const { *in_deriv = out_deriv; }
 private:
  int32 dim_;  
  BaseFloat stddev_;
};


/// Functions used in Init routines.  Suppose name=="foo", if "string" has a
/// field like foo=12, this function will set "param" to 12 and remove that
/// element from "string".  It returns true if the parameter was read.
bool ParseFromString(const std::string &name, std::string *string,
                     int32 *param);
/// This version is for parameters of type BaseFloat.
bool ParseFromString(const std::string &name, std::string *string,
                     BaseFloat *param);
/// This version is for parameters of type std::vector<int32>; it expects
/// them as a colon-separated list, without spaces.
bool ParseFromString(const std::string &name, std::string *string,
                     std::vector<int32> *param);
/// This version is for parameters of type bool, which can appear
/// as any string beginning with f, F, t or T.
bool ParseFromString(const std::string &name, std::string *string,
                     bool *param);


} // namespace nnet2
} // namespace kaldi


#endif
