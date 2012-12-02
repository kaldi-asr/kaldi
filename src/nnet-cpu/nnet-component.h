// nnet-cpu/nnet-component.h

// Copyright 2011-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_COMPONENT_H_
#define KALDI_NNET_CPU_COMPONENT_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

#include <iostream>

namespace kaldi {


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
  Component() { }
  
  virtual std::string Type() const = 0; // each type should return a string such as
  // "SigmoidComponent".

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
  virtual int32 LeftContext() { return 0; }

  /// Number of right-context frames the component sees for each output frame;
  /// nonzero only for splicing layers.
  virtual int32 RightContext() { return 0; }

  /// Perform forward pass propagation Input->Output.  Each row is
  /// one frame or training example.  Interpreted as "num_chunks"
  /// equally sized chunks of frames; this only matters for layers
  /// that do things like context splicing.  Typically this variable
  /// will either be 1 (when we're processing a single contiguous
  /// chunk of data) or will be the same as in.NumFrames(), but
  /// other values are possible if some layers do splicing.
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const = 0; 
  
  /// Perform backward pass propagation of the derivative, and
  /// also either update the model (if to_update == this) or
  /// update another model or compute the model derivative (otherwise).
  /// Note: in_value and out_value are the values of the input and output
  /// of the component, and these may be dummy variables if respectively
  /// BackpropNeedsInput() or BackpropNeedsOutput() return false for
  /// that component (not all components need these).
  ///
  /// chunk_weights is a vector, indexed by chunk (i.e. the same size as the
  /// "num_chunks" argument to Propagate()), that gives a weighting for each
  /// chunk; in the normal case each of these would be equal to the number of
  /// labels in the chunk (one label per chunk, for standard SGD); but we
  /// support weighting of samples so there may be an additional factor.  This
  /// is only needed for reasons relating to l2 regularization and the storing
  /// of occupation counts.  For SGD we don't need this information, because the
  /// code that computes the objective-function derivative at the output layer
  /// incorporates this weighting. 
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,                        
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const = 0;
  
  virtual bool BackpropNeedsInput() const { return true; } // if this returns false,
  // the "in_value" to Backprop may be a dummy variable.
  virtual bool BackpropNeedsOutput() const { return true; } // if this returns false,
  // the "out_value" to Backprop may be a dummy variable.
  
  /// Read component from stream
  static Component* ReadNew(std::istream &is, bool binary);

  /// Copy component (deep copy).
  virtual Component* Copy() const = 0;

  /// By default this does nothing; it's used in a couple of classes.
  /// For things like zeroing the stored average occupation count.
  virtual void ZeroOccupancy() { }
  
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
  KALDI_DISALLOW_COPY_AND_ASSIGN(Component);
};


/**
 * Class UpdatableComponent is a Component which has
 * trainable parameters and contains some global 
 * parameters for stochastic gradient descent
 * (learning rate, L2 regularization constant).
 * This is a base-class for Components with parameters.
 */
class UpdatableComponent : public Component {
 public:
  void Init(BaseFloat learning_rate) {
    learning_rate_ = learning_rate;
  }
  UpdatableComponent(BaseFloat learning_rate) {
    Init(learning_rate);
  }

  /// Set parameters to zero,
  /// and if treat_as_gradient is true, we'll be treating this as a gradient
  /// so set the learning rate to 1 and l2_penalty to zero and make any other
  /// changes necessary (there's a variable we have to set for the
  /// MixtureProbComponent).
  virtual void SetZero(bool treat_as_gradient) = 0;
  
  /// Note: l2_penalty is per frame.
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
  /// by this amount.  It's used in "parameter shrinkage",
  /// which is related to l2 regularization.
  virtual void Scale(BaseFloat scale) = 0;

  /// This new virtual function adds the parameters of another
  /// updatable component, times some constant, to the current
  /// parameters.
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other) = 0;
  
  /// Sets the learning rate of gradient descent
  void SetLearningRate(BaseFloat lrate) {  learning_rate_ = lrate; }
  /// Gets the learning rate of gradient descent
  BaseFloat LearningRate() const { return learning_rate_; }
 protected:
  BaseFloat learning_rate_; ///< learning rate (0.0..0.01)
};

/// This kind of Component is a base-class for things like
/// sigmoid and softmax.
class NonlinearComponent: public Component {
 public:
  void Init(int32 dim) { dim_ = dim; }
  NonlinearComponent(int32 dim) { Init(dim); }
  NonlinearComponent(): dim_(0) { } // e.g. prior to Read().
  
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  
  /// We implement InitFromString at this level.
  virtual void InitFromString(std::string args);
  
  /// We implement Read at this level as it just needs the Type().
  virtual void Read(std::istream &is, bool binary);
  
  /// Write component to stream.
  virtual void Write(std::ostream &os, bool binary) const;
 
 protected:
  int32 dim_;
};

class SigmoidComponent: public NonlinearComponent {
 public:
  SigmoidComponent(int32 dim): NonlinearComponent(dim) { }
  SigmoidComponent() { }
  virtual std::string Type() const { return "SigmoidComponent"; }
  virtual bool BackpropNeedsInput() { return false; }
  virtual bool BackpropNeedsOutput() { return true; }
  virtual Component* Copy() const { return new SigmoidComponent(dim_); }
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const; 
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(SigmoidComponent);
};

class TanhComponent: public NonlinearComponent {
 public:
  TanhComponent(int32 dim): NonlinearComponent(dim) { }
  TanhComponent() { }
  virtual std::string Type() const { return "TanhComponent"; }
  virtual Component* Copy() const { return new TanhComponent(dim_); }
  virtual bool BackpropNeedsInput() { return false; }
  virtual bool BackpropNeedsOutput() { return true; }
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const; 
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(TanhComponent);
};

class SoftmaxComponent: public NonlinearComponent {
 public:
  SoftmaxComponent(int32 dim) { Init(dim); }
  SoftmaxComponent() { }
  virtual std::string Type() const { return "SoftmaxComponent"; }  // Make it lower case
  // because each type of Component needs a different first letter.
  virtual Component* Copy() const { return new SoftmaxComponent(dim_); }
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const; 
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
  
  virtual void ZeroOccupancy() { counts_.SetZero(); }
  
  // The functions below are already implemented at the
  // NonlinearComponent level, but we override them for reasons relating
  // to the occupation counts.
  void Init(int32 dim) { dim_ = dim; counts_.Resize(dim); }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
 private:
  Vector<BaseFloat> counts_; // Occupation counts per dim.
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(SoftmaxComponent);
};


// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// function as a base-class for more specialized versions of
// AffineComponent.
class AffineComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev,
            bool precondition);
  virtual std::string Info() const;
  virtual void InitFromString(std::string args);
  
  AffineComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "AffineComponent"; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value, // dummy
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
  virtual void SetZero(bool treat_as_gradient);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual Component* Copy() const;
  virtual void PerturbParams(BaseFloat stddev);
  virtual void ZeroOccupancy();
 protected:
  // This function Update() is for extensibility; child classes override this.
  virtual void Update(
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  void UpdateSimple(
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv);  
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(AffineComponent);
  Matrix<BaseFloat> linear_params_;
  Vector<BaseFloat> bias_params_;

  // The following is stored mostly for diagnostics: the
  // average input value in each dimension.
  Vector<BaseFloat> avg_input_;
  double avg_input_count_;
  bool is_gradient_; // If true, treat this as just a gradient.
};

class AffineComponentNobias: public AffineComponent {
 public:
  virtual std::string Type() const { return "AffineComponentNobias"; }
  // The Read and Write functions are shared with AffineComponent,
  // which calls the virtual Type() function.
  virtual Component* Copy() const;
 private:
  virtual void Update(
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv);
};

// This is an idea Dan is trying out, a little bit like
// preconditioning the update with the Fisher matrix.
class AffineComponentPreconditioned: public AffineComponent {
 public:
  virtual std::string Type() const { return "AffineComponentPreconditioned"; }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(BaseFloat learning_rate,
            int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev,
            bool precondition, BaseFloat alpha);
  virtual void InitFromString(std::string args);
  virtual std::string Info() const;
  virtual Component* Copy() const;  

 private:
  BaseFloat alpha_;
  virtual void Update(
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv);
};


/// Splices a context window of frames together.
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
  virtual int32 LeftContext() { return left_context_; }
  virtual int32 RightContext() { return right_context_; }
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const;
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
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
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value, // dummy
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
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
  Vector<BaseFloat> input_precision_; // Inverse variance of input features; used
  // to precondition the update.
};



// Affine means a linear function plus an offset.  "Block" means
// here that we support a number of equal-sized blocks of parameters,
// in the linear part, so e.g. 2 x 500 would mean 2 blocks of 500 each.
class BlockAffineComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const { return linear_params_.NumCols() * num_blocks_; }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }
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
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const; 
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".                        
                        Matrix<BaseFloat> *in_deriv) const;
  virtual void SetZero(bool treat_as_gradient);
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual Component* Copy() const;
  virtual void PerturbParams(BaseFloat stddev);
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BlockAffineComponent);
  // The matrix linear_parms_ has a block structure, with num_blocks_ blocks fo
  // equal size.  The blocks are stored in linear_params_ as
  // [ M
  //   N
  //   O ] but we actually treat it as the matrix:
  // [ M 0 0
  //   0 N 0
  //   0 0 O ]
  Matrix<BaseFloat> linear_params_;
  Vector<BaseFloat> bias_params_;
  int32 num_blocks_;
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
// between a gradient and the parameters), it behaves as if the parameters
// were stored as unnormalized log-prob and the gradients were taken
// w.r.t. that representation.  This is the only way for the Scale() function
// to make sense.  In reality, the parameters are stored as actual
// probabilities (normalized to sum to one for each row).

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
  virtual std::string Type() const { return "MixtureComponent"; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const;
  // Note: in_value and out_value are both dummy variables.
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
  virtual Component* Copy() const;
  
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void PerturbParams(BaseFloat stddev);
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(MixtureProbComponent);
  std::vector<Matrix<BaseFloat> > params_;
  int32 input_dim_;
  int32 output_dim_;
  bool is_gradient_; // true if we're treating this as just a store for the gradient.
};

/// PermuteComponent does a random permutation of the dimensions.  Useful in
/// conjunction with block-diagonal transforms.
class PermuteComponent: public Component {
 public:
  void Init(int32 dim);
  PermuteComponent(int32 dim) { Init(dim); }
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
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const; 
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value, // dummy
                        const MatrixBase<BaseFloat> &out_value, // dummy
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // dummy
                        Matrix<BaseFloat> *in_deriv) const;
  
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
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const;
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
 private:
  void Reorder(MatrixBase<BaseFloat> *mat, bool reverse) const;
  int32 dim_; // The input dimension of the (sub)vector.

  bool reorder_; // If true, transformation matrix we use is not
  // block diagonal but is block diagonal after reordering-- so
  // effectively we transform with the Kronecker product D x I,
  // rather than a matrix with D's on the diagonal (i.e. I x D,
  // where x is the Kronecker product).  We'll set reorder_ to
  // true if we want to use this to transform in the time domain,
  // because the SpliceComponent splices blocks of e.g. MFCCs
  // together so each time is a dimension of the block.

  Matrix<BaseFloat> dct_mat_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DctComponent);
};


/// FixedLinearComponent is a linear transform that is supplied
/// at network initialization time and is not trainable.
class FixedLinearComponent: public Component {
 public:
  FixedLinearComponent() { } 
  virtual std::string Type() const { return "FixedLinearComponent"; }
  virtual std::string Info() const;
  
  void Init(const MatrixBase<BaseFloat> &matrix) { mat_ = matrix; }

  // InitFromString takes only the option matrix=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromString(std::string args);
  
  virtual int32 InputDim() const { return mat_.NumCols(); }
  virtual int32 OutputDim() const { return mat_.NumRows(); }
  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                         int32 num_chunks,
                         Matrix<BaseFloat> *out) const;
  virtual void Backprop(const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        const VectorBase<BaseFloat> &chunk_weights,
                        Component *to_update, // may be identical to "this".
                        Matrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
 private:
  Matrix<BaseFloat> mat_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(FixedLinearComponent);
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


} // namespace kaldi


#endif
