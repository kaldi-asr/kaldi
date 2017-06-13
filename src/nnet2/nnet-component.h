// nnet2/nnet-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2014  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//                2014  Vijayaditya Peddinti
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

#ifndef KALDI_NNET2_NNET_COMPONENT_H_
#define KALDI_NNET2_NNET_COMPONENT_H_

#include <mutex>
#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "nnet2/nnet-precondition-online.h"

#include <iostream>

namespace kaldi {
namespace nnet2 {


/**
   ChunkInfo is a class whose purpose is to describe the structure of matrices
   holding features.  This is useful mostly in training time.
   The main reason why we have this is to support efficient
   training for networks which we have splicing components that splice in a
   non-contiguous way, e.g. frames -5, 0 and 5.  We also have in mind future
   extensibility to convnets which might have similar issues.  This class
   describes the structure of a minibatch of features, or of a single
   contiguous block of features.
   Examples are as follows, and offsets is empty if not mentioned:
     When decoding, at input to the network:
       feat_dim = 13, num_chunks = 1, first_offset = 0, last_offset = 691
      and in the middle of the network (assuming splicing is +-7):
       feat_dim = 1024, num_chunks = 1, first_offset = 7, last_offset = 684
    When training, at input to the network:
      feat_dim = 13, num_chunks = 512, first_offset = 0, last_offset= 14
     and in the middle of the network:
      feat_dim = 1024, num_chunks = 512, first_offset = 7, last_offset = 7
   The only situation where offsets would be nonempty would be if we do
   splicing with gaps in.  E.g. suppose at network input we splice +-2 frames
   (contiguous) and somewhere in the middle we splice frames {-5, 0, 5}, then
   we would have the following while training
     At input to the network:
      feat_dim = 13, num_chunks = 512, first_offset = 0, last_offset = 14
     After the first hidden layer:
      feat_dim = 1024, num_chunks = 512, first_offset = 2, last_offset = 12,
       offsets = {2, 10, 12}
     At the output of the last hidden layer (after the {-5, 0, 5} splice):
      feat_dim = 1024, num_chunks = 512, first_offset = 7, last_offset = 7
   (the decoding setup would still look pretty normal, so we don't give an example).

*/
class ChunkInfo {
 public:
  ChunkInfo()  // default constructor we assume this object will not be used
      : feat_dim_(0), num_chunks_(0),
        first_offset_(0), last_offset_(0),
        offsets_() { }

  ChunkInfo(int32 feat_dim, int32 num_chunks,
            int32 first_offset, int32 last_offset )
      : feat_dim_(feat_dim), num_chunks_(num_chunks),
        first_offset_(first_offset), last_offset_(last_offset),
        offsets_() { Check(); }

  ChunkInfo(int32 feat_dim, int32 num_chunks,
            const std::vector<int32> offsets)
      : feat_dim_(feat_dim), num_chunks_(num_chunks),
        first_offset_(offsets.front()), last_offset_(offsets.back()),
        offsets_(offsets) { if (last_offset_ - first_offset_ + 1 == offsets_.size())
                              offsets_.clear();
          Check(); }

  // index : actual row index in the current chunk
  // offset : the time offset of feature frame at current row in the chunk
  // As described above offsets can take a variety of values, we see the indices
  // corresponding to the offsets in each case
  // 1) if first_offset = 0 & last_offset = 691, then chunk has data
  // corresponding to time offsets 0:691, so index = offset
  // 2) if first_offset = 7 & last_offset = 684,
  //      then index = offset - first offset
  // 3) if offsets = {2, 10, 12} then indices for these offsets are 0, 1 and 2

  // Returns the chunk row index corresponding to given time offset
  int32 GetIndex (int32 offset) const;

  // Returns time offset at the current row index in the chunk
  int32 GetOffset (int32 index) const;

  // Makes the offsets vector empty, to ensure that the chunk is processed as a
  // contiguous chunk with the given first_offset and last_offset
  void MakeOffsetsContiguous () { offsets_.clear(); Check(); }

  // Returns chunk size, meaning the number of distinct frame-offsets we
  // have for each chunk (they don't have to be contiguous).
  inline int32 ChunkSize() const { return NumRows() / num_chunks_; }

  // Returns number of chunks we expect the feature matrix to have
  inline int32 NumChunks() const { return num_chunks_; }

  /// Returns the number of rows that we expect the feature matrix to have.
  int32 NumRows() const {
    return num_chunks_ * (!offsets_.empty() ? offsets_.size() :
                                         last_offset_ - first_offset_ + 1); }

  /// Returns the number of columns that we expect the feature matrix to have.
  int32 NumCols() const { return feat_dim_; }

  /// Checks that the matrix has the size we expect, and die if not.
  void CheckSize(const CuMatrixBase<BaseFloat> &mat) const;

  /// Checks that the data in the ChunkInfo is valid, and die if not.
  void Check() const;

 private:
  int32 feat_dim_;  // Feature dimension.
  int32 num_chunks_;  // Number of separate equal-sized chunks of features
  int32 first_offset_;  // Start time offset within each chunk, numbered so that at
                      // the input to the network, the first_offset of the first
                      // feature would always be zero.
  int32 last_offset_;  // End time offset within each chunk.
  std::vector<int32> offsets_; // offsets is only nonempty if the chunk contains
                             // a non-contiguous sequence.  If nonempty, it must
                             // be sorted, and offsets.front() == first_offset,
                             // offsets.back() == last_offset.

};

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

  /// Return a vector describing the temporal context this component requires
  /// for each frame of output, as a sorted list.  The default implementation
  /// returns a vector ( 0 ), but a splicing layer might return e.g. (-2, -1, 0,
  /// 1, 2), but it doesn't have to be contiguous.  Note : The context needed by
  /// the entire network is a function of the contexts needed by all the
  /// components.  It is required that Context().front() <= 0 and
  /// Context().back() >= 0.
  virtual std::vector<int32> Context() const { return std::vector<int32>(1, 0); }

  /// Perform forward pass propagation Input->Output.  Each row is
  /// one frame or training example.  Interpreted as "num_chunks"
  /// equally sized chunks of frames; this only matters for layers
  /// that do things like context splicing.  Typically this variable
  /// will either be 1 (when we're processing a single contiguous
  /// chunk of data) or will be the same as in.NumFrames(), but
  /// other values are possible if some layers do splicing.
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const = 0;

  /// A non-virtual propagate function that first resizes output if necessary.
  void Propagate(const ChunkInfo &in_info,
                 const ChunkInfo &out_info,
                 const CuMatrixBase<BaseFloat> &in,
                 CuMatrix<BaseFloat> *out) const {
    if (out->NumRows() != out_info.NumRows() ||
        out->NumCols() != out_info.NumCols()) {
      out->Resize(out_info.NumRows(), out_info.NumCols());
    }

    // Cast to CuMatrixBase to use the virtual version of propagate function.
    Propagate(in_info, out_info, in,
              static_cast<CuMatrixBase<BaseFloat>*>(out));
  }

  /// Perform backward pass propagation of the derivative, and
  /// also either update the model (if to_update == this) or
  /// update another model or compute the model derivative (otherwise).
  /// Note: in_value and out_value are the values of the input and output
  /// of the component, and these may be dummy variables if respectively
  /// BackpropNeedsInput() or BackpropNeedsOutput() return false for
  /// that component (not all components need these).
  ///
  /// num_chunks lets us treat the input matrix as contiguous-in-time
  /// chunks of equal size; it only matters if splicing is involved.
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  friend class LogSoftmaxComponent;
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
  // The mutex is used in UpdateStats, only for resizing vectors.
  std::mutex mutex_;
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &,  //out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  void Init(int32 input_dim, int32 output_dim,
            int32 pool_size, int32 pool_stride);
  explicit MaxpoolingComponent(int32 input_dim, int32 output_dim,
                               int32 pool_size, int32 pool_stride) {
    Init(input_dim, output_dim, pool_size, pool_stride);
  }
  MaxpoolingComponent(): input_dim_(0), output_dim_(0),
    pool_size_(0), pool_stride_(0) { }
  virtual std::string Type() const { return "MaxpoolingComponent"; }
  virtual void InitFromString(std::string args);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &,  //out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return true; }
  virtual Component* Copy() const {
    return new MaxpoolingComponent(input_dim_, output_dim_,
                               pool_size_, pool_stride_); }

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

  virtual std::string Info() const;
 protected:
  int32 input_dim_;
  int32 output_dim_;
  int32 pool_size_;
  int32 pool_stride_;
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &,  //out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  virtual bool BackpropNeedsOutput() const { return false; }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
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



class SumGroupComponent;  // Forward declaration.
class AffineComponent;  // Forward declaration.
class FixedScaleComponent;  // Forward declaration.

class SoftmaxComponent: public NonlinearComponent {
 public:
  explicit SoftmaxComponent(int32 dim): NonlinearComponent(dim) { }
  explicit SoftmaxComponent(const SoftmaxComponent &other): NonlinearComponent(other) { }
  SoftmaxComponent() { }
  virtual std::string Type() const { return "SoftmaxComponent"; }
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return true; }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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

class LogSoftmaxComponent: public NonlinearComponent {
 public:
  explicit LogSoftmaxComponent(int32 dim): NonlinearComponent(dim) { }
  explicit LogSoftmaxComponent(const LogSoftmaxComponent &other): NonlinearComponent(other) { }
  LogSoftmaxComponent() { }
  virtual std::string Type() const { return "LogSoftmaxComponent"; }
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return true; }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;

  virtual Component* Copy() const { return new LogSoftmaxComponent(*this); }
 private:
  LogSoftmaxComponent &operator = (const LogSoftmaxComponent &other); // Disallow.
};


class FixedAffineComponent;

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// functions as a base-class for more specialized versions of
// AffineComponent.
class AffineComponent: public UpdatableComponent {
  friend class SoftmaxComponent; // Friend declaration relates to mixing up.
 public:
  AffineComponent(const AffineComponent &other);
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

  virtual std::string Info() const;
  virtual void InitFromString(std::string args);

  AffineComponent(): is_gradient_(false) { } // use Init to really initialize.
  virtual std::string Type() const { return "AffineComponent"; }
  virtual bool BackpropNeedsInput() const { return true; }
  virtual bool BackpropNeedsOutput() const { return false; }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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


// This is an idea Dan is trying out, a little bit like
// preconditioning the update with the Fisher matrix, but the
// Fisher matrix has a special structure.
// [note: it is currently used in the standard recipe].
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


/// Keywords: natural gradient descent, NG-SGD, naturalgradient.  For
/// the top-level of the natural gradient code look here, and also in
/// nnet-precondition-online.h.
/// AffineComponentPreconditionedOnline is, like AffineComponentPreconditioned,
/// a version of AffineComponent that has a non-(multiple of unit) learning-rate
/// matrix.  See nnet-precondition-online.h for a description of the technique.
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

  virtual void Resize(int32 input_dim, int32 output_dim);

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

class RandomComponent: public Component {
 public:
  // This function is required in testing code and in other places we need
  // consistency in the random number generation (e.g. when optimizing
  // validation-set performance), but check where else we call sRand().  You'll
  // need to call srand as well as making this call.
  void ResetGenerator() { random_generator_.SeedGpu(); }
 protected:
  CuRand<BaseFloat> random_generator_;
};

/// Splices a context window of frames together [over time]
class SpliceComponent: public Component {
 public:
  SpliceComponent() { }  // called only prior to Read() or Init().
  // Note: it is required that the elements of "context" be in
  // strictly increasing order, that the lowest element of component
  // be nonpositive, and the highest element be nonnegative.
  void Init(int32 input_dim,
            std::vector<int32> context,
            int32 const_component_dim=0);
  virtual std::string Type() const { return "SpliceComponent"; }
  virtual std::string Info() const;
  virtual void InitFromString(std::string args);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const;
  virtual std::vector<int32> Context() const { return context_; }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  std::vector<int32> context_;
  int32 const_component_dim_;
};

/// This is as SpliceComponent but outputs the max of
/// any of the inputs (taking the max across time).
class SpliceMaxComponent: public Component {
 public:
  SpliceMaxComponent() { }  // called only prior to Read() or Init().
  void Init(int32 dim,
            std::vector<int32> context);
  virtual std::string Type() const { return "SpliceMaxComponent"; }
  virtual std::string Info() const;
  virtual void InitFromString(std::string args);
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual std::vector<int32> Context() const  { return context_; }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  std::vector<int32> context_;
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  // Note: in_value and out_value are both dummy variables.
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
/// model conversion).
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsInput() const { return false; }
  virtual bool BackpropNeedsOutput() const { return false; }
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

  void Init(const CuVectorBase<BaseFloat> &scales);

  // InitFromString takes only the option bias=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromString(std::string args);

  virtual int32 InputDim() const { return bias_.Dim(); }
  virtual int32 OutputDim() const { return bias_.Dim(); }
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const ;
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
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
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const { *in_deriv = out_deriv; }
 private:
  int32 dim_;
  BaseFloat stddev_;
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

  int32 InputDim() const;
  int32 OutputDim() const;
  void Init(BaseFloat learning_rate, int32 input_dim, int32 output_dim,
            int32 patch_dim, int32 patch_step, int32 patch_stride,
            BaseFloat param_stddev, BaseFloat bias_stddev, bool appended_conv);
  void Init(BaseFloat learning_rate,
            int32 patch_dim, int32 patch_step, int32 patch_stride,
            std::string matrix_filename, bool appended_conv);

  // resize the component, setting the parameters to zero, while
  // leaving any other configuration values the same
  void Resize(int32 input_dim, int32 output_dim);
  std::string Info() const;
  void InitFromString(std::string args);
  std::string Type() const { return "Convolutional1dComponent"; }
  bool BackpropNeedsInput() const { return true; }
  bool BackpropNeedsOutput() const { return false; }
  using Component::Propagate; // to avoid name hiding
  void Propagate(const ChunkInfo &in_info,
                 const ChunkInfo &out_info,
                 const CuMatrixBase<BaseFloat> &in,
                 CuMatrixBase<BaseFloat> *out) const;
  void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update_in,
                        CuMatrix<BaseFloat> *in_deriv) const;
  void SetZero(bool treat_as_gradient);
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  Component* Copy() const;
  void PerturbParams(BaseFloat stddev);
  void SetParams(const VectorBase<BaseFloat> &bias,
                 const MatrixBase<BaseFloat> &filter);
  const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return filter_params_; }
  int32 GetParameterDim() const;
  void Update(const CuMatrixBase<BaseFloat> &in_value,
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
  // When appending convolutional1dcomponents, appended_conv_ should be
  // set ture for the appended convolutional1dcomponents.
  bool appended_conv_;
  bool is_gradient_;
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
