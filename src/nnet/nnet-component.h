// nnet/nnet-component.h

// Copyright 2011-2016  Brno University of Technology (Author: Karel Vesely)

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



#ifndef KALDI_NNET_NNET_COMPONENT_H_
#define KALDI_NNET_NNET_COMPONENT_H_

#include <iostream>
#include <string>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "nnet/nnet-trnopts.h"

namespace kaldi {
namespace nnet1 {

/**
 * Abstract class, building block of the network.
 * It is able to propagate (PropagateFnc: compute the output based on its input)
 * and backpropagate (BackpropagateFnc: i.e. transform loss derivative w.r.t. output to derivative w.r.t. the input)
 * the formulas are implemented in descendant classes (AffineTransform,Sigmoid,Softmax,...).
 */
class Component {
 /// Component type identification mechanism,
 public:
  /// Types of Components,
  typedef enum {
    kUnknown = 0x0,

    kUpdatableComponent = 0x0100,
    kAffineTransform,
    kLinearTransform,
    kConvolutionalComponent,
    kLstmProjected,
    kBlstmProjected,
    kRecurrentComponent,

    kActivationFunction = 0x0200,
    kSoftmax,
    kHiddenSoftmax,
    kBlockSoftmax,
    kSigmoid,
    kTanh,
    kParametricRelu,
    kDropout,
    kLengthNormComponent,

    kTranform = 0x0400,
    kRbm,
    kSplice,
    kCopy,
    kTranspose,
    kBlockLinearity,
    kAddShift,
    kRescale,

    kKlHmm = 0x0800,
    kSentenceAveragingComponent, /* deprecated */
    kSimpleSentenceAveragingComponent,
    kAveragePoolingComponent,
    kMaxPoolingComponent,
    kFramePoolingComponent,
    kParallelComponent,
    kMultiBasisComponent
  } ComponentType;

  /// A pair of type and marker,
  struct key_value {
    const Component::ComponentType key;
    const char *value;
  };

  /// The table with pairs of Component types and markers
  /// (defined in nnet-component.cc),
  static const struct key_value kMarkerMap[];

  /// Converts component type to marker,
  static const char* TypeToMarker(ComponentType t);

  /// Converts marker to component type (case insensitive),
  static ComponentType MarkerToType(const std::string &s);

 /// Generic interface of a component,
 public:
  Component(int32 input_dim, int32 output_dim):
    input_dim_(input_dim),
    output_dim_(output_dim)
  { }

  virtual ~Component()
  { }

  /// Copy component (deep copy),
  virtual Component* Copy() const = 0;

  /// Get Type Identification of the component,
  virtual ComponentType GetType() const = 0;

  /// Check if componeny has 'Updatable' interface (trainable components),
  virtual bool IsUpdatable() const {
    return false;
  }

  /// Check if component has 'Recurrent' interface (trainable and recurrent),
  virtual bool IsMultistream() const {
    return false;
  }

  /// Get the dimension of the input,
  int32 InputDim() const {
    return input_dim_;
  }

  /// Get the dimension of the output,
  int32 OutputDim() const {
    return output_dim_;
  }

  /// Perform forward-pass propagation 'in' -> 'out',
  void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out);

  /// Perform backward-pass propagation 'out_diff' -> 'in_diff'.
  /// Note: 'in' and 'out' will be used only sometimes...
  void Backpropagate(const CuMatrixBase<BaseFloat> &in,
                     const CuMatrixBase<BaseFloat> &out,
                     const CuMatrixBase<BaseFloat> &out_diff,
                     CuMatrix<BaseFloat> *in_diff);

  /// Initialize component from a line in config file,
  static Component* Init(const std::string &conf_line);

  /// Read the component from a stream (static method),
  static Component* Read(std::istream &is, bool binary);

  /// Write the component to a stream,
  void Write(std::ostream &os, bool binary) const;

  /// Print some additional info (after <ComponentName> and the dims),
  virtual std::string Info() const { return ""; }

  /// Print some additional info about gradient (after <...> and dims),
  virtual std::string InfoGradient() const { return ""; }


 /// Abstract interface for propagation/backpropagation
 protected:
  /// Forward pass transformation (to be implemented by descending class...)
  virtual void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) = 0;

  /// Backward pass transformation (to be implemented by descending class...)
  virtual void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                const CuMatrixBase<BaseFloat> &out,
                                const CuMatrixBase<BaseFloat> &out_diff,
                                CuMatrixBase<BaseFloat> *in_diff) = 0;

 /// Virtual interface for initialization and I/O,
 protected:
  /// Initialize internal data of a component
  virtual void InitData(std::istream &is) { }

  /// Reads the component content
  virtual void ReadData(std::istream &is, bool binary) { }

  /// Writes the component content
  virtual void WriteData(std::ostream &os, bool binary) const { }

 /// Data members,
 protected:
  int32 input_dim_;  ///< Dimension of the input of the Component,
  int32 output_dim_;  ///< Dimension of the output of the Component,

 /// Private members (descending classes cannot call this),
 private:
  /// Create a new intance of component,
  static Component* NewComponentOfType(
    ComponentType t, int32 input_dim, int32 output_dim
  );
};


/**
 * Class UpdatableComponent is a Component which has trainable parameters,
 * it contains SGD training hyper-parameters in NnetTrainOptions.
 * The constants 'learning_rate_coef_' and 'bias_learn_rate_coef_'
 * are separate, and should be stored by ::WriteData(...),
 */
class UpdatableComponent : public Component {
 public:
  UpdatableComponent(int32 input_dim, int32 output_dim):
    Component(input_dim, output_dim),
    learn_rate_coef_(1.0),
    bias_learn_rate_coef_(1.0)
  { }

  virtual ~UpdatableComponent()
  { }

  /// Check if contains trainable parameters,
  bool IsUpdatable() const {
    return true;
  }

  /// Number of trainable parameters,
  virtual int32 NumParams() const = 0;

  /// Get gradient reshaped as a vector,
  virtual void GetGradient(VectorBase<BaseFloat> *gradient) const = 0;

  /// Get the trainable parameters reshaped as a vector,
  virtual void GetParams(VectorBase<BaseFloat> *params) const = 0;

  /// Set the trainable parameters from, reshaped as a vector,
  virtual void SetParams(const VectorBase<BaseFloat> &params) = 0;

  /// Compute gradient and update parameters,
  virtual void Update(const CuMatrixBase<BaseFloat> &input,
                      const CuMatrixBase<BaseFloat> &diff) = 0;

  /// Set the training options to the component,
  virtual void SetTrainOptions(const NnetTrainOptions &opts) {
    opts_ = opts;
  }

  /// Get the training options from the component,
  const NnetTrainOptions& GetTrainOptions() const {
    return opts_;
  }

  /// Set the learn-rate coefficient,
  virtual void SetLearnRateCoef(BaseFloat val) {
    learn_rate_coef_ = val;
  }

  /// Set the learn-rate coefficient for bias,
  virtual void SetBiasLearnRateCoef(BaseFloat val) {
    bias_learn_rate_coef_ = val;
  }

  /// Initialize the content of the component by the 'line' from the prototype,
  virtual void InitData(std::istream &is) = 0;

 protected:
  /// Option-class with training hyper-parameters,
  NnetTrainOptions opts_;

  /// Scalar applied to learning rate for weight matrices
  /// (to be used in ::Update method),
  BaseFloat learn_rate_coef_;

  /// Scalar applied to learning rate for bias
  /// (to be used in ::Update method),
  BaseFloat bias_learn_rate_coef_;
};


/**
 * Class MultistreamComponent is an extension of UpdatableComponent
 * for recurrent networks, which are trained with parallel sequences.
 */
class MultistreamComponent : public UpdatableComponent {
 public:
  MultistreamComponent(int32 input_dim, int32 output_dim):
    UpdatableComponent(input_dim, output_dim)
  { }

  bool IsMultistream() const {
    return true;
  }

  virtual void SetSeqLengths(const std::vector<int32>& sequence_lengths) {
    sequence_lengths_ = sequence_lengths;
  }

  int32 NumStreams() const {
    return std::max<int32>(1, sequence_lengths_.size());
  }

  /// Optional function to reset the transfer of context (not used for BLSTMs
  virtual void ResetStreams(const std::vector<int32>& stream_reset_flag)
  { }

 protected:
  std::vector<int32> sequence_lengths_;
};


/*
 * Inline methods for ::Component,
 */
inline void Component::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 CuMatrix<BaseFloat> *out) {
  // Check the dims
  if (input_dim_ != in.NumCols()) {
    KALDI_ERR << "Non-matching dims on the input of " << TypeToMarker(GetType())
              << " component. The input-dim is " << input_dim_
              << ", the data had " << in.NumCols() << " dims.";
  }
  // Allocate target buffer
  out->Resize(in.NumRows(), output_dim_, kSetZero);  // reset
  // Call the propagation implementation of the component
  PropagateFnc(in, out);
}

inline void Component::Backpropagate(const CuMatrixBase<BaseFloat> &in,
                                     const CuMatrixBase<BaseFloat> &out,
                                     const CuMatrixBase<BaseFloat> &out_diff,
                                     CuMatrix<BaseFloat> *in_diff) {
  // Check the dims,
  if (OutputDim() != out_diff.NumCols()) {
    KALDI_ERR << "Non-matching dims! Component output dim " << OutputDim()
              << ", the dim of output derivatives " << out_diff.NumCols();
  }

  int32 num_frames = out_diff.NumRows();
  KALDI_ASSERT(num_frames == in.NumRows());
  KALDI_ASSERT(num_frames == out.NumRows());

  KALDI_ASSERT(InputDim() == in.NumCols());
  KALDI_ASSERT(OutputDim() == out.NumCols());

  // Allocate target buffer,
  KALDI_ASSERT(in_diff != NULL);
  in_diff->Resize(num_frames, InputDim(), kSetZero);  // reset,

  // Call the 'virtual' backprop function,
  BackpropagateFnc(in, out, out_diff, in_diff);
}


}  // namespace nnet1
}  // namespace kaldi


#endif  // KALDI_NNET_NNET_COMPONENT_H_
