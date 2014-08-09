// nnet/nnet-component.h

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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


#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "nnet/nnet-trnopts.h"

#include <iostream>

namespace kaldi {
namespace nnet1 {

/**
 * Abstract class, building block of the network.
 * It is able to propagate (PropagateFnc: compute the output based on its input)
 * and backpropagate (BackpropagateFnc: i.e. transform loss derivative w.r.t. output to derivative w.r.t. the input)
 * the formulas are implemented in descendant classes (AffineTransform,Sigmoid,Softmax,...).
 */ 
class Component {

 /// Component type identification mechanism
 public: 
  /// Types of Components
  typedef enum {
    kUnknown = 0x0,
     
    kUpdatableComponent = 0x0100, 
    kAffineTransform,
    kLinearTransform,
    kConvolutionalComponent,
    kConvolutional2DComponent,

    kActivationFunction = 0x0200, 
    kSoftmax, 
    kSigmoid,
    kTanh,
    kDropout,

    kTranform = 0x0400,
    kRbm,
    kSplice,
    kCopy,
    kTranspose,
    kBlockLinearity,
    kAddShift,
    kRescale,
    
    kKlHmm = 0x0800,
    kSentenceAveragingComponent,
    kAveragePoolingComponent,
    kAveragePooling2DComponent,
    kMaxPoolingComponent,
    kMaxPooling2DComponent,
    kParallelComponent
  } ComponentType;
  /// A pair of type and marker 
  struct key_value {
    const Component::ComponentType key;
    const char *value;
  };
  /// Mapping of types and markers (the table is defined in nnet-component.cc) 
  static const struct key_value kMarkerMap[];
  /// Convert component type to marker
  static const char* TypeToMarker(ComponentType t);
  /// Convert marker to component type (case insensitive)
  static ComponentType MarkerToType(const std::string &s);
 
 /// General interface of a component  
 public:
  Component(int32 input_dim, int32 output_dim) 
      : input_dim_(input_dim), output_dim_(output_dim) { }
  virtual ~Component() { }

  /// Copy component (deep copy).
  virtual Component* Copy() const = 0;

  /// Get Type Identification of the component
  virtual ComponentType GetType() const = 0;  
  /// Check if contains trainable parameters 
  virtual bool IsUpdatable() const { 
    return false; 
  }

  /// Get size of input vectors
  int32 InputDim() const { 
    return input_dim_; 
  }  
  /// Get size of output vectors 
  int32 OutputDim() const { 
    return output_dim_; 
  }
 
  /// Perform forward pass propagation Input->Output
  void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out); 
  /// Perform backward pass propagation, out_diff -> in_diff
  /// '&in' and '&out' will sometimes be unused... 
  void Backpropagate(const CuMatrixBase<BaseFloat> &in,
                     const CuMatrixBase<BaseFloat> &out,
                     const CuMatrixBase<BaseFloat> &out_diff,
                     CuMatrix<BaseFloat> *in_diff); 

  /// Initialize component from a line in config file
  static Component* Init(const std::string &conf_line);
  /// Read component from stream
  static Component* Read(std::istream &is, bool binary);
  /// Write component to stream
  void Write(std::ostream &os, bool binary) const;

  /// Optionally print some additional info
  virtual std::string Info() const { return ""; }
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

  /// Initialize internal data of a component
  virtual void InitData(std::istream &is) { }

  /// Reads the component content
  virtual void ReadData(std::istream &is, bool binary) { }

  /// Writes the component content
  virtual void WriteData(std::ostream &os, bool binary) const { }

 /// Data members
 protected:
  int32 input_dim_;  ///< Size of input vectors
  int32 output_dim_; ///< Size of output vectors

 private:
  /// Create new intance of component
  static Component* NewComponentOfType(ComponentType t, 
                      int32 input_dim, int32 output_dim);
  
 protected:
  //KALDI_DISALLOW_COPY_AND_ASSIGN(Component);
};


/**
 * Class UpdatableComponent is a Component which has trainable parameters,
 * contains SGD training hyper-parameters in NnetTrainOptions.
 */
class UpdatableComponent : public Component {
 public: 
  UpdatableComponent(int32 input_dim, int32 output_dim)
    : Component(input_dim, output_dim) { }
  virtual ~UpdatableComponent() { }

  /// Check if contains trainable parameters 
  bool IsUpdatable() const { 
    return true; 
  }

  /// Number of trainable parameters
  virtual int32 NumParams() const = 0;
  virtual void GetParams(Vector<BaseFloat> *params) const = 0;

  /// Compute gradient and update parameters
  virtual void Update(const CuMatrixBase<BaseFloat> &input,
                      const CuMatrixBase<BaseFloat> &diff) = 0;

  /// Sets the training options to the component
  virtual void SetTrainOptions(const NnetTrainOptions &opts) {
    opts_ = opts;
  }
  /// Gets the training options from the component
  const NnetTrainOptions& GetTrainOptions() const { 
    return opts_; 
  }

  virtual void InitData(std::istream &is) = 0;

 protected:
  /// Option-class with training hyper-parameters
  NnetTrainOptions opts_; 
};


inline void Component::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 CuMatrix<BaseFloat> *out) {
  // Check the dims
  if (input_dim_ != in.NumCols()) {
    KALDI_ERR << "Non-matching dims! " << TypeToMarker(GetType()) 
              << " input-dim : " << input_dim_ << " data : " << in.NumCols();
  }
  // Allocate target buffer
  out->Resize(in.NumRows(), output_dim_, kSetZero); // reset
  // Call the propagation implementation of the component
  PropagateFnc(in, out);
}


inline void Component::Backpropagate(const CuMatrixBase<BaseFloat> &in,
                                     const CuMatrixBase<BaseFloat> &out,
                                     const CuMatrixBase<BaseFloat> &out_diff,
                                     CuMatrix<BaseFloat> *in_diff) {
  // Check the dims
  if (output_dim_ != out_diff.NumCols()) {
    KALDI_ERR << "Non-matching output dims, component:" << output_dim_ 
              << " data:" << out_diff.NumCols();
  }
  
  // Target buffer NULL : backpropagate only through components with nested nnets.
  if (in_diff == NULL) {
    if (GetType() == kParallelComponent ||
        GetType() == kSentenceAveragingComponent) {
      BackpropagateFnc(in, out, out_diff, NULL);
    } else {
      return;
    }
  } else {
    // Allocate target buffer
    in_diff->Resize(out_diff.NumRows(), input_dim_, kSetZero); // reset
    // Asserts on the dims
    KALDI_ASSERT((in.NumRows() == out.NumRows()) &&
                 (in.NumRows() == out_diff.NumRows()) &&
                 (in.NumRows() == in_diff->NumRows()));
    KALDI_ASSERT(in.NumCols() == in_diff->NumCols());
    KALDI_ASSERT(out.NumCols() == out_diff.NumCols());
    // Call the backprop implementation of the component
    BackpropagateFnc(in, out, out_diff, in_diff);
  }
}


} // namespace nnet1
} // namespace kaldi


#endif
