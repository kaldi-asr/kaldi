// nnet/nnet-component.h

// Copyright 2011-2013 Brno University of Technology (Author: Karel Vesely)

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

// declare the nnet class so we can declare pointer
struct NnetTrainOptions;
class Nnet;
   

/**
 * Abstract class, basic element of the network,
 * it is a box with defined inputs, outputs,
 * and tranformation functions interface.
 *
 * It is able to propagate and backpropagate
 * exact implementation is to be implemented in descendants.
 *
 * The data buffers are not included 
 * and will be managed from outside.
 */ 
class Component {

  // Polymorphic Component RTTI
 public: 
  /// Types of the net components
  typedef enum {
    kUnknown = 0x0,
     
    kUpdatableComponent = 0x0100, 
    kAffineTransform,

    kActivationFunction = 0x0200, 
    kSoftmax, 
    kSigmoid,
    kTanh,
    kDropout,

    kTranform =  0x0400,
    kRbm,
    kSplice,
    kCopy,
    kTranspose,
    kBlockLinearity,
    kAddShift,
    kRescale,
    kLog
  } ComponentType;
  /// Pair of type and marker
  struct key_value {
    const Component::ComponentType key;
    const char *value;
  };
  /// Mapping of types and markers 
  static const struct key_value kMarkerMap[];
  /// Convert component type to marker
  static const char* TypeToMarker(ComponentType t);
  /// Convert marker to component type
  static ComponentType MarkerToType(const std::string &s);

  Component(int32 input_dim, int32 output_dim, Nnet *nnet) 
      : input_dim_(input_dim), output_dim_(output_dim), nnet_(nnet) { }
  virtual ~Component() { }
   
 public:
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
 
  /// Perform forward pass propagateion Input->Output
  void Propagate(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out); 
  /// Perform backward pass propagation, out_diff -> in_diff
  /// '&in' and '&out' will often be unused... 
  void Backpropagate(const CuMatrix<BaseFloat> &in,
                     const CuMatrix<BaseFloat> &out,
                     const CuMatrix<BaseFloat> &out_diff,
                     CuMatrix<BaseFloat> *in_diff); 

  /// Read component from stream
  static Component* Read(std::istream &is, bool binary, Nnet *nnet);
  /// Write component to stream
  void Write(std::ostream &os, bool binary) const;


  // abstract interface for propagation/backpropagation 
 protected:
  /// Forward pass transformation (to be implemented by descendents...)
  virtual void PropagateFnc(const CuMatrix<BaseFloat> &in,
                            CuMatrix<BaseFloat> *out) = 0;
  /// Backward pass transformation (to be implemented by descendents...)
  virtual void BackpropagateFnc(const CuMatrix<BaseFloat> &in,
                                const CuMatrix<BaseFloat> &out,
                                const CuMatrix<BaseFloat> &out_diff,
                                CuMatrix<BaseFloat> *in_diff) = 0;

  /// Reads the component content
  virtual void ReadData(std::istream &is, bool binary) { }

  /// Writes the component content
  virtual void WriteData(std::ostream &os, bool binary) const { }


  // data members
 protected:
  int32 input_dim_;  ///< Size of input vectors
  int32 output_dim_; ///< Size of output vectors
  
  Nnet *nnet_; ///< Pointer to the whole network
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(Component);
};


/**
 * Class UpdatableComponent is a Component which has
 * trainable parameters and contains SGD training 
 * hyper-parameters (learnrate, momenutm, L2, L1)
 */
class UpdatableComponent : public Component {
 public: 
  UpdatableComponent(int32 input_dim, int32 output_dim, Nnet *nnet)
    : Component(input_dim, output_dim, nnet) { }
  virtual ~UpdatableComponent() { }

  /// Check if contains trainable parameters 
  bool IsUpdatable() const { 
    return true; 
  }

  /// Compute gradient and update parameters
  virtual void Update(const CuMatrix<BaseFloat> &input,
                      const CuMatrix<BaseFloat> &diff) = 0;

  /// Sets the training options to the component
  void SetTrainOptions(const NnetTrainOptions &opts) {
    opts_ = opts;
  }
  /// Gets the training options from the component
  const NnetTrainOptions& GetTrainOptions() const { 
    return opts_; 
  }

 protected:
  /// Option-class with training hyper-parameters
  NnetTrainOptions opts_; 
};




inline void Component::Propagate(const CuMatrix<BaseFloat> &in,
                                 CuMatrix<BaseFloat> *out) {
  if (input_dim_ != in.NumCols()) {
    KALDI_ERR << "Nonmatching dims, component:" << input_dim_ << " data:" << in.NumCols();
  }
  
  if (output_dim_ != out->NumCols() || in.NumRows() != out->NumRows()) {
    out->Resize(in.NumRows(), output_dim_);
  }

  PropagateFnc(in, out);
}


inline void Component::Backpropagate(const CuMatrix<BaseFloat> &in,
                                     const CuMatrix<BaseFloat> &out,
                                     const CuMatrix<BaseFloat> &out_diff,
                                     CuMatrix<BaseFloat> *in_diff) {
  //check the dims
  if (output_dim_ != out_diff.NumCols()) {
    KALDI_ERR << "Nonmatching output dims, component:" << output_dim_ 
              << " data:" << out_diff.NumCols();
  }
  //allocate buffer
  if (input_dim_ != in_diff->NumCols() || out_diff.NumRows() != in_diff->NumRows()) {
    in_diff->Resize(out_diff.NumRows(), input_dim_);
  }
  //asserts on the dims
  KALDI_ASSERT((in.NumRows() == out.NumRows()) &&
               (in.NumRows() == out_diff.NumRows()) &&
               (in.NumRows() == in_diff->NumRows()));
  KALDI_ASSERT(in.NumCols() == in_diff->NumCols());
  KALDI_ASSERT(out.NumCols() == out_diff.NumCols());
  //call the backprop implementation of the component
  BackpropagateFnc(in, out, out_diff, in_diff);
}



} // namespace nnet1
} // namespace kaldi


#endif
