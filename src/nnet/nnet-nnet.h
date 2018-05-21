// nnet/nnet-nnet.h

// Copyright 2011-2016  Brno University of Technology (Author: Karel Vesely)
//           2018 Alibaba.Inc (Author: ShiLiang Zhang) 

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

#ifndef KALDI_NNET_NNET_NNET_H_
#define KALDI_NNET_NNET_NNET_H_

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-component.h"

namespace kaldi {
namespace nnet1 {

class Nnet {
 public:
  Nnet();
  ~Nnet();

  Nnet(const Nnet& other);  // Allow copy constructor.
  Nnet& operator= (const Nnet& other);  // Allow assignment operator.

 public:
  /// Perform forward pass through the network,
  void Propagate(const CuMatrixBase<BaseFloat> &in,
                 CuMatrix<BaseFloat> *out);
  /// Perform backward pass through the network,
  void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff,
                     CuMatrix<BaseFloat> *in_diff);
  /// Perform forward pass through the network (with 2 swapping buffers),
  void Feedforward(const CuMatrixBase<BaseFloat> &in,
                   CuMatrix<BaseFloat> *out);

  /// Dimensionality on network input (input feature dim.),
  int32 InputDim() const;
  /// Dimensionality of network outputs (posteriors | bn-features | etc.),
  int32 OutputDim() const;

  /// Returns the number of 'Components' which form the NN.
  /// Typically a NN layer is composed of 2 components:
  /// the <AffineTransform> with trainable parameters
  /// and a non-linearity like <Sigmoid> or <Softmax>.
  /// Usually there are 2x more Components than the NN layers.
  int32 NumComponents() const {
    return components_.size();
  }

  /// Component accessor,
  const Component& GetComponent(int32 c) const;

  /// Component accessor,
  Component& GetComponent(int32 c);

  /// LastComponent accessor,
  const Component& GetLastComponent() const;

  /// LastComponent accessor,
  Component& GetLastComponent();

  /// Replace c'th component in 'this' Nnet (deep copy),
  void ReplaceComponent(int32 c, const Component& comp);

  /// Swap c'th component with the pointer,
  void SwapComponent(int32 c, Component** comp);

  /// Append Component to 'this' instance of Nnet (deep copy),
  void AppendComponent(const Component& comp);

  /// Append Component* to 'this' instance of Nnet by a shallow copy
  /// ('this' instance of Nnet over-takes the ownership of the pointer).
  void AppendComponentPointer(Component *dynamically_allocated_comp);

  /// Append other Nnet to the 'this' Nnet (copy all its components),
  void AppendNnet(const Nnet& nnet_to_append);

  /// Remove c'th component,
  void RemoveComponent(int32 c);

  /// Remove the last of the Components,
  void RemoveLastComponent();

  /// Access to the forward-pass buffers
  const std::vector<CuMatrix<BaseFloat> >& PropagateBuffer() const {
    return propagate_buf_;
  }
  /// Access to the backward-pass buffers
  const std::vector<CuMatrix<BaseFloat> >& BackpropagateBuffer() const {
    return backpropagate_buf_;
  }

  /// Get the number of parameters in the network,
  int32 NumParams() const;

  /// Get the gradient stored in the network,
  void GetGradient(Vector<BaseFloat>* gradient) const;

  /// Get the network weights in a supervector,
  void GetParams(Vector<BaseFloat>* params) const;

  /// Set the network weights from a supervector,
  void SetParams(const VectorBase<BaseFloat>& params);

  /// Set the dropout rate
  void SetDropoutRate(BaseFloat r);

  /// Reset streams in multi-stream training,
  void ResetStreams(const std::vector<int32> &stream_reset_flag);

  /// Set sequence length in LSTM multi-stream training,
  void SetSeqLengths(const std::vector<int32> &sequence_lengths);

  /// Initialize the Nnet from the prototype,
  void Init(const std::string &proto_file);

  /// Read Nnet from 'rxfilename',
  void Read(const std::string &rxfilename);
  /// Read Nnet from 'istream',
  void Read(std::istream &in, bool binary);

  /// Write Nnet to 'wxfilename',
  void Write(const std::string &wxfilename, bool binary) const;
  /// Write Nnet to 'ostream',
  void Write(std::ostream &out, bool binary) const;

  /// Create string with human readable description of the nnet,
  std::string Info() const;
  /// Create string with per-component gradient statistics,
  std::string InfoGradient(bool header = true) const;
  /// Create string with propagation-buffer statistics,
  std::string InfoPropagate(bool header = true) const;
  /// Create string with back-propagation-buffer statistics,
  std::string InfoBackPropagate(bool header = true) const;
  /// Consistency check,
  void Check() const;
  /// Relese the memory,
  void Destroy();

  /// Set hyper-parameters of the training (pushes to all UpdatableComponents),
  void SetTrainOptions(const NnetTrainOptions& opts);
  /// Get training hyper-parameters from the network,
  const NnetTrainOptions& GetTrainOptions() const {
    return opts_;
  }

  /// For FSMN component
  void SetFlags(const Vector<BaseFloat> &flags);

 private:
  /// Vector which contains all the components composing the neural network,
  /// the components are for example: AffineTransform, Sigmoid, Softmax
  std::vector<Component*> components_;

  /// Buffers for forward pass (on demand initialization),
  std::vector<CuMatrix<BaseFloat> > propagate_buf_;
  /// Buffers for backward pass (on demand initialization),
  std::vector<CuMatrix<BaseFloat> > backpropagate_buf_;

  /// Option class with hyper-parameters passed to UpdatableComponent(s)
  NnetTrainOptions opts_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_NNET_H_


