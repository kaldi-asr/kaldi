// nnet/nnet-nnet.h

// Copyright 2011  Karel Vesely

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



#ifndef KALDI_NNET_NNET_H
#define KALDI_NNET_NNET_H


#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet/nnet-component.h"

#include <iostream>
#include <sstream>
#include <vector>


namespace kaldi {

class Nnet {
 public:
  Nnet() { }
  
  ~Nnet(); 

 public:
  /// Perform forward pass through the network
  void Propagate(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>* out); 
  /// Perform backward pass through the network
  void Backpropagate(const CuMatrix<BaseFloat>& in_err, CuMatrix<BaseFloat>* out_err);
  /// Perform forward pass through the network, don't keep buffers (use it when not training)
  void Feedforward(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>* out); 

  MatrixIndexT InputDim() const; ///< Dimensionality of the input features
  MatrixIndexT OutputDim() const; ///< Dimensionality of the desired vectors

  MatrixIndexT LayerCount() const { ///< Get number of layers
    return nnet_.size(); 
  }
  Component* Layer(MatrixIndexT index) { ///< Access to individual layer
    return nnet_[index]; 
  }
  int IndexOfLayer(const Component& comp) const; ///< Get the position of layer in network

  /// Access to forward pass buffers
  const std::vector<CuMatrix<BaseFloat> >& PropagateBuffer() const { 
    return propagate_buf_; 
  }

  /// Access to backward pass buffers
  const std::vector<CuMatrix<BaseFloat> >& BackpropagateBuffer() const { 
    return backpropagate_buf_; 
  }
  
  /// Read the MLP from file (can add layers to exisiting instance of Nnet)
  void Read(const std::string& file);  
  /// Read the MLP from stream (can add layers to exisiting instance of Nnet)
  void Read(std::istream& in, bool binary);  
  /// Write MLP to file
  void Write(const std::string& file, bool binary); 
  /// Write MLP to stream 
  void Write(std::ostream& out, bool binary);    
  
  /// Set the learning rate values to trainable layers, 
  /// factors can disable training of individual layers
  void SetLearnRate(BaseFloat lrate, const char* lrate_factors); 
  /// Get the global learning rate value
  BaseFloat GetLearnRate() { 
    return learn_rate_; 
  }
  /// Get the string with real learning rate values
  std::string GetLearnRateString();  

  void SetMomentum(BaseFloat mmt);
  void SetL2Penalty(BaseFloat l2);
  void SetL1Penalty(BaseFloat l1);

 private:
  /// Creates a component by reading from stream, return NULL if no more components
  static Component* ComponentFactory(std::istream& in, bool binary, Nnet* nnet);
  /// Dumps individual component to stream
  static void ComponentDumper(std::ostream& out, bool binary, const Component& comp);

  typedef std::vector<Component*> NnetType;
  
  NnetType nnet_;     ///< vector of all Component*, represents layers

  std::vector<CuMatrix<BaseFloat> > propagate_buf_; ///< buffers for forward pass
  std::vector<CuMatrix<BaseFloat> > backpropagate_buf_; ///< buffers for backward pass

  BaseFloat learn_rate_; ///< global learning rate

  KALDI_DISALLOW_COPY_AND_ASSIGN(Nnet);
};
  

inline Nnet::~Nnet() {
  // delete all the components
  NnetType::iterator it;
  for(it=nnet_.begin(); it!=nnet_.end(); ++it) {
    delete *it;
  }
}

   
inline MatrixIndexT Nnet::InputDim() const { 
  if (LayerCount() > 0) {
   return nnet_.front()->InputDim(); 
  } else {
   KALDI_ERR << "No layers in MLP"; 
  }
}


inline MatrixIndexT Nnet::OutputDim() const { 
  if (LayerCount() > 0) {
    return nnet_.back()->OutputDim(); 
  } else {
    KALDI_ERR << "No layers in MLP"; 
  }
}


inline int32 Nnet::IndexOfLayer(const Component& comp) const {
  for(int32 i=0; i<LayerCount(); i++) {
    if (&comp == nnet_[i]) return i;
  }
  KALDI_ERR << "Component:" << &comp 
            << " type:" << comp.GetType() 
            << " not found in the MLP";
  return -1;
}
 
  
inline void Nnet::Read(const std::string& file) {
  bool binary;
  Input in(file, &binary);
  Read(in.Stream(), binary);
  in.Close();
}


inline void Nnet::Write(const std::string& file, bool binary) {
  Output out(file, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}


inline void Nnet::Write(std::ostream& out, bool binary) {
  for(int32 i=0; i<LayerCount(); i++) {
    nnet_[i]->Write(out, binary);
  }
}

    
inline void Nnet::SetMomentum(BaseFloat mmt) {
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(nnet_[i])->SetMomentum(mmt);
    }
  }
}


inline void Nnet::SetL2Penalty(BaseFloat l2) {
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(nnet_[i])->SetL2Penalty(l2);
    }
  }
}


inline void Nnet::SetL1Penalty(BaseFloat l1) {
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(nnet_[i])->SetL1Penalty(l1);
    }
  }
}




} // namespace kaldi

#endif


