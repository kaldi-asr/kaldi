// nnet3/nnet-nnet.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_NNET_H_
#define KALDI_NNET3_NNET_NNET_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-descriptor.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>


namespace kaldi {
namespace nnet3 {

// NetworkNode is used to represent, in a neural net, either an input of the
// network, an output of the network, or an instance of a Component (note: if
// you want an output of the network is an output of a component, there is a way
// to represent that in the InputDescriptor).
struct NetworkNode {
  enum NodeType { kInput, kOutput, kComponent } node_type;

  // This is relevant only for kOutput and kComponent.  It describes
  // where it gets its input from; see type InputDescriptor for details.
  InputDescriptor input;

  // For kComponent, the index of the component in the network's components_
  // vector; otherwise -1.
  int32 component_index;
  
  int32 OutputDim(const Nnet &nnet);  // Dimension that this node outputs.
};




class Nnet {

 private:
  // names of components, used only in reading and writing code.  Internally we
  // always use integers.
  std::vector<std::string> component_names_;
  // the components of the nnet, in arbitrary order.  The network topology is
  // defined separately, below; a given Component may appear more than once in
  // the network if necessary for parameter tying.
  std::vector<Component*> components_;  

  // the names of the network-nodes, used only in reading and writing
  // code.  Internally we always use integers.
  std::vector<std::string> node_names_;

  // the network nodes.
  std::vector<NetworkNode> nodes_;
  
};



} // namespace nnet2
} // namespace kaldi

#endif
