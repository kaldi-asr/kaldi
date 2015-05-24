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
// network, an output of the network, an input to a Component, or an instance of
// a Component.
// Note: for each instance of a component in the network, there are always
// two nodes one of type kComponentInput and one of type kComponent, and the
// kComponent comes directly after the corresponding kComponentInput.  So the
// input to a component always comes from the network-node directly before it.
struct NetworkNode {
  enum NodeType { kInput, kOutput, kComponent, kComponentInput } node_type;

  // This is relevant only for kOutput and kComponentInput.  It describes which
  // other network nodes it gets its input from, and how those inputs are
  // combined together; see type Descriptor in nnet-descriptor.h for
  // details.
  Descriptor descriptor;

  union {
    // For kComponent, the index of the component in the network's components_
    // vector.
    int32 component_index;

    // for kInput, the dimension of the input feature.
    int32 dim;
  } u;
  
  int32 Dim(const Nnet &nnet) const;  // Dimension that this node outputs.
};



class Nnet {
 public:
  void Init(std::istream &config_file);
  
  int32 NumComponents() const { return components_.size(); }

  int32 NumNodes() const { return nodes_.size(); }

  /// return component indexed c.  not a copy; not owned by caller.
  Component *GetComponent(int32 c);

  /// return component indexed c (const version).  not a copy; not owned by
  /// caller.
  const Component *GetComponent(int32 c) const;

  /// returns const reference to a particular numbered network node.
  const NetworkNode &GetNode(int32 node) const;

  /// returns vector of node names (needed by some parsing code, for instance).
  const std::vector<std::string> &GetNodeNames() const { return node_names_; }

  // returns index associated with this node name, or -1 if no such index.
  int32 IndexOfNode(const std::string &node_name) const;
  
  void Read(std::istream &istream, bool binary);

  void Write(std::ostream &ostream, bool binary) const;

  /// note to self: one thing of many that we need to check is that no output
  /// nodes are referred to in Descriptors.  This might mess up the combination
  /// of each output node into a single step, as dependencies would be messed
  /// up.
  void Check()const;

  /// returns some human-readable information about the network, mostly for
  /// debugging purposes.
  std::string Info() const;
 private:
  // the names of the components of the network.  Note, these may be distinct
  // from the network node names below (and live in a different namespace); the
  // same component may be used in multiple network nodes, to define parameter
  // sharing.
  std::vector<std::string> names_;
  
  // the components of the nnet, in arbitrary order.  The network topology is
  // defined separately, below; a given Component may appear more than once in
  // the network if necessary for parameter tying.
  std::vector<Component*> components_;  

  // names of network nodes, i.e. inputs, components and outputs, used only in
  // reading and writing code.  Indexed by network-node index.  Note,
  // components' names are always listed twice, once as foo-input and once as
  // foo, because the input to a component always gets its own NetworkNode index.
  std::vector<std::string> node_names_;

  // the network nodes.
  std::vector<NetworkNode> nodes_;
  
};



} // namespace nnet3
} // namespace kaldi

#endif
