// nnet3/nnet-graph.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-graph.h"

namespace kaldi {
namespace nnet3 {



void NnetToDirectedGraph(const Nnet &nnet,
                         std::vector<std::vector<int32> > *graph) {
  graph->clear();
  int32 num_nodes = nnet.NumNodes();
  graph->resize(num_nodes);
  for (int32 n = 0; n < num_nodes; n++) {
    const NetworkNode &node = nnet.GetNode(n);
    // handle dependencies of this node.
    std::vector<int32> dependencies;
    switch (node.node_type) {
      case NetworkNode::kInput: break;  // no dependencies.
      case NetworkNode::kDescriptor: 
        node.descriptor.ComputeDependencies(&dependencies);
        break;
      case NetworkNode::kComponent:
        dependencies.push_back(n - 1);
    }
    SortAndUniq(&dependencies);
    for (size_t i = 0; i < dependencies.size(); i++) {
      int32 dep_n = dependencies[i];
      KALDI_ASSERT(dep_n >= 0 && dep_n < num_nodes);
      (*graph)[dep_n].push_back(n);
    }
  }
}

void ComputeGraphTranspose(const std::vector<std::vector<int32> > &graph,
                           std::vector<std::vector<int32> > *graph_transpose) {
  int32 size = graph.size();
  graph_transpose->clear();
  graph_transpose->resize(size);
  for (int32 n = 0; n < size; n++) {
    const std::vector<int32> &nodes = graph[n];
    std::vector<int32>::const_iterator iter = nodes.begin(), end = nodes.end();
    for (; iter != end; ++iter) {
      int32 dest = *iter;
      (*graph_transpose)[dest].push_back(n);
    }
  }
}
          

} // namespace nnet3
} // namespace kaldi
