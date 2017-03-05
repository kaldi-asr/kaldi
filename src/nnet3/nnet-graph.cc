// nnet3/nnet-graph.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen

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
    std::vector<int32> node_dependencies;
    switch (node.node_type) {
      case kInput:
        break;  // no node dependencies.
      case kDescriptor:
        node.descriptor.GetNodeDependencies(&node_dependencies);
        break;
      case kComponent:
        node_dependencies.push_back(n - 1);
        break;
      case kDimRange:
        node_dependencies.push_back(node.u.node_index);
        break;
      default:
        KALDI_ERR << "Invalid node type";
    }
    SortAndUniq(&node_dependencies);
    for (size_t i = 0; i < node_dependencies.size(); i++) {
      int32 dep_n = node_dependencies[i];
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

struct TarjanNode {
  int32 index;
  int32 lowlink;
  bool on_stack;
  TarjanNode() : index(-1), lowlink(-1), on_stack(false) {}
};

void TarjanSccRecursive(int32 node,
                        const std::vector<std::vector<int32> > &graph,
                        int32 *global_index,
                        std::vector<TarjanNode> *tarjan_nodes,
                        std::vector<int32> *tarjan_stack,
                        std::vector<std::vector<int32> > *sccs) {
  KALDI_ASSERT(sccs != NULL);
  KALDI_ASSERT(tarjan_nodes != NULL);
  KALDI_ASSERT(tarjan_stack != NULL);
  KALDI_ASSERT(global_index != NULL);
  KALDI_ASSERT(node >= 0 && node < graph.size());

  // Initializes the current Tarjan node.
  (*tarjan_nodes)[node].index = *global_index;
  (*tarjan_nodes)[node].lowlink = *global_index;
  *global_index += 1;
  (*tarjan_nodes)[node].on_stack = true;
  tarjan_stack->push_back(node);

  // DFS from the current node.
  for (int32 i = 0; i < graph[node].size(); ++i) {
    int32 next = graph[node][i];

    if ((*tarjan_nodes)[next].index == -1) {
      // First time we see this node.
      TarjanSccRecursive(next, graph,
                         global_index, tarjan_nodes, tarjan_stack, sccs);
      (*tarjan_nodes)[node].lowlink = std::min((*tarjan_nodes)[node].lowlink,
                                               (*tarjan_nodes)[next].lowlink);
    } else if ((*tarjan_nodes)[next].on_stack) {
      // Next node is on the stack -- back edge. We can't use the lowlink of
      // next node, because that may point to the index of the root, while the
      // current node can't be the root.
      (*tarjan_nodes)[node].lowlink = std::min((*tarjan_nodes)[node].lowlink,
                                               (*tarjan_nodes)[next].index);
    }
  }

  // Output SCC.
  if ((*tarjan_nodes)[node].index == (*tarjan_nodes)[node].lowlink) {
    std::vector<int32> scc;
    int32 pop_node;
    do {
      pop_node = tarjan_stack->back();
      tarjan_stack->pop_back();
      (*tarjan_nodes)[pop_node].on_stack = false;
      scc.push_back(pop_node);
    } while (pop_node != node);
    KALDI_ASSERT(pop_node == node);
    sccs->push_back(scc);
  }
}

void FindSccsTarjan(const std::vector<std::vector<int32> > &graph,
                    std::vector<std::vector<int32> > *sccs) {
  KALDI_ASSERT(sccs != NULL);

  // Initialization.
  std::vector<TarjanNode> tarjan_nodes(graph.size());
  std::vector<int32> tarjan_stack;
  int32 global_index = 0;

  // Calls the recursive function.
  for (int32 n = 0; n < graph.size(); ++n) {
    if (tarjan_nodes[n].index == -1) {
      TarjanSccRecursive(n, graph,
                         &global_index, &tarjan_nodes, &tarjan_stack, sccs);
    }
  }
}

void FindSccs(const std::vector<std::vector<int32> > &graph,
              std::vector<std::vector<int32> > *sccs) {
  // Internally we call Tarjan's SCC algorithm, as it only requires one DFS. We
  // can change this to other methods later on if necessary.
  KALDI_ASSERT(sccs != NULL);
  FindSccsTarjan(graph, sccs);
}

void MakeSccGraph(const std::vector<std::vector<int32> > &graph,
                  const std::vector<std::vector<int32> > &sccs,
                  std::vector<std::vector<int32> > *scc_graph) {
  KALDI_ASSERT(scc_graph != NULL);
  scc_graph->clear();
  scc_graph->resize(sccs.size());

  // Hash map from node to SCC index.
  std::vector<int32> node_to_scc_index(graph.size());
  for (int32 i = 0; i < sccs.size(); ++i) {
    for (int32 j = 0; j < sccs[i].size(); ++j) {
      KALDI_ASSERT(sccs[i][j] >= 0 && sccs[i][j] < graph.size());
      node_to_scc_index[sccs[i][j]] = i;
    }
  }

  // Builds graph.
  for (int32 i = 0; i < sccs.size(); ++i) {
    for (int32 j = 0; j < sccs[i].size(); ++j) {
      int32 node = sccs[i][j];
      KALDI_ASSERT(node >= 0 && node < graph.size());
      for (int32 k = 0; k < graph[node].size(); ++k) {
        if (node_to_scc_index[graph[node][k]] != i) { // Exclucding self.
          (*scc_graph)[i].push_back(node_to_scc_index[graph[node][k]]);
        }
      }
    }
    // If necessary, we can use a hash maps to avoid this sorting.
    SortAndUniq(&((*scc_graph)[i]));
  }
}

void ComputeTopSortOrderRecursive(int32 node,
                                  const std::vector<std::vector<int32> > &graph,
                                  std::vector<bool> *cycle_detector,
                                  std::vector<bool> *is_visited,
                                  std::vector<int32> *reversed_orders) {
  KALDI_ASSERT(node >= 0 && node < graph.size());
  KALDI_ASSERT(cycle_detector != NULL);
  KALDI_ASSERT(is_visited != NULL);
  KALDI_ASSERT(reversed_orders != NULL);
  if ((*cycle_detector)[node]) {
    KALDI_ERR << "Cycle detected when computing the topological sorting order";
  }

  if (!(*is_visited)[node]) {
    (*cycle_detector)[node] = true;
    for (int32 i = 0; i < graph[node].size(); ++i) {
      ComputeTopSortOrderRecursive(graph[node][i], graph,
                                   cycle_detector, is_visited, reversed_orders);
    }
    (*cycle_detector)[node] = false;
    (*is_visited)[node] = true;
    // At this point we have added all the children to <reversed_orders>, so we
    // can add the current now.
    reversed_orders->push_back(node);
  }
}

void ComputeTopSortOrder(const std::vector<std::vector<int32> > &graph,
                         std::vector<int32> *node_to_order) {
  // Internally we use DFS, but we only put the node to <node_to_order> when all
  // its parents have been visited.
  KALDI_ASSERT(node_to_order != NULL);
  node_to_order->resize(graph.size());

  std::vector<bool> cycle_detector(graph.size(), false);
  std::vector<bool> is_visited(graph.size(), false);

  std::vector<int32> reversed_orders;
  for(int32 i = 0; i < graph.size(); ++i) {
    if (!is_visited[i]) {
      ComputeTopSortOrderRecursive(i, graph, &cycle_detector,
                                   &is_visited, &reversed_orders);
    }
  }

  KALDI_ASSERT(node_to_order->size() == reversed_orders.size());
  for (int32 i = 0; i < reversed_orders.size(); ++i) {
    KALDI_ASSERT(reversed_orders[i] >= 0 && reversed_orders[i] < graph.size());
    (*node_to_order)[reversed_orders[i]] = graph.size() - i - 1;
  }
}

std::string PrintGraphToString(const std::vector<std::vector<int32> > &graph) {
  std::ostringstream os;
  int32 num_nodes = graph.size();
  for (int32 i = 0; i < num_nodes; i++) {
    os << i << " -> (";
    const std::vector<int32> &vec = graph[i];
    int32 size = vec.size();
    for (int32 j = 0; j < size; j++) {
      os << vec[j];
      if (j + 1 < size) os << ",";
    }
    os << ")";
    if (i + 1 < num_nodes) os << "; ";
  }
  return os.str();
}

void ComputeNnetComputationEpochs(const Nnet &nnet,
                                  std::vector<int32> *node_to_epoch) {
  KALDI_ASSERT(node_to_epoch != NULL);

  std::vector<std::vector<int32> > graph;
  NnetToDirectedGraph(nnet, &graph);
  KALDI_VLOG(6) << "graph is: " << PrintGraphToString(graph);

  std::vector<std::vector<int32> > sccs;
  FindSccs(graph, &sccs);

  std::vector<std::vector<int32> > scc_graph;
  MakeSccGraph(graph, sccs, &scc_graph);
  KALDI_VLOG(6) << "scc graph is: " << PrintGraphToString(scc_graph);

  std::vector<int32> scc_node_to_epoch;
  ComputeTopSortOrder(scc_graph, &scc_node_to_epoch);
  if (GetVerboseLevel() >= 6) {
    std::ostringstream os;
    for (int32 i = 0; i < scc_node_to_epoch.size(); i++)
      os << scc_node_to_epoch[i] << ", ";
    KALDI_VLOG(6) << "scc_node_to_epoch is: " << os.str();
  }

  node_to_epoch->clear();
  node_to_epoch->resize(graph.size());
  for (int32 i = 0; i < sccs.size(); ++i) {
    for (int32 j = 0; j < sccs[i].size(); ++j) {
      int32 node = sccs[i][j];
      KALDI_ASSERT(node >= 0 && node < graph.size());
      (*node_to_epoch)[node] = scc_node_to_epoch[i];
    }
  }
}

bool GraphHasCycles(const std::vector<std::vector<int32> > &graph) {
  std::vector<std::vector<int32> > sccs;
  FindSccs(graph, &sccs);
  for (size_t i = 0; i < sccs.size(); i++) {
    if (sccs[i].size() > 1)
      return true;
  }
  // the next code checks for links from a state to itself.
  int32 num_nodes = graph.size();
  for (size_t i = 0; i < num_nodes; i++)
    for (std::vector<int32>::const_iterator iter = graph[i].begin(),
             end = graph[i].end(); iter != end; ++iter)
      if (*iter == i) return true;
  return false;
}

} // namespace nnet3
} // namespace kaldi
