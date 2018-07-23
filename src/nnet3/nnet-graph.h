// nnet3/nnet-graph.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)
//                  2015  Guoguo Chen

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

#ifndef KALDI_NNET3_NNET_GRAPH_H_
#define KALDI_NNET3_NNET_GRAPH_H_

#include "nnet3/nnet-nnet.h"


namespace kaldi {
namespace nnet3 {

/**
   \file nnet-graph.h

   This file contains a few functions that treat the neural net as a graph on nodes:
   e.g. to find strongly-connected components in the graph, and from there to
   compute an ordering on the graph nodes.
*/


/// Prints a graph to a string in a pretty way for human readability, e.g. as 0
/// -> 1,2; 1 -> 2; 2 -> 3,4,5; etc.
std::string PrintGraphToString(const std::vector<std::vector<int32> > &graph);

/// This function takes an nnet and turns it to a directed graph on nodes.  This
/// is the reverse of the dependency graph.  The nodes will be numbered from 0
/// to graph->size() - 1, where graph->size() == nnet.NumNodes().  For each
/// node-index n, the vector in (*graph)[n] will contain a list of all the nodes
/// that have a direct dependency on node n (in order to compute them).  For
/// instance, if n is the output node, (*graph)[n] will be the empty list
/// because no other node will depend on it.
void NnetToDirectedGraph(const Nnet &nnet,
                         std::vector<std::vector<int32> > *graph);


/// Given a directed graph (where each std::vector<int32> is a list
/// of destination-nodes of arcs coming from the current node),
/// partition it into strongly connected components (i.e. within
/// each SCC, all nodes are reachable from all other nodes).
/// Each element of 'sccs' is a list of node indexes that are
/// in that scc.
void FindSccs(const std::vector<std::vector<int32> > &graph,
              std::vector<std::vector<int32> > *sccs);


/// This function returns 'true' if the graph represented in 'graph'
/// contains cycles (including cycles where a single node has an arc
/// to itself).
bool GraphHasCycles(const std::vector<std::vector<int32> > &graph);


/// Given a list of sccs of a graph (e.g. as computed by FindSccs), compute a
/// directed graph on the sccs.  Of course this directed graph will be acyclic.
void MakeSccGraph(const std::vector<std::vector<int32> > &graph,
                  const std::vector<std::vector<int32> > &sccs,
                  std::vector<std::vector<int32> > *scc_graph);

/// Given an acyclic graph (where each std::vector<int32> is a list of
/// destination-nodes of arcs coming from the current node), compute a
/// topological ordering of the graph nodes.  The output format is that
/// node_to_order[n] contains an integer t = 0, 1, ... which is the order of
/// node n in a topological sorting.  node_to_order should contain some
/// permutation of the numbers 0 ... graph.size() - 1.  This function should
/// crash if the graph contains cycles.
void ComputeTopSortOrder(const std::vector<std::vector<int32> > &graph,
                         std::vector<int32> *node_to_order);


/// Outputs a graph in which the order of arcs is reversed.
void ComputeGraphTranspose(const std::vector<std::vector<int32> > &graph,
                           std::vector<std::vector<int32> > *graph_transpose);

/// This function computes the order in which we need to compute each node in
/// the graph, where each node-index n maps to an epoch-index t = 0, 1, ... that
/// says when we should compute it.  Nodes that are part of a strongly connected
/// component (SCC) will all be computed at the same time, but any two nodes
/// that are not part of an SCC will have different epoch-index, and these
/// epoch-indexes will be such that a node computed at a larger epoch-index may
/// depend on a node computed at a smaller epoch-index, but not vice versa.
///
/// Internally it calls NnetToDirectedGraph, FindSccs, MakeSccGraph and
/// ComputeTopSortOrder.
void ComputeNnetComputationEpochs(const Nnet &nnet,
                                  std::vector<int32> *node_to_epoch);


} // namespace nnet3
} // namespace kaldi

#endif
