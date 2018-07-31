// nnet3/nnet-graph-test.cc

// Copyright 2015  Guoguo Chen

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

#include "nnet3/nnet-graph.h"

namespace kaldi {
namespace nnet3 {

// In this particular, we don't worry about tolerance and ordering.
bool AssertGraphEqual(const std::vector<std::vector<int32> > &graph1,
                      const std::vector<std::vector<int32> > &graph2) {
  if (graph1.size() != graph2.size()) { return false; }
  for (int32 i = 0; i < graph1.size(); ++i) {
    if (graph1[i].size() != graph2[i].size()) { return false; }
    for (int32 j = 0; j < graph1[i].size(); ++j) {
      if (graph1[i][j] != graph2[i][j]) { return false; }
    }
  }
  return true;
}

bool AssertVectorEqual(const std::vector<int32> &vec1,
                       const std::vector<int32> &vec2) {
  if (vec1.size() != vec2.size()) { return false; }
  for (int32 i = 0; i < vec1.size(); ++i) {
    if (vec1[i] != vec2[i]) { return false; }
  }
  return true;
}

void BuildTestGraph(std::vector<std::vector<int32> > *graph) {
  KALDI_ASSERT(graph != NULL);
  graph->clear();
  graph->resize(8);

  // We create the following graph for testing.
  // 0 --> 4
  // 1 --> 0
  // 2 --> 1 3
  // 3 --> 2
  // 4 --> 1
  // 5 --> 1 4 6
  // 6 --> 5
  // 7 --> 7 3 6
  std::vector<int32> tmp;
  tmp.resize(1); tmp[0] = 4; (*graph)[0] = tmp;
  tmp.resize(1); tmp[0] = 0; (*graph)[1] = tmp;
  tmp.resize(2); tmp[0] = 1; tmp[1] = 3; (*graph)[2] = tmp;
  tmp.resize(1); tmp[0] = 2; (*graph)[3] = tmp;
  tmp.resize(1); tmp[0] = 1; (*graph)[4] = tmp;
  tmp.resize(3); tmp[0] = 1; tmp[1] = 4; tmp[2] = 6; (*graph)[5] = tmp;
  tmp.resize(1); tmp[0] = 5; (*graph)[6] = tmp;
  tmp.resize(3); tmp[0] = 7; tmp[1] = 3; tmp[2] = 6; (*graph)[7] = tmp;
}

void BuildTestGraphTranspose(std::vector<std::vector<int32> > *graph) {
  KALDI_ASSERT(graph != NULL);
  graph->clear();
  graph->resize(8);

  // We create the following graph for testing.
  // 0 --> 1
  // 1 --> 2 4 5
  // 2 --> 3
  // 3 --> 2 7
  // 4 --> 0 5
  // 5 --> 6
  // 6 --> 5 7
  // 7 --> 7
  std::vector<int32> tmp;
  tmp.resize(1); tmp[0] = 1; (*graph)[0] = tmp;
  tmp.resize(3); tmp[0] = 2; tmp[1] = 4; tmp[2] = 5; (*graph)[1] = tmp;
  tmp.resize(1); tmp[0] = 3; (*graph)[2] = tmp;
  tmp.resize(2); tmp[0] = 2; tmp[1] = 7; (*graph)[3] = tmp;
  tmp.resize(2); tmp[0] = 0; tmp[1] = 5; (*graph)[4] = tmp;
  tmp.resize(1); tmp[0] = 6; (*graph)[5] = tmp;
  tmp.resize(2); tmp[0] = 5; tmp[1] = 7; (*graph)[6] = tmp;
  tmp.resize(1); tmp[0] = 7; (*graph)[7] = tmp;
}

void BuildTestSccs(std::vector<std::vector<int32> > *sccs) {
  KALDI_ASSERT(sccs != NULL);
  sccs->clear();
  sccs->resize(4);

  // We create the following SCCs for testing.
  // 0 --> 1 4 0
  // 1 --> 3 2
  // 2 --> 6 5
  // 3 --> 7
  std::vector<int32> tmp;
  tmp.resize(3); tmp[0] = 1; tmp[1] = 4; tmp[2] = 0; (*sccs)[0] = tmp;
  tmp.resize(2); tmp[0] = 3; tmp[1] = 2; (*sccs)[1] = tmp;
  tmp.resize(2); tmp[0] = 6; tmp[1] = 5; (*sccs)[2] = tmp;
  tmp.resize(1); tmp[0] = 7; (*sccs)[3] = tmp;
}

void BuildTestSccGraph(std::vector<std::vector<int32> > *scc_graph) {
  KALDI_ASSERT(scc_graph != NULL);
  scc_graph->clear();
  scc_graph->resize(4);

  // We create the following SCC graph for testing.
  // 0 -->
  // 1 --> 0
  // 2 --> 0
  // 3 --> 1 2
  std::vector<int32> tmp;
  tmp.resize(0); (*scc_graph)[0] = tmp;
  tmp.resize(1); tmp[0] = 0; (*scc_graph)[1] = tmp;
  tmp.resize(1); tmp[0] = 0; (*scc_graph)[2] = tmp;
  tmp.resize(2); tmp[0] = 1; tmp[1] = 2; (*scc_graph)[3] = tmp;
}

void BuildTestTopSortOrder(std::vector<int32> *node_to_order) {
  KALDI_ASSERT(node_to_order != NULL);
  node_to_order->clear();
  node_to_order->resize(4);

  // The topological sorting order of the above SCC graph is as follows (from
  // our particular algorithm):
  // 0 --> 3
  // 1 --> 2
  // 2 --> 1
  // 3 --> 0
  (*node_to_order)[0] = 3;
  (*node_to_order)[1] = 2;
  (*node_to_order)[2] = 1;
  (*node_to_order)[3] = 0;
}

void UnitTestComputeGraphTranspose() {
  std::vector<std::vector<int32> > graph;
  BuildTestGraph(&graph);

  std::vector<std::vector<int32> > graph_transpose;
  ComputeGraphTranspose(graph, &graph_transpose);

  std::vector<std::vector<int32> > ref_graph_transpose;
  BuildTestGraphTranspose(&ref_graph_transpose);
  KALDI_ASSERT(AssertGraphEqual(graph_transpose, ref_graph_transpose));
}

void UnitTestFindSccs() {
  std::vector<std::vector<int32> > graph;
  BuildTestGraph(&graph);

  std::vector<std::vector<int32> > sccs;
  FindSccs(graph, &sccs);

  std::vector<std::vector<int32> > ref_sccs;
  BuildTestSccs(&ref_sccs);
  KALDI_ASSERT(AssertGraphEqual(sccs, ref_sccs));
}

void UnitTestMakeSccGraph() {
  std::vector<std::vector<int32> > graph;
  BuildTestGraph(&graph);

  std::vector<std::vector<int32> > sccs;
  BuildTestSccs(&sccs);

  std::vector<std::vector<int32> > scc_graph;
  MakeSccGraph(graph, sccs, &scc_graph);

  std::vector<std::vector<int32> > ref_scc_graph;
  BuildTestSccGraph(&ref_scc_graph);
  KALDI_ASSERT(AssertGraphEqual(scc_graph, ref_scc_graph));
}

void UnitTestComputeTopSortOrder() {
  std::vector<std::vector<int32> > scc_graph;
  BuildTestSccGraph(&scc_graph);

  std::vector<int32> node_to_order;
  ComputeTopSortOrder(scc_graph, &node_to_order);

  std::vector<int32> ref_node_to_order;
  BuildTestTopSortOrder(&ref_node_to_order);
  KALDI_ASSERT(AssertVectorEqual(node_to_order, ref_node_to_order));
}

void UnitTestComputeTopSortOrder2() {
  // The outer vector is indexed by node ID, and each nested vector contains
  // the node IDs for its successors in the graph. For example, if there are
  // arcs from node 0 to nodes 1 and 2, then the vector at graph[0] will be (1, 2)
  std::vector<std::vector<int32> > graph;

  // Build a test graph:
  // 0 ---> 1 ---> 2 ---> 4
  //   `--> 3 -----^
  graph.resize(5);
  graph[0].push_back(1); graph[0].push_back(3);
  graph[1].push_back(2);
  graph[2].push_back(4);
  graph[3].push_back(2);
  // graph[4] is empty(has no successors)

  // fill in the desired(topological) mapping node->order
  std::vector<int32> ref_node_to_order;
  ref_node_to_order.push_back(0); // node 0 comes first
  ref_node_to_order.push_back(2); // node 1 comes third
  ref_node_to_order.push_back(3); // node 2 comes fourth
  ref_node_to_order.push_back(1); // node 3 comes second
  ref_node_to_order.push_back(4); // node 4 comes last

  std::vector<int32> computed_node_to_order;
  ComputeTopSortOrder(graph, &computed_node_to_order);
  KALDI_ASSERT(AssertVectorEqual(ref_node_to_order, computed_node_to_order));
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  UnitTestComputeGraphTranspose();
  UnitTestFindSccs();
  UnitTestMakeSccGraph();
  UnitTestComputeTopSortOrder();
  UnitTestComputeTopSortOrder2();

  KALDI_LOG << "Nnet graph tests succeeded.";

  return 0;
}
