// nnet3/nnet-computation-graph.h

// Copyright 2015    Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_COMPUTATION_GRAPH_H_
#define KALDI_NNET3_NNET_COMPUTATION_GRAPH_H_

#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {

// The first step in compilation is to turn the ComputationSpecification
// into a ComputationGraph, where for each Cindex we have a list of
// other Cindexes that it depends on, and compute the shortest distance to
// the input.
struct ComputationGraph {

  // This is the reverse mapping of cindex_to_cindex_id: it maps from cindex_id
  // to Cindex.
  std::vector<Cindex> cindexes;
  
  // dependencies[cindex_id] gives you the list of other cindex_ids that this
  // particular cindex_id directly depends on to compute it.
  std::vector<std::vector<int32> > dependencies;

  // tells us whether each dependency is optional (true) rather than required
  // (false).  if a vector doesn't go to a certain size, assume required..
  // Note: most dependencies will required; only non-simple Components (which we
  // haven't written yet) will have optional dependencies.  These will be used
  // for doing things like aggregating over all possible frames, when we don't
  // know (or don't want to have to know exactly) the list of frames that are
  // available.  Also, if a component has only optional dependencies, we will
  // view it as computable only if at least one dependency can be computed.  We
  // can change this rule later if needed; see function IsComputable is
  // nnet-computation-graph.cc.  Note: after calling PruneComputationGraph we
  // will set this to the empty vector.
  std::vector<std::vector<bool> > optional;

  // Maps a Cindex to an integer cindex_id.  If not present, then add it and set
  // *is_new to true.  If present, set is_new to false and return the existing
  // cindex_id.
  int32 GetCindexId(Cindex cindex, bool *is_new);

  // Const version of the above. Accepts no bool argument; it will return -1 if
  // the Cindex is not present, and the user should check for this.
  int32 GetCindexId(Cindex cindex) const;

  // This function renumbers the cindex-ids, keeping only for which keep[c] is
  // true.  Note, it first asserts that the optional array is empty as it does
  // not handle that (we didn't code it since we don't need it in our
  // application of this function).
  void Renumber(const std::vector<bool> &keep);
 private:
  // Maps each Cindex to an integer (call this cindex_id) that uniquely
  // identifies it (obviously these cindex_id values are specific to the
  // computation graph).
  // We don't make this public because it's better to access it via
  // GetCindexId.
  unordered_map<Cindex, int32, CindexHasher> cindex_to_cindex_id_;
  
  
};

/// Computes an initial version of the computation-graph, with dependencies
/// listed.  Does not check whether all inputs we depend on are contained in the
/// input of the computation_request; that is done in PruneComputatationGraph.
void ComputeComputationGraph(const Nnet &nnet,
                             const ComputationRequest &computation_request,
                             ComputationGraph *computation_graph);

/// Prunes a computatation graph by removing any Cindex that is not computable
/// from the supplied input.  This will only ever successfully remove some
/// Cindexes if we have Components optional dependencies (i.e. we are using some
/// non-simple Components that list some dependencies as optional).  It is an
/// error if the output cannot be computed from the input.
void PruneComputationGraph(const Nnet &nnet,
                           const ComputationRequest &computation_request,
                           ComputationGraph *computation_graph);


/// Compute the order in which we can compute each cindex in the computation.
/// This is 0 for input cindexes, and in general order n for any cindex that can
/// be computed immediately from cindexes less than n.  It is an error if some
/// cindexes cannot be computed (we assume that you have called
/// PruneComputationGraph before this function).  If the "order" parameter is
/// non-NULL, it will output a vector giving the order for each cindex_id.  If
/// the "by_order" parameter is non-NULL, it will output for each order 0, 1 and
/// so on, a vector of sorted cindex_ids that have that order.
void ComputeComputationOrder(
    const Nnet &nnet,
    const ComputationRequest &request,
    const ComputationGraph &computation_graph,
    const std::vector<int32> &shortest_distance,
    std::vector<int32> *order,
    std::vector<std::vector<int32> > *by_order);




} // namespace nnet3
} // namespace kaldi


#endif

