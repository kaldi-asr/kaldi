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

/// The first step in compilation is to turn the ComputationSpecification
/// into a ComputationGraph, where for each Cindex we have a list of
/// other Cindexes that it depends on.  All the stages of compilation
/// use the ComputationGraph representation; they are mostly manipulations
/// of it.
///
/// For efficiency, we give each Cindex its own integer identifier, called a
/// "cindex_id".  A cindex_id is only interpretable relative to a
/// ComputationGraph; it's an index into the "cindexes" array of the
/// ComputationGraph.  The GetCindexId() functions perform the reverse mapping.
struct ComputationGraph {

  /// This is the mapping of cindex_id to Cindex.
  std::vector<Cindex> cindexes;

  /// For each Cindex this tells us whether it is an input to the network.  This
  /// would not be necessary if we didn't allow the outputs of Components to be
  /// supplied as inputs [this may be useful for things like RNNs if we want to
  /// do the computation piece by piece].
  std::vector<bool> is_input;

  /// dependencies[cindex_id] gives you the list of other cindex_ids that this
  /// particular cindex_id directly depends on to compute it.  Note, some of
  /// these dependencies may be optional dependencies; in early stages of
  /// compilation this will contain all "desired" inputs and later we will prune
  /// the dependencies contain just those that are used (which will vary
  /// depending on availability).
  std::vector<std::vector<int32> > dependencies;

  /// Maps a Cindex to an integer cindex_id.  If not present, then add it (with
  /// the corresponding "is_input" flag set to the value "input") and set
  /// *is_new to true.  If present, set is_new to false and return the existing
  /// cindex_id.
  int32 GetCindexId(const Cindex &cindex, bool input, bool *is_new);

  /// Const version of GetCindexId that does not add CindexIds.  It will return
  /// -1 if the Cindex is not present, and the user should check for this.
  int32 GetCindexId(const Cindex &cindex) const;
  
  /// This function renumbers the cindex-ids, keeping only for which keep[c] is
  /// true.  The "keep" array must be the same size as this->cindexes.
  void Renumber(const std::vector<bool> &keep);
 private:
  /// Maps each Cindex to an integer cindex_id: reverse mapping of "cindexes".
  /// Must be accessed via the GetCindexId() functions.
  unordered_map<Cindex, int32, CindexHasher> cindex_to_cindex_id_;
};


/// An abstract representation of a set of Cindexes.
class CindexSet {
 public:
  /// Parenthesis operator; returns true if this cindex exists in the set.
  bool operator () (const Cindex &cindex) const;

  /// with this constructor, represents the set of all Cindexes that exist
  /// in the graph.
  CindexSet(const ComputationGraph &graph);

  /// with this constructor, represents the set of all Cindexes that exist
  /// in the graph and is_computable_[cindex_id] == true.
  CindexSet(const ComputationGraph &graph,
            const std::vector<bool> &is_computable);
 private:
  const ComputationGraph &graph_;
  const std::vector<bool> *is_computable_;
};


/// An abstract representation of a set of Cindexes.
class IndexSet {
 public:
  /// Returns true if this Index exists in the set. 
  bool operator () (const Index &index) const;

  /// This constructor creates the set of all Indexes x such that a Cindex
  /// (node_id, x) which is computable exists in this graph.
  IndexSet(const ComputationGraph &graph,
           const std::vector<bool> &is_computable,
           int32 node_id);
 private:
  const ComputationGraph &graph_;
  const std::vector<bool> &is_computable_;  
  int32 node_id_;


};




/// Computes an initial version of the computation-graph, with all dependencies
/// including optional dependencies listed.  Does not check whether all inputs
/// we depend on are contained in the input of the computation_request; that is
/// done in ComputeComputableArray.
void ComputeComputationGraph(const Nnet &nnet,
                             const ComputationRequest &computation_request,
                             ComputationGraph *computation_graph);

/// Computes an array "computable" that says for each Cindex in the graph
/// (indexed by cindex_id) whether it is provided in, or computable from, the
/// supplied inputs.
void ComputeComputableArray(const Nnet &nnet,
                            const ComputationRequest &computation_request,
                            const ComputationGraph &computation_graph,
                            std::vector<bool> *computable);

/// This function prunes the "dependencies" array of each cindex_id in the
/// computation graph, to include just those that are used in the computation
/// (rather than all dependencies that are desired).  It uses the IsComputable
/// function of the Descriptors and Components to work out which inputs are
/// used; this does something interesting only if we have non-simple components
/// or descriptors that involve optional inputs.  It also clears the
/// dependencies for all non-computable cindexes.
void PruneDependencies(const Nnet &nnet,
                       const ComputationRequest &computation_request,
                       const std::vector<bool> &computable,
                       ComputationGraph *computation_graph);
                       

/// Computes an array "required" that says for each Cindex in the graph (indexed
/// by cindex_id) whether it is required to compute the outputs of the
/// computation.  Normally this vector will be all true because we only populate
/// the graph with things that we needed in the first place; but there are
/// instances where this might not be the case: if "dependencies" for a certain
/// output contains A and B, but the Descriptor has an expression like (A if
/// present; if not, B) then because A is present, B is not required.  The same
/// type of thing could occur with non-simple Components.  This function just
/// looks at the "dependnecies" arrays; it must only be called after you have
/// called "PruneDependencies" on the graph, otherwise it will just say
/// everything is required.
void ComputeRequiredArray(const Nnet &nnet,
                          const ComputationGraph &computation_graph,
                          const std::vector<bool> &computable,
                          std::vector<bool> *required);


/// Prunes a computatation graph by removing some Cindexes (this involves a
/// renumbering).  Any Cindex that is not computable, or that is not required
/// and is not an input (!graph.is_input[cindex_id]) is removed in the
/// renumbering.  If this function detects that not all the requested outputs
/// can be computed from the inputs, it returns false and the answer is
/// undefined; otherwise it returns true and does the renumbering.  The caller
/// should check the return status.
bool PruneComputationGraph(
    const Nnet &nnet,
    const std::vector<bool> &computable,
    const std::vector<bool> &required,
    ComputationGraph *computation_graph);


/// Compute the order in which we can compute each cindex in the computation;
/// each cindex will map to an order-index.  This is a prelude to computing the
/// steps of the computation (ComputeComputationSteps), and gives a coarser
/// ordering than the steps.  The order-index is 0 for input cindexes, and in
/// general is n for any cindex that can be computed immediately from cindexes
/// with order-index less than n.  It is an error if some cindexes cannot be
/// computed (we assume that you have called PruneComputationGraph before this
/// function).  If the "order" parameter is non-NULL, it will output to "order"
/// a vector mapping cindex_id to order-index.  If the "by_order" parameter is
/// non-NULL, it will output for each order-index 0, 1 and so on, a sorted
/// vector of cindex_ids that have that order-index.
void ComputeComputationOrder(
    const Nnet &nnet,
    const ComputationGraph &computation_graph,
    std::vector<int32> *order,
    std::vector<std::vector<int32> > *by_order);


/// Once the computation order has been computed by ComputeComputationOrder,
/// this function computes the "steps" of the computation.  These are a finer
/// measure than the "order" because if there are cindexes with a particular
/// order-index and different node-ids (i.e. they belong to different nodes of
/// the nnet), they need to be separated into different steps.  Also, if the
/// cindexes for a particular output node are computed in multiple steps, they
/// are all combined into a single step whose numbering is the same as the last
/// of the steps.  [we'll later delete the other unnecessary steps].
///
/// Also this function makes sure that the order of cindex_ids in each step is
/// correct.  For steps corresponding to input and output nodes, this means that
/// the order is the same as specified in the ComputationRequest; for other
/// steps, it means that they are sorted using the order of struct Index (but
/// this order may be modified by components that defined ReorderIndexes()).
///
/// The reason why computation_graph is not provided as a const argument is
/// that in order to process nodes of type kDimRange, which will end up
/// just being a sub-matrix looking at nodes of type kComponent or kInput,
/// we may in some circumstances need to add some new cindexes to the graph,
/// to fill in any gaps.
void ComputeComputationSteps(
    const Nnet &nnet,
    const ComputationRequest &request,
    const std::vector<std::vector<int32> > &by_order,
    ComputationGraph *computation_graph,
    std::vector<std::vector<int32> > *steps);


} // namespace nnet3
} // namespace kaldi


#endif

