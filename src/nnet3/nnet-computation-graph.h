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
#include <deque>

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

  /// The mapping of cindex_id to Cindex.
  std::vector<Cindex> cindexes;

  /// For each Cindex this tells us whether it was provided as an input to the
  /// network.  This is necessary for a couple of reasons: firstly, the
  /// framework allows users to provide values for nodes of type kComponent
  /// (e.g. for RNN context).  Also, Cindexes for input nodes that were not
  /// provided by the user may be created during computation graph creation
  /// (although they will not be computable), and we need to distinguish these
  /// from the provided Cindexes.
  std::vector<bool> is_input;

  /// dependencies[cindex_id] gives you the list of other cindex_ids that this
  /// particular cindex_id directly depends on to compute it.  No repeats will
  /// be present.  Note, some of these dependencies may be optional
  /// dependencies; in early stages of compilation this will contain all
  /// "desired" inputs and later we will prune the dependencies contain just
  /// those that are used (which will vary depending on availability).
  std::vector<std::vector<int32> > dependencies;

  /// This variable is only of particular interest in a 'multi-segment'
  /// computation, which is used while creating computations for 'online'
  /// operation (for the kind of situation where you provide some input; run the
  /// computation; get some output, provide some more input for larger 't'
  /// values, etc.).  In this context, a 'segment' is a continuous range of
  /// cindex_ids, and a segment_end is one past the end of each segment, which
  /// is the same as the beginning of the next segment, if there is one.  In the
  /// case of a fully-created computation graph with only one segment, this will
  /// contain just one value which equals the number of cindex_ids.
  /// This information is needed to correctly order the computation, because
  ///
  /// the computation graph itself does not contain dependencies that encode the
  /// ordering of segments (and even if it did contain those dependencies, it's
  /// not really compatible with the way we use the scc's in the graph structure
  /// of the network to order the computation).
  std::vector<int32> segment_ends;

  /// Maps a Cindex to an integer cindex_id.  If not present, then add it (with
  /// the corresponding "is_input" flag set to the value "input") and set
  /// *is_new to true.  If present, set is_new to false and return the existing
  /// cindex_id.
  int32 GetCindexId(const Cindex &cindex, bool is_input, bool *is_new);

  /// Const version of GetCindexId that does not add CindexIds.  It will return
  /// -1 if the Cindex is not present, and the user should check for this.
  int32 GetCindexId(const Cindex &cindex) const;

  /// This function renumbers the cindex-ids (but only those with index c >= start_cindex_id,
  // keeping only for which keep[c - start_cindex_id] is
  /// true.  The "keep" array must be the same size as this->cindexes.size() -
  /// start_cindex_id.
  void Renumber(int32 start_cindex_id,
                const std::vector<bool> &keep);


  /// This function, useful for debugging/visualization purposes,
  /// prints out a summary of the computation graph (which will take up
  /// multiple lines).
  /// Format is: [ cindex1 -> dep-cindex1 dep-cindex2 ] [ cindex2 -> dep-cindex3 dep-cindex4 ]
  /// showing each Cindex and the Cindexes it depends on.  cindexes from different network
  /// nodes are shown on different lines.
  void Print(std::ostream &os, const std::vector<std::string> &node_names);

 private:
  /// Maps each Cindex to an integer cindex_id: reverse mapping of "cindexes".
  /// Must be accessed via the GetCindexId() functions.
  unordered_map<Cindex, int32, CindexHasher> cindex_to_cindex_id_;
};


/// An abstract representation of a set of Cindexes.
/// See \ref dnn3_compile_graph_building.
class ComputationGraphBuilder {
 public:
  ComputationGraphBuilder(const Nnet &nnet,
                          ComputationGraph *graph);

  // Does the initial computation (populating the graph and computing whether
  // each required cindex_id is computable), without the pruning.  In the normal
  // case you call this just once with one 'request', but in the 'online' case
  // you call Compute() [then maybe check AllOutputsAreComputable()] then
  // Prune() multiple times, with a sequence of different requests for
  // increasing time values.
  // Note: it sets the class member request_ to the address of 'request', so
  // you should not let 'request' go out of scope while this class might
  // still use it (e.g. until you call Compute() with a different
  void Compute(const ComputationRequest &request);

  // Returns true if all requested outputs are computable.  To be called after
  // Compute() but before Prune(().
  bool AllOutputsAreComputable() const;

  // Prints logging info to explain why all outputs are not computable.
  // To be called only if AllOutputsAreComputable() returned false.
  void ExplainWhyAllOutputsNotComputable() const;

  // This function outputs to "computable" information about whether each
  // requested element of each output was computable.  "computable" will have
  // the same size as request_->outputs, and each element will have the same
  // size as request_->outputs[i].indexes.size().  May only be called after
  // Compute() but before Prune().  If you have already called Prune(), you can
  // just assume everything was computable, or else Prune() would have crashed.
  void GetComputableInfo(std::vector<std::vector<bool> > *computable) const;

  // to be called after Compute(), this prunes away unused cindex_ids.
  // If not all the outputs are computable, this will die;
  // you can check the return status of AllOutputsAreComputable() first if
  // you want to avoid this.
  void Prune();

  // This enum says for each cindex_id, whether we can compute it from the given
  // inputs or not.  Note that there may be situations where before adding
  // dependencies of a particular cindex_id we realize that we won't be able to
  // use this cindex_id (i.e. it may be computable but it's not used) because
  // its usable_count is zero, and in those cases we change the status to
  // kWillNotCompute even though the cindex-id may be computable- for most
  // purposes this status is treated the same as kNotComputable.
  enum ComputableInfo {
    kUnknown = 0,
    kComputable = 1,
    kNotComputable = 2,
    kWillNotCompute = 3
  };
 private:
  // This function, called from ExplainWhyNotComputable(), prints to "os"
  // a human-readable form of a given cindex_id, that looks like
  // some_network_node(n, t, x), e.g. "final_logsoftmax(0, -4, 0)".
  void PrintCindexId(std::ostream &os, int32 cindex_id) const;

  // This function, typically to be called just before dying, prints logging
  // information to explain why the given cindex_id is not computable.
  void ExplainWhyNotComputable(int32 cindex_id) const;

  // called at the start of Compute(), this populates the graph (and member
  // variables) for all the inputs specified in the computation request.
  void AddInputs();

  // called at the start of Compute(), this populates the graph (and member
  // variables, including current_queue_) with all the outputs specified in the
  // computation request.
  void AddOutputs();

  // this does one iteration of building the graph, and increases
  // current_distance_ by one, i.e. it searches at one more remove from
  // the output.
  void BuildGraphOneIter();

  // make sure the "computable_info_" array is up to date.
  void UpdateAllComputableInfo();

  // (called from UpdateAllComputableInfo); make sure the computable_info for
  // cindex_id is up to date.  As a side effect this may also update the
  // usable_count_ array.
  void UpdateComputableInfo(int32 cindex_id);

  // (called from BuildGraphOneIter()), this function sets the cindex_id to
  // status kWillNotCompute and places members of depend_on_this_ into the
  // computable queue if needed.
  void SetAsWillNotCompute(int32 cindex_id);

  // compute and return the ComputableInfo for this cindex_id (kUnknown,
  // kComputable or kNotComputable).
  ComputableInfo ComputeComputableInfo(int32 cindex_id) const;

  // To be called when this cindex_id has just been newly added to graph_, this
  // function adds various initial variables associated with it, to *this.
  // is_input should be set to true if this cindex-id is being added as an input
  // (from request_.inputs), and is_output should be set to true if this
  // cindex-id is being added as an output (from request_.outputs).
  inline void AddCindexId(int32 cindex_id, bool is_input, bool is_output);

  // Add cindex_ids that this cindex_id depends on.
  void AddDependencies(int32 cindex_id);

  // increment the "usable" value of this cindex_id.
  void IncrementUsableCount(int32 cindex_id);

  // decrement the "usable" value of this cindex_id.
  void DecrementUsableCount(int32 cindex_id);

  // This function, called from Prune(), modifies the members of
  // graph_->dependencies-- it removes those cindexes that are not used in the
  // computation for the current cindex_id.  This will only do something
  // interesting in cases where there are optional dependencies.
  // It also clears the dependencies of those cindexes that are not computable.
  void PruneDependencies(int32 cindex_id);

  // This function, called from Prune(), computes an array "required", with an
  // element for each cindex_id that says whether it is required to compute the
  // requested outputs.  This is similar in function to the "usable_count_"
  // array, but it's more exact because it's computed after we have done
  // PruneDependencies() to remove unused dependencies, so it will only say
  // something is required if it is really accessed in the computation.
  // We'll later use this to remove unnecessary cindexes.
  // 'start_cindex_id' is the cindex_id from which the 'required' array is
  // to start (normally zero, but may be nonzero in multi-segment computations);
  // so 'required' is indexed by cindex_id - start_cindex_id.
  void ComputeRequiredArray(int32 start_cindex_id,
                            std::vector<bool> *required) const;

  // this function, to be called from Compute(), does some sanity checks to
  // verify that the internal state is consistent.  It only does this for the
  // current 'segment' of the computation, starting from 'start_cindex_id' (this
  // will be 0 in normal, single-segment computations).
  void Check(int32 start_cindex_id) const;

  const Nnet &nnet_;
  const ComputationRequest *request_;
  ComputationGraph *graph_;

  // this is the transpose of graph_->dependencies; it tells us
  // for each cindex_id, which other cindex_ids depend on it.
  std::vector<std::vector<int32> > depend_on_this_;

  // this vector, indexed by cindex_id, contains our information about whether
  // each cindex_id is computable; it's ComputableInfo, cast to char.
  std::vector<char> computable_info_;

  // this is a queue of cindex_ids that we need to re-compute whether they are
  // computable or not (because either they are new and haven't had dependencies
  // added, or their dependencies' computable status has changed since we last
  // computed their computable_ value).
  std::deque<int32> computable_queue_;
  // this vector tells us whether a cindex_id is in computable_queued_; it
  // stops us from adding things twice.
  std::vector<bool> computable_queued_;

  // usable_count_[i] for a cindex_id i is defined as 1 if i is a requested
  // output, and otherwise as the number of other cindex_ids j such that
  // computable_info_[j] is not kNotComputable AND usable_count_[j] > 0 AND i is
  // a member of graph->dependencies[j].  A cindex_id is termed "usable"
  // (meaning it could potentially participate in the computation of the output)
  // if its usable_count_ is > 0.  This quantity is designed to be easy to keep
  // updated as we add cindex_ids.
  std::vector<int32> usable_count_;

  // current_distance_ >= 0 is the distance to the output, of the cindex_ids in
  // current_queue_.
  int32 current_distance_;
  // the cindex_ids in current_queue_ are at distance "current_distance" to the
  // output and have not yet had their dependencies processed.
  std::vector<int32> current_queue_;
  // the cindex_ids in next_queue_ are at distance current_distance + 1 to the
  // output and have not yet had their dependencies processed.
  std::vector<int32> next_queue_;
};

/// This is to be used in logging only.
std::ostream& operator << (std::ostream &os,
                           const ComputationGraphBuilder::ComputableInfo &info);


class CindexSet {
 public:
  /// Parenthesis operator; returns true if this cindex exists in the set.
  bool operator () (const Cindex &cindex) const;

  /// with this constructor, represents the set of all Cindexes that exist
  /// in the graph.
  CindexSet(const ComputationGraph &graph);

  /// with this constructor, represents the set of all Cindexes that exist in
  /// the graph and which are computable.  If treat_unknown_as_computable is
  /// true then we consider kComputable and kUnknown to be computable, else we
  /// consider just nodes that are kComputable to be computable.
  CindexSet(const ComputationGraph &graph,
            const std::vector<char> &is_computable,
            bool treat_unknown_as_computable);
 private:
  const ComputationGraph &graph_;
  const std::vector<char> *is_computable_;
  bool treat_unknown_as_computable_;
};


/// An abstract representation of a set of Indexes.
class IndexSet {
 public:
  /// Returns true if this Index exists in the set.
  bool operator () (const Index &index) const;

  /// This constructor creates the set of all Indexes x such that a Cindex
  /// (node_id, x) which is computable exists in this graph.  If
  /// treat_unknown_as_computable is true then we consider kComputable and kUnknown
  /// to be computable, else we consider just nodes that are kComputable to be
  /// computable.
  IndexSet(const ComputationGraph &graph,
           const std::vector<char> &computable_info,
           int32 node_id,
           bool treat_unknown_as_computable);
 private:
  const ComputationGraph &graph_;
  const std::vector<char> &is_computable_;
  int32 node_id_;
  bool treat_unknown_as_computable_;
};




/**
   This function divides a computation into 'phases', where a 'phase' is a
   collection of cindexes which can (as far as the computation graph is
   concerned) all be computed at the same time, and depend only on cindexes
   previously computed in earlier phases.  So the phases are an ordering of the
   Cindexes in the computation, but an ordering that depends on graph-theoretic
   considerations only, and not practical concerns like whether the cindexes
   belong to the same node [for that, see the notion of steps].

   @param [in] nnet  The neural network this computation is for
   @param [in] graph  The computation graph that we're computing phases for.
   @param [out] phases_per_segment  The phases, listed separately for each
                segment of the computation [there will be just one segment in
                the normal case, more in the online-recognition case].  Consider
                just one segment for now.  Suppose the computation can be
                completed in 20 phases, then (*phases)[0].size() will be 20 at
                exit, and (*phases)[0][0] will be a sorted list of cindex_ids.
                that belong to the first phase, and so on. (Remember, a
                cindex_id is an index into graph->cindexes; it compactly
                identifies a cindex.)  The sets represented by the int32's in
                'phases_per_segment' will be disjoint and will cover all
                elements in [0 .. computation.cindexes.size() - 1].

                Note: we assume you have called PruneComputationGraph() before
                this function.  Even so, this function will be crash if the
                computation cannot actually be computed-- there are some
                mal-formed computations where you can build the computation graph
                but not the ordering of cindexes because there are dependencies
                forward and backward in time that intertwine.
*/
void ComputeComputationPhases(
    const Nnet &nnet,
    const ComputationGraph &computation_graph,
    std::vector<std::vector<std::vector<int32> > > *phases_per_segment);


/**
   This class arranges the cindex_ids of the computation into a sequence of
   lists called "steps", which will correspond roughly to the commands in the
   compiled computation.  The steps are finer than phases.  (See \ref
   dnn3_compile_steps for more info).  To summarize the properties that
   these steps will satisfy:

  - All cindex_ids within a given step correspond to the same node in the graph
  - All dependencies of cindex_ids within a given step have been computed in
    earlier steps.
  - All cindex_ids within a given step share the same location when
    computed (i.e. a matrix or submatix)

 There are also some extra, more obscure properties that the sequence of steps
 must satisfy:

  - Any input or output specified in a ComputationRequest must be in one step,
    with the Indexes in the same order as specified in the ComputationRequest.
    (Note: inputs can be for nodes of type kComponent as well as kInput).
  - If a step corresponds to a node of type kComponent (and does not
    correspond to an input in the ComputationRequest), then the immediately
    preceding step must correspond to a node of type kDescriptor, and the
    sequence of Indexes in the two steps must be identical.
  - If a step corresponds to a node of type kDimRange, then there must be
    a preceding step corresponding to the source node, with exactly the same
    Indexes appearing in the same order.  (This lets us use a sub-matrix for
    the kDimRange node).  We guarantee this by adding extra cindexes to the
    kDimRange steps as needed.

 The reason why computation_graph is not provided as a const argument is that in
 order to ensure the final property we may have to add a few new cindex_ids.
*/

class ComputationStepsComputer {
 public:
  /// Constructor.
  ///  @param [in] nnet        The neural network that this computation is for.
  ///  @param [in,out]  graph  The computation graph that we're computing the steps
  ///                          for.  It's only non-const because there are couple of
  ///                          unusual situations in which we may need to add new
  ///                          cindexes:
  ///                              (1) For nodes of type kDimRange, we may add
  ///                          cindexes to enable the node to span a contiguous range
  ///                          of the input indexes.
  ///                              (2) For certain non-simple Components
  ///                          (principally CNN-related components), we may need
  ///                          to add 'blank' cindexes (defined as cindexes
  ///                          where the 't' indexes are replaced by kNoTime),
  ///                          for zero-padding and to turn irregularly
  ///                          structured computations into regularly structured
  ///                          ones as needed by the component's implementation.
  ///  @param [out] steps     The main output of this class, which is
  ///                         a sequence of steps, each step being an ordered list of cindex_ids.
  ///                         It just gets cleared in the constructor; it's set up
  ///                         when you call ComputeForSegment().
  ///  @param [out] locations The additional output of this class, which is a function
  ///                         of the information in 'steps'.  The array
  ///                         'locations' is indexed by cindex_id, and each one is a pair
  ///                         (step-index, index-into-step), so that for any cindex_id c,
  ///                         (*steps)[locations[c].first][locations[c].second] == c.
  ///                          It's possible in principle if there are non-simple
  ///                          Components, that for nodes corresponding to component-input
  ///                          descriptors, a cindex might be present in more than one step,
  ///                          so it doesn't follow that if (*steps)[i][j] == c, then
  ///                          locations[c] == (i,j).
  ComputationStepsComputer(const Nnet &nnet,
                           ComputationGraph *graph,
                           std::vector<std::vector<int32> > *steps,
                           std::vector<std::pair<int32, int32> > *locations);

  /// You call this once for each segment, in order (note: for normal,
  /// non-online computations, there is only one segment).
  void ComputeForSegment(const ComputationRequest &request,
                         const std::vector<std::vector<int32> > &phases);

  /// This is only to be called after you have called ComputeForSegment
  /// for all the segments.
  void Check() const;
 private:

  // Adds step(s) for one "sub-phase".  A sub-phase is the set of cindex_ids from
  // one phase that have the same node index.  Note: for nodes that are
  // component-input descriptors, we don't actually create the step here, we
  // create it just before creating the step for its component, and we recreate
  // the list of cindexes from those from the component.  The reason is that
  // there are situations where doing it directly from the raw_step would not do
  // the right thing (especially with non-simple components, it's possible that
  // the cindexes component-input descriptors could be used twice by two
  // different components)..
  void ProcessSubPhase(const ComputationRequest &request,
                       const std::vector<Cindex> &sub_phase);

  // Called from ProcessSubPhase- for the case where it's a DimRangeNode.
  void ProcessDimRangeSubPhase(const std::vector<Cindex> &sub_phase);

  // Called from ProcessSubPhase- for the case where it's an input or output node.
  void ProcessInputOrOutputStep(const ComputationRequest &request,
                                bool is_output,
                                const std::vector<Cindex> &sub_phase);

  // Called from ProcessSubPhase- for the case where it's a component node.
  void ProcessComponentStep(const std::vector<Cindex> &step);


  // Splits a phase up into multiple "sub-phases", which are just the cindexes
  // from a phase that are from a single node, sorted.  At this point we
  // represent them as Cindexes, not cindex_ids.  For efficiency and because it
  // would be discarded anyway, it discards any raw steps that correspond to
  // component-input descriptors because these are not processed inside
  // ProcessSubPhase().
  void SplitIntoSubPhases(const std::vector<int32> &phase,
                          std::vector<std::vector<Cindex> > *sub_phase) const;

  // This low-level function used by functions like ProcessComponentStep,
  // ProcessInputStep and so on, adds one step to 'steps_' (converting from
  // Cindex to cindex_ids), and updates 'locations' appropriately.  It returns
  // the step index that we just added (== size of steps_ at entry).
  // If you specify add_if_absent = true, it will add any Cindexes that were
  // not already present, to the graph.  [this option is only to be used
  // in processing dim-range nodes.
  int32 AddStep(const std::vector<Cindex> &cindexes,
                bool add_if_absent = false);

  // This is an alternative interface to AddStep() that takes a list of
  // cindex_ids instead of cindexes (it's destructive of that list).
  int32 AddStep(std::vector<int32> *cindex_ids);


  // This utility function uses graph_ to convert a vector of cindex_ids into
  // Cindexes.
  void ConvertToCindexes(const std::vector<int32> &cindex_ids,
                         std::vector<Cindex> *cindexes) const;

  // Converts a vector of Cindexes to a vector of Indexes, by
  // stripping out the node index.
  static void ConvertToIndexes(const std::vector<Cindex> &cindexes,
                               std::vector<Index> *indexes);

  // Converts a vector of Indexes to Cindexes, using a supplied
  // node index.
  static void ConvertToCindexes(const std::vector<Index> &indexes,
                                int32 node_index,
                                std::vector<Cindex> *cindexes);


  // This utility function uses graph_ to convert a vector of cindex_ids into
  // Cindexes.   It will crash if the cindexes were not present in the graph.
  void ConvertToCindexIds(const std::vector<Cindex> &cindexes,
                          std::vector<int32> *cindex_ids) const;

  // This utility function uses the 'locations_' array to convert the cindex_ids
  // in 'cindex_ids' into an array (of the same length) of locations, i.e. of
  // pairs (step, index-into-step), so that if cindex_ids[i] = c, then
  // (*locations)[i] will be set to (*locations_)[c].  It will die if
  // one of the locations was not defined, i.e. was the pair (-1, -1).
  void ConvertToLocations(
      const std::vector<int32> &cindex_ids,
      std::vector<std::pair<int32, int32> > *locations) const;


  const Nnet &nnet_;
  ComputationGraph *graph_;
  /// steps_ is a pointer to an output that's passed in in the constructor.
  std::vector<std::vector<int32> > *steps_;
  /// locations_ is a map from cindex_id to the pair of indexes into steps_ where
  /// that cindex_id resides, so if (*locations_)[c] = (i,j), then
  /// (*steps_)[i][j] == c.  This is also an output (we get the pointer in
  /// the constructor).
  std::vector<std::pair<int32, int32> > *locations_;


  /// dim_range_nodes_ is used when allocating steps for nodes of type kDimRangeNode.
  /// This is a set of (source_step, dim_range_node_index),
  /// where source_step is the step in which we computed of the input
  /// of the dim-range node (this step will be for a node of type kComponentNode).
  /// This just tells us whether we've already added a particular dim-range node
  /// for this step, so we know whether we need to add it again.
  std::unordered_set<std::pair<int32, int32>, PairHasher<int32> > dim_range_nodes_;
};



} // namespace nnet3
} // namespace kaldi


#endif
