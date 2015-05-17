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

#include <deque>
#include "nnet3/nnet-computation-graph.h"

namespace kaldi {
namespace nnet3 {


int32 ComputationGraph::GetCindexId(Cindex cindex, bool *is_new) {
  typedef unordered_map<Cindex, int32, CindexHasher> map_type;
  int32 new_index = cindexes.size();  // we'll add this if we don't find it.
  std::pair<map_type::iterator, bool> p = cindex_to_cindex_id_.insert(
      std::pair<Cindex, int32>(cindex, new_index));
  if (p.second == true) {  // We added something to the hash.
    *is_new = true;
    cindexes.push_back(cindex);
    // make room for this "dependencies" entry.
    dependencies.resize(new_index + 1);
    return new_index;
  } else { // We did not add anything.
    *is_new = false;
    return p.first->second;
  }
}
int32 ComputationGraph::GetCindexId(Cindex cindex) const {
  typedef unordered_map<Cindex, int32, CindexHasher> map_type;
  map_type::const_iterator iter = cindex_to_cindex_id_.find(cindex);
  if (iter == cindex_to_cindex_id_.end())
    return -1;
  else
    return iter->second;
}


void ComputationGraph::Renumber(const std::vector<bool> &keep) {
  KALDI_ASSERT(optional.empty());
  int32 num_cindex_ids = cindexes.size();
  KALDI_ASSERT(keep.size() == num_cindex_ids);
  ComputationGraph temp_graph;
  std::vector<int32> old2new(num_cindex_ids, -1), new2old;
  new2old.reserve(num_cindex_ids);
  for (int32 j = 0; j < num_cindex_ids; j++) {
    if (keep[j]) {
      old2new[j] = new2old.size();      
      new2old.push_back(j);
    }
  }
  int32 new_num_cindex_ids = new2old.size();
  if (new_num_cindex_ids == num_cindex_ids) {
    // this is an optimization for when we are not deleting any
    // cindex-ids.
    return;
  }
  temp_graph.cindexes.resize(new_num_cindex_ids);
  temp_graph.dependencies.resize(new_num_cindex_ids);
  for (int32 c = 0; c < new_num_cindex_ids; c++) {
    int32 d = new2old[c];
    temp_graph.cindexes[c] = cindexes[d];
    temp_graph.dependencies[c].reserve(dependencies[d].size());
    std::vector<int32>::const_iterator
        iter = dependencies[d].begin(), end = dependencies[d].end();
    for (; iter != end; ++iter) {
      int32 old_dep = *iter, new_dep = old2new[old_dep];
      if (new_dep != -1)
        temp_graph.dependencies[c].push_back(new_dep);
    }
  }
  // at this point, rather than setting up cindex_to_cindex_id_ on the
  // temporary graph, we copy cindexes and dependencies to this graph,
  // and then set up cindex_to_cindex_id_ locally.
  cindexes.swap(temp_graph.cindexes);
  dependencies.swap(temp_graph.dependencies);
  optional.clear();
  cindex_to_cindex_id_.clear();
  for (int32 c = 0; c < new_num_cindex_ids; c++)
    cindex_to_cindex_id_[cindexes[c]] = c;
}


// make our own namespace for helper functions of ComputeComputationGraph.
namespace computation_graph {


// This function adds cindex_ids corresponding to each output
// index, to the graph.
void AddOutputToGraph(const ComputationRequest &request,
                      const Nnet &nnet,                      
                      ComputationGraph *graph) {
  int32 num_added = 0;
  for (int32 i = 0; i < request.outputs.size(); i++) {
    int32 n = nnet.IndexOfNode(request.outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.outputs[i].name;
    for (int32 j = 0; j < request.outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.outputs[i].indexes[j]);
      bool is_new;
      graph->GetCindexId(cindex, &is_new);  // ignore the return value.
      KALDI_ASSERT(is_new && "Output index seems to be listed more than once");
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddOutputToGraph: nothing to add.");
}  


// This function adds cindex_ids corresponding to each input
// index, to the graph.
void AddInputToGraph(const ComputationRequest &request,
                     const Nnet &nnet,                      
                     ComputationGraph *graph) {
  int32 num_added = 0;
  for (int32 i = 0; i < request.inputs.size(); i++) {
    int32 n = nnet.IndexOfNode(request.inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.inputs[i].name;
    for (int32 j = 0; j < request.inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.inputs[i].indexes[j]);
      bool is_new;
      graph->GetCindexId(cindex, &is_new);  // ignore the return value.
      KALDI_ASSERT(is_new && "Input index seems to be listed more than once");
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddOutputToGraph: nothing to add.");
}  


// compute, for each cindex_id, a list of other cindex_ids that
// directly depend on it in order to be computed.
// used in ComputeComputationOrder.
static void ComputeDependsOn(
    const ComputationGraph &graph,
    std::vector<std::vector<int32> > *depend_on_this) {
  int32 num_cindex_ids = graph.cindexes.size();  
  depend_on_this->clear();
  depend_on_this->resize(num_cindex_ids);
  // next block computes "depend_on_this".
  for (int32 c = 0; c < num_cindex_ids; c++) {
    std::vector<int32>::const_iterator
        iter = graph.dependencies[c].begin(),
        end = graph.dependencies[c].end();
    for (; iter != end; ++iter) {
      int32 d = *iter;
      (*depend_on_this)[d].push_back(c);
    }
  }
}

// compute a sorted (and naturally unique) list of cindex_ids in the
// graph that are inputs in the "request".
static void ComputeInputCindexIds(
    const ComputationRequest &request,
    const Nnet &nnet,
    const ComputationGraph &graph,
    std::vector<int32> *inputs) {
  inputs->clear();
  for (int32 i = 0; i < request.inputs.size(); i++) {
    int32 n = nnet.IndexOfNode(request.inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no input with name "
                << request.inputs[i].name;
    for (int32 j = 0; j < request.inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.inputs[i].indexes[j]);
      int32 cindex_id = graph.GetCindexId(cindex);
      if (cindex_id != -1)
        inputs->push_back(cindex_id);
    }
  }
  std::sort(inputs->begin(), inputs->end());
  KALDI_ASSERT(IsSortedAndUniq(*inputs) &&
               "Computation contains duplicate indexes.");
}  

// compute a sorted (and naturally unique) list of cindex_ids in the
// graph that are outputs in the "request".
static void ComputeOutputCindexIds(
    const ComputationRequest &request,
    const Nnet &nnet,
    const ComputationGraph &graph,
    std::vector<int32> *outputs) {
  outputs->clear();
  for (int32 i = 0; i < request.outputs.size(); i++) {
    int32 n = nnet.IndexOfNode(request.outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.outputs[i].name;
    for (int32 j = 0; j < request.outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.outputs[i].indexes[j]);
      int32 cindex_id = graph.GetCindexId(cindex);
      // outputs are not allowed to be missing.
      KALDI_ASSERT(cindex_id != -1 && "Output not present in graph.");
      if (cindex_id != -1)
        outputs->push_back(cindex_id);
    }
  }
  std::sort(outputs->begin(), outputs->end());
  KALDI_ASSERT(IsSortedAndUniq(*outputs) &&
               "Computation contains duplicate indexes.");
}  

static void CheckOutputsAreComputable(
    const ComputationRequest &request,
    const Nnet &nnet,
    const ComputationGraph &graph,
    const std::vector<bool> &computable) {
  std::vector<int32> output_cindex_ids;
  ComputeOutputCindexIds(request, nnet, graph, &output_cindex_ids);
  int32 num_cindex_ids = graph.cindexes.size(),
      num_output_cindex_ids = output_cindex_ids.size(),
      num_output_cindex_ids_computable = 0,
      num_cindex_ids_computable = 0;
  for (int32 i = 0; i < num_output_cindex_ids; i++)
    if (computable[output_cindex_ids[i]])
      num_output_cindex_ids_computable++;
  for (int32 i = 0; i < num_cindex_ids; i++)
    if (computable[i])
      num_cindex_ids_computable++;
  if (num_output_cindex_ids_computable < num_output_cindex_ids) {
    KALDI_ERR << "Cannot do requested computation: can only compute "
              << num_output_cindex_ids_computable << " out of "
              << num_output_cindex_ids << ".  In the whole computation, "
              << num_cindex_ids_computable << " out of " << num_cindex_ids
              << " are computable.";
  }      
}

// Sorts and uniq's a vector of int32, while keeping
// an associated vector of bool in sync with the sorting if it is nonempty.
// It is an error if the same int32 appears with two different boolean values.
void SortAndUniqInSync(std::vector<int32> *int_vec,
                       std::vector<bool> *bool_vec) {
  if (bool_vec->empty()) {
    SortAndUniq(int_vec);
  } else {
    int32 size = int_vec->size();
    KALDI_ASSERT(bool_vec->size() == size);
    std::vector<std::pair<int32, bool> > pairs(size);
    for (int32 i = 0; i < size; i++) {
      pairs[i].first = (*int_vec)[i];
      pairs[i].second = (*bool_vec)[i];
    }
    SortAndUniq(&pairs);
    int32 new_size = pairs.size();
    if (new_size != size) {
      int_vec->resize(new_size);
      bool_vec->resize(new_size);
    }
    for (int32 i = 0; i < new_size; i++) {
      (*int_vec)[i] = pairs[i].first;
      (*bool_vec)[i] = pairs[i].second;
      if (i > 0) {
        KALDI_ASSERT((*int_vec)[i] != (*int_vec)[i-1] && "Component lists the "
                     "same input as both optional and not optional");
      }      
    }
  }
}


void ComputeComputationGraph(const ComputationRequest &request,
                             const Nnet &nnet,
                             ComputationGraph *graph) {
  using namespace computation_graph;
  // make sure graph is empty at the start.
  KALDI_ASSERT(graph->cindexes.empty());

  AddInputToGraph(request, nnet, graph);
  AddOutputToGraph(request, nnet, graph);

  // queue of cindex_ids to process.
  std::vector<int32> queue(graph->cindexes.size());
  for (int32 i = 0; i < graph->cindexes.size(); i++)
    queue.push_back(i);

  while (!queue.empty()) {
    int32 cindex_id = queue.back();
    queue.pop_back();    
    Cindex cindex = graph->cindexes[cindex_id];
    // find the dependencies of this cindex.
    int32 n = cindex.first;
    const Index &index = cindex.second;
    const NetworkNode &node = nnet.GetNode(n);

    std::vector<Cindex> input_cindexes;
    std::vector<bool> is_optional;  // says whether each required cindex is
                                    // optional or not (only for kComponent).
    
    // the following switch statement sets up "required_cindexes"
    // which is a list of this cindex's dependencies.
    switch (node.node_type) {
      case NetworkNode::kInput:
        // once we reach input nodes there will be no dependencies, so nothing to do.        
        break;
      case NetworkNode::kOutput: case NetworkNode::kComponentInput:
        {
          // desc describes how this node obtains its input from other nodes.
          const Descriptor &desc = node.descriptor;
          desc.GetInputCindexes(index, &input_cindexes);
        }
        break;
      case NetworkNode::kComponent:
        int32 c = node.u.component_index;
        const Component *component = nnet.GetComponent(c);
        std::vector<Index> input_indexes;
        component->GetInputIndexes(request.misc_info, index,
                                   &input_indexes, &is_optional);
        // each Component node should be preceded by a node that describes its
        // input, of type kComponentInput.
        KALDI_ASSERT(nnet.GetNode(n-1).node_type ==
                     NetworkNode::kComponentInput);
        input_cindexes.resize(input_indexes.size());
        for (size_t i = 0; i < input_indexes.size(); i++) {
          input_cindexes[i].first = n - 1;  // preceding node.
          input_cindexes[i].second = input_indexes[i];
        }
        break;
    }
    if (static_cast<int32>(graph->dependencies.size()) <= cindex_id) {
      // the "dependencies" and "optional" arrays always maintain the same size.
      graph->dependencies.resize(cindex_id + 1);
      graph->optional.resize(cindex_id + 1);
    }
    std::vector<int32> &this_dep = graph->dependencies[cindex_id];
    std::vector<bool> &this_opt = graph->optional[cindex_id];
    int32 num_dependencies = input_cindexes.size();
    this_dep.resize(num_dependencies);
    for (size_t i = 0; i < num_dependencies; i++) {
      bool is_new;
      int32 dep_cindex_id = graph->GetCindexId(input_cindexes[i], &is_new);
      this_dep[i] = dep_cindex_id;
      if (is_new)
        queue.push_back(dep_cindex_id);
    }
    this_opt = is_optional;
    // Make sure we are not listing any dependency twice.
    SortAndUniqInSync(&this_dep, &this_opt);
  }
}

} // end namespace computation_graph

void ComputeComputationOrder(
    const Nnet &nnet,    
    const ComputationRequest &request,
    const ComputationGraph &graph,
    std::vector<int32> *order,
    std::vector<std::vector<int32> > *by_order) {
  using namespace computation_graph;
  if (order == NULL) {  // ensure order != NULL by recursing if it's NULL.
    std::vector<int32> order_tmp;
    ComputeComputationOrder(nnet, request, graph,
                            &order_tmp, by_order);
    return;
  }
  KALDI_ASSERT(graph.optional.empty() &&
               "You must call PruneComputationGraph before "
               "ComputeComputationOrder.");
  int32 num_cindex_ids = graph.cindexes.size();
  order->clear();
  order->resize(num_cindex_ids, -1);
  // "depend_on_this" is, for each cindex_id, a list of cindex_ids that depend on
  // it.  this is used to help us evaluate only for those cindex_ids that might
  // only now have become computable (i.e. to stop the algorithm taking
  // potentially quadratic time for things like RNNs).
  std::vector<std::vector<int32> > depend_on_this(num_cindex_ids);
  ComputeDependsOn(graph, &depend_on_this);

  int32 num_computed = 0;
  int32 cur_order = 0;
  std::vector<int32> this_order;  // list of elements of this order.

  ComputeInputCindexIds(request, nnet, graph, &this_order);
  for (int32 i = 0; i < this_order.size(); i++)
    (*order)[this_order[i]] = 0;  // set order of inputs.
  
  if (by_order) {
    by_order->clear();
    by_order->reserve(50);  // minimize un-needed copies.  50 is very
                            // arbitrarily chosen; it's the maximum context we
                            // anticipate ever having for something like an RNN.
  }
  std::vector<int32> next_order_candidates;  
  for (; !this_order.empty(); cur_order++) {
    if (by_order) by_order->push_back(this_order);
    num_computed += this_order.size();
    // next_order_candidates is a list of cindexes that we should check whether
    // they are computable now, because one of the things they depend on just
    // became computable.  We declared it outside to avoid reallocation.
    next_order_candidates.clear();  
    for (int32 i = 0; i < this_order.size(); i++) {
      int32 c = this_order[i];  // c is a cindex_id with order cur_order.
      std::vector<int32>::const_iterator iter = depend_on_this[c].begin(),
          end = depend_on_this[c].end();
      for (; iter != end; ++iter) {
        int32 d = *iter;  // cindex_id that depends on c.
        next_order_candidates.push_back(d);
      }
    }
    SortAndUniq(&next_order_candidates);
    this_order.clear();
    // now check the candidates that might be of the next order, and put any
    // things that we are able to compute in "this_order".
    for (int32 i = 0; i < next_order_candidates.size(); i++) {
      int32 c = next_order_candidates[i];
      std::vector<int32>::const_iterator
          iter = graph.dependencies[c].begin(),
          end = graph.dependencies[c].end();
      for (; iter != end; ++iter) {
        int32 d = *iter;  // d is cindex_id that c depends on.
        if ((*order)[d] < 0)  // we can't compute c yet as something we depend
          break;              // on is not yet computed.
      }
      if (iter == end) { // we reached the end and did not break, so all
                         // dependencies satisfied
        this_order.push_back(c);  // cindex_id c can be computed at this time.
      }
    }
  }
  // make sure everything was computable.  If the next assert fails it's likely
  // a bug in this function or in PruneComputataionGraph.
  KALDI_ASSERT(num_computed == num_cindex_ids);
}

// This helper function used in PruneComputationGraph returns true if
// we can compute cindex_id from the cindex_ids that are already computable
// (i.e. whose entries in the "computable" array are true).
static bool IsComputable(const Nnet &nnet,
                         const ComputationRequest &request,
                         const ComputationGraph &graph,                         
                         const std::vector<bool> &computable,
                         int32 cindex_id) {
  KALDI_ASSERT(cindex_id >= 0 &&
               cindex_id < graph.dependencies.size());
  const std::vector<int32> &this_dep = graph.dependencies[cindex_id];
  const std::vector<bool> *optional = NULL;
  if (!graph.optional.empty() && !graph.optional[cindex_id].empty())
    optional = &(graph.optional[cindex_id]);
  int32 num_dependencies = this_dep.size();
  int32 num_computable = 0;
  for (int32 i = 0; i < num_dependencies; i++) {
    int32 d = this_dep[i];
    if (computable[d]) {  // this dependency is currently computable.
      num_computable++;  // keep track of how many dependencies we could compute.
    } else {  // this dependency can't currently be computed.
      bool is_optional = (optional != NULL && (*optional)[i]);
      if (!is_optional)  // A non-optional dependency can't be computed,
        return false;    // so this cindex_id can't be computed.
    }
  }
  // At this point, we would have already returned false if a non-optional
  // dependency couldn't be computed.  The rule at this point is: if any
  // dependency can be computed, or there are no dependencies at all,
  // return true; else return false.  The reason to insist t
  return (num_computable > 0 || num_dependencies == 0);
}



void PruneComputationGraph(const Nnet &nnet,
                           const ComputationRequest &request,
                           ComputationGraph *graph) {
  using namespace computation_graph;  
  int32 num_cindex_ids = graph->cindexes.size();
  // "depend_on_this" is, for each cindex_id, a list of cindex_ids that depend on it
  // (optionally or not).  this is used to help us evaluate only for those
  // cindex_ids that might only now have become computable (i.e. to stop the
  // algorithm taking potentially quadratic time for things like RNNs).
  std::vector<std::vector<int32> > depend_on_this(num_cindex_ids);
  ComputeDependsOn(*graph, &depend_on_this);
  
  std::vector<bool> computable(num_cindex_ids, false);

  std::vector<int32> input_cindex_ids;  // list of input elements.
  ComputeInputCindexIds(request, nnet, *graph,
                        &input_cindex_ids);

  unordered_set<int32> is_queued;
  std::deque<int32> queue;  // if this is slow we could try making this a deque
                            // and popping from the front.
  for (size_t i = 0; i < input_cindex_ids.size(); i++) {
    int32 c = input_cindex_ids[i];
    computable[c] = true;
    for (size_t j = 0; j < depend_on_this[c].size(); j++) {
      int32 d = depend_on_this[c][j];
      // note: d cannot be an input since it has a dependency.      
      KALDI_ASSERT(!computable[d]);      
      if (is_queued.insert(d).second)  // if not already there..
        queue.push_back(d);
    }
  }
  while (!queue.empty()) {
    int32 c = queue.front();
    queue.pop_front();
    is_queued.erase(c);
    KALDI_ASSERT(!computable[c]);
    if (IsComputable(nnet, request, *graph,
                     computable, c)) {
      computable[c] = true;
      for (size_t j = 0; j < depend_on_this[c].size(); j++) {
        int32 d = depend_on_this[c][j];
        if (!computable[d]) {  // d depends not yet known to be computable.
          if (is_queued.insert(d).second)  // if not already there
            queue.push_back(d);
        }
      }
    }
  }

  CheckOutputsAreComputable(request, nnet, *graph, computable);
  
  // At this point we can forget the information about which dependencies were
  // optional.  (Renumber requires us to do this as it doesn't renumber that).
  graph->optional.clear();  
  // Renumber to keep only computable cindex-ids;
  graph->Renumber(computable);
}

namespace compute_computation_steps {
// namespace for some helper functions for ComputeComputationSteps.

/// Adds a "step" for each of the inputs in the ComputationRequest.
/// Does this in the same order in which they were declared in
/// the request (this won't matter at all).
void AddInputSteps(const Nnet &nnet,
                   const ComputationRequest &request,
                   const ComputationGraph &graph,                   
                   std::vector<std::vector<int32> > *by_step) {
  KALDI_ASSERT(by_step->empty());
  by_step->reserve(50);  // will minimize unnecessary copies of vectors.
  std::set<int32> all_nodes;  // to make sure nothing listed twice.
  for (int32 i = 0; i < request.inputs.size(); i++) {
    int32 n = nnet.IndexOfNode(request.inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.inputs[i].name;
    // ensure no input node is listed twice.
    KALDI_ASSERT(all_nodes.count(n) == 0 && "Invalid computation request: "
                 "double listing of node.");
    all_nodes.insert(n);
    KALDI_ASSERT(!request.inputs[i].indexes.empty() &&
                 "Computation request had no indexes for input ");
    by_step->push_back(std::vector<int32>());
    std::vector<int32> &this_step = by_step->back();
    this_step.resize(request.inputs[i].indexes.size());
    for (int32 j = 0; j < request.inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.inputs[i].indexes[j]);
      int32 cindex_id = graph.GetCindexId(cindex);
      KALDI_ASSERT(cindex_id != -1);  // would be code error.
      this_step[j] = cindex_id;
    }
  }
}


/// Adds a "step" for each of the outputs in the ComputationRequest.  This will
/// be done after adding steps for all the inputs and then all the
/// non(input/output)s.  Does this in the same order in which they were declared
/// in the request (this won't matter at all).
void AddOutputSteps(const Nnet &nnet,
                    const ComputationRequest &request,
                    const ComputationGraph &graph,                   
                    std::vector<std::vector<int32> > *by_step) {
  std::set<int32> all_nodes;  // to make sure nothing listed twice.
  for (int32 i = 0; i < request.outputs.size(); i++) {
    int32 n = nnet.IndexOfNode(request.outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.outputs[i].name;
    // ensure no output node is listed twice.
    KALDI_ASSERT(all_nodes.count(n) == 0 && "Invalid computation request: "
                 "double listing of node.");
    all_nodes.insert(n);
    KALDI_ASSERT(!request.outputs[i].indexes.empty() &&
                 "Computation request had no indexes for output ");
    by_step->push_back(std::vector<int32>());
    std::vector<int32> &this_step = by_step->back();
    this_step.resize(request.outputs[i].indexes.size());
    for (int32 j = 0; j < request.outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.outputs[i].indexes[j]);
      int32 cindex_id = graph.GetCindexId(cindex);
      KALDI_ASSERT(cindex_id != -1);  // would be code error.
      this_step[j] = cindex_id;
    }
  }
}

/// Adds steps corresponding to everything that is not an input or output in the
/// ComputataionRequest.  The way this works is: for each order-index, it takes
/// all Cindexes that don't correspond to either inputs or outputs of the network,
/// and it separates them into groups based on node-index so that things with the
/// same order but different node-index will be in separate steps.
void AddIntermediateSteps(
    const Nnet &nnet,
    const ComputationRequest &request,
    const ComputationGraph &graph,
    const std::vector<std::vector<int32> > &by_order,
    std::vector<std::vector<int32> > *by_step) {
  int32 num_order_indexes = by_order.size();

  std::vector<char> is_input_or_output(nnet.NumNodes(), '\0');
  for (int32 node_index = 0; node_index < nnet.NumNodes(); node_index++) {
    NetworkNode::NodeType t = nnet.GetNode(node_index).node_type;
    if (t == NetworkNode::kInput || t == NetworkNode::kOutput)
      is_input_or_output[node_index] = static_cast<char>(1);
  }


  std::vector<Cindex> cindexes;  
  for (int32 order_index = 0; order_index < num_order_indexes; order_index++) {
    const std::vector<int32> &this_cindex_ids = by_order[order_index];
    
    cindexes.clear();
    cindexes.reserve(this_cindex_ids.size());
    int32 num_cindex_ids = this_cindex_ids.size();
    for (int32 i = 0; i < num_cindex_ids; i++) {
      int32 cindex_id = this_cindex_ids[i],
          node_index = graph.cindexes[cindex_id].first;
      if (!is_input_or_output[node_index])
        cindexes.push_back(graph.cindexes[cindex_id]);
    }
    // now "cindexes" contains all Cindexes that are not from input or output nodes.
    // Sorting this array gives us the ordering we want, where Cindexes from different
    // node-ids are separated into contiguous ranges, and within each range, they
    // are sorted by Index.
    std::sort(cindexes.begin(), cindexes.end());

    std::vector<Cindex>::iterator iter = cindexes.begin(), end = cindexes.end();
    while (iter != end) {
      // each pass through this while loop processes one batch of cindex_ids;
      // each batch has a particular node-index.
      std::vector<Cindex>::iterator cur_end = iter;
      int32 this_node_id = iter->first;
      while (cur_end != end && cur_end->first == this_node_id)
        cur_end++;
      // the range [iter, cur_end) is nonempty and contains all the same node-id.
      int32 size = cur_end - iter;
      by_step->push_back(std::vector<int32>());
      std::vector<int32> &this_step = by_step->back();
      this_step.resize(size);
      for (int32 i = 0; i < size; i++, iter++)
        this_step[i] = graph.GetCindexId(*iter);
      // at this point iter will be equal to cur_end; it will point to either
      // the end of the "cindexes" vector, or the beginning of the next set
      // of Cindexes to process.
    }
  }
}

} // namespace compute_computation_steps.


void ComputeComputationSteps(
    const Nnet &nnet,
    const ComputationRequest &request,
    const ComputationGraph &graph,
    const std::vector<std::vector<int32> > &by_order,
    std::vector<std::vector<int32> > *by_step) {
  using namespace compute_computation_steps;
  by_step->clear();
  AddInputSteps(nnet, request, graph, by_step);
  AddIntermediateSteps(nnet, request, graph, by_order, by_step);
  AddOutputSteps(nnet, request, graph, by_step);
}


} // namespace nnet3
} // namespace kaldi
