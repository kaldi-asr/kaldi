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



namespace computation_graph {

// make our own namespace for helper functions of ComputeComputationGraph.
void AddOutputToGraph(const ComputationRequest &request,
                      ComputationGraph *graph) {
  // TODO.
  // adds just the requested output cindexes as cindexes in the computation graph.
}

// compute, for each cindex_id, a list of other cindex_ids that
// directly depend on it in order to be computed.
// used in ComputeComputationOrder.
static void ComputeDependsOn(
    const ComputationGraph &graph,
    std::vector<std::vector<int32> > *depends_on) {
  int32 num_cindex_ids = graph.cindexes.size();  
  depends_on->clear();
  depends_on->resize(num_cindex_ids);
  // next block computes "depends_on".
  for (int32 c = 0; c < num_cindex_ids; c++) {
    std::vector<int32>::const_iterator
        iter = graph.dependencies[c].begin(),
        end = graph.dependencies[c].end();
    for (; iter != end; ++iter) {
      int32 d = *iter;
      (*depends_on)[d].push_back(c);
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



void ComputeComputationGraph(const ComputationRequest &request,
                             const Nnet &nnet,
                             ComputationGraph *graph) {
  using namespace computation_graph;
  // make sure graph is empty at start.
  KALDI_ASSERT(graph->cindexes.empty());

  AddOutputToGraph(request, graph);

  // queue of cindex_ids to process.
  std::vector<int32> queue(graph->cindexes.size());
  for (int32 i = 0; i < graph->cindexes.size(); i++)
    queue.push_back(i);

  // TODO.  This code needs to be changed now that we made the inputs separate
  // nodes.
  while (!queue.empty()) {
    int32 cindex_id = queue.back();
    queue.pop_back();    
    Cindex cindex = graph->cindexes[cindex_id];
    // find the dependencies of this cindex.
    int32 n = cindex.first;
    const Index &index = cindex.second;
    const NetworkNode &node = nnet.GetNode(n);
    // once we reach input nodes there will be no dependencies, so nothing to do.
    if (node.node_type == NetworkNode::kInput)
      continue;
    
    // desc describes how this node obtains its input from other nodes.
    const Descriptor &desc = node.input; 
    int32 c = node.component_index;
    const Component *component = nnet.GetComponent(c);
    // list of inputs we require, in cindex_id form.
    std::vector<int32> required_cindex_ids;
    
    // input_indexes is the indexes at the input of the component (for simple
    // components, will have length one).
    std::vector<Index> input_indexes;
    component->GetInputIndexes(request.misc_info, index,
                               &input_indexes);
    for (size_t i = 0; i < input_indexes.size(); i++) {
      std::vector<Cindex> required_cindexes;
      desc.GetInputCindexes(input_indexes[i], &required_cindexes);
      std::vector<Cindex>::iterator iter = required_cindexes.begin(),
          end = required_cindexes.end();
      for (; iter != end; ++iter) {
        bool is_new;
        int32 dep_cindex_id = graph.GetCindexId(*iter, &is_new);
        required_cindex_ids.push_back(dep_cindex_id);
        if (is_new)
          queue.push_back(dep_cindex_id);
      }
    }
    SortAndUniq(&required_cindex_ids);
    dependencies[cindex_id] = required_cindex_ids;
  }
}


void ComputeComputationOrder(
    const Nnet &nnet,    
    const ComputationRequest &request,
    const ComputationGraph &graph,
    const std::vector<int32> &shortest_distance,
    std::vector<int32> *order,
    std::vector<std::vector<int32> > *by_order) {
  if (order == NULL) {  // ensure order != NULL by recursing if it's NULL.
    std::vector<int32> order_tmp;
    ComputeComputationOrder(request, nnet, graph, shortest_distance,
                            order_tmp, by_order);
    return;
  }
  KALDI_ASSERT(graph.optional.empty() &&
               "You must call PruneComputationGraph before "
               "ComputeComputationOrder.");
  int32 num_cindex_ids = graph.cindexes.size();
  order->clear();
  order->resize(num_cindex_ids, -1);
  // "depends_on" is, for each cindex_id, a list of cindex_ids that depend on
  // it.  this is used to help us evaluate only for those cindex_ids that might
  // only now have become computable (i.e. to stop the algorithm taking
  // potentially quadratic time for things like RNNs).
  std::vector<std::vector<int32> > depends_on(num_cindex_ids);
  ComputeDependsOn(graph, &depends_on);
  
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
    // next_order_candidates is a list of cindexes that we should check whether
    // they are computable now, because one of the things they depend on just
    // became computable.  We declared it outside to avoid reallocation.
    next_order_candidates.clear();  
    for (int32 i = 0; i < this_order.size(); i++) {
      int32 c = this_order[i];  // c is a cindex_id with order cur_order.
      std::vector<int32>::const_iterator iter = depends_on[c].begin(),
          end = depends_on[c].end();
      for (; iter != end; ++iter) {
        int32 d = *iter;  // cindex_id that depends on c.
        next_order_candidates.push_back(d);
      }
    }
    SortAndUniq(next_order_candidates);
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
      if (iter == end) { // we reached the end so all dependencies satisfied.
        this_order.push_back(c);  // cindex_id c can be computed at this time.
      }
    }
  }
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
  if (!computable.optional.empty() && !computable.optional[cindex_id].empty())
    optional = &(computable.optional[cindex_id]);
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
  int32 num_cindex_ids = graph.cindexes.size();
  // "depends_on" is, for each cindex_id, a list of cindex_ids that depend on it
  // (optionally or not).  this is used to help us evaluate only for those
  // cindex_ids that might only now have become computable (i.e. to stop the
  // algorithm taking potentially quadratic time for things like RNNs).
  std::vector<std::vector<int32> > depends_on(num_cindex_ids);
  ComputeDependsOn(graph, &depends_on);
  
  std::vector<bool> computable(num_cindex_ids, false);

  std::vector<int32> input_cindex_ids;  // list of input elements.
  ComputeInputCindexIds(request, nnet, graph,
                        &input_cindex_ids);

  std::unordered_set<int32> is_queued;
  std::deque<int32> queue;  // if this is slow we could try making this a deque
                            // and popping from the front.
  for (size_t i = 0; i < input_cindex_ids.size(); i++) {
    int32 c = input_cindex_ids[i];
    computable[c] = true;
    for (size_t j = 0; j < depends_on[c].size(); j++) {
      int32 d = depends_on[c][j];
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
    if (IsComputable(nnet, request, graph,
                     computable, c)) {
      computable[c] = true;
      for (size_t j = 0; j < depends_on[c].size(); j++) {
        int32 d = depends_on[c][j];
        if (!computable[d]) {  // d depends not yet known to be computable.
          if (is_queued.insert(d).second)  // if not already there
            queue.push_back(d);
        }
      }
    }
  }

  CheckOutputsAreComputable(request, nnet, computataion_graph, computable);
  
  // At this point we can forget the information about which dependencies were
  // optional.  (Renumber requires us to do this as it doesn't renumber that).
  graph->optional.clear();  
  // Renumber to keep only computable cindex-ids;
  graph->Renumber(computable);
}


} // namespace nnet3
} // namespace kaldi
