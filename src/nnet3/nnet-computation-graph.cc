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
#include "nnet3/nnet-graph.h"

namespace kaldi {
namespace nnet3 {


int32 ComputationGraph::GetCindexId(const Cindex &cindex,
                                    bool input, bool *is_new) {
  typedef unordered_map<Cindex, int32, CindexHasher> map_type;
  int32 new_index = cindexes.size();  // we'll add this if we don't find it.
  std::pair<map_type::iterator, bool> p = cindex_to_cindex_id_.insert(
      std::pair<Cindex, int32>(cindex, new_index));
  if (p.second == true) {  // We added something to the hash.
    *is_new = true;
    KALDI_ASSERT(is_input.size() == cindexes.size());
    cindexes.push_back(cindex);
    is_input.push_back(input);
    // make room for this "dependencies" entry.
    dependencies.resize(new_index + 1);
    return new_index;
  } else { // We did not add anything.
    *is_new = false;
    return p.first->second;
  }
}
int32 ComputationGraph::GetCindexId(const Cindex &cindex) const {
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
  temp_graph.is_input.resize(new_num_cindex_ids);  
  temp_graph.dependencies.resize(new_num_cindex_ids);
  for (int32 c = 0; c < new_num_cindex_ids; c++) {
    int32 d = new2old[c];
    temp_graph.cindexes[c] = cindexes[d];
    temp_graph.is_input[c] = is_input[d];
    temp_graph.dependencies[c].reserve(dependencies[d].size());
    std::vector<int32>::const_iterator
        iter = dependencies[d].begin(), end = dependencies[d].end();
    for (; iter != end; ++iter) {
      int32 old_dep = *iter, new_dep = old2new[old_dep];
      if (new_dep != -1)
        temp_graph.dependencies[c].push_back(new_dep);
    }
  }
  // at this point, rather than setting up cindex_to_cindex_id_ on the temporary
  // graph, we copy cindexes, is_input and dependencies to this graph, and then
  // set up cindex_to_cindex_id_ locally.
  cindexes.swap(temp_graph.cindexes);
  is_input.swap(temp_graph.is_input);
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
      bool is_input = false, is_new;
      graph->GetCindexId(cindex, is_input, &is_new);  // ignore the return value.
      KALDI_ASSERT(is_new && "Output index seems to be listed more than once");
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddOutputToGraph: nothing to add.");
}  


// This function adds cindex_ids corresponding to each input index, to the
// graph.
void AddInputToGraph(const ComputationRequest &request,
                     const Nnet &nnet,                      
                     ComputationGraph *graph) {
  int32 num_added = 0;
  for (int32 i = 0; i < request.inputs.size(); i++) {
    int32 n = nnet.IndexOfNode(request.inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no input with name "
                << request.inputs[i].name;
    NetworkNode::NodeType t = nnet.GetNode(n).node_type;
    KALDI_ASSERT(t == NetworkNode::kInput || t == NetworkNode::kComponent &&
                 "Inputs to graph only allowed for Input and Component nodes.");
    
    for (int32 j = 0; j < request.inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.inputs[i].indexes[j]);
      bool is_input = true, is_new;
      graph->GetCindexId(cindex, is_input, &is_new);  // ignore the return value.
      KALDI_ASSERT(is_new && "Input index seems to be listed more than once");
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddInputToGraph: nothing to add.");
}  


// This function outputs to dependencies_subset[c], for each cindex_id c,
// the subset of elements d of graph.dependencies[c] such that
// cindex_id_to_order[d] == cindex_id_to_order[d].
static void ComputeDependenciesSubset(
    const ComputationGraph &graph,
    const std::vector<int32> &cindex_id_to_order,
    std::vector<std::vector<int32> > *dependencies_subset) {
  int32 num_cindex_ids = graph.cindexes.size();
  KALDI_ASSERT(cindex_id_to_order.size() == num_cindex_ids);
  dependencies_subset->resize(num_cindex_ids);
  for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++) {
    int32 order_index = cindex_id_to_order[cindex_id];
    const std::vector<int32> &dependencies = graph.dependencies[cindex_id];
    std::vector<int32> &dep_subset = (*dependencies_subset)[cindex_id];
    int32 num_dep = dependencies.size();
    for (int32 i = 0; i < num_dep; i++) {
      int32 d = dependencies[i];
      if (cindex_id_to_order[d] == order_index)
        dep_subset.push_back(d);
    }
  }
}


/// computes a sorted (and naturally unique) list of cindex_ids in the graph
/// that are outputs in the "request".
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


// This function assumes indexes and optional have the same size, with each
// element of "optional" saying whether this dependency is optional.  What it
// does is to remove each element of "indexes" that is optional.  It's called if
// the use_optional_dependencies member of the ComputationRequest is false,
// which is only the case when the user is trying to discover the amount of
// left-context and right context the network has.
static void RemoveOptionalInputs(std::vector<Index> *indexes,
                                 std::vector<bool> *optional) {
  KALDI_ASSERT(indexes->size() == optional->size());
  int32 size = indexes->size();
  // "in" is the location we read from, "out" is the location we write to, as we
  // copy only the non-optional elements.
  int32 in = 0, out = 0;
  for (; in != size; in++) {
    if (! (*optional)[in]) {
      if (out != in)
        (*indexes)[out] = (*indexes)[in];
      out++;
    }
  }
  if (out != size) {
    indexes->resize(out);
    optional->clear();
    optional->resize(out, false);
  }
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
    if (static_cast<int32>(graph->dependencies.size()) <= cindex_id) {
      graph->dependencies.resize(cindex_id + 1);
      graph->optional.resize(cindex_id + 1);
    }
    if (graph->is_input[cindex_id])
      continue;
    Cindex cindex = graph->cindexes[cindex_id];

    // find the dependencies of this cindex.
    int32 n = cindex.first;
    const Index &index = cindex.second;
    const NetworkNode &node = nnet.GetNode(n);

    std::vector<Cindex> input_cindexes;
    std::vector<bool> is_optional;  // says whether each required cindex is
                                    // optional or not (only for kComponent).
    
    // the following switch statement sets up "input_cindexes" and
    // "is_optional".
    switch (node.node_type) {
      case NetworkNode::kDescriptor: {
        // desc describes how this node obtains its input from other nodes.
        const Descriptor &desc = node.descriptor;
        desc.GetInputCindexes(index, &input_cindexes);
        break;
      }
      case NetworkNode::kComponent: {
        int32 c = node.u.component_index;
        const Component *component = nnet.GetComponent(c);
        std::vector<Index> input_indexes;
        component->GetInputIndexes(request.misc_info, index,
                                   &input_indexes, &is_optional);
        // each Component node should be preceded by a node that describes its
        // input, of type kDescriptor
        KALDI_ASSERT(nnet.GetNode(n-1).node_type ==
                     NetworkNode::kDescriptor);
        if (!request.use_optional_dependencies)
          RemoveOptionalInputs(&input_indexes, &is_optional);
        
        input_cindexes.resize(input_indexes.size());
        for (size_t i = 0; i < input_indexes.size(); i++) {
          input_cindexes[i].first = n - 1;  // preceding node.
          input_cindexes[i].second = input_indexes[i];
        }
        break;
      }
      case NetworkNode::kInput: default:
        // for kInput, you should have hit the "continue" statement above.
        KALDI_ERR << "Invalid node type";
    }
    std::vector<int32> &this_dep = graph->dependencies[cindex_id];
    std::vector<bool> &this_opt = graph->optional[cindex_id];
    int32 num_dependencies = input_cindexes.size();
    this_dep.resize(num_dependencies);
    for (size_t i = 0; i < num_dependencies; i++) {
      bool is_input = false, is_new;
      int32 dep_cindex_id = graph->GetCindexId(input_cindexes[i],
                                               is_input, &is_new);
      this_dep[i] = dep_cindex_id;
      if (is_new)
        queue.push_back(dep_cindex_id);
    }
    this_opt = is_optional;
    // Make sure we are not listing any dependency twice.
    SortAndUniqInSync(&this_dep, &this_opt);
  }
}

/// This function computes certain information about "super-order" of cindex_ids.
/// The function ComputeComputationGraphOrder() from nnet-graph.h gives us a map
/// from the NetworkNode index to an index we call the "super-order" index:
/// basically, nodes that are computed first have a lower super-order index, and
/// all nodes that are part of strongly connected components have the same
/// super-order index.
/// The overall computation order that we compute, will respect this super-order
/// (except that outputs of nodes of type kComponent that are actually provided
/// as inputs to the network, won't be subject to these limitations but will
/// come first in the order)... we will just ignore the output of this function
/// as it concerns cindex-ids that are provided as input to the network.
///
///  \param nnet [in] The neural net
///  \param graph [in] The computation graph
///  \param cindex_id_to_super_order [out] A vector that maps cindex_id to
///            super_order index, as obtained by adding one to the output of
///            ComputeNnetComputationOrder; however, input cindex_ids (those for
///            which is_input[cindex_id] is true) always map to 0.
///  \param by_super_order [out] The same information as
///            cindex_id_to_super_order, but in a different format: for each
///            super_order, a list of cindex_ids with that super_order index.
///  \param super_order_is_trivial [out] A vector of bool, indexed by
///            super_order index that's true if this super_order index corresponds
///            to just a single NetworkNode. (and also true for super_order index 0,
///            which corresponds only to inputs to the network).
static void ComputeSuperOrderInfo(
    const Nnet &nnet,    
    const ComputationGraph &graph,
    std::vector<int32> *cindex_id_to_super_order,
    std::vector<std::vector<int32 > > *by_super_order,
    std::vector<bool> *super_order_is_trivial) {

  // node_to_super_order maps each nnet node to an index >= 0that tells
  // us what order to compute them in... but we may need to compute
  // a finer ordering at the cindex_id level in cases like RNNs.
  std::vector<int32> node_to_super_order;
  ComputeNnetComputationOrder(nnet, &node_to_super_order);
  // Add one to the super-order info, because we will be reserving
  // zero for inputs to the network, and we don't want to have to
  // prove that super-order-index 0 would correspond only to inputs.
  for (int32 i = 0; i < node_to_super_order.size(); i++)
    node_to_super_order[i]++;
  int32 num_nodes = nnet.NumNodes(),
      num_cindex_ids = graph.cindexes.size(),
      num_super_order_indexes = 1 + *std::max_element(node_to_super_order.begin(),
                                                      node_to_super_order.end());
  KALDI_ASSERT(node_to_super_order.size() == num_nodes);  

  // super_order_to_num_nodes is only used so we know whether each super_order
  // index corresponds to multiple nodes; if it's just one node then we know
  // the computation is very simple and we can do an optimization.
  std::vector<int32> super_order_to_num_nodes(num_super_order_indexes, 0);
  for (int32 n = 0; n < num_nodes; n++)
    super_order_to_num_nodes[node_to_super_order[n]]++;

  super_order_is_trivial->resize(num_super_order_indexes);
  for (int32 o = 0; o < num_super_order_indexes; o++) {
    KALDI_ASSERT(o == 0 || super_order_to_num_nodes[o] > 0);
    (*super_order_is_trivial)[o] = (super_order_to_num_nodes[o] <= 1);
  }
  
  cindex_id_to_super_order->resize(num_cindex_ids);
  by_super_order->resize(num_super_order_indexes);
  for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++) {
    int32 node_index = graph.cindexes[cindex_id].first,
        super_order_index = (graph.is_input[cindex_id] ? 0 :
                             node_to_super_order[node_index]);
    (*cindex_id_to_super_order)[cindex_id] = super_order_index;
    (*by_super_order)[super_order_index].push_back(cindex_id);
  }
}
    

} // end namespace computation_graph

void ComputeComputationOrder(
    const Nnet &nnet,    
    const ComputationGraph &graph,
    std::vector<int32> *order,
    std::vector<std::vector<int32> > *by_order) {
  using namespace computation_graph;
  if (order == NULL) {  // ensure order != NULL by recursing if it's NULL.
    std::vector<int32> order_tmp;
    ComputeComputationOrder(nnet, graph, &order_tmp, by_order);
    return;
  }
  KALDI_ASSERT(graph.optional.empty() &&
               "You must call PruneComputationGraph before "
               "ComputeComputationOrder.");
  
  std::vector<int32> cindex_id_to_super_order;
  std::vector<std::vector<int32 > > by_super_order;
  std::vector<bool> super_order_is_trivial;
  ComputeSuperOrderInfo(nnet, graph, &cindex_id_to_super_order,
                        &by_super_order, &super_order_is_trivial);
  
  // dependencies_subset contains just the subset of dependencies
  // of each cindex_id, that have the same super_order index.
  std::vector<std::vector<int32> > dependencies_subset;
  ComputeDependenciesSubset(graph, cindex_id_to_super_order,
                            &dependencies_subset);
  

  // depend_on_subset is a subset of the normal "depend_on" list (i.e. a list of
  // all cindex_ids that depend on the current cindex_id), limited to just those
  // cindex_ids that have the same super_order index.
  std::vector<std::vector<int32> > depend_on_subset;
  ComputeGraphTranspose(dependencies_subset, &depend_on_subset);

  int32 num_cindex_ids = graph.cindexes.size(),
      num_super_order_indexes = super_order_is_trivial.size();
  order->clear();
  order->resize(num_cindex_ids, -1);
  if (by_order) {
    by_order->clear();
    by_order->reserve(50);  // minimize unnecessary copies.  50 is very
                            // arbitrarily chosen.
  }

  std::vector<int32> this_order, next_order_candidates;
  int32 num_computed = 0, cur_order = 0;
  
  for (int32 super_order = 0; super_order < num_super_order_indexes; super_order++) {
    // Each super-order index will correspond to one or more order indexes.
    // we start out with those that have no dependencies.
    const std::vector<int32> &this_cindexes = by_super_order[super_order];
    if (by_super_order[super_order].empty())
      continue;  

    // this_order is the list of elements of this order.  Start out with all
    // elements of this super-order that have no dependencies within the same
    // super-order.
    {
      std::vector<int32>::const_iterator iter = this_cindexes.begin(),
          end = this_cindexes.end();
      for (; iter != end; ++iter) {
        int32 cindex_id = *iter;
        if (dependencies_subset[cindex_id].empty())
          this_order.push_back(cindex_id);
      }
    }
    // if the next assert fails, the graph at the level of cindex_ids is not acyclic.
    KALDI_ASSERT(!this_order.empty() &&
                 "Trying to process computation with cycles");

    for (; !this_order.empty(); cur_order++) {
      if (by_order)
        by_order->push_back(this_order);
      num_computed += this_order.size();

      // The next if-statement is an optimization: if for this super-order index
      // there is just one node, there will be just one order-index for this
      // super-order index, so we can skip the rest of this loop.  Note: if
      // super_order == 0, even if there is just one node, cindex_ids from
      // multiple nodes may be put here because of the rule that cindex_ids which
      // are inputs always get super_order 0.  But it's still true that they
      // will have no dependencies, so we can still skip the code below.
      if (super_order_is_trivial[super_order])
        break;
      
      // next_order_candidates is a list of cindexes that we should check
      // whether they are computable now, because one of the things they depend
      // on just became computable.  We declared it outside to avoid
      // reallocation.
      next_order_candidates.clear();  
      for (int32 i = 0; i < this_order.size(); i++) {
        int32 c = this_order[i];  // c is a cindex_id with order cur_order.
        (*order)[c] = cur_order;
        std::vector<int32>::const_iterator iter = depend_on_subset[c].begin(),
            end = depend_on_subset[c].end();
        for (; iter != end; ++iter) {
          int32 d = *iter;  // cindex_id that depends on c.
          next_order_candidates.push_back(d);
        }
      }
      SortAndUniq(&next_order_candidates);
      this_order.clear();
      // now check the candidates that might be of the next order, and put any
      // things that we are able to compute in "this_order".
      std::vector<int32>::const_iterator iter = next_order_candidates.begin(),
          end = next_order_candidates.end();
      for (; iter != end; ++iter) {
        int32 c = *iter;
        std::vector<int32>::const_iterator
            dep_iter = dependencies_subset[c].begin(),
            dep_end = dependencies_subset[c].end();
        for (; dep_iter != dep_end; ++dep_iter) {
          int32 d = *dep_iter;  // d is cindex_id that c depends on.
          if ((*order)[d] < 0)  // we can't compute c yet as something we depend
            break;              // on is not yet computed.
        }
        if (dep_iter == dep_end) {
          // we reached the end and did not break -> all dependencies satisfied
          this_order.push_back(c);
        }
      }
    }
  }
  // make sure everything was computable.  If the next assert fails it's likely
  // a bug in this function or in PruneComputataionGraph.
  KALDI_ASSERT(num_computed == num_cindex_ids);
}

// This helper function used in PruneComputationGraph returns true if we can
// compute cindex_id from the cindex_ids that are already computable (i.e. whose
// entries in the "computable" array are true).
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
  // return true.
  if (num_computable > 0 || num_dependencies == 0)
    return true;
  // At this point we have established that there were optional dependencies
  // and none of them could be computed.  The return status now depends on
  // the component type.  We shouldn't have reached here at all if this node is
  // not a Component, as there are no optional dependencies for descriptors.
  int32 node_index = graph.cindexes[cindex_id].first;
  const NetworkNode &node = nnet.GetNode(node_index);
  KALDI_ASSERT(node.node_type == NetworkNode::kComponent);
  const Component *component = nnet.GetComponent(node.u.component_index);
  return (component->Properties()&kAllowNoOptionalDependencies) != 0;
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
  ComputeGraphTranspose(graph->dependencies, &depend_on_this);
  
  std::vector<bool> computable = graph->is_input;

  unordered_set<int32> is_queued;
  std::deque<int32> queue;  // if this is slow we could try making this a deque
                            // and popping from the front.
  for (int32 c = 0; c < num_cindex_ids; c++) {
    // First iterate over only the input cindex_ids (which may be from nodes of
    // type kInput, but also of type kComponent).
    if (graph->is_input[c]) {
      for (size_t j = 0; j < depend_on_this[c].size(); j++) {
        int32 d = depend_on_this[c][j];
        // note: d cannot be an input since it has a dependency.      
        KALDI_ASSERT(!computable[d]);      
        if (is_queued.insert(d).second)  // if not already there..
          queue.push_back(d);
      }
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
/// the request (this order won't matter at all).
/// returns the total number of cindex_ids that correspond to inputs.
int32 AddInputSteps(const Nnet &nnet,
                    const ComputationRequest &request,
                    const ComputationGraph &graph,                   
                    std::vector<std::vector<int32> > *by_step) {
  KALDI_ASSERT(by_step->empty());
  by_step->reserve(50);  // will minimize unnecessary copies of vectors.
  unordered_set<int32> all_nodes;  // to make sure nothing is listed twice.
  int32 num_cindex_ids = 0;
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
    num_cindex_ids += request.inputs[i].indexes.size();
  }
  return num_cindex_ids;
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
/// ComputationRequest.  The way this works is: for each order-index in turn,
/// it takes all Cindexes that don't correspond to either inputs or outputs of
/// the network, and it separates them into groups based on node-index so that
/// things with the same order but different node-index will be in separate
/// steps.
void AddIntermediateSteps(
    const Nnet &nnet,
    const ComputationRequest &request,
    const ComputationGraph &graph,
    const std::vector<std::vector<int32> > &by_order,
    std::vector<std::vector<int32> > *by_step) {
  int32 num_order_indexes = by_order.size();

  std::vector<char> is_output(nnet.NumNodes(), '\0');
  for (int32 node_index = 0; node_index < nnet.NumNodes(); node_index++) {
    if (nnet.IsOutput(node_index))
      is_output[node_index] = static_cast<char>(1);
  }
  
  std::vector<Cindex> cindexes;
  // We don't include order_index = 0, because all inputs to the network
  // (whether the node index is type kInput or kComponent) will be assigned to
  // order_index 0, and no non-inputs should be there (we checked this).
  for (int32 order_index = 1; order_index < num_order_indexes; order_index++) {
    const std::vector<int32> &this_cindex_ids = by_order[order_index];
    
    cindexes.clear();
    cindexes.reserve(this_cindex_ids.size());
    int32 num_cindex_ids = this_cindex_ids.size();
    for (int32 i = 0; i < num_cindex_ids; i++) {
      int32 cindex_id = this_cindex_ids[i],
          node_index = graph.cindexes[cindex_id].first;
      KALDI_ASSERT(!graph.is_input[cindex_id]);
      if (!is_output[node_index])
        cindexes.push_back(graph.cindexes[cindex_id]);
    }
    // now "cindexes" contains all Cindexes that are not from output nodes [and
    // we already eliminated all inputs].  Sorting this array gives us the
    // ordering we want, where Cindexes from different node-ids are separated
    // into contiguous ranges, and within each range, they are sorted by Index.
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

/// This function would not be necessary if we had not added the ModifyIndexes
/// function to class Component.  It is responsible for possibly modifying the
/// order of the inputs and outputs of non-simple Components, and also possibly
/// removing some inputs if the Component has decided it doesn't need them.  It
/// may be a while before this is ever used for something.  An example use is
/// that maybe in convolutional nets or simple models, some components may want
/// a certain ordering of the input that differs from the normal order.
///
/// Note: right now we don't take any steps to further prune the computation
/// graph in case the components' ModifyIndexes functions remove any input
/// indexes.  We can add this later if it becomes necessary.
void ModifyIndexes(const Nnet &nnet,
                   const ComputationRequest &request,
                   const ComputationGraph &graph,                   
                   std::vector<std::vector<int32> > *by_step) {
  
  // cindex_id_to_step will map a cindex_id to the step it appears in.  For
  // efficiency we only populate it with cindex_ids that are from a node of type
  // kDescriptor whose corresponding Component modifies its input.  The
  // other entries are left undefined.  This means that we need to be a bit
  // careful when interpreting its values.
  std::vector<int32> cindex_id_to_step(graph.cindexes.size());
  for (int32 step = 0; step < by_step->size(); step++) {
    std::vector<int32> &cindex_ids = (*by_step)[step];
    KALDI_ASSERT(!cindex_ids.empty());
    int32 first_cindex_id = cindex_ids.front();
    int32 node_index = graph.cindexes[first_cindex_id].first;
    const NetworkNode &node = nnet.GetNode(node_index);
    if (node.node_type != NetworkNode::kDescriptor)
      continue;  // nothing to do if not a component input.
    // the corresponding Component is always numbered 1 more
    const NetworkNode &next_node = nnet.GetNode(node_index + 1);
    KALDI_ASSERT(next_node.node_type == NetworkNode::kComponent);
    int32 c = next_node.u.component_index;
    const Component *component = nnet.GetComponent(c);
    if (!(component->Properties() & kModifiesIndexes))
      continue;  // nothing to do if it doesn't modify indexes.
    int32 size = cindex_ids.size();
    for (int32 i = 0; i < size; i++) {
      int32 cindex_id = cindex_ids[i];
      cindex_id_to_step[cindex_id] = step;
    }
  }
  
  for (int32 step = 0; step < by_step->size(); step++) {
    std::vector<int32> &cindex_ids = (*by_step)[step];
    int32 cindex_id = cindex_ids.front();
    int32 node_index = graph.cindexes[cindex_id].first;
    const NetworkNode &node = nnet.GetNode(node_index);
    if (node.node_type != NetworkNode::kComponent ||
        graph.is_input[cindex_id])
      continue;  // nothing to do if an input, or if not a Component.
    int32 c = node.u.component_index;
    const Component *component = nnet.GetComponent(c);
    if (!(component->Properties() & kModifiesIndexes))
      continue;  // nothing to do if it doesn't modify indexes.
    const std::vector<int32> &this_dep = graph.dependencies[cindex_id];
    KALDI_ASSERT(!this_dep.empty());
    int32 input_cindex_id = this_dep.front();
    int32 input_step = cindex_id_to_step[input_cindex_id];
    KALDI_ASSERT(input_step >= 0 && input_step < step);
    int32 input_node_index = graph.cindexes[input_cindex_id].first;
    KALDI_ASSERT(input_node_index == node_index - 1);
    // the following assert makes sure that the input_step is plausibly correct-
    // we check this because the code that set up the cindex_id_to_step array
    // was a bit complex and left un-needed elements undefined.  note that all
    // cindex_ids in a step will share the node-index.
    KALDI_ASSERT(graph.cindexes[(*by_step)[input_step].front()].first ==
                 input_node_index);
    std::vector<int32> &input_cindex_ids = (*by_step)[input_step];
    KALDI_ASSERT(cindex_ids.size() == input_cindex_ids.size());
    int32 size = cindex_ids.size();
    std::vector<Index> input_indexes(size), indexes(size);
    for (int32 i = 0; i < size; i++) {
      input_indexes[i] = graph.cindexes[input_cindex_ids[i]].second;
      indexes[i] = graph.cindexes[cindex_ids[i]].second;
    }
    component->ModifyIndexes(&input_indexes, &indexes);
    input_cindex_ids.resize(input_indexes.size());
    for (int32 i = 0; i < input_cindex_ids.size(); i++) {
      Cindex cindex(input_node_index, input_indexes[i]);
      input_cindex_ids[i] = graph.GetCindexId(cindex);
    }
    cindex_ids.resize(indexes.size());
    for (int32 i = 0; i < cindex_ids.size(); i++) {
      Cindex cindex(node_index, indexes[i]);
      cindex_ids[i] = graph.GetCindexId(cindex);
    }
    // note: cindex_ids and input_cindex_ids are references.
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
  int32 num_input_cindex_ids = AddInputSteps(nnet, request, graph, by_step);
  // If the following assert fails, it means that one of our assumptions was
  // wrong and would indicate a problem in the code somewhere.  by_order[0] is
  // supposed to contain all the inputs, and only inputs.
  KALDI_ASSERT(num_input_cindex_ids == by_order[0].size());
  AddIntermediateSteps(nnet, request, graph, by_order, by_step);
  ModifyIndexes(nnet, request, graph, by_step);
  AddOutputSteps(nnet, request, graph, by_step);
}


} // namespace nnet3
} // namespace kaldi
