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
    int32 n = nnet.GetNodeIndex(request.outputs[i].name);
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
    int32 n = nnet.GetNodeIndex(request.inputs[i].name);
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
    if (static_cast<int32>(graph->dependencies.size()) <= cindex_id)
      graph->dependencies.resize(cindex_id + 1);

    if (graph->is_input[cindex_id])
      continue;
    Cindex cindex = graph->cindexes[cindex_id];

    // find the dependencies of this cindex.
    int32 n = cindex.first;
    const Index &index = cindex.second;
    const NetworkNode &node = nnet.GetNode(n);

    std::vector<Cindex> input_cindexes;
    
    // the following switch statement sets up "input_cindexes" and
    // "is_optional".
    switch (node.node_type) {
      case NetworkNode::kDescriptor: {
        // desc describes how this node obtains its input from other nodes.
        const Descriptor &desc = node.descriptor;
        desc.GetDependencies(index, &input_cindexes);
        break;
      }
      case NetworkNode::kComponent: {
        int32 c = node.u.component_index;
        const Component *component = nnet.GetComponent(c);
        std::vector<Index> input_indexes;
        component->GetInputIndexes(request.misc_info, index,
                                   &input_indexes);
        // each Component node should be preceded by a node that describes its
        // input, of type kDescriptor
        KALDI_ASSERT(nnet.GetNode(n-1).node_type ==
                     NetworkNode::kDescriptor);
        
        input_cindexes.resize(input_indexes.size());
        for (size_t i = 0; i < input_indexes.size(); i++) {
          input_cindexes[i].first = n - 1;  // preceding node.
          input_cindexes[i].second = input_indexes[i];
        }
        break;
      }
      case NetworkNode::kDimRange: {
        input_cindexes.resize(1);
        input_cindexes[0] = Cindex(node.u.node_index, index);
        break;
      }
      case NetworkNode::kInput: default:
        // for kInput, you should have hit the "continue" statement above.
        KALDI_ERR << "Invalid node type";
    }
    std::vector<int32> &this_dep = graph->dependencies[cindex_id];

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

    // remove duplicates of dependencies.
    SortAndUniq(&this_dep);
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
  
  std::vector<int32> cindex_id_to_super_order;
  std::vector<std::vector<int32 > > by_super_order;
  std::vector<bool> super_order_is_trivial;
  ComputeSuperOrderInfo(nnet, graph, &cindex_id_to_super_order,
                        &by_super_order, &super_order_is_trivial);
  
  // dependencies_subset contains just the subset of dependencies
  // of each cindex_id, that have the same super_order index as
  // cindex_id itself.  This will be used to correctly order
  // cindexes that have a certain super-order index (i.e. they
  // likely come from the same strongly connected component of
  // the graph of nodes).
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
      // on just became computable.  We declared it outside the current loop to
      // avoid reallocation.
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
      // members that we are currently able to compute into "this_order".
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
      if (!next_order_candidates.empty() && this_order.empty())  {
        // this should have been caught earlier so likely a code error rather than
        // a problem with user input.
        KALDI_ERR << "Possibly some cindexes were not reachable (code error?)";
      }
    }
  }
  // make sure everything was computable.  If the next assert fails it's likely
  // a bug in this function or in PruneComputataionGraph.
  KALDI_ASSERT(num_computed == num_cindex_ids);
}

// This helper function used in PruneComputationGraph returns true if we can
// compute cindex_id from the cindex_ids that are already computable (i.e. whose
// entries in the current "computable" array are true).
static bool IsComputable(const Nnet &nnet,
                         const ComputationRequest &request,
                         const ComputationGraph &graph,                         
                         const std::vector<bool> &computable,
                         int32 cindex_id) {
  KALDI_ASSERT(static_cast<size_t>(cindex_id) < graph.dependencies.size());

  const Cindex &cindex = graph.cindexes[cindex_id];
  int32 node_id = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet.GetNode(node_id);
  switch (node.node_type) {
    case NetworkNode::kDescriptor: {
      const Descriptor &desc = node.descriptor;
      CindexSet cindex_set(graph, computable);
      return desc.IsComputable(index, cindex_set, NULL);
    }
    case NetworkNode::kComponent: {
      const Component *c = nnet.GetComponent(node.u.component_index);
      IndexSet index_set(graph, computable, node_id);
      return c->IsComputable(request.misc_info, index, index_set, NULL);
    }
    case NetworkNode::kDimRange: {
      Cindex input_cindex(node.u.node_index, index);
      int32 cindex_id = graph.GetCindexId(input_cindex);
      return (cindex_id != -1 && computable[cindex_id]);
    }      
    case NetworkNode::kInput: default:
      // we shouldn't reach here because Cindexes from input nodes have
      // no dependencies, and dependencies becoming computable are what
      // triggers a call to this function.
      KALDI_ERR << "Not expecting IsComputable to be called for inputs";
      return true;  // suppress compiler warning.
  }
}




void ComputeComputableArray(const Nnet &nnet,
                            const ComputationRequest &request,
                            const ComputationGraph &graph,
                            std::vector<bool> *computable) {
  using namespace computation_graph;  
  int32 num_cindex_ids = graph.cindexes.size();

  // "depend_on_this" is, for each cindex_id, a list of cindex_ids that depend on it
  // (optionally or not).  this is used to help us evaluate only for those
  // cindex_ids that might only now have become computable (i.e. to stop the
  // algorithm taking potentially quadratic time for things like RNNs).
  std::vector<std::vector<int32> > depend_on_this(num_cindex_ids);
  ComputeGraphTranspose(graph.dependencies, &depend_on_this);
  
  *computable = graph.is_input;
  
  unordered_set<int32> is_queued;
  std::deque<int32> queue;

  for (int32 c = 0; c < num_cindex_ids; c++) {
    // First iterate over only the input cindex_ids (which may be from nodes of
    // type kInput, but also of type kComponent).
    if (graph.is_input[c]) {
      for (size_t j = 0; j < depend_on_this[c].size(); j++) {
        int32 d = depend_on_this[c][j];
        if (!(*computable)[d] && is_queued.insert(d).second)
          queue.push_back(d);
      }
    }
  }
  while (!queue.empty()) {
    int32 c = queue.front();
    queue.pop_front();
    is_queued.erase(c);
    KALDI_ASSERT(!(*computable)[c]);
    if (IsComputable(nnet, request, graph, *computable, c)) {
      (*computable)[c] = true;
      for (size_t j = 0; j < depend_on_this[c].size(); j++) {
        int32 d = depend_on_this[c][j];
        if (!(*computable)[d] && is_queued.insert(d).second)
          queue.push_back(d);
      }
    }
  }
}
  
/// This function must be called only for cindexes that are computable
/// (i.e. computable[cindex_id] == true).  It removes from
/// graph->dependencies[cindex_id] any indexes which are not actually being used
/// in the computation of that quantity.  This will only do something
/// interesting in cases where there are optional dependencies.
static void PruneDependenciesForCindex(
    const Nnet &nnet,
    const ComputationRequest &request,
    const std::vector<bool> &computable,    
    int32 cindex_id,
    ComputationGraph *graph) {
  KALDI_ASSERT(static_cast<size_t>(cindex_id) < graph->dependencies.size() &&
               computable[cindex_id]);
  const Cindex &cindex = graph->cindexes[cindex_id];
  int32 node_id = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet.GetNode(node_id);

  std::vector<int32> &dependencies = graph->dependencies[cindex_id];
  std::sort(dependencies.begin(), dependencies.end());
  std::vector<int32> used_cindex_ids;
  
  switch (node.node_type) {
    case NetworkNode::kDescriptor: {
      const Descriptor &desc = node.descriptor;
      CindexSet cindex_set(*graph, computable);
      std::vector<Cindex> used_cindexes;
      bool ans = desc.IsComputable(index, cindex_set, &used_cindexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      std::vector<int32> &dependencies = graph->dependencies[cindex_id];
      std::sort(dependencies.begin(), dependencies.end());
      size_t size = used_cindexes.size();
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        int32 dep_cindex_id = graph->GetCindexId(used_cindexes[i]);
        KALDI_ASSERT(dep_cindex_id != -1);
        used_cindex_ids[i] = dep_cindex_id;
        KALDI_ASSERT(std::binary_search(dependencies.begin(),
                                        dependencies.end(),
                                        dep_cindex_id));
      }
      break;      
    }
    case NetworkNode::kComponent: {
      const Component *c = nnet.GetComponent(node.u.component_index);
      IndexSet index_set(*graph, computable, node_id);
      std::vector<Index> used_indexes;
      bool ans = c->IsComputable(request.misc_info, index, index_set,
                                 &used_indexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      size_t size = used_indexes.size();
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        Cindex dep_cindex(node_id, used_indexes[i]);
        int32 dep_cindex_id = graph->GetCindexId(dep_cindex);
        KALDI_ASSERT(dep_cindex_id != -1);
        used_cindex_ids[i] = dep_cindex_id;
        KALDI_ASSERT(std::binary_search(dependencies.begin(),
                                        dependencies.end(),
                                        dep_cindex_id));
      }
      break;
    }
    case NetworkNode::kDimRange:
      // there should be exactly one dependency and it is required, not
      // optional, so leave it.
      KALDI_ASSERT(dependencies.size() == 1);
      break;
    case NetworkNode::kInput: default:
      // we shouldn't reach here because Cindexes from input nodes have
      // no dependencies, and dependencies becoming computable are what
      // triggers a call to this function.
      KALDI_ERR << "Not expecting IsComputable to be called for inputs";
  }

  std::sort(used_cindex_ids.begin(), used_cindex_ids.end());
  // make sure there are no repeats; this is not currently allowed.  It
  // is a limitation on what expressions we allow the user to create.
  for (size_t i = 0; i + 1 < used_cindex_ids.size(); i++) {
    if (used_cindex_ids[i] == used_cindex_ids[i+1]) {
      KALDI_ERR << "Repeat detected in dependencies: cindex "
                << graph->cindexes[used_cindex_ids[i]] << " appears twice. "
                << "This means you have used a Descriptor that allows the "
                << "same quantity to appear twice in a sum, which is "
                << "disallowed in order to make implementation easier.";
    }
  }
  // the next statement modifies the graph.
  dependencies.swap(used_cindex_ids);
}


// see comment by declaration in header.
void PruneDependencies(const Nnet &nnet,
                       const ComputationRequest &request,
                       const std::vector<bool> &computable,
                       ComputationGraph *graph) {
  int32 num_cindex_ids = graph->cindexes.size();
  KALDI_ASSERT(computable.size() == num_cindex_ids);

  for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++) {
    if (computable[cindex_id]) {
      PruneDependenciesForCindex(nnet, request, computable, cindex_id, graph);
    } else {
      // no point keeping the dependencies of non-computable cindex_ids,
      // it might just slow us down later.
      graph->dependencies[cindex_id].clear();
    }
  }
}

void ComputeRequiredArray(const Nnet &nnet,
                          const ComputationGraph &graph,
                          const std::vector<bool> &computable,
                          std::vector<bool> *required) {
  int32 num_cindex_ids = graph.cindexes.size();
  KALDI_ASSERT(computable.size() == num_cindex_ids);
  required->clear();
  required->resize(num_cindex_ids, false);

  std::vector<int32> queue;
  for (int32 c = 0; c < num_cindex_ids; c++) {
    // First put the output cindex_ids into the queue.
    int32 node_id = graph.cindexes[c].first;
    if (nnet.IsOutput(node_id)) {
      (*required)[c] = true;
      queue.push_back(c);
    }
  }
  while (!queue.empty()) {
    int32 c = queue.back();
    queue.pop_back();
    const std::vector<int32> &dependencies = graph.dependencies[c];
    std::vector<int32>::const_iterator iter = dependencies.begin(),
        end = dependencies.end();
    for (; iter != end; ++iter) {
      int32 d = *iter;
      if (!(*required)[d]){
        (*required)[d] = true;
        queue.push_back(d);
      }
    }
  }
}


bool PruneComputationGraph(
    const Nnet &nnet,
    const std::vector<bool> &computable,
    const std::vector<bool> &required,
    ComputationGraph *graph) {
  int32 num_cindex_ids = graph->cindexes.size();
  std::vector<bool> keep(num_cindex_ids);
  for (int32 c = 0; c < num_cindex_ids; c++) {
    // we can't remove any of the inputs because the ordering is supplied by the
    // user and changing it would make it hard to interpret the input.
    if (required[c] && !computable[c])
      return false;
    keep[c] = (graph->is_input[c] || (computable[c] && required[c]));
  }
  graph->Renumber(keep);
  return true;
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
                    std::vector<std::vector<int32> > *steps) {
  KALDI_ASSERT(steps->empty());
  steps->reserve(50);  // will minimize unnecessary copies of vectors.
  unordered_set<int32> all_nodes;  // to make sure nothing is listed twice.
  int32 num_cindex_ids = 0;
  for (int32 i = 0; i < request.inputs.size(); i++) {
    int32 n = nnet.GetNodeIndex(request.inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.inputs[i].name;
    // ensure no input node is listed twice.
    KALDI_ASSERT(all_nodes.count(n) == 0 && "Invalid computation request: "
                 "double listing of node.");
    all_nodes.insert(n);
    KALDI_ASSERT(!request.inputs[i].indexes.empty() &&
                 "Computation request had no indexes for input ");
    steps->push_back(std::vector<int32>());
    std::vector<int32> &this_step = steps->back();
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
                    std::vector<std::vector<int32> > *steps) {
  std::set<int32> all_nodes;  // to make sure nothing listed twice.
  for (int32 i = 0; i < request.outputs.size(); i++) {
    int32 n = nnet.GetNodeIndex(request.outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.outputs[i].name;
    // ensure no output node is listed twice.
    KALDI_ASSERT(all_nodes.count(n) == 0 && "Invalid computation request: "
                 "double listing of node.");
    all_nodes.insert(n);
    KALDI_ASSERT(!request.outputs[i].indexes.empty() &&
                 "Computation request had no indexes for output ");
    steps->push_back(std::vector<int32>());
    std::vector<int32> &this_step = steps->back();
    this_step.resize(request.outputs[i].indexes.size());
    for (int32 j = 0; j < request.outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.outputs[i].indexes[j]);
      int32 cindex_id = graph.GetCindexId(cindex);
      KALDI_ASSERT(cindex_id != -1);  // would be code error.
      this_step[j] = cindex_id;
    }
  }
}

/// Outputs into component_steps, steps corresponding to all Cindexes that
/// correspond to Component nodes and that are not inputs to the network.  (note
/// that a Cindex for a Component node that's provided as an input to the
/// network is not case we anticipate being common, but it's possible in the
/// framework).  Note, a step is just a list of cindex_ids that can all be computed
/// at the same time.
void AddComponentSteps(
    const Nnet &nnet,
    const ComputationGraph &graph,
    const std::vector<std::vector<int32> > &by_order,
    std::vector<std::vector<int32> > *component_steps) {
  int32 num_order_indexes = by_order.size();

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
      NetworkNode::NodeType t = nnet.GetNode(node_index).node_type;
      if (t == NetworkNode::kComponent) {
        // the following assert is only possible because order_index > 1.
        KALDI_ASSERT(!graph.is_input[cindex_id]);
        cindexes.push_back(graph.cindexes[cindex_id]);
      }
    }
    // now "cindexes" contains all Cindexes that are from Component nodes (and
    // we have made sure that none of these are being provided as inputs).
    // Sorting this array gives us the ordering we want, where Cindexes from
    // different node-ids are separated into contiguous ranges, and within each
    // range, they are sorted by Index.
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
      component_steps->push_back(std::vector<int32>());
      std::vector<int32> &this_step = component_steps->back();
      this_step.resize(size);
      for (int32 i = 0; i < size; i++, iter++)
        this_step[i] = graph.GetCindexId(*iter);
      KALDI_ASSERT(iter == cur_end);
      // at this point iter will point to either the end of the "cindexes"
      // vector, or the beginning of the next set of Cindexes to process.
    }
  }
}


/// You call this function after calling AddInputSteps to add steps for inputs
/// to "all_steps", then calling AddComponentSteps to output steps for
/// components to "component_steps".  This function moves the component steps
/// from "component_steps" to "all_steps", while preceding each component step
/// with a corresponding step for setting up the input to that component (i.e. a
/// step for the preceding Descriptor).  The reason we do it like this is (a) to
/// ensure that the step for the input to the Component, which comes from a
/// Descriptor, comes immediately before it, which is convenient; and (b)
/// because it's possible in certain rather weird setups, some Cindexes
/// corresponding to the Descriptors at the inputs of Components will end up
/// being listed in two separate steps; and if we added the input-descriptor
/// steps using the same mechanism as AddComponentSteps, we wouldn't be able to
/// correctly capture this duplication.
static void AddComponentInputSteps(
    const ComputationGraph &graph,
    std::vector<std::vector<int32> > *component_steps,
    std::vector<std::vector<int32> > *all_steps) {

  int32 space_for_outputs = 10;  // arbitrary.
  all_steps->reserve(all_steps->size() +
                     component_steps->size() * 2 + space_for_outputs);
  

  for (size_t i = 0; i < component_steps->size(); i++) {
    std::vector<int32> &component_step = (*component_steps)[i];
    KALDI_ASSERT(!component_step.empty());
    // First make a step for the descriptor at the input of this Component.
    unordered_set<int32> descriptor_cindex_ids;
    std::vector<int32>::iterator iter = component_step.begin(),
        end = component_step.end();
    for (; iter != end; ++iter) {
      int32 c = *iter;
      const std::vector<int32> &dependencies = graph.dependencies[c];
      std::vector<int32>::const_iterator dep_iter = dependencies.begin(),
          dep_end = dependencies.end();
      for (; dep_iter != dep_end; ++dep_iter) {
        int32 d = *dep_iter;
        descriptor_cindex_ids.insert(d);
      }
    }
    // Convert to Cindexes so we can sort them as Cindexes.
    std::vector<Cindex> descriptor_cindexes;
    descriptor_cindexes.reserve(descriptor_cindex_ids.size());
    unordered_set<int32>::iterator set_iter = descriptor_cindex_ids.begin(),
        set_end = descriptor_cindex_ids.end();
    for (; set_iter != set_end; ++set_iter) {
      int32 c = *set_iter;
      descriptor_cindexes.push_back(graph.cindexes[c]);
    }
    // sort the cindexes.
    std::sort(descriptor_cindexes.begin(), descriptor_cindexes.end());

    // We technically allow a Component with no input, e.g. in case where for
    // some reason it decides it has no dependencies, e.g. it has a constant
    // output.  In this case we create an empty step, to preserve the property
    // that the step for the Component's input comes immediately before the step
    // for the Component itself.
    if (!descriptor_cindexes.empty()) {
      // Make sure all these cindexes come from the same node_id, which should
      // be the one immediately preceding the Component node_id of
      // "component_step".
      int32 node_id = descriptor_cindexes.front().first;
      KALDI_ASSERT(descriptor_cindexes.back().first == node_id &&
                   graph.cindexes[component_step.front()].first == node_id + 1);
    }
    // Now that we've sorted, convert back to cindex_ids (this list will be
    // the "step").
    int32 size = descriptor_cindexes.size();      
    std::vector<int32> descriptor_step(size);
    for (int32 i = 0; i < size; i++) {
      descriptor_step[i] = graph.GetCindexId(descriptor_cindexes[i]);
      KALDI_ASSERT(descriptor_step[i] != -1);
    }
    // efficiently add descriptor_step to the end of all_steps.
    all_steps->push_back(std::vector<int32>());
    all_steps->back().swap(descriptor_step);
    
    // efficiently add component_step to the end of all_steps (this destroys the
    // input, which we won't be needing any more).
    all_steps->push_back(std::vector<int32>());
    all_steps->back().swap(component_step);
  }
  component_steps->clear();
}


static void CreateCindexIdToStep(
    const ComputationGraph &graph,    
    const std::vector<std::vector<int32> > &all_steps,
    std::vector<int32> *cindex_id_to_step) {
  int32 num_cindex_ids = graph.cindexes.size();
  cindex_id_to_step->clear();
  cindex_id_to_step->resize(num_cindex_ids, -1);
  int32 num_steps = all_steps.size();
  for (int32 step = 0; step < num_steps; step++) {
    std::vector<int32>::const_iterator iter = all_steps[step].begin(),
        end = all_steps[step].end();
    for (; iter != end; ++iter) {
      int32 cindex_id = *iter;
      (*cindex_id_to_step)[cindex_id] = step;
    }
  }
}

/// This function inserts into "all_steps", which at this point should contain
/// all but the output steps, steps corresponding to any nodes of type kDimRange.
/// "graph" is non-const as there are situations in which we might need to
/// add cindexes for nodes of type kDimRange.
static void AddDimRangeSteps(
    const Nnet &nnet,    
    ComputationGraph *graph,
    std::vector<std::vector<int32> > *all_steps) {
  int32 num_nodes = nnet.NumNodes();
  bool dim_range_node_exists = false;
  std::vector<char> is_dim_range_node(num_nodes, '\0');
  for (int32 n = 0; n < num_nodes; n++) {
    if (nnet.GetNode(n).node_type == NetworkNode::kDimRange) {
      is_dim_range_node[n] = (char)1;
      dim_range_node_exists = true;
    }
  }
  if (!dim_range_node_exists)
    return;

  std::vector<int32> cindex_id_to_step;
  CreateCindexIdToStep(*graph, *all_steps, &cindex_id_to_step);
  int32 num_steps = all_steps->size();

  // We are going to insert steps for nodes of type kDimRange just after the
  // kInput or kComponent steps that the kDimRange nodes refer to.
  // new_nodes_per_step will be a list of any nodes of type kDimRange that
  // have input corresponding to something in that step.
  std::vector<std::set<int32> > new_nodes_per_step(num_steps);
  int32 num_cindex_ids = graph->cindexes.size();
  std::vector<Cindex>::const_iterator iter = graph->cindexes.begin();
  for (int32 i = 0; i < num_cindex_ids; i++,iter++) {
    const Cindex &cindex = *iter;
    int32 node_index = cindex.first;
    if (!is_dim_range_node[node_index])
      continue;
    const NetworkNode &node = nnet.GetNode(node_index);
    Cindex input_cindex(node.u.node_index, cindex.second);
    int32 input_cindex_id = graph->GetCindexId(input_cindex);
    KALDI_ASSERT(input_cindex_id != -1);
    int32 input_step = cindex_id_to_step[input_cindex_id];
    KALDI_ASSERT(input_step != -1);
    new_nodes_per_step[input_step].insert(node_index);
  }
  int32 num_new_steps = 0, space_for_output = 10;
  for (int32 step = 0; step < num_steps; step++)
    num_new_steps += new_nodes_per_step[step].size();

  // we'll later swap all_steps_out with all_steps.
  std::vector<std::vector<int32> > all_steps_out;
  all_steps_out.reserve(num_steps + num_new_steps + space_for_output);
  for (int32 step = 0; step < num_steps; step++) {
    std::vector<int32> &this_step = (*all_steps)[step];
    int32 cur_out_index = all_steps_out.size();
    all_steps_out.push_back(std::vector<int32>());  // make space for this step.
    std::set<int32>::iterator iter = new_nodes_per_step[step].begin(),
        end = new_nodes_per_step[step].end();
    for (; iter != end; ++iter) {
      int32 node = *iter, size = this_step.size();
      std::vector<int32> new_step(size);
      for (int32 i = 0; i < size; i++) {
        int32 cindex_id = this_step[i];
        Cindex dimrange_cindex(node, graph->cindexes[cindex_id].second);
        bool input = false, is_new;
        int32 dimrange_cindex_id = graph->GetCindexId(dimrange_cindex,
                                                      input, &is_new);
        // actually we don't care about is_new's value.  some new ones are
        // allowed.
        new_step[i] = dimrange_cindex_id;
      }
      all_steps_out.push_back(std::vector<int32>());
      all_steps_out.back().swap(new_step);
    }
    all_steps_out[cur_out_index].swap(this_step);
  }
  all_steps->swap(all_steps_out);
}



/// This function would not be necessary if we had not added the ReorderIndexes
/// function to class Component.  It is responsible for possibly modifying the
/// order of the inputs and outputs of non-simple Components, and also possibly
/// removing some inputs if the Component has decided it doesn't need them.  It
/// may be a while before this is ever used for something.  An example use is
/// that maybe in convolutional nets or simple models, some components may want,
/// efficiency or convenience, a certain ordering of the input that differs from
/// the normal order.
void ReorderIndexes(const Nnet &nnet,
                    const ComputationRequest &request,
                    const ComputationGraph &graph,                   
                    std::vector<std::vector<int32> > *steps) {

  for (int32 step = 0; step < steps->size(); step++) {
    std::vector<int32> &cindex_ids = (*steps)[step];
    if (cindex_ids.empty()) continue;
    int32 cindex_id = cindex_ids.front();
    int32 node_index = graph.cindexes[cindex_id].first;
    const NetworkNode &node = nnet.GetNode(node_index);
    if (node.node_type != NetworkNode::kComponent ||
        graph.is_input[cindex_id])
      continue;  // nothing to do if an input, or if not a Component.
    
    int32 c = node.u.component_index;
    const Component *component = nnet.GetComponent(c);
    if (!(component->Properties() & kReordersIndexes))
      continue;  // nothing to do if it doesn't modify indexes.
    KALDI_ASSERT(step > 0);  // or should have continued already.

    // preceding step will be Cindexes from the input Descriptor.
    std::vector<int32> &input_cindex_ids = (*steps)[step - 1];
        
    int32 size = cindex_ids.size(), input_size = input_cindex_ids.size();
    std::vector<Index> indexes(size), input_indexes(input_size);

    for (int32 i = 0; i < size; i++)
      indexes[i] = graph.cindexes[cindex_ids[i]].second;
    for (int32 i = 0; i < input_size; i++)    
      input_indexes[i] = graph.cindexes[input_cindex_ids[i]].second;

    component->ReorderIndexes(&input_indexes, &indexes);
    // size should not change.
    KALDI_ASSERT(input_indexes.size() == input_size && indexes.size() == size);

    if (size > 0) {
      int32 node_index = graph.cindexes[cindex_ids.front()].first;
      for (int32 i = 0; i < size; i++) {
        Cindex cindex(node_index, indexes[i]);
        cindex_ids[i] = graph.GetCindexId(cindex);
      }
    }
    if (input_size > 0) {
      int32 input_node_index = graph.cindexes[input_cindex_ids.front()].first;
      for (int32 i = 0; i < input_size; i++) {
        Cindex cindex(input_node_index, input_indexes[i]);
        input_cindex_ids[i] = graph.GetCindexId(cindex);
      }
    }
    // note: cindex_ids and input_cindex_ids are references, so we have
    // changed *steps by writing to them in the above two loops.
  }
}

} // namespace compute_computation_steps.

void ComputeComputationSteps(
    const Nnet &nnet,
    const ComputationRequest &request,
    const std::vector<std::vector<int32> > &by_order,
    ComputationGraph *graph,
    std::vector<std::vector<int32> > *steps) {
  using namespace compute_computation_steps;
  steps->clear();
  AddInputSteps(nnet, request, *graph, steps);
  {
    std::vector<std::vector<int32> > component_steps;
    AddComponentSteps(nnet, *graph, by_order, &component_steps);
    AddComponentInputSteps(*graph, &component_steps, steps);
  }
  // output steps don't get reordered so we do the reordering before adding
  // them.
  ReorderIndexes(nnet, request, *graph, steps);
  AddDimRangeSteps(nnet, graph, steps);
  AddOutputSteps(nnet, request, *graph, steps);

  int32 num_cindexes = 0;
  for (int32 i = 0; i < steps->size(); i++)
    num_cindexes += (*steps)[i].size();
  // The next line has ">=" not "==" because it is possible (although unlikely
  // in normal setups) that some cindexes of Descriptors which are at the inputs
  // of Components,
  KALDI_ASSERT(num_cindexes >= graph->cindexes.size());
}


} // namespace nnet3
} // namespace kaldi
