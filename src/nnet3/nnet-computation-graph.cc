// nnet3/nnet-computation-graph.cc

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
      else
        KALDI_ERR << "Dependency on nonexistent cindex-id";
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

void ComputationGraphBuilder::PrintCindexId(std::ostream &os,
                                            int32 cindex_id) const {
  KALDI_ASSERT(static_cast<size_t>(cindex_id) < graph_->cindexes.size());
  const Cindex &cindex = graph_->cindexes[cindex_id];
  const std::string &node_name = nnet_.GetNodeName(cindex.first);
  os << node_name << '(' << cindex.second.n << ", " << cindex.second.t
     << ", " << cindex.second.x << ')';
}

void ComputationGraphBuilder::ExplainWhyNotComputable(
    int32 first_cindex_id) const {
  int32 max_lines_print = 100;
  std::deque<int32> cindexes_to_explain;
  cindexes_to_explain.push_back(first_cindex_id);
  KALDI_ASSERT(graph_->cindexes.size() == graph_->dependencies.size());
  std::ostringstream os;
  os << "*** cindex ";
  PrintCindexId(os, first_cindex_id);
  os << " is not computable for the following reason: ***\n";
  for (int32 num_lines_printed = 0;
       num_lines_printed < max_lines_print && !cindexes_to_explain.empty();
       num_lines_printed++) {
    int32 cindex_id = cindexes_to_explain.front();
    cindexes_to_explain.pop_front();
    KALDI_ASSERT(static_cast<size_t>(cindex_id) < graph_->cindexes.size());
    PrintCindexId(os, cindex_id);
    os << " is " << static_cast<ComputableInfo>(
        computable_info_[cindex_id]) << ", dependencies: ";
    const std::vector<int32> dependencies = graph_->dependencies[cindex_id];
    std::vector<int32>::const_iterator iter = dependencies.begin(),
        end = dependencies.end();
    for (; iter != end; iter++) {
      int32 dep_cindex_id = *iter;
      PrintCindexId(os, dep_cindex_id);
      ComputableInfo status = static_cast<ComputableInfo>(
          computable_info_[cindex_id]);
      if (status != kComputable) {
        os << '[' << status << ']';
        cindexes_to_explain.push_back(dep_cindex_id);
      }
      if (iter+2 != end)
        os << ", ";
    }
    os << "\n";
  }
  os << "\n";
  KALDI_LOG << os.str();
}


void ComputationGraph::Print(std::ostream &os,
                             const std::vector<std::string> &node_names) {
  int32 max_cindexes_per_line = 50, max_dependencies = 5,
      num_cindexes = cindexes.size();

  std::vector<std::pair<Cindex, std::vector<Cindex> > > pairs;
  pairs.reserve(num_cindexes);
  for (int32 cindex_id = 0; cindex_id < num_cindexes; cindex_id++) {
    int32 size = dependencies[cindex_id].size();
    std::vector<Cindex> deps(size);
    for (size_t i = 0; i < size; i++)
      deps[i] = cindexes[dependencies[cindex_id][i]];
    std::sort(deps.begin(), deps.end());
    pairs.push_back(std::pair<Cindex, std::vector<Cindex> >(cindexes[cindex_id],
                                                            deps));
  }
  std::sort(pairs.begin(), pairs.end());
  int32 cur_start = 0;
  for (int32 i = 0; i < num_cindexes; i++) {
    if (pairs[i].first.first != pairs[cur_start].first.first) {
      cur_start = i;
      os << "\n";
    }
    if (i - cur_start < max_cindexes_per_line) {
      os << "[ ";
      PrintCindex(os, pairs[i].first, node_names);
      if (! is_input[GetCindexId(pairs[i].first)]) {
        // only print out dependences for cindexes that
        // were not provided as inputs.
        os << " -> ";
        for (int32 j = 0; j < pairs[i].second.size(); j++) {
          if (j < max_dependencies) {
            PrintCindex(os, pairs[i].second[j], node_names);
            if (j + 1 < pairs[i].second.size())
              os << ", ";
          } else if (j == max_dependencies) {
            os << "...";
          }
        }
      }
      os << " ] ";
    } else if (i - cur_start == max_cindexes_per_line) {
      os << "...";
    }
  }
  os << "\n";

}


// inline
void ComputationGraphBuilder::AddCindexId(int32 cindex_id,
                                          bool is_input,
                                          bool is_output) {
  // If this cindex_id has just now been added to the graph, the following
  // assert should succeed.
  KALDI_PARANOID_ASSERT(cindex_id == computable_queued_.size() &&
                        cindex_id == computable_info_.size() &&
                        cindex_id == depend_on_this_.size() &&
                        cindex_id == usable_count_.size());
  if (is_input) {
    computable_info_.push_back(kComputable);
    computable_queued_.push_back(false);
  } else {
    computable_info_.push_back(kUnknown);
    // add to the queue of things for which we need to compute their computable
    // status.
    computable_queued_.push_back(false);
    next_queue_.push_back(cindex_id);
  }
  depend_on_this_.push_back(std::vector<int32>());
  usable_count_.push_back(is_output ? 1 : 0);
}


void ComputationGraphBuilder::AddInputs() {
  int32 num_added = 0;
  for (int32 i = 0; i < request_.inputs.size(); i++) {
    int32 n = nnet_.GetNodeIndex(request_.inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no input with name "
                << request_.inputs[i].name;
    NodeType t = nnet_.GetNode(n).node_type;
    KALDI_ASSERT((t == kInput || t == kComponent) &&
                 "Inputs to graph only allowed for Input and Component nodes.");

    for (int32 j = 0; j < request_.inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request_.inputs[i].indexes[j]);
      bool is_input = true, is_new;
      int32 cindex_id = graph_->GetCindexId(cindex, is_input, &is_new);
      KALDI_ASSERT(is_new && "Input index seems to be listed more than once");
      AddCindexId(cindex_id, true, false);
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddInputToGraph: nothing to add.");
}

void ComputationGraphBuilder::AddOutputs() {
  int32 num_added = 0;
  for (int32 i = 0; i < request_.outputs.size(); i++) {
    int32 n = nnet_.GetNodeIndex(request_.outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request_.outputs[i].name;
    for (int32 j = 0; j < request_.outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request_.outputs[i].indexes[j]);
      bool is_input = false, is_new;
      int32 cindex_id = graph_->GetCindexId(cindex, is_input, &is_new);
      KALDI_ASSERT(is_new && "Output index seems to be listed more than once");
      AddCindexId(cindex_id, false, true);
      num_added++;
    }
  }
  if (num_added == 0) {
    KALDI_ERR << "Cannot process computation request with no outputs";
  }
  current_distance_ = 0;
  // the calls to AddCindexId in this function will have added to next_queue_.
  KALDI_ASSERT(current_queue_.empty());
  current_queue_.swap(next_queue_);
}

bool ComputationGraphBuilder::AllOutputsAreComputable() const {
  char is_computable_char = static_cast<char>(kComputable);
  std::vector<char>::const_iterator iter = computable_info_.begin(),
      end = computable_info_.end();
  for (int32 cindex_id = 0; iter != end; ++iter, ++cindex_id) {
    if (*iter != is_computable_char) {  // is not computable.
      int32 network_node = graph_->cindexes[cindex_id].first;
      if (nnet_.IsOutputNode(network_node))
        return false;
    }
  }
  return true;
}

std::ostream& operator << (std::ostream &os,
                           const ComputationGraphBuilder::ComputableInfo &info) {
  switch (info) {
    case ComputationGraphBuilder::kUnknown: os << "kUnknown";
      break;
    case ComputationGraphBuilder::kComputable: os << "kComputable";
      break;
    case ComputationGraphBuilder::kNotComputable: os << "kNotComputable";
      break;
    case ComputationGraphBuilder::kWillNotCompute: os << "kWillNotCompute";
      break;
    default: os << "[invalid enum value]"; break;
  }
  return os;
}


// Prints logging info to explain why all outputs are not computable.
void ComputationGraphBuilder::ExplainWhyAllOutputsNotComputable() const {
  std::vector<int32> outputs_not_computable;
  int32 num_outputs_total = 0;

  std::vector<Cindex>::const_iterator iter = graph_->cindexes.begin(),
      end = graph_->cindexes.end();
  for (int32 cindex_id = 0; iter != end; ++iter,++cindex_id) {
    int32 network_node = iter->first;
    ComputableInfo c = static_cast<ComputableInfo>(computable_info_[cindex_id]);
    if (nnet_.IsOutputNode(network_node)) {
      num_outputs_total++;
      if (c != kComputable)
        outputs_not_computable.push_back(cindex_id);
    }
  }
  KALDI_ASSERT(!outputs_not_computable.empty() &&
               "You called this function when everything was computable.");
  int32 num_print = 10, num_not_computable = outputs_not_computable.size();
  KALDI_LOG << num_not_computable << " output cindexes out of "
            << num_outputs_total << " were not computable.";
  std::ostringstream os;
  request_.Print(os);
  KALDI_LOG << "Computation request was: " << os.str();
  if (num_not_computable > num_print)
    KALDI_LOG << "Printing the reasons for " << num_print << " of these.";
  for (int32 i = 0; i < num_not_computable && i < num_print; i++)
    ExplainWhyNotComputable(outputs_not_computable[i]);
}



// this function limits the dependencies of cindex_id "cindex_id" to just those
// which are actually used in computing it.  It also clears the dependencies
// of those cindexes that are not computable.
void ComputationGraphBuilder::PruneDependencies(int32 cindex_id) {
  ComputableInfo c = static_cast<ComputableInfo>(computable_info_[cindex_id]);
  // by the time this is called, there should be no cindexes with unknown state.
  KALDI_ASSERT(c != kUnknown);
  if (c == kNotComputable || c == kWillNotCompute) {
    // if something is not computable, there is no point
    // keeping around its dependencies.
    graph_->dependencies[cindex_id].clear();
    return;
  }
  KALDI_ASSERT(c == kComputable);
  const Cindex &cindex = graph_->cindexes[cindex_id];
  int32 node_id = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet_.GetNode(node_id);

  std::vector<int32> &dependencies = graph_->dependencies[cindex_id];
  std::sort(dependencies.begin(), dependencies.end());
  std::vector<int32> used_cindex_ids;

  switch (node.node_type) {
    case kDescriptor: {
      const Descriptor &desc = node.descriptor;
      bool dont_care = false;  // there should be no kUnknown, and we check this
      CindexSet cindex_set(*graph_, computable_info_, dont_care);
      std::vector<Cindex> used_cindexes;
      bool ans = desc.IsComputable(index, cindex_set, &used_cindexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      size_t size = used_cindexes.size();
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        int32 dep_cindex_id = graph_->GetCindexId(used_cindexes[i]);
        KALDI_ASSERT(dep_cindex_id != -1);
        used_cindex_ids[i] = dep_cindex_id;
        KALDI_ASSERT(std::binary_search(dependencies.begin(),
                                        dependencies.end(),
                                        dep_cindex_id));
      }
      break;
    }
    case kComponent: {
      const Component *c = nnet_.GetComponent(node.u.component_index);
      bool dont_care = false;  // there should be no kUnknown, and we check this
      // In the line below, node_id - 1 is the index of the component-input
      // node-- the descriptor at the input to the component.  We are interested
      // in the set of inputs to the component that are computable.
      IndexSet index_set(*graph_, computable_info_, node_id - 1, dont_care);
      std::vector<Index> used_indexes;
      bool ans = c->IsComputable(request_.misc_info, index, index_set,
                                 &used_indexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      size_t size = used_indexes.size();
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        Cindex dep_cindex(node_id - 1, used_indexes[i]);
        int32 dep_cindex_id = graph_->GetCindexId(dep_cindex);
        KALDI_ASSERT(dep_cindex_id != -1);
        used_cindex_ids[i] = dep_cindex_id;
        KALDI_ASSERT(std::binary_search(dependencies.begin(),
                                        dependencies.end(),
                                        dep_cindex_id));
      }
      break;
    }
    case kDimRange:
      KALDI_ASSERT(dependencies.size() == 1);
      // there should be exactly one dependency and it is required, not
      // optional, so there is nothing to do here.  Return.
      return;
    case kInput:
      KALDI_ASSERT(dependencies.empty());
      // there is nothing to do; return.
      return;
    default:
      KALDI_ERR << "Invalid node type";
  }
  SortAndUniq(&used_cindex_ids);

  // the next statement modifies the graph.
  dependencies.swap(used_cindex_ids);
}

void ComputationGraphBuilder::Compute() {
  KALDI_ASSERT(current_distance_ == -1 && "Compute() called twice?");
  AddInputs();
  AddOutputs();  // sets current_distance_ to 0.
  // max_distance for debugging, to detect infinite recursion.
  int32 max_distance = 10000;
  while (current_distance_ < max_distance) {
    BuildGraphOneIter();
    // only check rarely if we're running at low verbose level.
    if (GetVerboseLevel() >= 3 || RandInt(1,  (current_distance_ + 1)) == 1)
      Check();
    // TODO: come up with a scheme to delay when we call
    // UpdateAllComputableInfo().
    UpdateAllComputableInfo();
    if (current_queue_.empty()) // we're done.
      break;
  }
  if (current_distance_ == max_distance)
    KALDI_ERR << "Loop detected while building computation graph (bad "
              << "network topology?)";
  Check();
}


void ComputationGraphBuilder::Check() const {
  int32 num_cindex_ids = graph_->cindexes.size();
  for (int32 cindex_id = 0; cindex_id < num_cindex_ids;
       cindex_id += 1 + RandInt(0, num_cindex_ids / 100)) {
    { // check depend_on_this.
      std::vector<int32> depend_on_this = depend_on_this_[cindex_id];
      int32 size = depend_on_this.size();
      std::sort(depend_on_this.begin(), depend_on_this.end());
      KALDI_ASSERT(IsSortedAndUniq(depend_on_this));
      for (size_t j = 0; j < size; j++) {
        int32 other_cindex_id = depend_on_this[j];
        // make sure appears in appropriate dependencies array.
        const std::vector<int32> &dep = graph_->dependencies[other_cindex_id];
        KALDI_ASSERT(std::count(dep.begin(), dep.end(), cindex_id) == 1);
      }
    }
    { // check dependencies.
      std::vector<int32> dependencies = graph_->dependencies[cindex_id];
      int32 size = dependencies.size();
      std::sort(dependencies.begin(), dependencies.end());
      KALDI_ASSERT(IsSortedAndUniq(dependencies));
      for (size_t j = 0; j < size; j++) {
        int32 dep_cindex_id = dependencies[j];
        // make sure appears in appropriate depend_on_this_ array.
        const std::vector<int32> &dep = depend_on_this_[dep_cindex_id];
        KALDI_ASSERT(std::count(dep.begin(), dep.end(), cindex_id) == 1);
      }
    }
    { // check usable_count_.
      int32 node_index = graph_->cindexes[cindex_id].first;
      int32 usable_count = usable_count_[cindex_id],
          usable_count_recomputed = nnet_.IsOutputNode(node_index) ? 1 : 0;
      std::vector<int32> depend_on_this = depend_on_this_[cindex_id];
      int32 size = depend_on_this.size();
      for (size_t j = 0; j < size; j++) {
        int32 other_cindex_id = depend_on_this[j];
        if (usable_count_[other_cindex_id] != 0 &&
            computable_info_[other_cindex_id] != kNotComputable)
          usable_count_recomputed++;
      }
      KALDI_ASSERT(usable_count == usable_count_recomputed);
    }
    // check computable_info_.  note: this will not be accurate
    // if the cindex_id is still queued to have dependencies added
    // (in cur_queue_ or next_queue_).
    if (computable_queue_.empty()) {
      ComputationGraphBuilder::ComputableInfo c =
          ComputeComputableInfo(cindex_id);
      // the status doesn't have to be correct if it's kWillNotCompute,
      // because these are cindex-ids that we chose not to compute
      // because we determined they would not be useful, and
      // ComputeComputableInfo() will never return this value.
      if (c != computable_info_[cindex_id] &&
          computable_info_[cindex_id] != kWillNotCompute) {
        int32 count_cur = std::count(current_queue_.begin(),
                                     current_queue_.end(), cindex_id),
            count_next = std::count(next_queue_.begin(),
                                    next_queue_.end(), cindex_id);
        // if it wasn't queued, then something is wrong.
        if (count_cur + count_next == 0)
          KALDI_ERR << "Mismatch in computable status";
      }
    }
    // check computable_queued_.
    // note, the following checks might be a bit slow.
    if (computable_queued_[cindex_id]) {
      KALDI_ASSERT(std::count(computable_queue_.begin(),
                              computable_queue_.end(),
                              cindex_id) == 1);
    } else {
      KALDI_ASSERT(std::count(computable_queue_.begin(),
                              computable_queue_.end(),
                              cindex_id) == 0);
    }
  }
}

void ComputationGraphBuilder::Prune() {
  int32 num_cindex_ids = graph_->cindexes.size();
  // Prune the dependencies to just those that are used (to remove
  // optional dependencies that don't end up getting used).

  for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++)
    PruneDependencies(cindex_id);
  depend_on_this_.clear();  // not valid any more after pruning dependencies.
  std::vector<bool> required;
  ComputeRequiredArray(&required);

  std::vector<bool> keep(num_cindex_ids, false);
  for (int32 c = 0; c < num_cindex_ids; c++) {
    if (required[c] || graph_->is_input[c]) {
      KALDI_ASSERT(computable_info_[c] == kComputable &&
                   "You are calling Prune when not everything is computable.");
      keep[c] = true;
    }
  }
  graph_->Renumber(keep);
  // The following variables will not be valid any more after the renumbering,
  // so clear them.
  computable_info_.clear();
  computable_queue_.clear();
  usable_count_.clear();
}

// Add cindex_ids that this cindex_id depends on.
void ComputationGraphBuilder::AddDependencies(int32 cindex_id) {
  if (static_cast<int32>(graph_->dependencies.size()) <= cindex_id) {
    graph_->dependencies.resize(2 * cindex_id + 1);
  }

  Cindex cindex = graph_->cindexes[cindex_id];

  // find the dependencies of this cindex.
  int32 node_index = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet_.GetNode(node_index);

  std::vector<Cindex> input_cindexes;

  // the following switch statement sets up "input_cindexes".
  switch (node.node_type) {
    case kDescriptor: {
      // desc describes how this node obtains its input from other nodes.
      const Descriptor &desc = node.descriptor;
      desc.GetDependencies(index, &input_cindexes);
      break;
    }
    case kComponent: {
      int32 c = node.u.component_index;
      const Component *component = nnet_.GetComponent(c);
      std::vector<Index> input_indexes;
      component->GetInputIndexes(request_.misc_info, index,
                                 &input_indexes);
      input_cindexes.resize(input_indexes.size());
      for (size_t i = 0; i < input_indexes.size(); i++) {
        input_cindexes[i].first = node_index  - 1;  // preceding node
        input_cindexes[i].second = input_indexes[i];
      }
      break;
    }
    case kDimRange: {
      input_cindexes.resize(1);
      input_cindexes[0] = Cindex(node.u.node_index, index);
      break;
    }
    case kInput:
      break;  // There will be no dependencies.
    default:
      KALDI_ERR << "Invalid node type";
  }

  int32 num_dependencies = input_cindexes.size();
  // this "reserve" statement is to make sure the reference
  // we declare below does not become invalid in the loop below
  // (the call to graph_->GetCindexId() could add up to
  // num_dependencies elements to the graph_->dependencies array
  // and we want to avoid allocation).
  // the RoundUpToNearestPowerOfTwo is for efficiency, to
  // avoid too-frequent resizes.
  graph_->dependencies.reserve(RoundUpToNearestPowerOfTwo(
      graph_->dependencies.size() +  num_dependencies));
  std::vector<int32> &this_dep = graph_->dependencies[cindex_id];

  this_dep.resize(num_dependencies);
  for (size_t i = 0; i < num_dependencies; i++) {
    bool is_input = false, is_new;
    int32 dep_cindex_id = graph_->GetCindexId(input_cindexes[i],
                                              is_input, &is_new);
    this_dep[i] = dep_cindex_id;
    if (is_new)
      AddCindexId(dep_cindex_id, false, false);
    // we will keep dependent's usable_count_ up to date below
  }
  // remove duplicates of dependencies.
  SortAndUniq(&this_dep);
  // set up the "depend_on_this_" array.
  std::vector<int32>::const_iterator iter = this_dep.begin(),
      end = this_dep.end();

  // Populate the "depend_on_this_" array, and append the
  // usable_count_ of things we depend on (see the definition
  // of this quantity next to where it is declared).
  // Note: before calling AddDependencies() we verified the following:
  //  computable_info_[cindex_id] == kUnknown
  // and
  //  usable_count_[cindex_id] != 0
  // which ensures that the conditions to increment the dependency's
  // usable_count_ are satisfied.
  for (; iter != end; ++iter) {
    int32 dep_cindex_id = *iter;
    depend_on_this_[dep_cindex_id].push_back(cindex_id);
    IncrementUsableCount(dep_cindex_id);
  }

  // Now that we've added the dependencies, we can put this into
  // the computable_queue_ to assess whether it's computable
  KALDI_ASSERT(computable_info_[cindex_id] == kUnknown &&
               !computable_queued_[cindex_id]);
  // we think it'll be faster in the next line to do push_front instead of
  // push_back; either one would be correct.
  computable_queue_.push_front(cindex_id);
  computable_queued_[cindex_id] = true;
}


ComputationGraphBuilder::ComputableInfo
ComputationGraphBuilder::ComputeComputableInfo(int32 cindex_id)
    const {
  const Cindex &cindex = graph_->cindexes[cindex_id];
  int32 node_id = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet_.GetNode(node_id);
  switch (node.node_type) {
    case kDescriptor: {
      const Descriptor &desc = node.descriptor;
      {
        CindexSet cindex_set(*graph_, computable_info_, false);
        if (desc.IsComputable(index, cindex_set, NULL)) {
          // it's computable even without counting kUnknown inputs as computable
          // [treat_unknown_as_computable = false] -> definitely computable.
          return kComputable;
        }
      }
      CindexSet cindex_set2(*graph_, computable_info_, true);
      if (!desc.IsComputable(index, cindex_set2, NULL)) {
        // it's not computable even when counting kUnknown inputs as
        // computable [treat_unknown_as_computable = true] -> definitely not
        // computable.
        return kNotComputable;
      }
      return kUnknown;
    }
    case kComponent: {
      const Component *c = nnet_.GetComponent(node.u.component_index);
      const int32 input_node_id = node_id - 1;
      {
        IndexSet index_set(*graph_, computable_info_, input_node_id, false);
        if (c->IsComputable(request_.misc_info, index, index_set, NULL)) {
          // it's computable even without counting kUnknown inputs as computable
          // [treat_unknown_as_computable = false] -> definitely computable.
          return kComputable;
        }
      }
      IndexSet index_set2(*graph_, computable_info_, input_node_id, true);
      if (!c->IsComputable(request_.misc_info, index, index_set2, NULL)) {
        // it's not computable even when counting kUnknown inputs as computable
        // [treat_unknown_as_computable = true] -> definitely not computable.
        return kNotComputable;
      }
      return kUnknown;
    }
    case kDimRange: {
      Cindex input_cindex(node.u.node_index, index);
      int32 input_cindex_id = graph_->GetCindexId(input_cindex);
      if (input_cindex_id != -1)
        return ComputableInfo(computable_info_[input_cindex_id]);
      else
        return kUnknown;
    }
    case kInput: {
      // cindexes for input nodes that are part of the computation request will
      // have graph_->is_input[cindex_id] == true; others will have
      // graph_->is_input[cindex_id] == true.
      return graph_->is_input[cindex_id] ? kComputable : kNotComputable;
    }
    default:
      KALDI_ERR << "Invalid node type.";
      return kUnknown;  // suppress compiler warning.
  }
}

void ComputationGraphBuilder::GetComputableInfo(
    std::vector<std::vector<bool> > *computable) const {
  KALDI_ASSERT(!graph_->cindexes.empty() &&
               "You need to call this after Compute()!");
  KALDI_ASSERT(!computable_info_.empty() &&
               "You need to call this before Prune()!");
  computable->clear();
  computable->resize(request_.outputs.size());
  for (size_t i = 0; i < request_.outputs.size(); i++) {
    const IoSpecification &output = request_.outputs[i];
    int32 n = nnet_.GetNodeIndex(output.name);
    KALDI_ASSERT(n != -1);
    int32 size = output.indexes.size();
    std::vector<bool> &this_vec = (*computable)[i];
    this_vec.resize(size);
    for (size_t j = 0; j < size; j++) {
      Cindex cindex(n, output.indexes[j]);
      int32 cindex_id = graph_->GetCindexId(cindex);
      KALDI_ASSERT(cindex_id != -1);
      this_vec[j] = (computable_info_[cindex_id] == kComputable);
    }
  }
}


void ComputationGraphBuilder::UpdateComputableInfo(int32 cindex_id) {
  // if the current computable_info_ for cindex_id value is not kUnknown, this
  // cindex_id should not have been in the queue.
  KALDI_ASSERT(static_cast<size_t>(cindex_id) < computable_info_.size());
  char &output = computable_info_[cindex_id];
  KALDI_ASSERT(output == kUnknown);

  output = static_cast<char>(ComputeComputableInfo(cindex_id));

  if (output != kUnknown) {
    // The computable status of cindexes that depend on this cindex and whose
    // status is currently kUnknown might now change, so if they are not in the
    // computable queue, put them there.
    std::vector<int32>::const_iterator iter = depend_on_this_[cindex_id].begin(),
        end = depend_on_this_[cindex_id].end();
    for (; iter != end; ++iter) {
      int32 other_cindex_id = *iter;
      if (computable_info_[other_cindex_id] == kUnknown &&
          !computable_queued_[other_cindex_id]) {
        computable_queue_.push_back(other_cindex_id);
        computable_queued_[other_cindex_id] = true;
      }
    }
    if (output == kNotComputable && usable_count_[cindex_id] != 0) {
      // If we have just changed the computable state from kUnknown to
      // kNotComputable, then given the way the usable_count_ is defined (see
      // the declaration), this means that we must decrement the
      // usable_count_ of all cindex_ids that we depend on.
      std::vector<int32>::const_iterator
          iter = graph_->dependencies[cindex_id].begin(),
          end = graph_->dependencies[cindex_id].end();
      for (; iter != end; ++iter) {
        int32 dep_cindex_id = *iter;
        DecrementUsableCount(dep_cindex_id);
      }
    }
  }
}

void ComputationGraphBuilder::SetAsWillNotCompute(int32 cindex_id) {
  KALDI_ASSERT(usable_count_[cindex_id] == 0);
  computable_info_[cindex_id] = kWillNotCompute;
  std::vector<int32>::const_iterator iter = depend_on_this_[cindex_id].begin(),
      end = depend_on_this_[cindex_id].end();
  for (; iter != end; ++iter) {
    int32 other_cindex_id = *iter;
    if (computable_info_[other_cindex_id] == kUnknown &&
        !computable_queued_[other_cindex_id]) {
      computable_queue_.push_back(other_cindex_id);
      computable_queued_[other_cindex_id] = true;
    }
  }
}


void ComputationGraphBuilder::UpdateAllComputableInfo() {
  while (!computable_queue_.empty()) {
    int32 cindex_id = computable_queue_.front();
    computable_queue_.pop_front();
    computable_queued_[cindex_id] = false;
    UpdateComputableInfo(cindex_id);
  }
}


void ComputationGraphBuilder::IncrementUsableCount(int32 cindex_id) {
  KALDI_PARANOID_ASSERT(static_cast<size_t>(cindex_id)<usable_count_.size());
  // the next line post-increments the reachable count.
  if (usable_count_[cindex_id]++ == 0 &&
      computable_info_[cindex_id] != kNotComputable) {
    std::vector<int32>::const_iterator
        iter = graph_->dependencies[cindex_id].begin(),
        end = graph_->dependencies[cindex_id].end();
    for (; iter != end; ++iter) {
      int32 dep_cindex_id = *iter;
      IncrementUsableCount(dep_cindex_id);
    }
  }
}


void ComputationGraphBuilder::DecrementUsableCount(int32 cindex_id) {
  KALDI_PARANOID_ASSERT(static_cast<size_t>(cindex_id)<usable_count_.size());
  KALDI_PARANOID_ASSERT(usable_count_[cindex_id] > 0);
  if (--usable_count_[cindex_id] == 0 &&
      computable_info_[cindex_id] != kNotComputable) {
    std::vector<int32>::const_iterator
        iter = graph_->dependencies[cindex_id].begin(),
        end = graph_->dependencies[cindex_id].end();
    for (; iter != end; ++iter) {
      int32 dep_cindex_id = *iter;
      DecrementUsableCount(dep_cindex_id);
    }
  }
}


void ComputationGraphBuilder::BuildGraphOneIter() {
  while (!current_queue_.empty()) {
    int32 cindex_id = current_queue_.back();
    current_queue_.pop_back();
    KALDI_ASSERT(computable_info_[cindex_id] == kUnknown);
    if (usable_count_[cindex_id] == 0)
      SetAsWillNotCompute(cindex_id);
    else
      AddDependencies(cindex_id);
  }
  current_queue_.swap(next_queue_);  // now next_queue_ will be empty.
  current_distance_++;
}

void ComputationGraphBuilder::ComputeRequiredArray(
    std::vector<bool> *required) const {

  int32 num_cindex_ids = graph_->cindexes.size();
  KALDI_ASSERT(computable_info_.size() == num_cindex_ids);
  required->clear();
  required->resize(num_cindex_ids, false);

  std::vector<int32> queue;
  for (int32 c = 0; c < num_cindex_ids; c++) {
    // First put the output cindex_ids into the queue.
    int32 node_id = graph_->cindexes[c].first;
    if (nnet_.IsOutputNode(node_id)) {
      (*required)[c] = true;
      queue.push_back(c);
    }
  }
  while (!queue.empty()) {
    int32 c = queue.back();
    queue.pop_back();
    const std::vector<int32> &dependencies = graph_->dependencies[c];
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
  // just check that we don't have any cindex_ids which are required but have
  // usable_count_ == 0; this would indicate a bug somewhere.
  for (int32 c = 0; c < num_cindex_ids; c++)
    KALDI_ASSERT(!((*required)[c] && (usable_count_[c] == 0)));

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
    NodeType t = nnet.GetNode(n).node_type;
    KALDI_ASSERT((t == kInput || t == kComponent) &&
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


/**
   This function outputs to dependencies_subset[c], for each cindex_id c,
   the subset of elements d of graph.dependencies[c] such that
   cindex_id_to_epoch[d] == cindex_id_to_epoch[c].  That is, it's
   the dependency graph of the entire computation, but removing
   links that go from one epoch to another epoch.  Topologically,
   'dependencies_subset' would therefor consist of a bunch of
   disconnected graphs.
*/
static void ComputeDependenciesSubset(
    const ComputationGraph &graph,
    const std::vector<int32> &cindex_id_to_epoch,
    std::vector<std::vector<int32> > *dependencies_subset) {
  int32 num_cindex_ids = graph.cindexes.size();
  KALDI_ASSERT(cindex_id_to_epoch.size() == num_cindex_ids);
  dependencies_subset->resize(num_cindex_ids);
  for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++) {
    int32 phase_index = cindex_id_to_epoch[cindex_id];
    const std::vector<int32> &dependencies = graph.dependencies[cindex_id];
    std::vector<int32> &dep_subset = (*dependencies_subset)[cindex_id];
    int32 num_dep = dependencies.size();
    for (int32 i = 0; i < num_dep; i++) {
      int32 d = dependencies[i];
      if (cindex_id_to_epoch[d] == phase_index)
        dep_subset.push_back(d);
    }
  }
}

/// This function computes certain information about "epochs" of cindex_ids.
/// The function ComputeNnetComputationEpochs() from nnet-graph.h gives us a map
/// from the NetworkNode index to an index we call the "epoch" index:
/// basically, nodes that are computed first have a lower epoch index, and
/// all nodes that are part of strongly connected components have the same
/// epoch index.  In an acyclic nnet graph each component will usually have
/// its own epoch index, but in things like LSTMs, each LSTM layer (with multiple
/// components) will have its own epoch index.
///
/// The overall computation order that we compute, will respect this ordering
/// into epochs (except that outputs of nodes of type kComponent that are
/// actually provided as inputs to the network, won't be subject to these
/// limitations but will come first in the order)... we will just ignore the
/// output of this function as it concerns cindex-ids that are provided as input
/// to the network.
///
///  \param nnet [in] The neural net
///  \param graph [in] The computation graph
///  \param cindex_id_to_epoch [out] A vector that maps cindex_id to
///            epoch index, as obtained by adding one to the output of
///            ComputeNnetComputationOrder; however, input cindex_ids (those for
///            which is_input[cindex_id] is true) always map to 0.
///            Note: the epoch-index only depends on the neural network's
///            topology of nodes; a node in the network should always map to
///            the same epoch-index regardless of the computation, and
///            we assign cindexes to epochs just based on what node the
///            cindexes are part of.
///  \param epochs [out] The same information as
///            cindex_id_to_epoch, but in a different format: for each
///            epoch, a list of cindex_ids with that epoch index.
///  \param epoch_is_trivial [out] A vector of bool, indexed by
///            epoch index that's true if this epoch index corresponds
///            to just a single NetworkNode. (and also true for epoch index 0,
///            which corresponds only to inputs to the network).
static void ComputeEpochInfo(
    const Nnet &nnet,
    const ComputationGraph &graph,
    std::vector<int32> *cindex_id_to_epoch,
    std::vector<std::vector<int32 > > *epochs,
    std::vector<bool> *epoch_is_trivial) {

  // node_to_epoch maps each nnet node to an index >= 0 that tells us coarsely
  // what order to compute them in... but we may need to compute a finer
  // ordering at the cindex_id level in cases like RNNs.
  std::vector<int32> node_to_epoch;
  ComputeNnetComputationEpochs(nnet, &node_to_epoch);
  {
    std::ostringstream os;
    PrintIntegerVector(os, node_to_epoch);
    KALDI_VLOG(6) << "node_to_epoch: " << os.str();
  }

  // Add one to the epoch numbering because we will be reserving
  // zero for inputs to the network, and we don't want to have to
  // prove that epoch number 0 would correspond only to inputs.
  for (int32 i = 0; i < node_to_epoch.size(); i++)
    node_to_epoch[i]++;
  int32 num_nodes = nnet.NumNodes(),
      num_cindex_ids = graph.cindexes.size(),
      num_epoch_indexes = 1 + *std::max_element(node_to_epoch.begin(),
                                                node_to_epoch.end());
  KALDI_ASSERT(node_to_epoch.size() == num_nodes);

  // epoch_to_num_nodes is only used so we know whether each epoch
  // index corresponds to multiple nodes; if it's just one node then we know
  // the computation is very simple and we can do an optimization.
  std::vector<int32> epoch_to_num_nodes(num_epoch_indexes, 0);
  for (int32 n = 0; n < num_nodes; n++)
    epoch_to_num_nodes[node_to_epoch[n]]++;

  epoch_is_trivial->resize(num_epoch_indexes);
  for (int32 o = 0; o < num_epoch_indexes; o++) {
    KALDI_ASSERT(o == 0 || epoch_to_num_nodes[o] > 0);
    (*epoch_is_trivial)[o] = (epoch_to_num_nodes[o] <= 1);
  }

  cindex_id_to_epoch->resize(num_cindex_ids);
  epochs->resize(num_epoch_indexes);
  for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++) {
    int32 node_index = graph.cindexes[cindex_id].first,
        epoch_index = (graph.is_input[cindex_id] ? 0 :
                             node_to_epoch[node_index]);
    (*cindex_id_to_epoch)[cindex_id] = epoch_index;
    (*epochs)[epoch_index].push_back(cindex_id);
  }
}


} // end namespace computation_graph


void ComputeComputationGraph(const Nnet &nnet,
                             const ComputationRequest &request,
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

    // the following switch statement sets up "input_cindexes".
    switch (node.node_type) {
      case kDescriptor: {
        // desc describes how this node obtains its input from other nodes.
        const Descriptor &desc = node.descriptor;
        desc.GetDependencies(index, &input_cindexes);
        break;
      }
      case kComponent: {
        int32 c = node.u.component_index;
        const Component *component = nnet.GetComponent(c);
        std::vector<Index> input_indexes;
        component->GetInputIndexes(request.misc_info, index,
                                   &input_indexes);
        // each Component node should be preceded by a node that describes its
        // input, of type kDescriptor
        KALDI_ASSERT(nnet.GetNode(n-1).node_type ==
                     kDescriptor);

        input_cindexes.resize(input_indexes.size());
        for (size_t i = 0; i < input_indexes.size(); i++) {
          input_cindexes[i].first = n - 1;  // preceding node.
          input_cindexes[i].second = input_indexes[i];
        }
        break;
      }
      case kDimRange: {
        input_cindexes.resize(1);
        input_cindexes[0] = Cindex(node.u.node_index, index);
        break;
      }
      case kInput: default:
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


static int32 SumVectorSizes(const std::vector<std::vector<int32> > &vec) {
  int32 ans = 0;
  std::vector<std::vector<int32> >::const_iterator iter = vec.begin(),
      end = vec.end();
  for (; iter != end; ++iter)
    ans += iter->size();
  return ans;
}

/*
  this function is called from ComputeComputationPhases; it handles the part of
  the computation from one epoch (this code was broken out to avoid that
  function being super-long).  Note: the phases are a numbered grouping of
  cindexes that say in what order we compute things, i.e. we first compute
  all the cindexes for phase 0, then for phase 1, and so on.

   @param [in] nnet       The neural net this computation is for
   @param [in] graph      The computation graph we're computing the phases for.

   @param [in] this_epoch The sorted list of the cindex_ids for this epoch; note,
                          cindex_ids are indexes into the array graph.cindexes.
                          Roughly speaking, this is a list of the cindex_ids that
                          correspond to one "layer" of the neural network, in
                          things like LSTMs, or for one part of one layer (the
                          affine component, the nonlinearity, or the splicing),
                          in things like TDNNs.
  @param [in] dependencies_subset  A subset of 'graph.dependencies' corresponding
                          just to dependencies within the same epoch (not specifically
                          this epoch; for all epochs).  E.g. for a cindex_id c
                          dependencies[c] is a list of other cindex_ids d1, d2,
                          such that in order to compute c we must first compute
                          d1, d2 and so on.
  @param [in] depends_on_subset  The graph-transpose of dependencies_subset;
                          for cindex_id c, depends_on_subset[c] is the list
                          of cindex_ids that directly depend on cindex_id c,
                          so c must be computed before them.
  @param [in] epoch_is_trivial  A bool that's true if this epoch is trivial
                          (meaning it consists of just one component)... this
                          enables a faster code path in this common case.
  @param [in,out] phase_indexes  This vector, to some elements of which this function writes
                          each time it is called, maps from cindex_id to the
                          'phase index'.  A phase index is a number identifying
                          the phases [like coarse steps] of the computation, with
                          zero for the first phase, one for the second, etc.
                          We work out how many phase indexes have been used already
                          by previous epochs, from phases->size().  Actually,
                          phase_indexes is really just a temporary variable used
                          by this function, that we allocate outside this
                          function for efficiency.  It is initialized to
                          -1 outside this function; different invocations of
                          this function work with different elements of the
                          vector.
  @param [in,out] phases  This is the output of this function.  Each time
                          we add a new phase, we append a vector to *phases.
                          E.g. (*phases)[0] is the sorted list of cindexes
                          in the first phase of the computation... and so on.
                          Note, this function is called multiple times, and
                          each time we add one or more phases to this vector,
                          so its size grows.
*/
static inline void ComputeComputationPhasesForEpoch(
    const Nnet &nnet,
    const ComputationGraph &graph,
    const std::vector<int32> &this_epoch,
    const std::vector<std::vector<int32> > &dependencies_subset,
    const std::vector<std::vector<int32> > &depend_on_subset,
    bool epoch_is_trivial,
    std::vector<int32> *phase_indexes,
    std::vector<std::vector<int32> > *phases) {
  std::vector<int32> this_phase, next_phase_candidates;

  if (this_epoch.empty())
    return;

  if (epoch_is_trivial) { // an optimization
    this_phase = this_epoch;
  } else {
    // Start out with all elements of this epoch that have no
    // dependencies within the same epoch (i.e. those that
    // can be computed first).
    std::vector<int32>::const_iterator iter = this_epoch.begin(),
        end = this_epoch.end();
    for (; iter != end; ++iter) {
      int32 cindex_id = *iter;
      if (dependencies_subset[cindex_id].empty())
        this_phase.push_back(cindex_id);
    }
  }

  // if the next assert fails, the graph at the level of cindex_ids is not acyclic.
  KALDI_ASSERT(!this_phase.empty() &&
               "Trying to process computation with cycles");

  while (!this_phase.empty()) {
    // The next two lines are a more efficient version of:
    // phases->push_back(this_phase);
    phases->resize(phases->size() + 1);
    phases->back().swap(this_phase);
    // The next if-statement is an optimization: if for this epoch index
    // there is just one node, we can skip the rest of this loop.  Note: if
    // epoch == 0, even if there is just one node, cindex_ids from
    // multiple nodes may be put here because of the rule that cindex_ids which
    // are inputs always get epoch 0.  But it's still true that they
    // will have no dependencies, so we can still skip the code below.
    if (epoch_is_trivial)
      return;

    int32 cur_phase_index = phases->size() - 1;

    // next_phases_candidates is a list of cindexes that we should check
    // whether they are computable now, because one of the things they depend
    // on just became computable.
    next_phase_candidates.clear();
    std::vector<int32>::const_iterator this_phase_iter = phases->back().begin(),
        this_phase_end = phases->back().end();

    for (; this_phase_iter != this_phase_end; ++this_phase_iter) {
      int32 c = *this_phase_iter;  // c is a cindex_id with phase cur_phase_index.
      (*phase_indexes)[c] = cur_phase_index;
      std::vector<int32>::const_iterator iter = depend_on_subset[c].begin(),
          end = depend_on_subset[c].end();
      for (; iter != end; ++iter) {
        int32 d = *iter;  // cindex_id that depends on c.
        next_phase_candidates.push_back(d);
      }
    }
    SortAndUniq(&next_phase_candidates);
    // note, at this point 'this_phase' will be the empty vector [see the 'swap'
    // above].
    this_phase.reserve(next_phase_candidates.size());
    // now check the candidates that might be in the next phase, and put any
    // members that we are currently able to compute into "this_phase".
    std::vector<int32>::const_iterator iter = next_phase_candidates.begin(),
        end = next_phase_candidates.end();
    for (; iter != end; ++iter) {
      int32 c = *iter;
      std::vector<int32>::const_iterator
          dep_iter = dependencies_subset[c].begin(),
          dep_end = dependencies_subset[c].end();
      for (; dep_iter != dep_end; ++dep_iter) {
        int32 d = *dep_iter;  // d is cindex_id that c depends on.
        if ((*phase_indexes)[d] < 0)  // we can't compute c yet because something we depend
          break;                      // on has not yet been computed.
      }
      if (dep_iter == dep_end) {
        // we reached the end and did not break -> all dependencies satisfied
        this_phase.push_back(c);
      }
    }
    if (!next_phase_candidates.empty() && this_phase.empty())  {
      // this should have been caught earlier so likely a code error rather than
      // a problem with user input.
      KALDI_ERR << "Your model has a type of recurrence that cannot be computed. "
                << "E.g. if x[t] depends on both x[t+1] and x[t-1]... no order "
                << "of computation will work.";
    }
  }
}

void ComputeComputationPhases(
    const Nnet &nnet,
    const ComputationGraph &graph,
    std::vector<std::vector<int32> > *phases) {
  using namespace computation_graph;
  int32 num_cindex_ids = graph.cindexes.size();

  std::vector<int32> cindex_id_to_epoch;
  std::vector<std::vector<int32 > > epochs;
  std::vector<bool> epoch_is_trivial;
  ComputeEpochInfo(nnet, graph, &cindex_id_to_epoch,
                   &epochs, &epoch_is_trivial);

  KALDI_ASSERT(SumVectorSizes(epochs) == num_cindex_ids);

  // dependencies_subset contains just the subset of dependencies
  // of each cindex_id, that have the same epoch index as
  // cindex_id itself.  This will be used to correctly order
  // cindexes within a certain epoch (relevant for things like
  // LSTMs).
  std::vector<std::vector<int32> > dependencies_subset;
  ComputeDependenciesSubset(graph, cindex_id_to_epoch,
                            &dependencies_subset);

  // depend_on_subset is a subset of the normal "depend_on" list (i.e. a list of
  // all cindex_ids that depend on the current cindex_id), limited to just those
  // cindex_ids that have the same epoch index.
  std::vector<std::vector<int32> > depend_on_subset;
  ComputeGraphTranspose(dependencies_subset, &depend_on_subset);

  int32 num_epoch_indexes = epoch_is_trivial.size();

  // "phase_indexes" is used inside ComputeComputationPhasesForEpoch.
  std::vector<int32> phase_indexes(num_cindex_ids, -1);

  if (phases) {
    phases->clear();
    phases->reserve(50);  // minimize unnecessary copies.  50 is very
                            // arbitrarily chosen.
  }

  for (int32 epoch = 0;
       epoch < num_epoch_indexes;
       epoch++)
    ComputeComputationPhasesForEpoch(nnet, graph,
                                     epochs[epoch],
                                     dependencies_subset,
                                     depend_on_subset,
                                     epoch_is_trivial[epoch],
                                     &phase_indexes, phases);


  // make sure everything was computable.  If the next assert fails it's likely
  // a bug in this function or in PruneComputataionGraph.
  KALDI_ASSERT(SumVectorSizes(*phases) == num_cindex_ids);
}

CindexSet::CindexSet(const ComputationGraph &graph):
    graph_(graph), is_computable_(NULL) { }

CindexSet::CindexSet(const ComputationGraph &graph,
                     const std::vector<char> &is_computable,
                     bool treat_unknown_as_computable):
    graph_(graph), is_computable_(&is_computable),
    treat_unknown_as_computable_(treat_unknown_as_computable) { }


bool CindexSet::operator () (const Cindex &cindex) const {
  int32 cindex_id = graph_.GetCindexId(cindex);
  if (cindex_id == -1) {
    return false;
  } else {
    if (is_computable_ == NULL) {
      return true;
    } else {
      ComputationGraphBuilder::ComputableInfo
          c = static_cast<ComputationGraphBuilder::ComputableInfo>(
              ((*is_computable_)[cindex_id]));
      if (treat_unknown_as_computable_)
        return (c == ComputationGraphBuilder::kComputable ||
                c == ComputationGraphBuilder::kUnknown);
      else
        return (c == ComputationGraphBuilder::kComputable);
    }
  }
}

IndexSet::IndexSet(const ComputationGraph &graph,
                   const std::vector<char> &is_computable,
                   int32 node_id,
                   bool treat_unknown_as_computable):
    graph_(graph), is_computable_(is_computable), node_id_(node_id),
    treat_unknown_as_computable_(treat_unknown_as_computable) { }

bool IndexSet::operator () (const Index &index) const {
  int32 cindex_id = graph_.GetCindexId(Cindex(node_id_, index));
  if (cindex_id == -1) {
    return false;
  } else {
    ComputationGraphBuilder::ComputableInfo
        c = static_cast<ComputationGraphBuilder::ComputableInfo>(
            is_computable_[cindex_id]);
    if (treat_unknown_as_computable_)
      return (c == ComputationGraphBuilder::kComputable ||
              c == ComputationGraphBuilder::kUnknown);
    else
      return (c == ComputationGraphBuilder::kComputable);
  }
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

/// Convert the cindex_ids in the vector "cindex_ids" to cindexes, but only
/// keeping those that correspond to nodes of type kComponent.
/// Asserts that none of these cindexes have the "is_input" set to true.
/// [this is possible because we call this only for phases >1, and inputs
/// should not be there.]
static void ExtractOnlyComponentCindexes(const std::vector<int32> &cindex_ids,
                                         const ComputationGraph &graph,
                                         const Nnet &nnet,
                                         std::vector<Cindex> *cindexes) {
  cindexes->clear();
  cindexes->reserve(cindex_ids.size());
  std::vector<int32>::const_iterator iter = cindex_ids.begin(),
                                      end = cindex_ids.end();
  for (; iter != end; ++iter) {
    int32 cindex_id = *iter;
    const Cindex &cindex = graph.cindexes[cindex_id];
    if (nnet.IsComponentNode(cindex.first)) {
      KALDI_ASSERT(!graph.is_input[cindex_id]);
      cindexes->push_back(cindex);
    }
  }
}

/// Outputs into component_steps, steps corresponding to all Cindexes that
/// correspond to Component nodes and that are not inputs to the network.  (note
/// that a Cindex for a Component node that's provided as an input to the
/// network is not case we anticipate being common, but it's possible in the
/// framework).  Note, a step is just a list of cindex_ids that can all be computed
/// at the same time.
static void AddComponentSteps(
    const Nnet &nnet,
    const ComputationGraph &graph,
    const std::vector<std::vector<int32> > &phases,
    std::vector<std::vector<int32> > *component_steps) {
  int32 num_phase_indexes = phases.size();

  std::vector<Cindex> cindexes;

  // We don't include phase_index = 0, because all inputs to the network
  // (whether the node index is type kInput or kComponent) will be assigned to
  // phase_index 0, and no non-inputs should be there (we checked this).
  for (int32 phase_index = 1; phase_index < num_phase_indexes; phase_index++) {
    ExtractOnlyComponentCindexes(phases[phase_index], graph, nnet, &cindexes);

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
    if (nnet.IsDimRangeNode(n)) {
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
        new_step[i] = dimrange_cindex_id;
        if (is_new) {  // if we newly added this cindex_id, note the dependency
                       // on its input.
          graph->dependencies[dimrange_cindex_id].push_back(cindex_id);
        }
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
    if (node.node_type != kComponent ||
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
    const std::vector<std::vector<int32> > &phases,
    ComputationGraph *graph,
    std::vector<std::vector<int32> > *steps) {
  using namespace compute_computation_steps;
  steps->clear();
  AddInputSteps(nnet, request, *graph, steps);
  {
    std::vector<std::vector<int32> > component_steps;
    AddComponentSteps(nnet, *graph, phases, &component_steps);
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
