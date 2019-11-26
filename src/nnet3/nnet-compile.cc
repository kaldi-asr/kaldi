// nnet3/nnet-compile.cc

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
#include "nnet3/nnet-compile.h"
#include "nnet3/nnet-compile-utils.h"
#include "nnet3/nnet-optimize.h"  // just for ConsolidateIoOperations().

namespace kaldi {
namespace nnet3 {

Compiler::Compiler(
    const ComputationRequest &request,
    const Nnet &nnet): nnet_(nnet) {
  requests_.push_back(&request);
}

Compiler::Compiler(
    const std::vector<const ComputationRequest*> &requests,
    const Nnet &nnet): requests_(requests), nnet_(nnet) {
  KALDI_ASSERT(requests_.size() >= 1);
  // We are currently not supporting getting model derivatives for multi-segment
  // (online) computations.
  if (requests_.size() != 1) {
    for (size_t i = 0; i < requests_.size(); i++) {
      KALDI_ASSERT(!requests_[i]->need_model_derivative);
      KALDI_ASSERT(requests_[i]->store_component_stats ==
                   requests_[0]->store_component_stats);
    }
  }
}

void Compiler::CreateComputation(const CompilerOptions &opts,
                                 NnetComputation *computation) {
  computation->Clear();
  ComputationGraphBuilder builder(nnet_, &graph_);
  // note: there are only >1 segments in a 'looped' computation.
  for (size_t segment = 0; segment < requests_.size(); segment++) {
    builder.Compute(*(requests_[segment]));
    if (!builder.AllOutputsAreComputable()) {
      builder.ExplainWhyAllOutputsNotComputable();  // prints logging info
      KALDI_ERR << "Not all outputs were computable, cannot create computation.";
    }
    builder.Prune();
  }
  // see function declaration's comment for more on the meaning of "phases" (a
  // phase will later be decomposed into one or more steps).  for each segment
  // s, phases_per_segment[s] is a list of phases; each phase is a list of
  // cindex_ids.
  std::vector<std::vector<std::vector<int32> > > phases_per_segment;
  ComputeComputationPhases(nnet_, graph_, &phases_per_segment);
  std::vector<std::vector<int32> > steps;
  steps.reserve(1000);

  // maps each step to the segment in which it appears.  in the normal case
  // (non-looped computation), a vector of all zeros.
  std::vector<int32> step_to_segment;


  {
    // note: this class will output to 'steps' and to 'cindex_id_to_location_'.
    // it may incidentally change 'graph_' by adding a few cindexes.
    ComputationStepsComputer steps_computer(nnet_, &graph_, &steps,
                                            &cindex_id_to_location_);

    for (size_t segment = 0; segment < requests_.size(); segment++) {
      steps_computer.ComputeForSegment(*(requests_[segment]),
                                       phases_per_segment[segment]);
      while (step_to_segment.size() < steps.size())
        step_to_segment.push_back(segment);

      // save memory, by deleting the phases we just consumed.  the
      // following two lines just exist to save memory.
      std::vector<std::vector<int32> > temp;
      phases_per_segment[segment].swap(temp);
    }
    steps_computer.Check();
  }
  std::vector<bool> deriv_needed;
  ComputeDerivNeeded(steps, step_to_segment, &deriv_needed);
  CreateStepInfo(deriv_needed, step_to_segment, &steps, computation);
  AddCommands(deriv_needed, step_to_segment, computation);
  // the following command reorders commands so kAcceptInput and kProvideOutput
  // appear in the desired places.
  ConsolidateIoOperations(nnet_, computation);
  if (opts.output_debug_info)
    OutputDebugInfo(computation);
}

void Compiler::AddCommands(const std::vector<bool> &deriv_needed,
                           const std::vector<int32> &step_to_segment,
                           NnetComputation *computation) {
  computation->need_model_derivative = requests_[0]->need_model_derivative;
  int32 arbitrary_factor = 8;
  computation->commands.reserve(computation->matrices.size()
                                * arbitrary_factor);

  std::vector<int32> whole_submatrices;
  computation->GetWholeSubmatrices(&whole_submatrices);
  AllocateMatrices(whole_submatrices, computation);
  SetUpPrecomputedIndexes(step_to_segment, computation);
  int32 num_steps = steps_.size();
  for (int32 step = 0; step < num_steps; step++) {
    CompileForward(step, computation);
    if (step + 1 < static_cast<int32>(step_to_segment.size()) &&
        step_to_segment[step + 1] != step_to_segment[step]) {
      // insert a marker that separates segments of the computation.
      computation->commands.push_back(
          NnetComputation::Command(kNoOperationMarker));
    }
  }

  // mark the end of the forward phase.
  computation->commands.push_back(
      NnetComputation::Command(kNoOperationMarker));

  for (int32 step = num_steps - 1; step >= 0; step--)
    if (deriv_needed[step])
      CompileBackward(step, computation);

  DeallocateMatrices(whole_submatrices, step_to_segment, computation);
}


void Compiler::ComputeStepDependencies(
    const std::vector<int32> &this_step,
    int32 step_index,
    unordered_set<int32> *dep_steps) {
  dep_steps->clear();
  if (this_step.empty())
    return;
  // steps always have a single node index, we can pick the first.
  int32 node_index = graph_.cindexes[this_step[0]].first;
  if (nnet_.IsComponentNode(node_index)) {
    // there is only one step that a component step depends on, and it's the
    // immediately preceding step (the component-input step).
    KALDI_ASSERT(step_index > 0);
    dep_steps->insert(step_index - 1);
    return;
  }
  std::vector<int32>::const_iterator step_iter = this_step.begin(),
      step_end = this_step.end();
  int32 prev_input_step = -1;  // this is an optimization for speed.
  for (; step_iter != step_end; ++step_iter) {
    int32 cindex_id = *step_iter;
    const std::vector<int32> &dep = graph_.dependencies[cindex_id];
    std::vector<int32>::const_iterator iter = dep.begin(), end = dep.end();
    for (; iter != end; ++iter) {
      int32 dep_cindex_id = *iter,
          input_step = cindex_id_to_location_[dep_cindex_id].first;
      if (input_step != prev_input_step) {  // optimization.
        prev_input_step = input_step;
        dep_steps->insert(input_step);
      }
    }
  }
}

void Compiler::ComputeDerivNeeded(
    const std::vector<std::vector<int32> > &steps,
    const std::vector<int32> &step_to_segment,
    std::vector<bool> *deriv_needed) {
  KALDI_ASSERT(steps.size() == step_to_segment.size() &&
               step_to_segment[0] == 0 &&
               step_to_segment.back() + 1 == requests_.size());
  deriv_needed->clear();
  int32 num_steps = steps.size();
  deriv_needed->resize(num_steps, false);

  for (int32 step = 0; step < num_steps; step++) {
    const std::vector<int32> &this_step = steps[step];
    if (this_step.empty())  // empty steps are theoretically possible, e.g.
      continue;             // if a non-simple Component requires no input.
    int32 cindex_id = this_step[0];
    int32 node_index = graph_.cindexes[cindex_id].first;
    bool is_input = graph_.is_input[cindex_id];

    std::string node_name = nnet_.GetNodeNames()[node_index];
    unordered_set<int32> input_steps;
    ComputeStepDependencies(this_step, step, &input_steps);

    unordered_set<int32>::iterator iter = input_steps.begin(),
        end = input_steps.end();
    // if some step that we depend on needs a derivative, we need the derivative.
    for (; iter != end; ++iter) {
      int32 dep_step = *iter;
      KALDI_ASSERT(dep_step < step);
      if ((*deriv_needed)[dep_step])
        (*deriv_needed)[step] = true;
    }
    // if this step is an input and the user requested the derivative w.r.t. that
    // input, we need the derivative.
    const ComputationRequest &request = *(requests_[step_to_segment[step]]);

    if (is_input) {
      int32 input_index = request.IndexForInput(node_name);
      KALDI_ASSERT(input_index != -1);
      if (request.inputs[input_index].has_deriv)
        (*deriv_needed)[step] = true;
    }
    // if this step is an output and the user is providing the derivative w.r.t. that
    // output, we need a place to store the derivative, so we set (*deriv_needed) to
    // true.
    if (nnet_.IsOutputNode(node_index)) {
      int32 output_index = request.IndexForOutput(node_name);
      KALDI_ASSERT(output_index != -1);
      if (request.outputs[output_index].has_deriv)
        (*deriv_needed)[step] = true;
    }

    // If this is an updatable Component node with a nonzero learning rate and
    // the user requested model derivatives (e.g. during training), we need this
    // step's derivative.
    if (nnet_.IsComponentNode(node_index) && request.need_model_derivative) {
      const NetworkNode &node = nnet_.GetNode(node_index);
      const Component *c = nnet_.GetComponent(node.u.component_index);
      if (c->Properties() & kUpdatableComponent) {
        const UpdatableComponent *u = dynamic_cast<const UpdatableComponent*>(c);
        KALDI_ASSERT(u != NULL);
        if (u->LearningRate() != 0)
          (*deriv_needed)[step] = true;
      }
    }
  }
  if (GetVerboseLevel() >= 5) {
    std::ostringstream os;
    os << "deriv_needed = ";
    for (int32 i = 0; i < deriv_needed->size(); i++)
      os << ((*deriv_needed)[i] ? "t" : "f");
    os << "\n";
    KALDI_VLOG(5) << os.str();
  }
}

MatrixStrideType Compiler::GetStrideType(int32 node_index) const {
  int32 component_node_index;
  bool is_input;
  if (nnet_.IsComponentInputNode(node_index)) {
    // this node is for the input to a component.
    component_node_index = node_index + 1;
    is_input = true;
  } else if (nnet_.IsComponentNode(node_index)) {
    component_node_index = node_index;
    is_input = false;
  } else {
    return kDefaultStride;
  }
  const NetworkNode &node = nnet_.GetNode(component_node_index);
  const Component *c = nnet_.GetComponent(node.u.component_index);
  if (is_input) {
    return (c->Properties() & kInputContiguous) ?
        kStrideEqualNumCols : kDefaultStride;
  } else {
    return (c->Properties() & kOutputContiguous) ?
        kStrideEqualNumCols : kDefaultStride;
  }
}


// Note: "by_step" is an input but is passed as a pointer because this
// function destroys it.
void Compiler::CreateStepInfo(
    const std::vector<bool> &deriv_needed,
    const std::vector<int32> &step_to_segment,
    std::vector<std::vector<int32> > *by_step,
    NnetComputation *computation) {
  KALDI_ASSERT(!by_step->empty());
  int32 num_steps = by_step->size();
  steps_.resize(num_steps);
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &this_info = steps_[step];
    this_info.output_cindex_ids.swap((*by_step)[step]);
    this_info.segment = step_to_segment[step];
    int32 num_ids = this_info.output_cindex_ids.size();
    this_info.output_indexes.resize(num_ids);
    for (int32 row_index = 0; row_index < num_ids; row_index++)
      this_info.output_indexes[row_index] =
          graph_.cindexes[this_info.output_cindex_ids[row_index]].second;
    if (num_ids > 0) {
      // node id's of all Cindexes are the same, so just use first one.
      this_info.node_index =
          graph_.cindexes[this_info.output_cindex_ids.front()].first;
    } else {
      // it's possible to have an empty step if it's the component-input step of
      // a GeneralComponent that does not always have dependencies, such as the
      // ConstantFunctionComponent.  This is just a kind of placeholder; it will
      // generate no commands.  The next command works because the next
      // step will be the propagate for that Component, whose node-index is one
      // more than the component-input node.
      KALDI_ASSERT((step+1) < by_step->size() && !(*by_step)[step+1].empty());
      this_info.node_index =
          graph_.cindexes[(*by_step)[step+1][0]].first - 1;
      KALDI_ASSERT(this_info.node_index >= 0);
      continue;  // we don't need to do anything else for this step.
    }
    const NetworkNode &node = nnet_.GetNode(this_info.node_index);
    int32 num_rows = num_ids, num_cols = node.Dim(nnet_);

    if (node.node_type != kDimRange) {
      MatrixStrideType stride_type = GetStrideType(this_info.node_index);
      this_info.value = computation->NewMatrix(num_rows, num_cols,
                                               stride_type);
      if (deriv_needed[step])
        this_info.deriv = computation->NewMatrix(num_rows, num_cols,
                                                 stride_type);
    } else {
      // kDimRange.  Will just be a sub-matrix of a Component or Input node.
      std::vector<int32>::const_iterator
          iter = this_info.output_cindex_ids.begin(),
          end = this_info.output_cindex_ids.end();
      int32 source_cindex_id = -1;
      for (; iter != end; ++iter) {
        int32 cindex_id = *iter;
        if (!graph_.dependencies[cindex_id].empty()) {
          KALDI_ASSERT(graph_.dependencies[cindex_id].size() == 1);
          source_cindex_id = graph_.dependencies[cindex_id][0];
          break;
        }
      }
      KALDI_ASSERT(source_cindex_id >= 0);
      int32 input_step = cindex_id_to_location_[source_cindex_id].first;
      KALDI_ASSERT(this_info.output_cindex_ids.size() ==
                   steps_[input_step].output_cindex_ids.size());
      KALDI_ASSERT(input_step >= 0 && input_step < step);
      KALDI_PARANOID_ASSERT(this_info.output_indexes ==
                            steps_[input_step].output_indexes);
      this_info.value = computation->NewSubMatrix(steps_[input_step].value,
                                                  0, -1,
                                                  node.dim_offset, node.dim);
      if (deriv_needed[step])
        this_info.deriv = computation->NewSubMatrix(steps_[input_step].deriv,
                                                    0, -1,
                                                    node.dim_offset, node.dim);
    }
    if (node.node_type == kDescriptor) {
      // we have a couple of things to do: set up input_locations_list which
      // says where we copy the data from, and also set up value_parts and
      // possibly deriv_parts.
      const Descriptor &desc = node.descriptor;
      int32 num_parts = desc.NumParts();
      KALDI_ASSERT(num_parts > 0);
      // set up input_locations_list.
      this_info.input_locations_list.resize(num_parts);
      for (int32 part = 0; part < num_parts; part++)
        ComputeInputLocationsList(step, part,
                                  &(this_info.input_locations_list[part]));
      // set up value_parts and deriv_parts.
      if (num_parts == 1) {
        this_info.value_parts.push_back(this_info.value);
        if (deriv_needed[step])
          this_info.deriv_parts.push_back(this_info.deriv);
      } else { // num_parts > 1.
        int32 cur_dim_offset = 0;
        // Have multiple parts, so need to set up sub-matrices.
        this_info.value_parts.resize(num_parts);
        if (deriv_needed[step])
          this_info.deriv_parts.resize(num_parts);
        for (int32 p = 0; p < num_parts; p++) {
          const SumDescriptor &this_part = desc.Part(p);
          int32 this_dim = this_part.Dim(nnet_);
          this_info.value_parts[p] =
              computation->NewSubMatrix(this_info.value,
                                        0, -1,
                                        cur_dim_offset, this_dim);
          if (deriv_needed[step])
            this_info.deriv_parts[p] =
                computation->NewSubMatrix(this_info.deriv,
                                          0, -1,
                                          cur_dim_offset, this_dim);
          cur_dim_offset += this_dim;
        }
        KALDI_ASSERT(cur_dim_offset == desc.Dim(nnet_));
      }
    }
    KALDI_ASSERT(static_cast<int32>(this_info.output_cindex_ids.size()) ==
                 computation->submatrices[this_info.value].num_rows);
  }
}

bool Compiler::IsInputStep(int32 step) const {
  KALDI_ASSERT(step >= 0);
  if (step >= steps_.size())
    return false;
  const StepInfo &step_info = steps_[step];
  const NetworkNode &node = nnet_.GetNode(step_info.node_index);
  return (node.node_type == kInput);
}

void Compiler::CompileForward(int32 step,
                                    NnetComputation *computation) const {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  const NetworkNode &node = nnet_.GetNode(step_info.node_index);
  switch (node.node_type) {
    case kInput:  // Note: input nodes appear before other node types.
      AddForwardStepInput(step, computation);
      if (!IsInputStep(step + 1))  // Make sure forward computation is nonempty.
        computation->commands.push_back(
            NnetComputation::Command(kNoOperationPermanent));
      break;
    case kDimRange: break;  // Nothing to do.
    case kComponent:
      AddForwardStepComponent(step, computation);
      break;
    case kDescriptor:
      CompileForwardDescriptor(step, computation);
      break;
    default:
      KALDI_ERR << "Invalid node type";
  }

}


void Compiler::CompileForwardDescriptor(
    int32 step, NnetComputation *computation) const {
  int32 num_parts = steps_[step].value_parts.size();
  for (int32 part = 0; part < num_parts; part++)
    CompileForwardSumDescriptor(step, part, computation);
  const StepInfo &step_info = steps_[step];
  if (nnet_.IsOutputNode(step_info.node_index)) {
    // If the node is an output then we need to add commands to provide the
    // output to the user, and possibly to get derivatives w.r.t. the output
    // from the user.
    int32 node_index = step_info.node_index,
        submatrix_index = step_info.value;
    KALDI_ASSERT(computation->IsWholeMatrix(submatrix_index));
    NnetComputation::Command c(kProvideOutput, submatrix_index, node_index);
    computation->commands.push_back(c);
  }
}


// The output vector "locations" is indexed first by output row-index i
// (i.e. the index of output_indexes or output_cindex_ids), and then is a list
// of input locations for that row-index, sorted in the natural order of
// Cindexes (but not necessarily unique).  The semantics is that the i'th row of
// the output becomes a sum over the rows in the i'th list (or zero if that list
// is empty).  These locations will be pairs [step-index, row-index].
void Compiler::ComputeInputLocationsList(
    int32 step, int32 part_index,
    std::vector<std::vector<std::pair<int32, int32> > > *submat_locations_list)
    const {

  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  const std::vector<Index> &output_indexes = step_info.output_indexes;
  const NetworkNode &node = nnet_.GetNode(step_info.node_index);
  const SumDescriptor &descriptor = node.descriptor.Part(part_index);
  int32 num_indexes = output_indexes.size();
  submat_locations_list->clear();
  submat_locations_list->resize(num_indexes);

  for (int32 i = 0; i < num_indexes; i++) {
    const Index &index = output_indexes[i];
    std::vector<std::pair<int32, int32> > &this_locations_list =
        (*submat_locations_list)[i];
    if (index.t != kNoTime) {
      // a real Index, not a 'blank' one
      // ('blank' indexes are inserted by some non-simple Components to
      // satisfy internal constraints.
      std::vector<int32> input_cindex_ids;
      std::vector<Cindex> input_cindexes;
      CindexSet cindex_set(graph_);
      bool ans = descriptor.IsComputable(index, cindex_set, &input_cindexes);
      // earlier compilation stages should have checked that it is computable,
      // and the graph should still contain required inputs.
      KALDI_ASSERT(ans);
      std::sort(input_cindexes.begin(), input_cindexes.end());
      int32 size = input_cindexes.size();
      input_cindex_ids.resize(size);
      for (int32 j = 0; j < size; j++) {
        int32 c = graph_.GetCindexId(input_cindexes[j]);
        KALDI_ASSERT(c != -1);
        input_cindex_ids[j] = c;
      }
      this_locations_list.resize(size);
      for (int32 j = 0; j < size; j++)
        this_locations_list[j] = cindex_id_to_location_[input_cindex_ids[j]];
    } else {
      this_locations_list.clear();
    }
  }
}

void Compiler::ComputeValueSubmatLocationsList(
const std::vector<std::vector<std::pair<int32, int32> > > &input_locations_list,
    std::vector<std::vector<std::pair<int32, int32> > >*submat_locations_list)
const {
  submat_locations_list->clear();
  submat_locations_list->resize(input_locations_list.size());
  int32 size = submat_locations_list->size();
  for (int32 i = 0; i < size; i++) {
    const std::vector<std::pair<int32, int32> > &this_list =
        input_locations_list[i];
    std::vector<std::pair<int32, int32> > &this_submat_list =
        (*submat_locations_list)[i];
    this_submat_list.resize(this_list.size());
    std::vector<std::pair<int32, int32> >::const_iterator
        input_iter = this_list.begin(), input_end = this_list.end();
    std::vector<std::pair<int32, int32> >::iterator iter =
        this_submat_list.begin();
    for (; input_iter != input_end; ++input_iter, ++iter) {
      int32 step = input_iter->first,
          value_submat_index = steps_[step].value,
          row = input_iter->second;
      iter->first = value_submat_index;
      iter->second = row;
    }
  }
}


void Compiler::ComputeDerivSubmatLocationsList(
 const std::vector<std::vector<std::pair<int32, int32> > > &input_locations_list,
    std::vector<std::vector<std::pair<int32, int32> > > *submat_locations_list)
    const {
  submat_locations_list->clear();
  submat_locations_list->resize(input_locations_list.size());
  int32 size = submat_locations_list->size();
  for (int32 i = 0; i < size; i++) {
    const std::vector<std::pair<int32, int32> > &this_list = input_locations_list[i];
    std::vector<std::pair<int32, int32> > &this_submat_list = (*submat_locations_list)[i];
    this_submat_list.reserve(this_list.size());
    std::vector<std::pair<int32, int32> >::const_iterator
        input_iter = this_list.begin(), input_end = this_list.end();
    for (; input_iter != input_end; ++input_iter) {
      int32 step = input_iter->first,
          deriv_submat_index = steps_[step].deriv,
          row = input_iter->second;
      if (deriv_submat_index > 0)
        this_submat_list.push_back(std::pair<int32,int32>(deriv_submat_index,
                                                          row));
    }
  }
}



BaseFloat Compiler::SplitByScale(
    const SumDescriptor &descriptor,
 const std::vector<std::vector<std::pair<int32,int32> > > &input_locations_list,
  std::vector<std::pair<BaseFloat,
    std::vector<std::vector<std::pair<int32,int32> > > > >
    *split_locations_lists) const {
  split_locations_lists->clear();
  // alpha_to_nodes maps from the scale alpha to the list of nodes which are
  // given that scale.
  std::map<BaseFloat, std::vector<int32> > alpha_to_nodes;
  { // This block compute `alpha_to_nodes`.
    std::vector<int32> nodes;
    descriptor.GetNodeDependencies(&nodes);
    SortAndUniq(&nodes);
    // Now `nodes` is a list of the graph node indexes that are referred to
    // in the descriptor.  E.g. if the Descriptor represents
    // 'Sum(tdnn1, Offset(tdnn2, -2))' then `nodes` would contain the
    // integer node indexes for graph-nodes 'tdnn1' and 'tdnn2'.
    for (size_t i = 0; i < nodes.size(); i++) {
      int32 node = nodes[i];
      BaseFloat alpha = descriptor.GetScaleForNode(node);
      KALDI_ASSERT(alpha - alpha == 0.0);  // check it's not infinity.
      alpha_to_nodes[alpha].push_back(node);
    }
  }

  if (alpha_to_nodes.size() == 1) {
    // If all the alpha values are the same we treat it as a special case
    // for efficiency, to avoid a redundant copy of the contents of
    // 'input_locations_list'.
    return alpha_to_nodes.begin()->first;
  }

  // `steps_used` will be a list of all step indexes that appear as `.first`
  // elements in `input_locations_list`.
  unordered_set<int32> steps_used;
  {  // This block computes `steps_used`.
    int32 cur_step = -1000;
    std::vector<std::vector<std::pair<int32,int32> > >::const_iterator
        iter = input_locations_list.begin(),
        end = input_locations_list.end();
    for (; iter != end; ++iter) {
      std::vector<std::pair<int32,int32> >::const_iterator
          pair_iter = iter->begin(),
          pair_end = iter->end();
      for (; pair_iter != pair_end; ++pair_iter) {
        if (pair_iter->first != cur_step) {
          cur_step = pair_iter->first;
          steps_used.insert(cur_step);
        }
      }
    }
  }

  // `node_to_steps` will be a map from graph node index to the list of steps
  // which are present in `steps_used` and which are associated with that graph
  // node.
  std::map<int32, std::vector<int32> > node_to_steps;
  {  // This block computes `node_to_steps`.
    unordered_set<int32>::const_iterator
        step_iter = steps_used.begin(), step_end = steps_used.end();
    for (; step_iter != step_end; ++step_iter) {
      int32 step_index = *step_iter;
      KALDI_ASSERT(static_cast<size_t>(step_index) < steps_.size());
      int32 node_index = steps_[step_index].node_index;
      node_to_steps[node_index].push_back(step_index);
    }
  }

  int32 num_rows = input_locations_list.size();
  split_locations_lists->resize(alpha_to_nodes.size());
  // `step_to_index` will map from the step-index to the index into
  // `split_locations_lists`; each index is associated with a different value of
  // the scale `alpha`.
  std::vector<int32> step_to_locations_index(steps_.size(), -1);
  {  // This block computes `step_to_index` and also sets the `alpha` values
     // which are present as (*split_locations_lists)[*].first.
    std::map<BaseFloat, std::vector<int32> >::const_iterator
        iter = alpha_to_nodes.begin(), end = alpha_to_nodes.end();
    int32 split_locations_index = 0;
    for (; iter != end; ++iter, ++split_locations_index) {
      BaseFloat alpha = iter->first;
      const std::vector<int32> &nodes = iter->second;
      (*split_locations_lists)[split_locations_index].first = alpha;
      (*split_locations_lists)[split_locations_index].second.resize(num_rows);
      for (size_t i = 0; i < nodes.size(); i++) {
        int32 node_index = nodes[i];
        KALDI_ASSERT(node_to_steps.count(node_index) != 0);
        const std::vector<int32> &steps = node_to_steps[node_index];
        for (size_t j = 0; j < steps.size(); j++) {
          int32 step_index = steps[j];
          KALDI_ASSERT(step_index >= 0 &&
                       step_to_locations_index[step_index] == -1);
          step_to_locations_index[step_index] = split_locations_index;
        }
      }
    }
  }

  {  // This block populates 'split_locations_lists[*].second' with the
     // split-by-alpha version of 'input_locations_list'
    for (int32 r = 0; r < num_rows; r++) {
      const std::vector<std::pair<int32,int32> > &this_list =
          input_locations_list[r];
      std::vector<std::pair<int32,int32> >::const_iterator
          pair_iter = this_list.begin(),
          pair_end = this_list.end();
      for (; pair_iter != pair_end; ++pair_iter) {
        int32 step = pair_iter->first,
            split_locations_index = step_to_locations_index[step];
        (*split_locations_lists)[split_locations_index].second[r].push_back(
            *pair_iter);
      }
    }
  }
  return std::numeric_limits<BaseFloat>::infinity();
}


void Compiler::CompileForwardSumDescriptor(
    int32 step, int32 part_index, NnetComputation *computation) const {
  const StepInfo &step_info = steps_[step];
  int32 value_submatrix_index = step_info.value_parts[part_index];
  const SumDescriptor &descriptor =
      nnet_.GetNode(step_info.node_index).descriptor.Part(part_index);

  BaseFloat offset_term = descriptor.GetScaleForNode(-1);
  if (offset_term != 0.0) {
    computation->commands.push_back(
        NnetComputation::Command(offset_term, kSetConst,
                                 value_submatrix_index));
    // if offset_term == 0.0 there's no need to do this, because
    // we zeroed the matrix when we allocated it; search in this
    // file for kSetConst to see the code.  If we are redundantly
    // setting the value, this will later be optimized out (in the
    // common cases).
  }


  // `input_locations_list` is a vector indexed by row-index, with each element
  // being a list of pairs (step, row_index) representing terms in a weighted
  // sum.
  const std::vector<std::vector<std::pair<int32,int32> > >
      &input_locations_list = step_info.input_locations_list[part_index];

  // `split_locations_lists` is a vector of pairs `(alpha, locations_list)`
  // where alpha is the scale in which these items appear in the
  // summation and `locations_list` is the same format as `input_locations_list`
  std::vector<std::pair<BaseFloat,
   std::vector<std::vector<std::pair<int32,int32> > > > > split_locations_lists;
  BaseFloat shared_alpha = SplitByScale(descriptor, input_locations_list,
                                 &split_locations_lists);
  if (shared_alpha - shared_alpha == 0.0) {
    // If the returned value 'shared_alpha' is finite, this indicates that there was no
    // need to split up 'input_locations_list' because all the alpha values
    // (scales) were the same.  We treat this case specially for efficiency
    // reasons; this branch will be the most common branch.
    std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
    ComputeValueSubmatLocationsList(input_locations_list,
                                    &submat_locations_list);
    CompileForwardFromSubmatLocationsList(
        value_submatrix_index,
        shared_alpha,
        submat_locations_list,
        computation);
  } else {
    for (size_t i = 0; i < split_locations_lists.size(); i++) {
      BaseFloat this_alpha = split_locations_lists[i].first;
      KALDI_ASSERT(this_alpha - this_alpha == 0.0);
      std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
      ComputeValueSubmatLocationsList(split_locations_lists[i].second,
                                      &submat_locations_list);
      CompileForwardFromSubmatLocationsList(
        value_submatrix_index,
        this_alpha,
        submat_locations_list,
        computation);
    }
  }
}

void Compiler::CompileForwardFromIndexes(
    int32 value_submatrix_index,
    int32 input_submatrix_index,
    BaseFloat alpha,
    const std::vector<int32> &indexes,
    NnetComputation *computation) const {

  int32 input_num_rows =
      computation->submatrices[input_submatrix_index].num_rows,
      num_rows = indexes.size();
  if (input_num_rows == num_rows) {
    int32 i;
    for (i = 0; i < num_rows; i++)
      if (indexes[i] != i)
        break;
    if (i == num_rows) {  // Simplest case: just matrix addition.
      computation->commands.push_back(
          NnetComputation::Command(alpha, kMatrixAdd,
                                   value_submatrix_index,
                                   input_submatrix_index));

      return;
    }
  }
  // if we got to here, it's not just a case of matrix-copy or matrix-add,
  // but it's still from a single source matrix.
  int32 indexes_index = computation->indexes.size();
  computation->indexes.push_back(indexes);
  computation->commands.push_back(
      NnetComputation::Command(alpha, kAddRows, value_submatrix_index,
                               input_submatrix_index, indexes_index));
  return;
}

void Compiler::CompileForwardFromSubmatLocations(
    int32 value_submatrix_index,
    BaseFloat alpha,
    const std::vector<std::pair<int32, int32> > &submat_locations,
    NnetComputation *computation) const {

  int32 input_submatrix_index = -1;
  std::vector<int32> indexes;
  if (ConvertToIndexes(submat_locations, &input_submatrix_index, &indexes)) {
    CompileForwardFromIndexes(value_submatrix_index,
                              input_submatrix_index,
                              alpha,
                              indexes,
                              computation);
    return;
  } else {
    // There are multiple source matrices.
    int32 indexes_multi_index = computation->indexes_multi.size();
    computation->indexes_multi.push_back(submat_locations);
    computation->commands.push_back(
        NnetComputation::Command(alpha, kAddRowsMulti,
                                 value_submatrix_index,
                                 indexes_multi_index));
    return;
  }
}

void Compiler::CompileForwardFromSubmatLocationsList(
    int32 value_submatrix_index,
    BaseFloat alpha,
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    NnetComputation *computation) const {
  std::vector<std::vector<std::pair<int32, int32> > > split_lists;
  SplitLocations(submat_lists, &split_lists);
  int32 size = split_lists.size();
  // note: `size` may be empty in unusual cases so don't assert that it's
  // nonzero.
  for (int32 i = 0; i < size; i++)
    CompileForwardFromSubmatLocations(
        value_submatrix_index,
        alpha,
        split_lists[i],
        computation);
}


void Compiler::CompileBackwardFromSubmatLocationsList(
    int32 deriv_submatrix_index,
    BaseFloat alpha,
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    NnetComputation *computation) const {
  std::vector<std::vector<std::pair<int32, int32> > > split_lists;
  SplitLocationsBackward(submat_lists, &split_lists);
  int32 size = split_lists.size();  // size may be zero e.g. for unused outputs.
  for (int32 i = 0; i < size; i++)
    CompileBackwardFromSubmatLocations(
        deriv_submatrix_index,
        alpha,
        split_lists[i],
        computation);
}


void Compiler::CompileBackwardSumDescriptor(
    int32 step, int32 part_index, NnetComputation *computation) const {
  const StepInfo &step_info = steps_[step];
  int32 deriv_submatrix_index = step_info.deriv_parts[part_index];
  KALDI_ASSERT(deriv_submatrix_index > 0);  // or should not have called this.
  const SumDescriptor &descriptor =
      nnet_.GetNode(step_info.node_index).descriptor.Part(part_index);
  // Note: `offset_term` appeared in the forward computation here but does not
  // come into the backward computation.

  // `input_locations_list` is a vector indexed by row-index, with each element
  // being a list of pairs (step, row_index) representing terms in a weighted
  // sum.
  const std::vector<std::vector<std::pair<int32,int32> > >
      &input_locations_list = step_info.input_locations_list[part_index];

  // `split_locations_lists` is a vector of pairs `(alpha, locations_list)`
  // where alpha is the scale in which these items appear in the
  // summation and `locations_list` is the same format as `input_locations_list`
  std::vector<std::pair<BaseFloat,
   std::vector<std::vector<std::pair<int32,int32> > > > > split_locations_lists;
  BaseFloat shared_alpha = SplitByScale(descriptor, input_locations_list,
                                 &split_locations_lists);
  if (shared_alpha - shared_alpha == 0.0) {
    // If the returned value 'shared_alpha' is finite, this indicates that there
    // was no need to split up 'input_locations_list' because all the alpha
    // values (scales) were the same.  We treat this case specially for
    // efficiency reasons; this branch will be the most common branch.
    std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
    ComputeDerivSubmatLocationsList(input_locations_list,
                                    &submat_locations_list);
    CompileBackwardFromSubmatLocationsList(deriv_submatrix_index,
                                           shared_alpha,
                                           submat_locations_list,
                                           computation);
  } else {
    for (size_t i = 0; i < split_locations_lists.size(); i++) {
      BaseFloat this_alpha = split_locations_lists[i].first;
      KALDI_ASSERT(this_alpha - this_alpha == 0.0);
      std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
      ComputeDerivSubmatLocationsList(split_locations_lists[i].second,
                                      &submat_locations_list);
      CompileBackwardFromSubmatLocationsList(deriv_submatrix_index,
                                             this_alpha,
                                             submat_locations_list,
                                             computation);
    }
  }
}



void Compiler::CompileBackwardFromSubmatLocations(
    int32 deriv_submatrix_index,
    BaseFloat alpha,
    const std::vector<std::pair<int32, int32> > &submat_locations,
    NnetComputation *computation) const {
  // This function creates a command to handle an individual piece of the
  // Descriptor, for backprop.  Note: because the backprop case is a little
  // trickier to implement efficiently on the GPU, there may be cases
  // which we will refuse to implement backprop for if we get here.



  int32 first_value;
  std::vector<int32> second_values;
  if (ConvertToIndexes(submat_locations, &first_value,
                       &second_values)) {
    int32 input_deriv_submatrix_index = first_value;
    CompileBackwardFromIndexes(deriv_submatrix_index,
                               input_deriv_submatrix_index,
                               alpha,
                               second_values,
                               computation);
    return;
  } else {
    // There are multiple source matrices.
    std::vector<std::pair<int32, int32> > submat_locations_sorted;
    std::sort(submat_locations_sorted.begin(), submat_locations_sorted.end());
    if (IsSortedAndUniq(submat_locations_sorted)) {
      // There are no repeats in any of the submat locations.  This means that
      // we can just use kAddToRowsMulti (i.e. AddToRows with pointer
      // destination).  If there were repeats, the CUDA kernel would require
      // special synchronization so we don't allow it.
      int32 indexes_multi_index = computation->indexes_multi.size();
      computation->indexes_multi.push_back(submat_locations);
      computation->commands.push_back(
          NnetComputation::Command(alpha,
                                   kAddToRowsMulti,
                                   deriv_submatrix_index,
                                   indexes_multi_index));
      return;
    }
    // If you reach this point, there is a case that wasn't handled.  Our
    // intended strategy to handle it, if it's ever needed, is to create a
    // temporary matrix consisting of all the unique submat_locations in the
    // input.  We would first recurse to CompileBackwardFromIndexes, and
    // let it write to this temporary matrix; and then do the kAddToRowsMulti
    // command as above to go from the temporary matrix to the multiple
    // matrices.
    KALDI_ERR << "This case not handled.";
  }
}

void Compiler::CompileBackwardFromIndexes(
    int32 deriv_submatrix_index,
    int32 input_deriv_submatrix_index,
    BaseFloat alpha,
    const std::vector<int32> &indexes,
    NnetComputation *computation) const {

  int32 num_rows = computation->submatrices[deriv_submatrix_index].num_rows,
      input_num_rows =
      computation->submatrices[input_deriv_submatrix_index].num_rows;
  KALDI_ASSERT(indexes.size() == num_rows);
  if (input_num_rows == num_rows) {
    int32 i;
    for (i = 0; i < num_rows; i++)
      if (indexes[i] != i)
        break;
    if (i == num_rows) {  // Simplest case: just matrix addition.
        computation->commands.push_back(
            NnetComputation::Command(alpha,
                                     kMatrixAdd,
                                     input_deriv_submatrix_index,
                                     deriv_submatrix_index));

      return;
    }
  }
  if (input_num_rows >= num_rows) {
    // If there are no repeated elements in the "indexes" array, we can reverse
    // the mapping and make it an operation of type kAddRows.  TODO: change this
    // to use kAddToRows, kCopyToRows, when implemented (will be more
    // efficient).
    std::vector<int32> reverse_indexes(input_num_rows, -1);
    int32 i;
    for (i = 0; i < num_rows; i++) {
      int32 index_i = indexes[i];
      KALDI_ASSERT(index_i >= -1 && index_i < input_num_rows);
      if (index_i >= 0) {
        if (reverse_indexes[index_i] == -1)
          reverse_indexes[index_i] = i;
        else
          break;
      }  // note: there may be -1's in 'indexes', meaning just use zero.
    }
    if (i == num_rows) {
      // There were no repeated elements, and this strategy will work.
      int32 indexes_index = computation->indexes.size();
      computation->indexes.push_back(reverse_indexes);
        computation->commands.push_back(
            NnetComputation::Command(alpha,
                                     kAddRows,
                                     input_deriv_submatrix_index,
                                     deriv_submatrix_index,
                                     indexes_index));
        return;
    }
  }
  std::vector<std::pair<int32, int32> > ranges;
  if (HasContiguousProperty(indexes, &ranges)) {
    // the operation can be set up as AddRowRanges.
    if (static_cast<int32>(ranges.size()) != input_num_rows) {
      KALDI_ASSERT(static_cast<int32>(ranges.size()) < input_num_rows);
      // extend with (-1, -1) pairs.
      ranges.resize(input_num_rows, std::pair<int32,int32>(-1, -1));
    }
    int32 indexes_ranges_index = computation->indexes_ranges.size();
    computation->indexes_ranges.push_back(ranges);
    computation->commands.push_back(
        NnetComputation::Command(alpha,
                                 kAddRowRanges,
                                 input_deriv_submatrix_index,
                                 deriv_submatrix_index,
                                 indexes_ranges_index));
    // TODO: if any of these ranges are quite long (summing over many rows), the
    // resulting code could be inefficient because the AddRowRanges kernels
    // takes time linear in the length of the range.  Using a temporary matrix
    // with an intermediate size would make this more efficient in that case, so
    // the one command would be two commands (plus commands to set up and
    // destroy the temporary matrix).
    return;
  }

  // If you ever reach here, it means someone has used a type of network that we
  // don't yet support in the backprop.  Basically this case can be handled by
  // creating a temporary matrix to reorder the matrix at deriv_submatrix_index,
  // (using CopyRows), and doing AddRowRanges from that.
  // It wouldn't be too much work.
  KALDI_ERR << "This case not implemented yet.";
}


void Compiler::CompileBackwardDescriptor(
    int32 step, NnetComputation *computation) {
  StepInfo &step_info = steps_[step];
  if (nnet_.IsOutputNode(step_info.node_index) &&
      step_info.deriv > 0) {
    int32 deriv_submatrix_index = step_info.deriv;
    KALDI_ASSERT(computation->IsWholeMatrix(deriv_submatrix_index));
    NnetComputation::Command c(kAcceptInput, deriv_submatrix_index,
                               step_info.node_index);
    computation->commands.push_back(c);
  }

  // the top-level descriptor has a bunch of parts that we concatenate features
  // over.
  int32 num_parts = step_info.value_parts.size();
  for (int32 part = 0; part < num_parts; part++)
    CompileBackwardSumDescriptor(step, part,
                                       computation);
}


void Compiler::CompileBackward(int32 step,
                                     NnetComputation *computation) {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case kInput:
      AddBackwardStepInput(step, computation);
      if (!IsInputStep(step + 1))  // Make sure backward computation is nonempty.
        computation->commands.push_back(
            NnetComputation::Command(kNoOperationPermanent));
      break;
    case kDimRange:
      break;  // Nothing to do.
    case kComponent:
      AddBackwardStepComponent(step, computation);
      break;
    case kDescriptor:
      CompileBackwardDescriptor(step, computation);
      break;
    default:
      KALDI_ERR << "Invalid node type";
  }
}

// This just adds a command of type kAcceptInput that directs the computer to
// expect input from the user.  Because inputs are always listed first in
// 'steps', these will precede the actual commands.
void Compiler::AddForwardStepInput(int32 step,
                                   NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index,
      submatrix_index = step_info.value;
  KALDI_ASSERT(computation->IsWholeMatrix(submatrix_index));

  const NetworkNode &node = nnet_.GetNode(node_index);
  // actually currently the node type would always be kInput.
  KALDI_ASSERT(node.node_type == kInput || node.node_type == kComponent);

  NnetComputation::Command c(kAcceptInput, submatrix_index, node_index);
  computation->commands.push_back(c);
}


void Compiler::AddForwardStepComponent(int32 step,
                                       NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 input_step = step - 1;
  const StepInfo &input_step_info = steps_[input_step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);
  KALDI_ASSERT(node.node_type == kComponent);
  int32 component_index = node.u.component_index;
  const Component *component = nnet_.GetComponent(component_index);

  // note RE memo_index: we'll renumber them in optimization to get rid of gaps.
  // The use of 'step' as the memo index is OK because step > 0 if we're doing
  // forward propagation, there must be preceding steps for inputs or for
  // component-input nodes).
  int32 properties = component->Properties(),
      input_submatrix_index = input_step_info.value,
      output_submatrix_index = step_info.value,
      memo_index = (step_info.deriv > 0 && (properties & kUsesMemo) ? step : 0),
      store_stats = (requests_[0]->store_component_stats &&
                     (properties & kStoresStats) ?  1 : 0);

  NnetComputation::Command c(kPropagate,
                             component_index,
                             step_info.precomputed_indexes_index,
                             input_submatrix_index,
                             output_submatrix_index,
                             memo_index,
                             store_stats);
  computation->commands.push_back(c);
}


void Compiler::AddBackwardStepInput(int32 step,
                                    NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index,
      deriv_submatrix_index = step_info.deriv;
  if (deriv_submatrix_index == 0)
    return;  // Nothing to do.
  KALDI_ASSERT(computation->IsWholeMatrix(deriv_submatrix_index));
  const NetworkNode &node = nnet_.GetNode(node_index);
  // actually, currently the node type would always be kInput.
  KALDI_ASSERT(node.node_type == kInput || node.node_type == kComponent);

  NnetComputation::Command c(kProvideOutput, deriv_submatrix_index, node_index);
  computation->commands.push_back(c);
}


void Compiler::AddBackwardStepComponent(int32 step,
                                        NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 input_step = step - 1;
  const StepInfo &input_step_info = steps_[input_step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);
  KALDI_ASSERT(node.node_type == kComponent);
  int32 component_index = node.u.component_index;
  const Component *component = nnet_.GetComponent(component_index);
  int32 properties = component->Properties();

  int32 input_submatrix_index = input_step_info.value,
      output_submatrix_index = step_info.value,
      input_deriv_submatrix_index = input_step_info.deriv,
      output_deriv_submatrix_index = step_info.deriv,
      memo_index = (properties & kUsesMemo ? step : 0);
  KALDI_ASSERT(output_deriv_submatrix_index > 0 &&
               (input_deriv_submatrix_index > 0 ||
                properties & kUpdatableComponent));

  if (! (properties & kBackpropNeedsInput))
    input_submatrix_index = 0;
  if (! (properties & kBackpropNeedsOutput))
    output_submatrix_index = 0;

  NnetComputation::Command c(kBackprop,
                             component_index,
                             step_info.precomputed_indexes_index,
                             input_submatrix_index,
                             output_submatrix_index,
                             output_deriv_submatrix_index,
                             input_deriv_submatrix_index,
                             memo_index);
  computation->commands.push_back(c);
}



void Compiler::AllocateMatrices(const std::vector<int32> &whole_submatrices,
                                NnetComputation *computation) const {
  KALDI_ASSERT(computation->commands.empty());
  // Work out which matrices are inputs to the computation (or output-derivs,
  // which are also supplied as inputs to the computation); we won't be setting
  // these up.
  unordered_set<int32> input_and_oderiv_matrices;
  int32 num_steps = steps_.size();
  for (int32 step = 0; step < num_steps; step++) {
    const StepInfo &this_info = steps_[step];
    if (this_info.output_cindex_ids.empty())
      continue;
    int32 first_cindex_id = this_info.output_cindex_ids.front(),
        node_index = this_info.node_index;
    bool is_input = graph_.is_input[first_cindex_id],
        is_output = nnet_.IsOutputNode(node_index);
    if (is_input) {
      int32 value_submatrix_index = this_info.value,
          value_matrix_index =
          computation->submatrices[value_submatrix_index].matrix_index;
      input_and_oderiv_matrices.insert(value_matrix_index);
    }
    if (is_output && this_info.deriv != 0) {
      int32 deriv_submatrix_index = this_info.deriv,
          deriv_matrix_index =
          computation->submatrices[deriv_submatrix_index].matrix_index;
      input_and_oderiv_matrices.insert(deriv_matrix_index);
    }
  }

  int32 num_matrices = computation->matrices.size();
  for (int32 m = 1; m < num_matrices; m++) {
    // We don't set up the matrices that are inputs to the computation;
    // this happens when the user provides the input.
    if (input_and_oderiv_matrices.count(m) == 0) {
      // get a submatrix index that refers to the entire matrix.
      int32 submatrix_index = whole_submatrices[m];

      computation->commands.push_back(
          NnetComputation::Command(kAllocMatrix, submatrix_index));
      // Later in the optimization phase, it turns out that zeroing is not
      // necessary for some matrices, we'll remove the redundant kSetConst
      // commands.
      computation->commands.push_back(
          NnetComputation::Command(0.0, kSetConst, submatrix_index));
    }
  }
}


void Compiler::SetUpPrecomputedIndexes(
    const std::vector<int32> &step_to_segment,
    NnetComputation *computation) {
  int32 num_steps = steps_.size();
  KALDI_ASSERT(computation->component_precomputed_indexes.empty());
  // the zeroth commponent is special, contains a NULL pointer.
  computation->component_precomputed_indexes.resize(1);
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &step_info = steps_[step];
    int32 node_index = step_info.node_index;
    const NetworkNode &node = nnet_.GetNode(node_index);
    // There is only something to do for nodes of type Component.
    if (node.node_type != kComponent)
      continue;
    const StepInfo &input_step_info = steps_[step - 1];
    int32 component_index = node.u.component_index;
    int32 input_node_index = input_step_info.node_index;
    KALDI_ASSERT(input_node_index == node_index - 1);
    const std::vector<Index> &input_indexes = input_step_info.output_indexes;
    const std::vector<Index> &output_indexes = step_info.output_indexes;

    const Component *component = nnet_.GetComponent(component_index);

    const ComputationRequest &request = *(requests_[step_to_segment[step]]);
    bool need_derivs = request.NeedDerivatives();
    ComponentPrecomputedIndexes *precomputed_indexes =
        component->PrecomputeIndexes(request.misc_info,
                                     input_indexes, output_indexes,
                                     need_derivs);
    if (precomputed_indexes == NULL) {
      // e.g. simple Components, and some other Components, will return NULL for
      // precomputed_indexes.
      step_info.precomputed_indexes_index = 0;
    } else {
      step_info.precomputed_indexes_index =
          computation->component_precomputed_indexes.size();

      NnetComputation::PrecomputedIndexesInfo info;
      info.data = precomputed_indexes;

      if (!input_indexes.empty() && input_indexes.back().n == 1 &&
          !output_indexes.empty() && output_indexes.back().n == 1) {
        // If these conditions are true, it's *possible* that we are doing
        // 'shortcut' compilation.  So just in case that's what's going on, we
        // store 'input_indexes' and 'output_indexes, which are needed by
        // the ExpandComputation() function that is used in that process.
        info.input_indexes = input_indexes;
        info.output_indexes = output_indexes;
      }
      computation->component_precomputed_indexes.push_back(info);
    }
  }
}

void Compiler::DeallocateMatrices(const std::vector<int32> &whole_submatrices,
                                  const std::vector<int32> &step_to_segment,
                                  NnetComputation *computation) {
  // This adds the commands to destroy all the matrices- but not the
  // ones that might be needed as outputs of the computation.  The ones that
  // are spared from destruction are those corresponding to outputs of the
  // computation, and those corresponding to input derivatives that were
  // requested by the user.
  int32 num_matrices = computation->matrices.size();
  std::vector<bool> will_destroy(num_matrices, true);

  int32 num_steps = steps_.size();
  for (int32 step = 0; step < num_steps; step++) {
    const StepInfo &step_info = steps_[step];
    const ComputationRequest &request = *(requests_[step_to_segment[step]]);
    if (nnet_.IsOutputNode(step_info.node_index)) {
      // steps corresponding to output nodes need to have their "value" kept.
      int32 value_matrix_index =
          computation->submatrices[step_info.value].matrix_index;
      will_destroy[value_matrix_index] = false;
    } else if (nnet_.IsInputNode(step_info.node_index)) {
      // steps corresponding to input nodes need to have their "deriv" kept, but
      // only if the corresponding input derivative was requested.  (we don't
      // need to worry about whether outputs were requested, because if they
      // were not requested we would not be computing them in the first place).
      std::string input_name = nnet_.GetNodeNames()[step_info.node_index];
      int32 i = 0, num_inputs = request.inputs.size();
      bool has_deriv = false;
      for (; i < num_inputs; i++) {
        if (input_name == request.inputs[i].name) {
          has_deriv = request.inputs[i].has_deriv;
          break;
        }
      }
      KALDI_ASSERT(i != num_inputs); // assert we found an input-request with
                                     // this name
      if (has_deriv) {
        int32 deriv_matrix_index =
          computation->submatrices[step_info.deriv].matrix_index;
        will_destroy[deriv_matrix_index] = false;
      }
    }
  }
  // note: matrix-index 0 is the empty matrix.
  for (int32 m = 1; m < num_matrices; m++) {
    if (will_destroy[m]) {
      int32 submatrix_index = whole_submatrices[m];
      computation->commands.push_back(
          NnetComputation::Command(kDeallocMatrix, submatrix_index));
    }
  }
}

void Compiler::OutputDebugInfo(NnetComputation *computation) const {
  int32 num_matrices = computation->matrices.size(),
      num_steps = steps_.size();
  computation->matrix_debug_info.resize(num_matrices);
  for (int32 step = 0; step < num_steps; step++) {
    const StepInfo &step_info = steps_[step];
    if (step_info.value == 0)
      continue;  // e.g. input step for ConstantComponent.
    if (!computation->IsWholeMatrix(step_info.value))
      continue;
    int32 value_matrix = computation->submatrices[step_info.value].matrix_index;
    int32 deriv_matrix = 0;
    if (step_info.deriv != 0 && computation->IsWholeMatrix(step_info.deriv))
      deriv_matrix = computation->submatrices[step_info.deriv].matrix_index;

    NnetComputation::MatrixDebugInfo &debug_info =
        computation->matrix_debug_info[value_matrix];
    debug_info.is_deriv = false;
    if (!debug_info.cindexes.empty()) {
      // This can happen if we created an alias for a node using a
      // dim-range-node that covers all the dimensions (would satisfy
      // IsWholeMatrix() above while not being a unique matrix).  We sometimes
      // do that to work around compiler constraints when creating expressions
      // that have the same quantity with more than one scaling value within the
      // same expression (like for computing deltas).
      continue;
    }
    AppendCindexes(step_info.node_index, step_info.output_indexes,
                   &debug_info.cindexes);
    if (deriv_matrix != 0) {
      NnetComputation::MatrixDebugInfo &deriv_debug_info =
          computation->matrix_debug_info[deriv_matrix];
      deriv_debug_info.is_deriv = true;
      deriv_debug_info.cindexes = debug_info.cindexes;
    }
  }
}

void AppendCindexes(int32 node, const std::vector<Index> &indexes,
                    std::vector<Cindex> *out) {
  size_t indexes_size = indexes.size();
  if (indexes_size > out->size())
    out->reserve(out->size() + indexes_size);
  for (size_t i = 0; i < indexes_size; i++)
    out->push_back(Cindex(node, indexes[i]));
}


} // namespace nnet3
} // namespace kaldi
