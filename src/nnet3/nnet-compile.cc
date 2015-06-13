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

namespace kaldi {
namespace nnet3 {

Compiler::Compiler(
    const ComputationRequest &request,
    const Nnet &nnet): request_(request), nnet_(nnet) { }


void Compiler::CreateComputation(NnetComputation *computation) {

  ComputeComputationGraph(nnet_, request_, &graph_);
  std::vector<bool> computable, required;
  ComputeComputableArray(nnet_, request_, graph_, &computable);
  PruneDependencies(nnet_, request_, computable, &graph_);
  ComputeRequiredArray(nnet_, graph_, computable, &required);
  if (!PruneComputationGraph(nnet_, computable, required, &graph_)) {
    // possible issue with graph topology, or not enough inputs provided.
    KALDI_ERR << "Computation cannot be done.";
  }
  PruneComputationGraph(nnet_, computable, required, &graph_);
  // see function declaration's comment for meaning of "by_order".
  std::vector<std::vector<int32> > by_order;
  ComputeComputationOrder(nnet_, graph_, NULL, &by_order);
  std::vector<std::vector<int32> > steps;
  ComputeComputationSteps(nnet_, request_, graph_, by_order, &steps);
  by_order.clear();
  CreateLocationInfo(steps);
  CreateStepInfo(&steps, computation);
  AddCommands(computation);
}

void Compiler::AddCommands(NnetComputation *computation) {
  DefineSubmatrices(computation);
  SetInputOutputInfo(computation);
  computation->need_model_derivative = request_.need_model_derivative;
  int32 arbitrary_factor = 8;
  computation->commands.reserve(computation->matrices.size()
                                * arbitrary_factor);
  SetUpMatrices(computation);
  int32 num_steps = steps_.size();
  for (int32 step = 0; step < num_steps; step++)
    DoForwardComputation(step, computation);
  // mark the end of the forward phase.
  computation->commands.push_back(
      NnetComputation::Command(NnetComputation::kNoOperationMarker));
  computation->forward_computation_end = computation->commands.size();
  if (request_.NeedDerivatives())
    for (int32 step = num_steps; step >= 0; step--)
      DoBackwardComputation(step, computation);
  DestroyMatrices(computation);
}

// Note: "by_step" is an input but is passed as a pointer because this
// function destroys it.
void Compiler::CreateStepInfo(
    std::vector<std::vector<int32> > *by_step,
    NnetComputation *computation) {
  KALDI_ASSERT(!by_step->empty());
  int32 num_steps = by_step->size();
  bool need_derivs = request_.NeedDerivatives();
  steps_.resize(num_steps);
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &this_info = steps_[step];
    this_info.output_cindex_ids.swap((*by_step)[step]);
    int32 num_ids = this_info.output_cindex_ids.size();
    this_info.output_indexes.resize(num_ids);
    for (int32 row_index = 0; row_index < num_ids; row_index++)
      this_info.output_indexes[row_index] =
          graph_.cindexes[this_info.output_cindex_ids[row_index]].second;
    KALDI_ASSERT(num_ids > 0);
    // node id's of all Cindexes are the same, so just use first one.
    this_info.node_index =
        graph_.cindexes[this_info.output_cindex_ids.front()].first;
    const NetworkNode &node = nnet_.GetNode(this_info.node_index);
    int32 num_rows = num_ids, num_cols = node.Dim(nnet_);
    
    if (node.node_type != NetworkNode::kDimRange) {    
      this_info.value = computation->NewMatrix(num_rows, num_cols);
      if (need_derivs)
        this_info.deriv = computation->NewMatrix(num_rows, num_cols);
    } else {
      // kDimRange.  Will just be a sub-matrix of a Component or Input node.
      int32 cindex_id = this_info.output_cindex_ids.front(),
          input_cindex_id = graph_.dependencies[cindex_id][0],
          input_step = cindex_id_to_location_[input_cindex_id].first;
      KALDI_ASSERT(input_step < step);
      this_info.value = computation->NewSubMatrix(steps_[input_step].value,
                                                  node.dim_offset, node.dim);
      if (need_derivs)
        this_info.value = computation->NewSubMatrix(steps_[input_step].value,
                                                    node.dim_offset, node.dim);
    }
  }
}

void Compiler::CreateLocationInfo(
    const std::vector<std::vector<int32> > &by_step) {
  KALDI_ASSERT(cindex_id_to_location_.empty());
  int32 num_steps = by_step.size();
  for (int32 step = 0; step < num_steps; step++) {
    const std::vector<int32> &output_cindex_ids = by_step[step];
    int32 num_rows = output_cindex_ids.size();
    for (int32 row = 0; row < num_rows; row++) {
      int32 cindex_id = output_cindex_ids[row];
      cindex_id_to_location_[cindex_id] = std::pair<int32,int32>(step, row);
    }
  }
}

void Compiler::DefineSubmatrices(NnetComputation *computation) {
  int32 num_steps = steps_.size();

  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &this_info = steps_[step];
    const NetworkNode &node = nnet_.GetNode(this_info.node_index);
    if (node.node_type == NetworkNode::kDescriptor) {
      const Descriptor &desc = node.descriptor;
      int32 num_parts = desc.NumParts();
      KALDI_ASSERT(num_parts > 0);
      if (num_parts == 1) {
        this_info.value_parts.push_back(this_info.value);
        if (this_info.deriv != 0)
          this_info.deriv_parts.push_back(this_info.deriv);
      } else { // num_parts > 1.
        int32 cur_dim_offset = 0;
        for (int32 part = 0; part < num_parts; part++) {
          // Have multiple parts, so need to set up sub-matrices.
          this_info.value_parts.resize(num_parts);
          if (this_info.deriv != 0)
            this_info.deriv_parts.resize(num_parts);
          for (int32 p = 0; p < num_parts; p++) {
            const SumDescriptor &this_part = desc.Part(p);
            int32 this_dim = this_part.Dim(nnet_);
            this_info.value_parts[p] =
                computation->NewSubMatrix(this_info.value,
                                          cur_dim_offset, this_dim);
            if (this_info.deriv != 0)
              this_info.deriv_parts[p] =
                  computation->NewSubMatrix(this_info.deriv,
                                            cur_dim_offset, this_dim);
            cur_dim_offset += this_dim;
          }
          KALDI_ASSERT(cur_dim_offset == desc.Dim(nnet_));
        }
      }
    }
  }
}


void Compiler::DoForwardComputation(int32 step,
                                    NnetComputation *computation) {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case NetworkNode::kInput: break;  // Nothing to do.
    case NetworkNode::kDescriptor:
      DoForwardComputationDescriptor(step, computation);
      break;
    case NetworkNode::kComponent:
      AddPropagateStep(step, computation);
      break;
    default:
      KALDI_ERR << "Invalid node type";
  }      
}


void Compiler::DoForwardComputationDescriptor(
    int32 step, NnetComputation *computation) {
  StepInfo &step_info = steps_[step];
  int32 num_parts = steps_[step].value_parts.size();
  step_info.submat_locations.resize(num_parts);
  for (int32 part = 0; part < num_parts; part++)
    DoForwardComputationSumDescriptor(step, part, computation);
  if (!request_.NeedDerivatives())
    step_info.submat_locations.clear();
}

void Compiler::ComputeSubmatLocationsList(
    int32 step, int32 part_index,
    const NnetComputation &computation,
    std::vector<std::vector<std::pair<int32, int32> > > *submat_locations)
    const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  const std::vector<Index> &output_indexes = step_info.output_indexes;
  const std::vector<int32> &output_cindex_ids = step_info.output_cindex_ids;
  const NetworkNode &node = nnet_.GetNode(step_info.node_index);
  const SumDescriptor &descriptor = node.descriptor.Part(part_index);
  bool is_only_part = (node.descriptor.NumParts() == 1);
  { // some checks.
    KALDI_ASSERT(node.node_type == NetworkNode::kDescriptor);
    int32 value_submatrix_index = step_info.value_parts[part_index];
    KALDI_ASSERT(descriptor.Dim(nnet_) ==
                 computation.sub_matrices[value_submatrix_index].num_cols);
  }
  int32 num_indexes = output_indexes.size();
  submat_locations->clear();
  submat_locations->resize(num_indexes);

  // This vector is indexed first by output row-index i (i.e. the index of
  // output_indexes or output_cindex_ids), and then is a list of input locations
  // for that row-index, sorted in the natural order of Cindexes.  The semantics
  // is that the i'th row of the output becomes a sum over the rows in the i'th
  // list (or zero if that list is empty).
  // These submat_locations will be pairs [submatrix-index, row-index].
  std::vector<std::vector<std::pair<int32, int32> > > input_submat_locations(
      num_indexes);
  for (int32 i = 0; i < num_indexes; i++) {
    int32 cindex_id = output_cindex_ids[i];
    const std::vector<int32> &dependencies = graph_.dependencies[cindex_id];

    std::vector<int32> input_cindex_ids;
    if (is_only_part) {
      // this is an optimization.
      input_cindex_ids = dependencies;
    } else {
      const Index &index = output_indexes[i];
      std::vector<Cindex> input_cindexes;
      CindexSet cindex_set(graph_);
      bool ans = descriptor.IsComputable(index, cindex_set, &input_cindexes);
      // earlier compilation stages should have checked that it is computable,
      // and the graph should still contain required inputs.
      KALDI_ASSERT(ans == true);
      std::sort(input_cindexes.begin(), input_cindexes.end());
      int32 size = input_cindexes.size();
      input_cindex_ids.resize(size);
      for (int32 j = 0; i < size; j++) {
        int32 c = graph_.GetCindexId(input_cindexes[j]);
        KALDI_ASSERT(c != -1);
        input_cindex_ids[i] = c;
      }
    }
    std::vector<std::pair<int32, int32> > &this_locations =
        input_submat_locations[i];
    int32 size = input_cindex_ids.size();
    this_locations.resize(size);
    for (int32 j = 0; j < size; j++) {
      std::pair<int32,int32> loc = cindex_id_to_location_[input_cindex_ids[j]];
      int32 input_step = loc.first, row_index = loc.second,
          submatrix_index = steps_[input_step].value;
      KALDI_ASSERT(input_step < step);
      this_locations[j].first = submatrix_index;
      this_locations[j].second = row_index;
    }
  }
}


void Compiler::DoForwardComputationSumDescriptor(
    int32 step,    
    int32 part_index,
    NnetComputation *computation) {
  StepInfo &step_info = steps_[step];
  // we store the submat_locations in step_info so that we can access it without
  // recomputing it in the backward phase.
  ComputeSubmatLocationsList(step, part_index, *computation,
                             &(step_info.submat_locations[part_index]));
  int32 value_submatrix_index = step_info.value_parts[part_index];
  DoForwardComputationFromSubmatLocationsList(
      value_submatrix_index,
      step_info.submat_locations[part_index],
      computation);
}

void Compiler::DoForwardComputationFromIndexes(
    int32 value_submatrix_index,
    int32 input_submatrix_index,    
    bool is_first_term_in_sum,
    const std::vector<int32> &indexes,
    NnetComputation *computation) const {
    
  int32 input_num_rows =
      computation->sub_matrices[input_submatrix_index].num_rows,
      num_rows = indexes.size();
  if (input_num_rows == num_rows) {
    int32 i;
    for (i = 0; i < num_rows; i++)
      if (indexes[i] != i)
        break;
    if (i == num_rows) {  // Simplest case: just matrix addition.
      NnetComputation::CommandType ctype =
          (is_first_term_in_sum ?
           NnetComputation::kMatrixCopy : NnetComputation::kMatrixAdd);
      computation->commands.push_back(
          NnetComputation::Command(ctype, input_submatrix_index,
                                   value_submatrix_index));
      return;
    }
  }
  // if we got to here, it's not just a case of matrix-copy or matrix-add,
  // but it's still from a single source matrix.
  int32 indexes_index = computation->indexes.size();
  computation->indexes.push_back(indexes);
  NnetComputation::CommandType ctype =
      (is_first_term_in_sum ?
       NnetComputation::kCopyRows : NnetComputation::kAddRows);
  computation->commands.push_back(
      NnetComputation::Command(ctype, input_submatrix_index,
                               value_submatrix_index, indexes_index));
  return;
}

void Compiler::DoForwardComputationFromSubmatLocations(
    int32 value_submatrix_index,
    bool is_first_term_in_sum,
    const std::vector<std::pair<int32, int32> > &submat_locations,        
    NnetComputation *computation) const {


  int32 input_submatrix_index = -1;
  std::vector<int32> indexes;

  if (ConvertToIndexes(submat_locations, &input_submatrix_index, &indexes)) {
    DoForwardComputationFromIndexes(value_submatrix_index,
                                    input_submatrix_index,
                                    is_first_term_in_sum,
                                    indexes,
                                    computation);
    return;
  } else {
    // There are multiple source matrices.
    NnetComputation::CommandType ctype =
        (is_first_term_in_sum ?
         NnetComputation::kCopyRowsMulti : NnetComputation::kAddRowsMulti);
    int32 indexes_multi_index = computation->indexes_multi.size();
    computation->indexes_multi.push_back(submat_locations);
    computation->commands.push_back(
        NnetComputation::Command(ctype, value_submatrix_index,
                                 indexes_multi_index));
    return;
  }
}

void Compiler::DoForwardComputationFromSubmatLocationsList(
    int32 value_submatrix_index,
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    NnetComputation *computation) const {
  std::vector<std::vector<std::pair<int32, int32> > > split_lists;  
  SplitLocations(submat_lists, &split_lists);
  int32 size = split_lists.size();
  KALDI_ASSERT(size > 0);
  for (int32 i = 0; i < size; i++)
    DoForwardComputationFromSubmatLocations(
        value_submatrix_index, (i == 0),
        split_lists[i],
        computation);
}



void Compiler::DoBackwardComputationSumDescriptor(
    int32 step, int32 part_index,
    NnetComputation *computation) const {

  const StepInfo &step_info = steps_[step];
  const std::vector<std::vector<std::pair<int32, int32> > >
      input_submat_locations = step_info.submat_locations[part_index];
  int32 deriv_submatrix_index = step_info.deriv_parts[part_index];
  DoBackwardComputationFromSubmatLocationsList(deriv_submatrix_index,
                                               input_submat_locations,
                                               computation);
}

void Compiler::DoBackwardComputationFromSubmatLocations(
    int32 deriv_submatrix_index,
    const std::vector<std::pair<int32, int32> > &submat_locations,        
    NnetComputation *computation) const {
  // This function creates a command to handle an individual piece of the
  // Descriptor, for backprop.  Note: because the backprop case is a little
  // trickier to implement efficiently on the GPU, there may be cases
  // which we will refuse to implement backprop for if we get here.
  
  int32 num_rows = submat_locations.size();
  std::vector<std::pair<int32, int32> >::const_iterator
      iter = submat_locations.begin(), end = submat_locations.end();
  int32 first_submat = iter->first;
  for (++iter; iter != end; ++iter)
    if (iter->first != first_submat)
      break;
  bool all_same_submatrix = (iter == end);
  if (all_same_submatrix) {
    int32 input_submatrix_index = first_submat;
    std::vector<int32> indexes(num_rows);
    for (int32 i = 0; i < num_rows; i++)
      indexes[i] = submat_locations[i].second;
    DoBackwardComputationFromIndexes(deriv_submatrix_index,
                                     input_submatrix_index,
                                     indexes,
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
          NnetComputation::Command(NnetComputation::kAddToRowsMulti,
                                   deriv_submatrix_index,
                                   indexes_multi_index));
      return;
    }
    // If you reach this point, there is a case that wasn't handled.  Our
    // intended strategy to handle it, if it's ever needed, is to create a
    // temporary matrix consisting of all the unique submat_locations in the
    // input.  We would first recurse to DoBackwardComputationFromIndexes, and
    // let it write to this temporary matrix; and then do the kAddToRowsMulti
    // command as above to go from the temporary matrix to the multiple
    // matrices.
    KALDI_ERR << "This case not handled.";
  }
}

// This function returns true if for each integer i, all the indexes j at which
// indexes[j] == i are consecutive with no gaps (more formally: if j1 < j2 < j3
// and indexes[j1] == indexes[j3], then indexes[j1] == indexes[j2]).  If so it
// also outputs to "reverse_indexes" the begin and end of these ranges, so that
// indexes[j] == i for all j such that (*reverse_indexes)[i].first <= j && j <
// (*reverse_indexes)[i].second.
static bool HasContiguousProperty(
    const std::vector<int32> &indexes,
    std::vector<std::pair<int32, int32> > *reverse_indexes) {
  int32 num_indexes = indexes.size(),
      num_input_indexes = *std::max_element(indexes.begin(), indexes.end()) + 1;
  reverse_indexes->resize(num_input_indexes);
  for (int32 i = 0; i < num_input_indexes; i++) {
    (*reverse_indexes)[i].first = -1;
    (*reverse_indexes)[i].second = -1;
  }
  // set each pair's "first" to the min index of all elements
  // of "indexes" with that value, and the "second" to the
  // max plus one.
  for (int32 i = 0; i < num_indexes; i++) {
    int32 j = indexes[i];
    KALDI_ASSERT(j >= 0);
    std::pair<int32, int32> &pair = (*reverse_indexes)[j];
    if (pair.first == -1) {
      pair.first = j;
      pair.second = j + 1;
    } else {
      pair.first = std::min(pair.first, j);
      pair.second = std::max(pair.second, j + 1);
    }
  }
  // check that the contiguous property holds.
  for (int32 i = 0; i < num_input_indexes; i++) {
    std::pair<int32, int32> pair = (*reverse_indexes)[i];
    if (pair.first != -1) {
      for (int32 j = pair.first; j < pair.second; j++)
        if (indexes[j] != i)
          return false;
    }
  }
  return true;
}

void Compiler::DoBackwardComputationFromIndexes(
    int32 deriv_submatrix_index,
    int32 input_deriv_submatrix_index,      
    const std::vector<int32> &indexes,
    NnetComputation *computation) const {
    
  int32 num_rows = computation->sub_matrices[deriv_submatrix_index].num_rows,
      input_num_rows =
      computation->sub_matrices[input_deriv_submatrix_index].num_rows;
  KALDI_ASSERT(indexes.size() == num_rows);
  if (input_num_rows == num_rows) {
    int32 i;
    for (i = 0; i < num_rows; i++)
      if (indexes[i] != i)
        break;
    if (i == num_rows) {  // Simplest case: just matrix addition.
        computation->commands.push_back(
            NnetComputation::Command(NnetComputation::kMatrixAdd,
                                     deriv_submatrix_index,
                                     input_deriv_submatrix_index));
      return;
    }
  }
  if (input_num_rows >= num_rows) {
    // If there are no repeated elements in the "indexes" array, we can
    // reverse the mapping and make it an operation of type kAddRows.
    std::vector<int32> reverse_indexes(input_num_rows, -1);
    int32 i;
    for (i = 0; i < num_rows; i++) {
      int32 index_i = indexes[i];
      KALDI_ASSERT(index_i >= 0 && index_i < input_num_rows);
      if (reverse_indexes[index_i] == -1)
        reverse_indexes[index_i] = i;
      else
        break;
    }
    if (i == num_rows) {
      // There were no repeated elements, and this strategy will work.
      int32 indexes_index = computation->indexes.size();
      computation->indexes.push_back(reverse_indexes);
        computation->commands.push_back(
            NnetComputation::Command(NnetComputation::kAddRows,
                                     deriv_submatrix_index,
                                     input_deriv_submatrix_index,
                                     indexes_index));
        return;
    }
  }
  std::vector<std::pair<int32, int32> > ranges;
  if (HasContiguousProperty(indexes, &ranges)) {
    // the operation can be set up as AddRowRanges.
    int32 indexes_multi_index = computation->indexes_multi.size();
    computation->indexes_multi.push_back(ranges);
    computation->commands.push_back(
        NnetComputation::Command(NnetComputation::kAddRowRanges,
                                 input_deriv_submatrix_index,
                                 deriv_submatrix_index,
                                 indexes_multi_index));
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
  

void Compiler::DoBackwardComputationDescriptor(
    int32 step, NnetComputation *computation) {
  StepInfo &step_info = steps_[step];
  // the top-level descriptor has a bunch of parts that we concatenate features
  // over.
  int32 num_parts = step_info.value_parts.size();
  for (int32 part = 0; part < num_parts; part++)
    DoBackwardComputationSumDescriptor(step, part,
                                       computation);
  step_info.submat_locations.clear();  // save memory.
}


void Compiler::DoBackwardComputation(int32 step,
                                     NnetComputation *computation) {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case NetworkNode::kInput: break;  // Nothing to do.
    case NetworkNode::kDescriptor:
      DoBackwardComputationDescriptor(step, computation);
      break;
    case NetworkNode::kComponent:
      AddBackpropStep(step, computation);
      break;
    default:
      KALDI_ERR << "Invalid node type";
  }      
}


void Compiler::AddPropagateStep(int32 step,
                                NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 input_step = step - 1;
  const StepInfo &input_step_info = steps_[input_step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);
  KALDI_ASSERT(node.node_type == NetworkNode::kComponent);  
  
  // in setting the following two variables, we use the fact that the submatrix
  // index of each submatrix that represents an entire matrix, is the same as
  // the matrix index of that matrix.
  int32 input_submatrix_index = input_step_info.value,
      output_submatrix_index = step_info.value;
  NnetComputation::Command c(NnetComputation::kPropagate,
                             node.u.component_index,
                             step_info.precomputed_indexes_index,
                             input_submatrix_index,
                             output_submatrix_index);
  computation->commands.push_back(c);
}


void Compiler::AddBackpropStep(int32 step,
                                         NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 input_step = step - 1;
  const StepInfo &input_step_info = steps_[input_step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);
  KALDI_ASSERT(node.node_type == NetworkNode::kComponent);
  int32 component_index = node.u.component_index;
  const Component *component = nnet_.GetComponent(component_index);  
  
  // in setting the following two variables, we use the fact that the submatrix
  // index of each submatrix that represents an entire matrix, is the same as
  // the matrix index of that matrix.
  int32 input_submatrix_index = input_step_info.value,
      output_submatrix_index = step_info.value,
      input_deriv_submatrix_index = input_step_info.deriv,
      output_deriv_submatrix_index = step_info.deriv;
  KALDI_ASSERT(input_deriv_submatrix_index > 0 &&
               output_deriv_submatrix_index > 0);
  if (! (component->Properties()&kBackpropNeedsInput))
    input_submatrix_index = 0;
  if (! (component->Properties()&kBackpropNeedsOutput))
    output_submatrix_index = 0;
  
  NnetComputation::Command c(NnetComputation::kBackprop,
                             node_index,
                             node.u.component_index,
                             step_info.precomputed_indexes_index,
                             input_submatrix_index,
                             output_submatrix_index,
                             input_deriv_submatrix_index,
                             output_deriv_submatrix_index);
  computation->commands.push_back(c);
}



void Compiler::SetUpMatrices(NnetComputation *computation) const {
  KALDI_ASSERT(computation->commands.empty());
  for (int32 m = 0; m < computation->matrices.size(); m++) {
    // Later in the optimization phase, it turns out that zeroing is not
    // necessary for some matrices, we'll turn these commands into
    // kResizeMatrixUndefined.
    NnetComputation::Command c(NnetComputation::kResizeMatrixZeroed, m);
    computation->commands.push_back(c);
  }
}


void Compiler::SetUpPrecomputedIndexes(
    NnetComputation *computation) {
  int32 num_steps = steps_.size();
  KALDI_ASSERT(computation->component_precomputed_indexes.empty());
  computation->component_precomputed_indexes.push_back(NULL);
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &step_info = steps_[step];
    int32 node_index = step_info.node_index;
    const NetworkNode &node = nnet_.GetNode(node_index);
    // There is only something to do for nodes of type Component.
    if (node.node_type != NetworkNode::kComponent)
      continue;
    const StepInfo &input_step_info = steps_[step - 1];
    int32 component_index = node.u.component_index;  
    int32 input_node_index = input_step_info.node_index;
    KALDI_ASSERT(input_node_index == node_index - 1);
    const std::vector<Index> &input_indexes = input_step_info.output_indexes;
    const std::vector<Index> &output_indexes = step_info.output_indexes;
    
    const Component *component = nnet_.GetComponent(component_index);

    bool need_derivs = request_.NeedDerivatives();
    ComponentPrecomputedIndexes *precomputed_indexes =
        component->PrecomputeIndexes(request_.misc_info,
                                     input_indexes, output_indexes,
                                     need_derivs);
    if (precomputed_indexes == NULL) {
      // e.g. simple Components, and some other Components, will return NULL for
      // precomputed_indexes.
      step_info.precomputed_indexes_index = 0;
    } else {
      step_info.precomputed_indexes_index =
          computation->component_precomputed_indexes.size();
      computation->component_precomputed_indexes.push_back(precomputed_indexes);
    }
  }
}


void Compiler::DestroyMatrices(NnetComputation *computation) {
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
    const NetworkNode &node = nnet_.GetNode(step_info.node_index);
    if (nnet_.IsOutput(step_info.node_index)) {
      // steps corresponding to output nodes need to have their "value" kept.
      will_destroy[step_info.value] = false;
    } else if (node.node_type == NetworkNode::kInput) {
      // steps corresponding to input nodes need to have their "deriv" kept, but
      // only if the corresponding input derivative was requested.  (we don't
      // need to worry about whether outputs were requested, because if they
      // were not requested we would not be computing them in the first place).
      std::string input_name = nnet_.GetNodeNames()[step_info.node_index];
      int32 i = 0, num_inputs = request_.inputs.size();
      bool has_deriv = false;
      for (; i < num_inputs; i++) {
        if (input_name == request_.inputs[i].name) {
          has_deriv = request_.inputs[i].has_deriv;
          break;
        }
      }
      KALDI_ASSERT(i != num_inputs); // assert we found an input-request with
                                     // this name
      if (has_deriv)
        will_destroy[step_info.deriv] = false;
    }
  }
  // note: matrix-index 0 is the empty matrix.
  for (int32 m = 1; m < num_matrices; m++)
    if (will_destroy[m])
      computation->commands.push_back(
          NnetComputation::Command(NnetComputation::kResizeMatrixEmpty, m));
}


} // namespace nnet3
} // namespace kaldi
