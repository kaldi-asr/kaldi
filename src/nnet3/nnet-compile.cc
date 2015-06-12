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
  CreateStepInfo(&steps);

  AddCommands(computation);
}

void Compiler::AddCommands(NnetComputation *computation) {
  DefineMatrices(computation);
  DefineSubmatrices(computation);
  SetInputOutputInfo(computation);
  computation->need_model_derivative = request_.need_model_derivative;
  int32 arbitrary_factor = 8;
  computation->commands.reserve(num_matrices_ * arbitrary_factor);
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

void Compiler::CreateStepInfo(
    std::vector<std::vector<int32> > *by_step) {
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
    this_info.node_index = graph_.cindexes[this_info.output_cindex_ids.front()].first;
    KALDI_ASSERT(this_info.node_index ==
                 graph_.cindexes[this_info.output_cindex_ids.back()].first);

    // set matrix indexes of value, and derivative if needed.  Matrix-index zero
    // is reserved for the empty matrix.
    if (need_derivs) {
      this_info.value = step * 2 + 1;
      this_info.deriv = step * 2 + 2;
    } else {
      this_info.value = step + 1;
    }
  }
  num_matrices_ = 1 + num_steps * (need_derivs ? 2 : 1);
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

// Adds to the computation object the information about the matrix sizes.
void Compiler::DefineMatrices(NnetComputation *computation) const {
  KALDI_ASSERT(computation->matrices.empty());
  computation->matrices.resize(num_matrices_);
  KALDI_ASSERT(num_matrices_ > 0);
  int32 num_steps = steps_.size();
  // matrix-index zero is reserved for the empty matrix.
  computation->matrices[0].num_rows = 0;
  computation->matrices[0].num_cols = 0;
  for (int32 step = 0; step < num_steps; step++) {
    const StepInfo &this_info = steps_[step];
    int32 num_rows = this_info.output_cindex_ids.size(),
        node_index = this_info.node_index,
        num_cols = nnet_.GetNode(node_index).Dim(nnet_);
    if (this_info.value != 0) {
      computation->matrices[this_info.value].num_rows = num_rows;
      computation->matrices[this_info.value].num_cols = num_cols;
    }
    if (this_info.deriv != 0) {
      computation->matrices[this_info.deriv].num_rows = num_rows;
      computation->matrices[this_info.deriv].num_cols = num_cols;
    }
  }
}

void Compiler::DefineSubmatrices(NnetComputation *computation) {
  // First add to the computation all the sub-matrix indexes that correspond to
  // an entire matrix.
  KALDI_ASSERT(computation->sub_matrices.empty());
  int32 num_matrices = computation->matrices.size();
  computation->sub_matrices.resize(num_matrices);
  for (size_t m = 0; m < num_matrices; m++) {
    NnetComputation::MatrixInfo &matrix_info = computation->matrices[m];
    NnetComputation::SubMatrixInfo &submatrix_info = computation->sub_matrices[m];
    submatrix_info.matrix_index = m;
    submatrix_info.row_offset = 0;
    submatrix_info.num_rows = matrix_info.num_rows;
    submatrix_info.col_offset = 0;
    submatrix_info.num_cols = matrix_info.num_cols;
  }
  // check index zero is empty.
  KALDI_ASSERT(computation->sub_matrices[0].num_rows == 0 &&
               computation->sub_matrices[0].num_cols == 0);

  int32 num_steps = steps_.size();
  // Now set up sub-matrices for matrices that are a concatenation over multiple
  // parts.
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
            const SumDescriptor *this_part = desc.Part(p);
            int32 this_dim = this_part->Dim(nnet_);
            int32 num_rows = this_info.output_indexes.size();
            int32 value_part_index = computation->sub_matrices.size();
            computation->sub_matrices.push_back(
                NnetComputation::SubMatrixInfo(this_info.value, 0, num_rows,
                                               cur_dim_offset, this_dim));
            this_info.value_parts[p] = value_part_index;
            if (this_info.deriv != 0) {
              int32 deriv_part_index = computation->sub_matrices.size();
              computation->sub_matrices.push_back(
                  NnetComputation::SubMatrixInfo(this_info.deriv, 0, num_rows,
                                                 cur_dim_offset, this_dim));
              this_info.deriv_parts[p] = deriv_part_index;
            }
            cur_dim_offset += this_dim;
          }
          KALDI_ASSERT(cur_dim_offset == desc.Dim(nnet_));
        }
      }
    }
  }
}


void Compiler::DoForwardComputation(int32 step,
                                              NnetComputation *computation) const {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case NetworkNode::kInput: break;  // Nothing to do.
    case NetworkNode::kDescriptor:
      DoForwardComputationDescriptor(step, node.descriptor, computation);
      break;
    case NetworkNode::kComponent:
      AddPropagateStep(step, computation);
      break;
    default:
      KALDI_ERR << "Invalid node type";
  }      
}


void Compiler::DoForwardComputationDescriptor(
    int32 step, const Descriptor &descriptor,
    NnetComputation *computation) const {
  const StepInfo &step_info = steps_[step];
  // the top-level descriptor has a bunch of parts that we concatenate features over.
  int32 num_parts = descriptor.NumParts();
  KALDI_ASSERT(num_parts == step_info.value_parts.size());
  for (int32 part = 0; part < num_parts; part++) {
    const SumDescriptor &sum_descriptor = descriptor.Part(part);
    int32 value_submatrix_index = step_info.value_parts[part];
    DoForwardComputationSumDescriptor(step,
                                      value_submatrix_index,
                                      (num_parts == 1),
                                      sum_descriptor,
                                      computation);
  }      
}

void Compiler::DoForwardComputationSumDescriptor(
    int32 step,    
    int32 value_submatrix_index,
    bool is_only_part,
    const SumDescriptor &descriptor,
    NnetComputation *computation) const {
  const StepInfo &step_info = steps_[step];
  const std::vector<Index> &output_indexes = step_info.output_indexes;
  const std::vector<int32> &output_cindex_ids = step_info.output_cindex_ids;
  KALDI_ASSERT(descriptor.Dim(nnet_) ==
               computation->sub_matrices[value_submatrix_index].num_cols);

  int32 num_indexes = output_indexes.size();
  
  // This vector is indexed first by output row-index i (i.e. the index of
  // output_indexes or output_cindex_ids), and then is a list of input locations
  // for that row-index, sorted in the natural order of Cindexes.  The semantics
  // is that the i'th row of the output becomes a sum over the rows in the i'th
  // list (or zero if that list is empty).
  // Note: these submat_locations will be pairs [submatrix-index, row-index]
  // rather than the "locations" [step-index, row-index].
  std::vector<std::vector<std::pair<int32, int32> > > input_submat_locations(
      num_indexes);
  for (int32 i = 0; i < num_indexes; i++) {
    int32 cindex_id = output_cindex_ids[i];
    const std::vector &dependencies = graph.dependencies[cindex_id];

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
  DoForwardComputationFromSubmatLocationsList(value_submatrix_index,
                                              is_first_term_in_sum,
                                              input_submat_locations,
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
  // First work out if all the input submatrix indexes are the same (i.e. there
  // is only one source).
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

void DoForwardComputationFromSubmatLocationsList(
    int32 value_submatrix_index,
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    NnetComputation *computation) const {
  
}



void Compiler::DoBackwardComputationForwardingDescriptor(
    int32 step,    
    int32 deriv_submatrix_index,    
    const ForwardingDescriptor &descriptor,
    NnetComputation *computation) const {
  const StepInfo &step_info = steps_[step];
  const std::vector<Index> &output_indexes = step_info.output_indexes;
  KALDI_ASSERT(descriptor.Dim(nnet_) ==
               computation->sub_matrices[deriv_submatrix_index].num_cols);
  
  // Note: these submat_locations will be pairs [submatrix-index, row-index]
  // rather than the more normal [step-index, row-index].
  std::vector<std::pair<int32, int32> > input_submat_locations(output_indexes.size());
  int32 num_indexes = output_indexes.size();
  for (int32 i = 0; i < num_indexes; i++) {
    const Index &index = output_indexes[i];
    Cindex input_cindex = descriptor.MapToInput(index);
    // The following call to GetCindexId will crash if the Cindex is not present
    // in computation graph.  That would be a bug anyway, so a crash is what we
    // want.
    int32 cindex_id = graph_.GetCindexId(input_cindex);
    std::pair<int32, int32> location = cindex_id_to_location_[cindex_id];
    int32 input_step = location.first, row_index = location.second,
        submatrix_index = steps_[input_step].value;
    input_submat_locations[i].first = submatrix_index;
    input_submat_locations[i].second = row_index;
  }
  DoBackwardComputationFromSubmatLocations(deriv_submatrix_index,
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
    int32 step, const Descriptor &descriptor,
    NnetComputation *computation) const {
  const StepInfo &step_info = steps_[step];
  // the top-level descriptor has a bunch of parts that we concatenate features over.
  KALDI_ASSERT(descriptor.parts.size() == step_info.value_parts.size());
  int32 num_parts = descriptor.parts.size();
  for (int32 part = 0; part < num_parts; part++) {
    const SumDescriptor &sum_descriptor = descriptor.parts[part];
    int32 deriv_submatrix_index = step_info.deriv_parts[part];
    KALDI_ASSERT(deriv_submatrix_index > 0);
    int32 num_terms = sum_descriptor.terms.size();
    for (int32 term = 0; term < num_terms; term++) {
      const ForwardingDescriptor &forwarding_descriptor =
          sum_descriptor.terms[term];
      DoBackwardComputationForwardingDescriptor(step,
                                                deriv_submatrix_index,
                                                forwarding_descriptor,
                                                computation);
    }
  }      
}


void Compiler::DoBackwardComputation(int32 step,
                                               NnetComputation *computation) const {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case NetworkNode::kInput: break;  // Nothing to do.
    case NetworkNode::kDescriptor:
      DoBackwardComputationDescriptor(step, node.descriptor, computation);
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
  int32 input_step = step_info.input_step;
  KALDI_ASSERT(input_step < step);
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
  int32 input_step = step_info.input_step;
  KALDI_ASSERT(input_step < step);
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
  for (int32 m = 0; m < num_matrices_; m++) {
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
    int32 input_step = step_info.input_step;
    KALDI_ASSERT(static_cast<size_t>(input_step) < steps_.size());
    const StepInfo &input_step_info = steps_[input_step];
    int32 node_index = step_info.node_index;
    const NetworkNode &node = nnet_.GetNode(node_index);
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
