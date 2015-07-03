// nnet3/nnet-analyze.cc

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

#include "nnet3/nnet-analyze.h"

namespace kaldi {
namespace nnet3 {

void ComputationVariables::ComputeSplitPoints(
    const NnetComputation &computation) {
  // note, these numbers are only valid if you include the empty zero-indexed
  // matrix/submatrix as a matrix.
  int32 num_matrices = computation.matrices.size(),
      num_submatrices = computation.submatrices.size();
  split_points_.resize(num_matrices);
  KALDI_ASSERT(computation.submatrices[0].num_rows == 0);
  for (int32 submatrix_index = 1;
       submatrix_index < num_submatrices;
       submatrix_index++) {
    const NnetComputation::SubMatrixInfo &s =
        computation.submatrices[submatrix_index];
    split_points_[s.matrix_index].push_back(s.col_offset);
    split_points_[s.matrix_index].push_back(s.col_offset + s.num_cols);
  }
  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    SortAndUniq(&(split_points_[matrix_index]));
    // should have at least 0 and num_rows included, so size >= 2.
    KALDI_ASSERT(split_points_[matrix_index].size() >= 2);
  }
  // note: the last split point of each matrix doesn't get its own variable index.
  matrix_to_variable_index_.resize(num_matrices + 1);
  matrix_to_variable_index_[0] = 0;
  matrix_to_variable_index_[1] = 0;
  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    int32 num_variables = split_points_[matrix_index].size();
    KALDI_ASSERT(num_variables >= 1);
    matrix_to_variable_index_[matrix_index+1] =
        matrix_to_variable_index_[matrix_index] + num_variables;
  }
  num_variables_ = matrix_to_variable_index_.back();
}

void ComputationVariables::ComputeVariableRanges(
    const NnetComputation &computation) {
  // note, these numbers are only valid if you include the empty zero-indexed
  // matrix/submatrix as a matrix.
  int32 num_submatrices = computation.submatrices.size();

  variable_ranges_.resize(num_submatrices);
  variable_ranges_[0] = std::pair<int32,int32>(0, 0);

  full_row_range_.resize(num_submatrices);
  submatrix_to_matrix_.resize(num_submatrices);
  submatrix_to_matrix_[0] = 0;
  
  for (int32 submatrix_index = 1;
       submatrix_index < num_submatrices;
       submatrix_index++) {
    const NnetComputation::SubMatrixInfo &s =
        computation.submatrices[submatrix_index];
    int32 matrix_index = s.matrix_index;
    submatrix_to_matrix_[submatrix_index] = matrix_index;
    int32 start_dim = s.col_offset, end_dim = start_dim + s.num_cols;
    const std::vector<int32> &split = split_points_[matrix_index];
    // std::lower_bound does a binary search -> faster than std::find.
    std::vector<int32>::const_iterator iter = std::lower_bound(
        split.begin(), split.end(), start_dim);
    KALDI_ASSERT(*iter == start_dim);  // or code error.
    int32 start_split_point_index = iter - split.begin();
    iter = std::lower_bound(iter, split.end(), end_dim);
    KALDI_ASSERT(*iter == end_dim);  // or code error.
    int32 end_split_point_index = iter - split.begin();
    int32 matrix_offset = matrix_to_variable_index_[matrix_index];
    int32 start_variable_index = matrix_offset + start_split_point_index,
        end_variable_index = matrix_offset + end_split_point_index;
    KALDI_ASSERT(end_variable_index > start_variable_index);
    variable_ranges_[submatrix_index].first = start_variable_index;
    variable_ranges_[submatrix_index].second = end_variable_index;
    full_row_range_[submatrix_index] =
        (s.row_offset == 0 && s.num_rows ==
         computation.matrices[matrix_index].num_rows);
  }
}

ComputationVariables::ComputationVariables(const NnetComputation &computation) {
  ComputeSplitPoints(computation);
  ComputeVariableRanges(computation);
}

void ComputationVariables::AppendVariablesForSubmatrix(
    int32 submatrix_index,
    std::vector<int32> *variable_indexes) const {
  KALDI_ASSERT(static_cast<size_t>(submatrix_index) < variable_ranges_.size());
  int32 start = variable_ranges_[submatrix_index].first,
      end = variable_ranges_[submatrix_index].second;
  for (int32 variable_index = start; variable_index < end; variable_index++)
    variable_indexes->push_back(variable_index);
}

void ComputationVariables::AppendVariablesForMatrix(
    int32 matrix_index,
    std::vector<int32> *variable_indexes) const {
  KALDI_ASSERT(static_cast<size_t>(matrix_index + 1) <
               matrix_to_variable_index_.size());
  int32 start = matrix_to_variable_index_[matrix_index],
      end = matrix_to_variable_index_[matrix_index + 1];

  for (int32 variable_index = start; variable_index < end; variable_index++)
    variable_indexes->push_back(variable_index);
}

void ComputationVariables::RecordAccessForSubmatrix(
    int32 submatrix_index,
    AccessType access_type,
    CommandAttributes *ca) const {
  switch (access_type) {
    case kReadAccess:
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_read));
      break;
    case kWriteAccess:
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_written));
      // if submatrix does not span the full row range of the matrix,
      // a write operation has to be considered a read/write operation
      // on the underlying variable.
      if (!full_row_range_[submatrix_index])
        AppendVariablesForSubmatrix(submatrix_index,
                                    &(ca->variables_read));
      break;
    case kReadWriteAccess:
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_written));
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_read));
  }
  ca->matrices_accessed.push_back(submatrix_to_matrix_[submatrix_index]);
}



/// given a vector of pairs from computation.indexes_multi_indexes
/// containing paris (submatrix-index, row-index), this function outputs
/// to "submatrix_indexes" all (unique) submatrix indexes that appear;
/// and it outputs to "contains_null_marker" true if the pair (-1, -1)
/// appears anywhere in indexes_multi, and false otherwise.
static void IndexesMultiToSubmatrixIndexes(
    const std::vector<std::pair<int32, int32> > &indexes_multi,
    std::vector<int32> *submatrix_indexes) {
  submatrix_indexes->clear();
  std::vector<std::pair<int32, int32> >::const_iterator
      iter = indexes_multi.begin(), end = indexes_multi.end();
  int32 cur_submatrix_index = -1; // an optimization.
  for (; iter != end; ++iter) {
    int32 submatrix_index = iter->first;
    if (submatrix_index != -1 && submatrix_index != cur_submatrix_index) {
      cur_submatrix_index = submatrix_index;
      submatrix_indexes->push_back(submatrix_index);
    }
  }
  SortAndUniq(submatrix_indexes);
}


void ComputeCommandAttributes(
    const Nnet &nnet,
    const NnetComputation &computation,
    const ComputationVariables &vars,
    std::vector<CommandAttributes> *attributes) {
  int32 num_commands = computation.commands.size();
  attributes->clear();
  attributes->resize(num_commands);
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    const NnetComputation::Command &c = computation.commands[command_index];
    CommandAttributes &attr = (*attributes)[command_index];
    switch (c.command_type) {
      case NnetComputation::kResizeMatrixZeroed:
        vars.AppendVariablesForMatrix(c.arg1, &attr.variables_written);
        break;
      case NnetComputation::kResizeMatrixUndefined: // nothing is written here. 
        break;
      case NnetComputation::kResizeMatrixEmpty: // ditto.
        break;
      case NnetComputation::kPropagate:
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        if (nnet.GetComponent(c.arg1)->Properties() & kPropagateAdds)
          vars.RecordAccessForSubmatrix(c.arg3, kReadWriteAccess, &attr);
        else
          vars.RecordAccessForSubmatrix(c.arg3, kWriteAccess, &attr);        
        break;
      case NnetComputation::kBackprop:
        vars.RecordAccessForSubmatrix(c.arg4, kReadAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg5, kReadAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg6, kReadAccess, &attr);
        if (nnet.GetComponent(c.arg1)->Properties() & kBackpropAdds)      
          vars.RecordAccessForSubmatrix(c.arg7, kReadWriteAccess, &attr);
        else
          vars.RecordAccessForSubmatrix(c.arg7, kWriteAccess, &attr);        
        if (nnet.GetComponent(c.arg2)->Properties() & kUpdatableComponent)
          attr.has_side_effects = true;
        break;
      case NnetComputation::kMatrixCopy:
        vars.RecordAccessForSubmatrix(c.arg1, kWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;
      case NnetComputation::kMatrixAdd:      
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;
      case NnetComputation::kAddRows:
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;      
      case NnetComputation::kCopyRows: {
        const std::vector<int32> &indexes = computation.indexes[c.arg3];
        // if there are -1's in "indexes", then the result of the operation
        // will depend on the initial value of the matrix, so it's
        // a "rw" operation, not a "write" operation.
        if (std::count(indexes.begin(), indexes.end(), -1) > 0)
          vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        else
          vars.RecordAccessForSubmatrix(c.arg1, kWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;
      }
      case NnetComputation::kAddRowsMulti: {
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        std::vector<int32> submatrix_indexes;
        IndexesMultiToSubmatrixIndexes(computation.indexes_multi[c.arg2],
                                       &submatrix_indexes);
        for (size_t i = 0; i < submatrix_indexes.size(); i++)
          vars.RecordAccessForSubmatrix(submatrix_indexes[i],
                                        kReadAccess, &attr);
        break;
      }
      case NnetComputation::kCopyRowsMulti: {
        std::vector<int32> submatrix_indexes;
        IndexesMultiToSubmatrixIndexes(computation.indexes_multi[c.arg2],
                                       &submatrix_indexes);
        // note: the CopyRows command assigns zero in cases where
        // there is no source for some row
        vars.RecordAccessForSubmatrix(c.arg1, kWriteAccess, &attr);
        for (size_t i = 0; i < submatrix_indexes.size(); i++)
          vars.RecordAccessForSubmatrix(submatrix_indexes[i],
                                        kReadAccess, &attr);
        break;
      }
      case NnetComputation::kAddToRowsMulti:
      case NnetComputation::kCopyToRowsMulti: {
        vars.RecordAccessForSubmatrix(c.arg1, kReadAccess, &attr);
        // if the submatrixes we're writing to (in kCopyToRowsMulti) had all
        // rows covered, it would be a pure write operation.
        std::vector<int32> submatrix_indexes;
        IndexesMultiToSubmatrixIndexes(computation.indexes_multi[c.arg2],
                                       &submatrix_indexes);
        for (size_t i = 0; i < submatrix_indexes.size(); i++)
          vars.RecordAccessForSubmatrix(submatrix_indexes[i], kReadWriteAccess,
                                        &attr);
        break;
      }
      case NnetComputation::kAddRowRanges: {
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
      }
      case NnetComputation::kNoOperation:
      case NnetComputation::kNoOperationMarker:
        break;
      default:
        KALDI_ERR << "Unknown command type.";
    }
  }
}

void ComputeVariableAccesses(
    const ComputationVariables &variables,
    const std::vector<CommandAttributes> &command_attributes,
    std::vector<VariableAccesses> *variable_accesses) {

}
        

ComputationChecker::ComputationChecker(
    const CheckComputationConfig &config,
    const Nnet &nnet,
    const ComputationRequest &request,
    const NnetComputation &computation):
    config_(config), nnet_(nnet), request_(request),
    computation_(computation), variables_(computation) {
}    


void ComputationChecker::Check() {
  CheckComputationIndexes();
  ComputeCommandAttributes(nnet_, computation_,
                           variables_, &attributes_);
  CheckComputationOrder();
  CheckComputationAllocation();
  CheckComputationUndefined();  
  if (config_.check_rewrite)
    CheckComputationRewrite();
  
  
}

/**
   Checks for accessing undefined variables: basically, that variables
   are always written to (e.g. zeroed) before being read, so it's never the case
   that we are reading an undefined value.
*/
void ComputationChecker::CheckComputationUndefined() const {
  KALDI_ERR << "todo";  
}


/**
   Checks for the situation where a read-only operation on a variable is
   followed by an operation that writes to the variable.  This should never
   occur prior to optimization, but after certain optimization we in effect
   "re-use" variables by doing things like propagate and backprop in-place, so
   this check shouldn't be performed after optimization.
*/
void ComputationChecker::CheckComputationRewrite() const {
  KALDI_ERR << "todo";
}


/**
   Checks that we never use variables before they
   are allocated or after they are deallocated.
*/
void ComputationChecker::CheckComputationAllocation() const {
  KALDI_ERR << "todo";  
}

/**
   This very basic check just makes sure that all indexes in the commands are
   within range, that dimensions agree with the request, that row/column dimensions
   agree with component dimensions.
*/
void ComputationChecker::CheckComputationIndexes() const {
  int32 num_commands = computation_.commands.size(),
      num_matrices = computation_.matrices.size(),
      num_submatrices = computation_.submatrices.size();
  const std::vector<NnetComputation::SubMatrixInfo> &submatrices =
      computation_.submatrices;
  
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    const NnetComputation::Command &c = computation_.commands[command_index];
    switch (c.command_type) {
      case NnetComputation::kResizeMatrixZeroed:
      case NnetComputation::kResizeMatrixUndefined:
      case NnetComputation::kResizeMatrixEmpty:
        if (c.arg1 < 1 || c.arg1 >= num_matrices)
          KALDI_ERR << "matrix index out of range.";
        break;
      case NnetComputation::kPropagate: {
        if (c.arg1 < 0 || c.arg1 >= nnet_.NumComponents())
          KALDI_ERR << "Component index out of range";
        const Component *component = nnet_.GetComponent(c.arg1);
        int32 properties = component->Properties();
        if (c.arg2 < 0 ||
            c.arg2 > computation_.component_precomputed_indexes.size())
          KALDI_ERR << "Precomputed-indexes index out of range";
        if (c.arg2 != 0 && (properties & kSimpleComponent))
          KALDI_ERR << "Precomputed-indexes index nonzero for simple component";
        // note: input may be the empty matrix (in unusual circumstances, for non-simple
        // components).
        if (c.arg3 < 0 || c.arg3 >= num_submatrices ||
            (c.arg3 == 0 && !(properties & kSimpleComponent)) ||
            c.arg4 < 1 || c.arg4 >= num_submatrices)
          KALDI_ERR << "Sub-matrix indexes out of range.";
        if (submatrices[c.arg3].num_cols != component->InputDim())
          KALDI_ERR << "Input-dim mismatch.";
        if (submatrices[c.arg4].num_cols != component->OutputDim())
          KALDI_ERR << "Input-dim mismatch.";
        if ((properties & kSimpleComponent) &&
            submatrices[c.arg3].num_rows !=
            submatrices[c.arg4].num_rows)
          KALDI_ERR << "Num-rows mismatch for simple component.";
        if (!(properties & kPropagateInPlace) &&
            c.arg3 == c.arg4)
          KALDI_ERR << "In-place propagation not supported for this component";
        break;
      }
      case NnetComputation::kBackprop: {
        if (c.arg1 < 0 || c.arg1 >= nnet_.NumNodes())
          KALDI_ERR << "Node index in backprop out of range";
        if (c.arg2 < 0 || c.arg2 >= nnet_.NumComponents());
        KALDI_ERR << "Component index in backprop out of range";
        const Component *component = nnet_.GetComponent(c.arg2);
        int32 properties = component->Properties();
        if (c.arg3 < 0 ||
            c.arg3 > computation_.component_precomputed_indexes.size())
          KALDI_ERR << "Precomputed-indexes index out of range";
        if (c.arg3 != 0 && (properties & kSimpleComponent))
          KALDI_ERR << "Precomputed-indexes index nonzero for simple component";
        // output-deriv (arg6) must be supplied; others could plausibly be zero.
        if (c.arg4 < 0 || c.arg4 >= num_submatrices ||
            c.arg5 < 0 || c.arg5 >= num_submatrices ||
            c.arg6 < 1 || c.arg6 >= num_submatrices ||
            c.arg7 < 0 || c.arg7 >= num_submatrices)
          KALDI_ERR << "Submatrix index out of range for backprop.";
        if ((properties & kBackpropNeedsInput) && c.arg4 == 0)
          KALDI_ERR << "Backprop input needed but not supplied.";
        if ((properties & kBackpropNeedsOutput) && c.arg5 == 0)
          KALDI_ERR << "Backprop output needed but not supplied.";
        if (c.arg7 == 0 && !(properties && kUpdatableComponent)) {
          // note: we could perhaps make this just a warning,
          // or optimize it away somehow.
          KALDI_ERR << "Backprop is done but has no effect.";
        }
        if (c.arg6 == c.arg7 && !(properties & kBackpropInPlace))
          KALDI_ERR << "In-place backprop used where not supported.";
        if (c.arg4 != 0 &&
            submatrices[c.arg4].num_cols != component->InputDim())
          KALDI_ERR << "Input-dim mismatch in backprop.";
        if (c.arg5 != 0 &&
            submatrices[c.arg5].num_cols != component->OutputDim())
          KALDI_ERR << "Output-dim mismatch in backprop.";
        if (c.arg6 != 0 &&
            submatrices[c.arg6].num_cols != component->OutputDim())
          KALDI_ERR << "Output-dim mismatch in backprop.";
        if (c.arg7 != 0 &&
            submatrices[c.arg7].num_cols != component->InputDim())
          KALDI_ERR << "Input-dim mismatch in backprop.";
        // check num-rows consistency for input.
        if (c.arg4 != 0 && c.arg7 != 0 &&
            submatrices[c.arg4].num_rows != submatrices[c.arg7].num_rows)
          KALDI_ERR << "Num-rows mismatch in backprop input";
        // check num-rows consistency for output
        if (c.arg5 != 0 ||
            submatrices[c.arg5].num_rows != submatrices[c.arg6].num_rows)
          KALDI_ERR << "Num-rows mismatch in backprop output";
        if ((properties & kSimpleComponent) && c.arg7 != 0 &&
            submatrices[c.arg6].num_rows != submatrices[c.arg7].num_rows)
          KALDI_ERR << "Num-rows mismatch in backprop input vs output.";
        break;
      }
      case NnetComputation::kMatrixCopy:
      case NnetComputation::kMatrixAdd:
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            c.arg2 < 1 || c.arg2 >= num_submatrices)
          KALDI_ERR << "Submatrix indexes out of range in matrix copy/add";
        if (submatrices[c.arg1].num_rows != submatrices[c.arg2].num_rows ||
            submatrices[c.arg1].num_cols != submatrices[c.arg2].num_cols)
          KALDI_ERR << "Submatrix indexes out of range in matrix copy/add";
        if (c.arg1 == c.arg2)
          KALDI_ERR << "Adding/copying to self";
        break;
      case NnetComputation::kAddRows:
      case NnetComputation::kCopyRows: {
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            c.arg2 < 1 || c.arg2 >= num_submatrices ||
            static_cast<size_t>(c.arg3) >= computation_.indexes.size())
          KALDI_ERR << "Index out of range in add-rows/copy-rows command.";
        const std::vector<int32> &indexes = computation_.indexes[c.arg3];
        if (indexes.size() != static_cast<size_t>(submatrices[c.arg1].num_rows))
          KALDI_ERR << "Indexes size mismatch in add-rows/copy-rows";
        if (submatrices[c.arg1].num_cols != submatrices[c.arg2].num_cols)
          KALDI_ERR << "Dimension mismatch in add-rows/copy-rows";
        if (*std::max_element(indexes.begin(), indexes.end()) >=
            submatrices[c.arg2].num_rows)
          KALDI_ERR << "Row-index out of range in add-rows/copy-rows";
        if (c.arg1 == c.arg2)
          KALDI_ERR << "Copying to self in add-rows/copy-rows command.";
        break;
      }
      case NnetComputation::kAddRowsMulti:
      case NnetComputation::kCopyRowsMulti:
      case NnetComputation::kAddToRowsMulti:
      case NnetComputation::kCopyToRowsMulti: {
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            static_cast<size_t>(c.arg2) >= computation_.indexes_multi.size())
          KALDI_ERR << "Index out of range in *-multi command";
        const std::vector<std::pair<int32, int32> > pairs =
            computation_.indexes_multi[c.arg2];
        int32 num_rows = submatrices[c.arg1].num_rows,
            num_cols =  submatrices[c.arg1].num_cols;
        if (pairs.size() != static_cast<size_t>(num_rows))
          KALDI_ERR << "Indexes dimension mismatch in *-multi command";
        std::vector<std::pair<int32, int32> >::const_iterator
            iter = pairs.begin(), end = pairs.end();
        for (; iter != end; ++iter) {
          int32 submatrix_index = iter->first, row_index = iter->second;
          if (submatrix_index == -1) {
            if (row_index != -1)
              KALDI_ERR << "Expected -1 row index if submatrix index is -1";
          } else {
            if (submatrix_index < 1 || submatrix_index >= num_submatrices)
              KALDI_ERR << "Submatrix index out of range in indexes_multi";
            if (row_index < 0 ||
                row_index >= submatrices[submatrix_index].num_rows)
              KALDI_ERR << "Row index out of range in indexes_multi";
            if (submatrix_index == c.arg1)
              KALDI_ERR << "Copying from self in *-multi command.";
            if (submatrices[submatrix_index].num_cols != num_cols)
              KALDI_ERR << "Mismatching dimension in *-multi command";
          }
        }
        if (c.command_type == NnetComputation::kAddToRowsMulti ||
            c.command_type == NnetComputation::kCopyToRowsMulti) {
          // check for duplicates; these are not allowed in kAddToRowsMulti
          // or kCopyToRowsMulti because they would necessitate extra work
          // in CUDA kernels.
          std::vector<std::pair<int32, int32> > pairs_copy(pairs);
          std::sort(pairs_copy.begin(), pairs_copy.end());
          std::vector<std::pair<int32, int32> >::const_iterator
              iter = pairs_copy.begin(), end = pairs_copy.end(),
              next_iter;
          for (; iter != end; ++iter) {
            next_iter = iter;
            ++next_iter;
            if (next_iter != end && *iter == *next_iter &&
                iter->first != -1) {
              KALDI_ERR << "Duplicate element "
                        << iter->first << ',' << iter->second << " found in "
                        << "indexes for {add,copy}-to-rows-multi command.";
            }
          }
        }
        break;
      }
      case NnetComputation::kAddRowRanges: {
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            c.arg2 < 1 || c.arg2 >= num_submatrices ||
            static_cast<size_t>(c.arg3) >= computation_.indexes_multi.size())          
          KALDI_ERR << "Index out of range in add-row-ranges command";
        const std::vector<std::pair<int32, int32> > pairs =
            computation_.indexes_multi[c.arg2];
        if (static_cast<size_t>(submatrices[c.arg1].num_rows) != pairs.size())
          KALDI_ERR << "Num-rows mismatch in add-row-ranges command";
        if (submatrices[c.arg1].num_cols != submatrices[c.arg2].num_cols)
          KALDI_ERR << "Dimension mismatch in add-row-ranges command";
        int32 src_num_rows = submatrices[c.arg2].num_rows;
        std::vector<std::pair<int32, int32> >::const_iterator
            iter = pairs.begin(), end = pairs.end();
        for (; iter != end; ++iter) {
          // note: -1's are not allowed.  To represent the empty range,
          // the user should use some valid index twice.
          if (iter->second < iter->first || iter->first < 0 ||
              iter->second > src_num_rows)
            KALDI_ERR << "Row range " << iter->first << ',' << iter->second
                      << " out of range in add-row-ranges command.";
        }
        break;
      }
      case NnetComputation::kNoOperation:
      case NnetComputation::kNoOperationMarker:
        break;
      default:
        KALDI_ERR << "Unknown command type.";
    }
  }
}


// make sure Propagate comes before kNoOpMarker and Backprop comes after it.
void ComputationChecker::CheckComputationOrder() const {
  KALDI_ERR << "todo";
}

void ComputationChecker::CheckComputationMatrixAccesses() const {
  KALDI_ERR << "todo";
}

} // namespace nnet3
} // namespace kaldi
