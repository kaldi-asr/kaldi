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
  row_split_points_.resize(num_matrices);
  column_split_points_.resize(num_matrices);
  KALDI_ASSERT(computation.submatrices[0].num_rows == 0);
  for (int32 submatrix_index = 1;
       submatrix_index < num_submatrices;
       submatrix_index++) {
    const NnetComputation::SubMatrixInfo &s =
        computation.submatrices[submatrix_index];
    row_split_points_[s.matrix_index].push_back(s.row_offset);
    row_split_points_[s.matrix_index].push_back(s.row_offset + s.num_rows);
    column_split_points_[s.matrix_index].push_back(s.col_offset);
    column_split_points_[s.matrix_index].push_back(s.col_offset + s.num_cols);
  }
  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    // Because it's possible for matrices not to have any submatrices (after
    // pruning), we need to make sure that the beginning and end dimensions are
    // in the split points.
    column_split_points_[matrix_index].push_back(0);
    column_split_points_[matrix_index].push_back(
        computation.matrices[matrix_index].num_cols);
    row_split_points_[matrix_index].push_back(0);
    row_split_points_[matrix_index].push_back(
        computation.matrices[matrix_index].num_rows);
    SortAndUniq(&(column_split_points_[matrix_index]));
    SortAndUniq(&(row_split_points_[matrix_index]));
  }
  // note: the last split point of each matrix doesn't get its own variable index.
  matrix_to_variable_index_.resize(num_matrices + 1);
  matrix_to_variable_index_[0] = 0;
  matrix_to_variable_index_[1] = 0;
  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    int32 num_row_variables = row_split_points_[matrix_index].size() - 1,
        num_column_variables = column_split_points_[matrix_index].size() - 1,
        num_variables = num_row_variables * num_column_variables;
    KALDI_ASSERT(num_variables >= 1);
    matrix_to_variable_index_[matrix_index+1] =
        matrix_to_variable_index_[matrix_index] + num_variables;
  }
  num_variables_ = matrix_to_variable_index_.back();
}

//static
int32 ComputationVariables::FindIndexOf(const std::vector<int32> &vec, int32 i) {
  // std::lower_bound does a binary search -> faster than std::find.
  std::vector<int32>::const_iterator iter = std::lower_bound(
      vec.begin(), vec.end(), i);
  KALDI_ASSERT(*iter == i);
  return iter - vec.begin();
}

void ComputationVariables::ComputeVariablesForSubmatrix(
    const NnetComputation &computation) {
  // note, these numbers are only valid if you include the empty zero-indexed
  // matrix/submatrix as a matrix.
  int32 num_submatrices = computation.submatrices.size();

  variables_for_submatrix_.resize(num_submatrices);

  submatrix_is_whole_matrix_.resize(num_submatrices, false);
  submatrix_to_matrix_.resize(num_submatrices);
  submatrix_to_matrix_[0] = 0;

  for (int32 submatrix_index = 1;
       submatrix_index < num_submatrices;
       submatrix_index++) {
    const NnetComputation::SubMatrixInfo &s =
        computation.submatrices[submatrix_index];
    int32 matrix_index = s.matrix_index;
    submatrix_to_matrix_[submatrix_index] = matrix_index;
    int32 start_col = s.col_offset, end_col = start_col + s.num_cols,
        start_row = s.row_offset, end_row = start_row + s.num_rows;
    int32 row_start = FindIndexOf(row_split_points_[matrix_index], start_row),
        row_end = FindIndexOf(row_split_points_[matrix_index], end_row),
        col_start = FindIndexOf(column_split_points_[matrix_index], start_col),
        col_end = FindIndexOf(column_split_points_[matrix_index], end_col),
        num_column_variables = column_split_points_[matrix_index].size() - 1,
        num_row_variables = row_split_points_[matrix_index].size() - 1,
        matrix_start_variable = matrix_to_variable_index_[matrix_index];
    KALDI_ASSERT(row_end > row_start && col_end > col_start &&
                 col_end <= num_column_variables);
    std::vector<int32> &variables = variables_for_submatrix_[submatrix_index];
    for (int32 r = row_start; r < row_end; r++)
      for (int32 c = col_start; c < col_end; c++)
        variables.push_back(matrix_start_variable + r*num_column_variables + c);
    if (row_start == 0 && row_end == num_row_variables &&
        col_start == 0 && col_end == num_column_variables)
      submatrix_is_whole_matrix_[submatrix_index] = true;
  }
}

void ComputationVariables::ComputeVariableToMatrix() {
  variable_to_matrix_.clear();
  variable_to_matrix_.resize(NumVariables());
  int32 num_matrices = matrix_to_variable_index_.size() - 1;
  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    int32 start_variable = matrix_to_variable_index_[matrix_index],
        end_variable = matrix_to_variable_index_[matrix_index + 1];
    for (int32 i = start_variable; i < end_variable; i++)
      variable_to_matrix_[i] = matrix_index;
  }
}

void ComputationVariables::Init(const NnetComputation &computation) {
  // don't call this twice on the same object..
  KALDI_ASSERT(row_split_points_.empty());
  ComputeSplitPoints(computation);
  ComputeVariablesForSubmatrix(computation);
  ComputeVariableToMatrix();
}

int32 ComputationVariables::GetMatrixForVariable(int32 variable) const {
  KALDI_ASSERT(static_cast<size_t>(variable) < variable_to_matrix_.size());
  return variable_to_matrix_[variable];
}

void ComputationVariables::AppendVariablesForSubmatrix(
    int32 submatrix_index,
    std::vector<int32> *variable_indexes) const {
  KALDI_ASSERT(static_cast<size_t>(submatrix_index) <
               variables_for_submatrix_.size());
  variable_indexes->insert(variable_indexes->end(),
                           variables_for_submatrix_[submatrix_index].begin(),
                           variables_for_submatrix_[submatrix_index].end());
}

void ComputationVariables::AppendVariablesForMatrix(
    int32 matrix_index,
    std::vector<int32> *variable_indexes) const {
  KALDI_ASSERT(static_cast<size_t>(matrix_index + 1) <
               matrix_to_variable_index_.size());
  int32 start = matrix_to_variable_index_[matrix_index],
      end = matrix_to_variable_index_[matrix_index + 1];
  variable_indexes->reserve(variable_indexes->size() + end - start);
  for (int32 variable_index = start; variable_index < end; variable_index++)
    variable_indexes->push_back(variable_index);
}

void ComputationVariables::RecordAccessForSubmatrix(
    int32 submatrix_index,
    AccessType access_type,
    CommandAttributes *ca) const {
  if (submatrix_index == 0)
    return;
  KALDI_ASSERT(static_cast<size_t>(submatrix_index) <
               submatrix_to_matrix_.size());
  int32 matrix_index = submatrix_to_matrix_[submatrix_index];
  bool is_whole_matrix = submatrix_is_whole_matrix_[submatrix_index];
  switch (access_type) {
    case kReadAccess:
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_read));
      ca->matrices_read.push_back(matrix_index);
      ca->submatrices_read.push_back(submatrix_index);
      break;
    case kWriteAccess:
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_written));
      ca->submatrices_written.push_back(submatrix_index);
      ca->matrices_written.push_back(matrix_index);
      // if submatrix does not span the full row range of the matrix,
      // a write operation has to be considered a read/write operation
      // on the underlying matrix
      if (!is_whole_matrix)
        ca->matrices_read.push_back(matrix_index);
      break;
    case kReadWriteAccess:
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_written));
      AppendVariablesForSubmatrix(submatrix_index,
                                  &(ca->variables_read));
      ca->submatrices_written.push_back(submatrix_index);
      ca->submatrices_read.push_back(submatrix_index);
      ca->matrices_written.push_back(matrix_index);
      ca->matrices_read.push_back(matrix_index);
  }
}

std::string ComputationVariables::DescribeVariable(int32 variable) const {
  KALDI_ASSERT(variable >= 0 && variable < num_variables_);
  int32 matrix_index = variable_to_matrix_[variable],
      offset = variable - matrix_to_variable_index_[matrix_index],
      num_column_variables = column_split_points_[matrix_index].size() - 1,
      num_row_variables = row_split_points_[matrix_index].size() - 1,
      column_variable = offset % num_column_variables,
      row_variable = offset / num_column_variables;
  KALDI_ASSERT(column_variable >= 0 && row_variable >= 0 &&
               row_variable < num_row_variables &&
               column_variable < num_column_variables);
  std::ostringstream os;
  os << 'm' << matrix_index;
  if (num_row_variables != 1 || num_column_variables != 1) {
    os << '(';
    if (num_row_variables == 1) {
      os << ':';
    } else {
      os << row_split_points_[matrix_index][row_variable] << ':'
         << row_split_points_[matrix_index][row_variable+1] - 1;
    }
    os << ',';
    if (num_column_variables == 1) {
      os << ':';
    } else {
      os << column_split_points_[matrix_index][column_variable] << ':'
         << column_split_points_[matrix_index][column_variable+1] - 1;
    }
    os << ')';
  }
  return os.str();
}

NnetComputation::SubMatrixInfo ComputationVariables::VariableInfo(
    int32 variable) const {
  KALDI_ASSERT(variable >= 0 && variable < num_variables_);
  int32 matrix_index = variable_to_matrix_[variable],
      offset = variable - matrix_to_variable_index_[matrix_index],
      num_column_variables = column_split_points_[matrix_index].size() - 1,
      column_variable = offset % num_column_variables,
      row_variable = offset / num_column_variables;
  int32 row_offset = row_split_points_[matrix_index][row_variable],
      num_rows = row_split_points_[matrix_index][row_variable+1] - row_offset,
      col_offset = column_split_points_[matrix_index][column_variable],
      num_cols = column_split_points_[matrix_index][column_variable+1] -
                  col_offset;
  return NnetComputation::SubMatrixInfo(matrix_index, row_offset, num_rows,
                                        col_offset, num_cols);
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
      case kAllocMatrix:
      case kDeallocMatrix:
      case kSwapMatrix:
        break;  // the commands above leave the matrix undefined.
      case kSetConst:
        vars.RecordAccessForSubmatrix(c.arg1, kWriteAccess, &attr);
        break;
      case kPropagate:
        vars.RecordAccessForSubmatrix(c.arg3, kReadAccess, &attr);
        if (nnet.GetComponent(c.arg1)->Properties() & kPropagateAdds)
          vars.RecordAccessForSubmatrix(c.arg4, kReadWriteAccess, &attr);
        else
          vars.RecordAccessForSubmatrix(c.arg4, kWriteAccess, &attr);
        break;
      case kBackprop:
      case kBackpropNoModelUpdate:
        vars.RecordAccessForSubmatrix(c.arg3, kReadAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg4, kReadAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg5, kReadAccess, &attr);
        if (nnet.GetComponent(c.arg1)->Properties() & kBackpropAdds)
          vars.RecordAccessForSubmatrix(c.arg6, kReadWriteAccess, &attr);
        else
          vars.RecordAccessForSubmatrix(c.arg6, kWriteAccess, &attr);
        if (c.command_type == kBackprop &&
            nnet.GetComponent(c.arg1)->Properties() & kUpdatableComponent)
          attr.has_side_effects = true;
        break;
      case kMatrixCopy:
        vars.RecordAccessForSubmatrix(c.arg1, kWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;
      case kMatrixAdd:
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;
      case kAddRows:
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;
      case kCopyRows: {
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
      case kAddRowsMulti: {
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        std::vector<int32> submatrix_indexes;
        IndexesMultiToSubmatrixIndexes(computation.indexes_multi[c.arg2],
                                       &submatrix_indexes);
        for (size_t i = 0; i < submatrix_indexes.size(); i++)
          vars.RecordAccessForSubmatrix(submatrix_indexes[i],
                                        kReadAccess, &attr);
        break;
      }
      case kCopyRowsMulti: {
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
      case kAddToRowsMulti:
      case kCopyToRowsMulti: {
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
      case kAddRowRanges: {
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        vars.RecordAccessForSubmatrix(c.arg2, kReadAccess, &attr);
        break;
      }
      case kCompressMatrix: {
        vars.RecordAccessForSubmatrix(c.arg1, kReadWriteAccess, &attr);
        break;
      }
      case kDecompressMatrix: {
        vars.RecordAccessForSubmatrix(c.arg1, kWriteAccess, &attr);
        break;
      }
      case kAcceptInput: {
        vars.RecordAccessForSubmatrix(c.arg1, kWriteAccess, &attr);
        break;
      }
      case kProvideOutput: {
        vars.RecordAccessForSubmatrix(c.arg1, kReadAccess, &attr);
        break;
      }
      case kNoOperation:
      case kNoOperationPermanent:
      case kNoOperationMarker:
      case kNoOperationLabel:
      case kGotoLabel:
        break;
      default:
        KALDI_ERR << "Unknown command type.";
    }
    SortAndUniq(&attr.variables_read);
    SortAndUniq(&attr.variables_written);
    SortAndUniq(&attr.submatrices_read);
    SortAndUniq(&attr.submatrices_written);
    SortAndUniq(&attr.matrices_read);
    SortAndUniq(&attr.matrices_written);
  }
}

void ComputeVariableAccesses(
    const ComputationVariables &variables,
    const std::vector<CommandAttributes> &command_attributes,
    std::vector<std::vector<Access> > *variable_accesses) {
  int32 num_variables = variables.NumVariables(),
      num_commands = command_attributes.size();
  variable_accesses->clear();
  variable_accesses->resize(num_variables);
  for (int32 c = 0; c < num_commands; c++) {
    const CommandAttributes &attr = command_attributes[c];
    KALDI_ASSERT(IsSortedAndUniq(attr.variables_read));
    KALDI_ASSERT(IsSortedAndUniq(attr.variables_written));
    std::vector<int32> all_variables;
    all_variables.reserve(attr.variables_read.size() +
                          attr.variables_written.size());
    all_variables.insert(all_variables.end(), attr.variables_read.begin(),
                         attr.variables_read.end());
    all_variables.insert(all_variables.end(), attr.variables_written.begin(),
                         attr.variables_written.end());
    SortAndUniq(&all_variables);

    std::vector<int32>::const_iterator iter = all_variables.begin(),
        end = all_variables.end();
    for (; iter != end; ++iter) {
      int32 variable_index = *iter;
      bool is_read = std::binary_search(attr.variables_read.begin(),
                                        attr.variables_read.end(),
                                        variable_index),
          is_written = (!is_read ? true :
                        std::binary_search(attr.variables_written.begin(),
                                           attr.variables_written.end(),
                                           variable_index));
      if (is_read && is_written) {
        (*variable_accesses)[variable_index].push_back(
            Access(c, kReadWriteAccess));
      } else if (is_read) {
        (*variable_accesses)[variable_index].push_back(
            Access(c, kReadAccess));
      } else {
        (*variable_accesses)[variable_index].push_back(
            Access(c, kWriteAccess));
      }
    }
  }
}

void ComputeMatrixAccesses(
    const Nnet &nnet,
    const NnetComputation &computation,
    const ComputationVariables &variables,
    const std::vector<CommandAttributes> &command_attributes,
    std::vector<MatrixAccesses> *matrix_accesses) {
  int32 num_matrices = computation.matrices.size(),
      num_commands = command_attributes.size();
  matrix_accesses->clear();
  matrix_accesses->resize(num_matrices);
  for (int32 c = 0; c < num_commands; c++) {
    const CommandAttributes &attr = command_attributes[c];
    KALDI_ASSERT(IsSortedAndUniq(attr.matrices_read));
    KALDI_ASSERT(IsSortedAndUniq(attr.matrices_written));
    std::vector<int32> all_matrices;
    all_matrices.reserve(attr.matrices_read.size() +
                          attr.matrices_written.size());
    all_matrices.insert(all_matrices.end(), attr.matrices_read.begin(),
                         attr.matrices_read.end());
    all_matrices.insert(all_matrices.end(), attr.matrices_written.begin(),
                         attr.matrices_written.end());
    SortAndUniq(&all_matrices);

    std::vector<int32>::const_iterator iter = all_matrices.begin(),
        end = all_matrices.end();
    for (; iter != end; ++iter) {
      int32 matrix_index = *iter;
      bool is_read = std::binary_search(attr.matrices_read.begin(),
                                        attr.matrices_read.end(),
                                        matrix_index),
          is_written = (!is_read ? true :
                        std::binary_search(attr.matrices_written.begin(),
                                           attr.matrices_written.end(),
                                           matrix_index));
      if (is_read && is_written) {
        (*matrix_accesses)[matrix_index].accesses.push_back(
            Access(c, kReadWriteAccess));
      } else if (is_read) {
        (*matrix_accesses)[matrix_index].accesses.push_back(
            Access(c, kReadAccess));
      } else {
        (*matrix_accesses)[matrix_index].accesses.push_back(
            Access(c, kWriteAccess));
      }
    }
    // Now set up allocate_command, deallocate_command,
    // is_input and is_output.
    const NnetComputation::Command &command = computation.commands[c];
    int32 matrix_index1, matrix_index2;

    switch (command.command_type) {
      case kAllocMatrix:
        if (!computation.IsWholeMatrix(command.arg1))
          KALDI_ERR << "Command does not operate on whole matrix";
        matrix_index1 = computation.submatrices[command.arg1].matrix_index;
        if ((*matrix_accesses)[matrix_index1].allocate_command != -1)
          KALDI_ERR << "Matrix " << matrix_index1 << " initialized twice.";
        (*matrix_accesses)[matrix_index1].allocate_command = c;
        break;
      case kSwapMatrix:
        if (!computation.IsWholeMatrix(command.arg1))
          KALDI_ERR << "Command does not operate on whole matrix";
        matrix_index1 = computation.submatrices[command.arg1].matrix_index;
        KALDI_ASSERT(computation.IsWholeMatrix(command.arg2));
        matrix_index2 = computation.submatrices[command.arg2].matrix_index;
        if ((*matrix_accesses)[matrix_index1].allocate_command != -1)
          KALDI_ERR << "Matrix " << matrix_index1 << " initialized twice.";
        (*matrix_accesses)[matrix_index1].allocate_command = c;
        if ((*matrix_accesses)[matrix_index2].deallocate_command != -1)
          KALDI_ERR << "Matrix " << matrix_index2 << " destroyed twice.";
        (*matrix_accesses)[matrix_index2].deallocate_command = c;
        break;
      case kDeallocMatrix:
        if (!computation.IsWholeMatrix(command.arg1))
          KALDI_ERR << "Command does not operate on whole matrix";
        matrix_index1 = computation.submatrices[command.arg1].matrix_index;
        if ((*matrix_accesses)[matrix_index1].deallocate_command != -1)
          KALDI_ERR << "Matrix " << matrix_index1 << " destroyed twice.";
        (*matrix_accesses)[matrix_index1].deallocate_command = c;
        break;
      case kAcceptInput:
        if (!computation.IsWholeMatrix(command.arg1))
          KALDI_ERR << "Command does not operate on whole matrix";
        matrix_index1 = computation.submatrices[command.arg1].matrix_index;
        (*matrix_accesses)[matrix_index1].is_input = true;
        // If a certain matrix is accepted as input multiple times, we
        // count the first one as allocating it (the second will just
        // allocate it again, which is harmless).
        if ((*matrix_accesses)[matrix_index1].allocate_command == -1)
          (*matrix_accesses)[matrix_index1].allocate_command = c;
        break;
      case kProvideOutput:
        if (!computation.IsWholeMatrix(command.arg1))
          KALDI_ERR << "Command does not operate on whole matrix";
        matrix_index1 = computation.submatrices[command.arg1].matrix_index;
        (*matrix_accesses)[matrix_index1].is_output = true;
        break;
      default:
        ;
    }
  }
}


ComputationChecker::ComputationChecker(
    const CheckComputationOptions &config,
    const Nnet &nnet,
    const NnetComputation &computation):
    config_(config), nnet_(nnet), computation_(computation) { }



void ComputationChecker::Check() {
  CheckComputationIndexes();
  a_.Init(nnet_, computation_);
  CheckComputationMatrixAccesses();
  CheckComputationCompression();
  CheckComputationUndefined();
  CheckComputationDebugInfo();
  if (config_.check_rewrite)
    CheckComputationRewrite();
}


/**
   Checks for the situation where a read-only operation on a variable is
   followed by an operation that writes to the variable.  This should never
   occur prior to optimization, but after certain optimization we in effect
   "re-use" variables by doing things like propagate and backprop in-place, so
   this check shouldn't be performed after optimization.
*/
void ComputationChecker::CheckComputationRewrite() const {
  int32 num_variables = a_.variable_accesses.size();
  for (int32 v = 0; v < num_variables; v++) {
    const std::vector<Access> &accesses = a_.variable_accesses[v];
    if (accesses.empty()) {
      if (config_.check_unused_variables) {
        KALDI_ERR << "Variable " << v << " = " << a_.variables.DescribeVariable(v)
                  << " is never used.";
      } else {
        continue;
      }
    }
    int32 num_accesses = accesses.size();
    int32 first_pure_read = -1;
    for (int32 access = 0; access < num_accesses; access++) {
      if (accesses[access].access_type == kReadAccess) {
        first_pure_read = access;
        break;
      }
    }
    if (first_pure_read != -1) {
      for (int32 access = first_pure_read + 1;
           access < num_accesses; access++) {
        if (accesses[access].access_type != kReadAccess) {
          KALDI_ERR << "Variable " << v << " = "
                    << a_.variables.DescribeVariable(v)
                    << " is modified after being read"
                    << " (this is not expected before optimization)";
        }
      }
    }
  }
}


/**
   Checks for the situation where a variable is read before being written.
*/
void ComputationChecker::CheckComputationUndefined() const {
  // the variable 'min_proportion' needs to be <= the min_proportion_ value in
  // class MatrixExtender, otherwise this code could spuriously reject a
  // computation.
  BaseFloat min_proportion = 0.8;

  int32 num_variables = a_.variable_accesses.size();
  for (int32 v = 0; v < num_variables; v++) {
    const std::vector<Access> &accesses = a_.variable_accesses[v];
    if (accesses.empty()) {
      if (config_.check_unused_variables) {
        NnetComputation::SubMatrixInfo info = a_.variables.VariableInfo(v);
        const NnetComputation::MatrixInfo &matrix_info =
            computation_.matrices[info.matrix_index];
        // Before we throw an error, we want to check that it isn't a case that
        // can be produced by the ExtendMatrices() optimization, that is
        // actually allowed.  This is a case when a variable is inside the last
        // few rows of a matrix, but not all columns of those last rows.
        if (info.row_offset >= min_proportion * matrix_info.num_rows &&
            !(info.col_offset == 0 && info.num_cols == matrix_info.num_cols)) {
          continue;
        }
        KALDI_ERR << "Variable " << v << " == "
                  << a_.variables.DescribeVariable(v) << " is never used.";
      }
    } else {
      // It's OK if part of a matrix is compressed, that is undefined;
      // likely that part won't be referred to when we uncompress.
      if (accesses[0].access_type != kWriteAccess &&
          !(computation_.commands[accesses[0].command_index].command_type ==
            kCompressMatrix))
        KALDI_ERR << "Variable " << v << " == "
                  << a_.variables.DescribeVariable(v)
                  << " is read before it is written to";
    }
  }
}


/**
   Checks that we never use variables before they are allocated or after they
   are deallocated, and some other checks that can be done from the
   MatrixAccesses.
*/
static bool computation_checker_warned_unused_input = false;

void ComputationChecker::CheckComputationMatrixAccesses() const {
  int32 num_matrices = a_.matrix_accesses.size();

  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    const MatrixAccesses &accesses = a_.matrix_accesses[matrix_index];
    if (accesses.allocate_command == -1)
      KALDI_ERR << "Matrix m" << matrix_index << " is not initialized.";
    if (accesses.accesses.empty()) {
      KALDI_ERR << "Matrix m" << matrix_index << " is never accessed.";
    } else if (accesses.accesses.front().command_index <
               accesses.allocate_command) {
      KALDI_ERR << "Matrix m" << matrix_index << " is accessed before "
          "it is initialized";
    }
    if (accesses.accesses.size() == 1 && config_.check_unused_variables) {
      int32 first_access_command = accesses.accesses[0].command_index;
      if (computation_.commands[first_access_command].command_type == kSetConst) {
        if (!config_.check_unused_variables)
        KALDI_ERR << "Matrix m" << matrix_index << " is only set to a constant "
                  << "value, but then never accessed.";
      }
    }

    if (accesses.accesses.empty()) {
      if (accesses.is_input) {
        // we allow there to be no accesses if it is an input, e.g. if an
        // output derivative is supplied for some reason but never used.
        // We'll warn, though (once).
        if (!computation_checker_warned_unused_input) {
          KALDI_WARN << "Matrix m" << matrix_index << " is never accessed. "
              "Allowing because it is an input (un-needed input or "
              "derivative?)  Will warn only once.";
          computation_checker_warned_unused_input = true;
        }
      } else {
        KALDI_ERR << "Matrix m" << matrix_index << " is never accessed.";
      }
    } else if (accesses.deallocate_command != -1 &&
               accesses.accesses.back().command_index >=
               accesses.deallocate_command) {
      KALDI_ERR << "Matrix m" << matrix_index << " is accessed after "
          "it is destroyed";
    }
  }
}

void ComputationChecker::CheckComputationCompression() const {
  int32 num_matrices = a_.matrix_accesses.size();

  // 'middle_command' will be the index of the command that separates
  // the forward and backward passes.
  int32 middle_command = -1;
  for (size_t i = 0; i < computation_.commands.size(); i++) {
    if (computation_.commands[i].command_type == kNoOperationMarker) {
        middle_command = static_cast<int32>(i);
        break;
    }
  }
  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    const MatrixAccesses &accesses = a_.matrix_accesses[matrix_index];
    int32 num_accesses = accesses.accesses.size();
    for (int32 a = 0; a < num_accesses; a++) {
      const Access &access = accesses.accesses[a];
      int32 command_index = access.command_index;
      const NnetComputation::Command &command =
          computation_.commands[command_index];
      if (command.command_type == kDecompressMatrix) {
        // check that the previous access to this matrix was a compression
        // command.
        KALDI_ASSERT(
            a > 0 && computation_.commands[
                accesses.accesses[a-1].command_index].command_type ==
            kCompressMatrix);
      }
      if (command.command_type == kCompressMatrix) {
        // check that the next access to this matrix is an uncompression
        // command.
        int32 next_command_index = accesses.accesses[a+1].command_index;
        KALDI_ASSERT(computation_.commands[next_command_index].command_type ==
                     kDecompressMatrix &&
                     command_index < middle_command &&
                     next_command_index > middle_command);
        if (command.alpha == 0.0) {
          // alpha == 0.0 means we're only retaining the sign; we should
          // only do this if this is the output of a ReLU.
          // make sure there are only 2 commands after this: the uncompress
          // command, and a relu backprop command.  (Any deallocation
          // command doesn't show up in the list of 'accesses').
          KALDI_ASSERT(a > 0 && command.arg2 == kCompressedMatrixUint8 &&
                       num_accesses == a + 3);
          // make sure the next access to that matrix, apart from the
          // uncompression command, is a ReLU propagation.
          int32 next_command_index = accesses.accesses[a+2].command_index;
          const NnetComputation::Command &next_command =
              computation_.commands[next_command_index];
          KALDI_ASSERT(next_command.command_type == kBackprop &&
                       nnet_.GetComponent(next_command.arg1)->Type() ==
                       "RectifiedLinearComponent");
        }
      }
    }
  }
}

/**
   This very basic check just makes sure that all indexes in the commands are
   within range, that dimensions agree with the request, that row/column dimensions
   agree with component dimensions.
*/
void ComputationChecker::CheckComputationIndexes() const {
  int32 num_commands = computation_.commands.size(),
      num_submatrices = computation_.submatrices.size();
  const std::vector<NnetComputation::SubMatrixInfo> &submatrices =
      computation_.submatrices;

  // This maps from the memo-index > 0 to the Propagate command
  // which created it.  When the corresponding Backprop command
  // is encountered, we delete the map element.
  std::unordered_map<int32, int32> memo_to_command;

  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    const NnetComputation::Command &c = computation_.commands[command_index];
    switch (c.command_type) {
      case kAllocMatrix:
      case kDeallocMatrix:
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            !computation_.IsWholeMatrix(c.arg1))
          KALDI_ERR << "submatrix index out of range or invalid";
        break;
      case kSetConst:
        if (c.arg1 < 1 || c.arg1 >= num_submatrices)
          KALDI_ERR << "submatrix index out of range or invalid";
        break;
      case kSwapMatrix:
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            !computation_.IsWholeMatrix(c.arg1) ||
            c.arg2 < 1 || c.arg2 >= num_submatrices ||
            !computation_.IsWholeMatrix(c.arg2))
          KALDI_ERR << "submatrix index out of range or invalid";
        if (computation_.submatrices[c.arg1].num_rows !=
            computation_.submatrices[c.arg2].num_rows ||
            computation_.submatrices[c.arg1].num_cols !=
            computation_.submatrices[c.arg2].num_cols)
          KALDI_ERR << "Dimension mismatch in kSwapMatrix command";
        break;
      case kPropagate: {
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
            (c.arg3 == 0 && (properties & kSimpleComponent)) ||
            c.arg4 < 1 || c.arg4 >= num_submatrices)
            KALDI_ERR << "Sub-matrix indexes out of range.";
        if (c.arg3 > 0 && submatrices[c.arg3].num_cols != component->InputDim())
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
        if (c.arg5 > 0) {
          KALDI_ASSERT(memo_to_command.count(c.arg5) == 0 &&
                       "Memo index re-used.");
          memo_to_command[c.arg5] = command_index;
        }
        KALDI_ASSERT(c.arg6 == 0 || c.arg6 == 1);
        break;
      }
      case kBackprop:
      case kBackpropNoModelUpdate: {
        if (c.arg1 < 0 || c.arg1 >= nnet_.NumComponents())
          KALDI_ERR << "Component index in backprop invalid or out of range";
        const Component *component = nnet_.GetComponent(c.arg1);
        int32 properties = component->Properties();
        if (c.arg2 < 0 ||
            c.arg2 > computation_.component_precomputed_indexes.size())
          KALDI_ERR << "Precomputed-indexes index out of range";
        if (c.arg2 != 0 && (properties & kSimpleComponent))
          KALDI_ERR << "Precomputed-indexes index nonzero for simple component";
        // output-deriv (arg5) must be supplied; others could plausibly be zero.
        if (c.arg3 < 0 || c.arg3 >= num_submatrices ||
            c.arg4 < 0 || c.arg4 >= num_submatrices ||
            c.arg5 < 1 || c.arg5 >= num_submatrices ||
            c.arg6 < 0 || c.arg6 >= num_submatrices)
          KALDI_ERR << "Submatrix index out of range for backprop.";
        if ((properties & kBackpropNeedsInput) && c.arg3 == 0)
          KALDI_ERR << "Backprop input needed but not supplied.";
        if ((properties & kBackpropNeedsOutput) && c.arg4 == 0)
          KALDI_ERR << "Backprop output needed but not supplied.";
        if (c.arg6 == 0 && !(properties & kUpdatableComponent)) {
          // note: we could perhaps make this just a warning,
          // or optimize it away somehow.
          KALDI_ERR << "Backprop is done but has no effect.";
        }
        if (c.arg5 == c.arg6 && !(properties & kBackpropInPlace))
          KALDI_ERR << "In-place backprop used where not supported.";
        if (c.arg3 != 0 &&
            submatrices[c.arg3].num_cols != component->InputDim())
          KALDI_ERR << "Input-dim mismatch in backprop.";
        if (c.arg4 != 0 &&
            submatrices[c.arg4].num_cols != component->OutputDim())
          KALDI_ERR << "Output-dim mismatch in backprop.";
        if (c.arg5 != 0 &&
            submatrices[c.arg5].num_cols != component->OutputDim())
          KALDI_ERR << "Output-dim mismatch in backprop.";
        if (c.arg6 != 0 &&
            submatrices[c.arg6].num_cols != component->InputDim())
          KALDI_ERR << "Input-dim mismatch in backprop.";
        // check num-rows consistency for input.
        if (c.arg3 != 0 && c.arg6 != 0 &&
            submatrices[c.arg3].num_rows != submatrices[c.arg6].num_rows)
          KALDI_ERR << "Num-rows mismatch in backprop input";
        // check num-rows consistency for output
        if (c.arg4 != 0 &&
            submatrices[c.arg4].num_rows != submatrices[c.arg5].num_rows)
          KALDI_ERR << "Num-rows mismatch in backprop output";
        if ((properties & kSimpleComponent) && c.arg6 != 0 &&
            submatrices[c.arg5].num_rows != submatrices[c.arg6].num_rows)
          KALDI_ERR << "Num-rows mismatch in backprop input vs output.";
        if (c.arg7 != 0) {
          KALDI_ASSERT(c.arg7 > 0);
          if (memo_to_command.count(c.arg7) == 0)
            KALDI_ERR << "Memo-index " << c.arg7 << " not used for propagate.";
          int32 propagate_command = memo_to_command[c.arg7];
          memo_to_command.erase(c.arg7);
          if (c.arg1 != computation_.commands[propagate_command].arg1)
            KALDI_ERR << "Mismatch in component-node for memo index";
          if (!(properties & kUsesMemo))
            KALDI_ERR << "Component not expected to use a memo.";
        }
        break;
      }
      case kMatrixCopy:
      case kMatrixAdd:
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            c.arg2 < 1 || c.arg2 >= num_submatrices)
          KALDI_ERR << "Submatrix indexes out of range in matrix copy/add";
        if (submatrices[c.arg1].num_rows != submatrices[c.arg2].num_rows ||
            submatrices[c.arg1].num_cols != submatrices[c.arg2].num_cols)
          KALDI_ERR << "Submatrix indexes out of range in matrix copy/add";
        if (c.arg1 == c.arg2) {
          // we allow copying to itself if alpha != 1.0; this is how we
          // implement scaling.
          if (!(c.command_type == kMatrixCopy && c.alpha != 1.0)) {
            KALDI_ERR << "Adding/copying to self";
          }
        }
        break;
      case kAddRows:
      case kCopyRows: {
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
      case kAddRowsMulti:
      case kCopyRowsMulti:
      case kAddToRowsMulti:
      case kCopyToRowsMulti: {
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
        if (c.command_type == kAddToRowsMulti ||
            c.command_type == kCopyToRowsMulti) {
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
      case kAddRowRanges: {
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            c.arg2 < 1 || c.arg2 >= num_submatrices ||
            static_cast<size_t>(c.arg3) >= computation_.indexes_ranges.size())
          KALDI_ERR << "Index out of range in add-row-ranges command";
        const std::vector<std::pair<int32, int32> > pairs =
            computation_.indexes_ranges[c.arg3];
        if (static_cast<size_t>(submatrices[c.arg1].num_rows) != pairs.size())
          KALDI_ERR << "Num-rows mismatch in add-row-ranges command";
        if (submatrices[c.arg1].num_cols != submatrices[c.arg2].num_cols)
          KALDI_ERR << "Dimension mismatch in add-row-ranges command";
        int32 src_num_rows = submatrices[c.arg2].num_rows;
        std::vector<std::pair<int32, int32> >::const_iterator
            iter = pairs.begin(), end = pairs.end();
        for (; iter != end; ++iter) {
          if (!((iter->first == -1 && iter->second == -1) ||
                (iter->second > iter->first &&
                 iter->first >= 0 && iter->second <= src_num_rows)))
            KALDI_ERR << "Row range " << iter->first << ',' << iter->second
                      << " is invalid in add-row-ranges command.";
        }
        break;
      }
      case kCompressMatrix: {
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            !computation_.IsWholeMatrix(c.arg1))
          KALDI_ERR << "submatrix index out of range or invalid";
        if (c.arg2 < static_cast<int32>(kCompressedMatrixInt8) ||
            c.arg2 > static_cast<int32>(kCompressedMatrixUint16))
          KALDI_ERR << "Invalid compressed-matrix type.";
        if (c.arg3 != 0 && c.arg3 != 1)
          KALDI_ERR << "Invalid 'truncate' option for compressing matrix.";
        if (c.alpha < 0.0 || c.alpha > 1000.0 ||
            (c.alpha == 0.0 && c.arg2 != kCompressedMatrixUint8))
          KALDI_ERR << "Invalid alpha in kCompressMatrix command.";
        break;
      }
      case kDecompressMatrix: {
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            !computation_.IsWholeMatrix(c.arg1))
          KALDI_ERR << "submatrix index out of range or invalid";
        break;
      }
      case kAcceptInput: case kProvideOutput: {
        if (c.arg1 < 1 || c.arg1 >= num_submatrices ||
            !computation_.IsWholeMatrix(c.arg1))
          KALDI_ERR << "submatrix index out of range or invalid";
        // note: we may later change the following condition to allow component
        // nodes.  we allow it on output node because of derivatives.
        if (!nnet_.IsInputNode(c.arg2) && !nnet_.IsOutputNode(c.arg2))
          KALDI_ERR << "Invalid network node";
        break;
      }
      case kNoOperation:
      case kNoOperationPermanent:
      case kNoOperationMarker:
      case kNoOperationLabel:
        break;
      case kGotoLabel: {
        int32 label_index = c.arg1;
        if (label_index < 0 || label_index >= command_index ||
            computation_.commands[label_index].command_type != kNoOperationLabel)
          KALDI_ERR << "kGotoLabel command has invalid destination index.";
        if (command_index + 1 != num_commands) {
          KALDI_ERR << "kGotoLabel is not the last command in the computation";
        }
        break;
      }
      default:
        KALDI_ERR << "Unknown command type.";
    }
  }
  if (!memo_to_command.empty()) {
    KALDI_ERR << "Memo was used in command "
              << memo_to_command.begin()->second
              << " but never consumed.";
  }
}

void ComputationChecker::CheckComputationDebugInfo() const {
  if (computation_.matrix_debug_info.empty()) return;
  if (computation_.matrix_debug_info.size() !=
      computation_.matrices.size())
    KALDI_ERR << "Debug info has wrong size";
  for (size_t i = 1; i < computation_.matrix_debug_info.size(); i++) {
    if (computation_.matrix_debug_info[i].cindexes.size() !=
        static_cast<size_t>(computation_.matrices[i].num_rows))
      KALDI_ERR << "Debug info for matrix m" << i
                << " has wrong num-rows.";
    std::vector<Cindex>::const_iterator
        iter = computation_.matrix_debug_info[i].cindexes.begin(),
        end = computation_.matrix_debug_info[i].cindexes.end();
    for (; iter != end; ++iter) {
      if (iter->second.n < 0) {
        KALDI_ERR << "Negative n index in debug info";
      }
    }
  }
}


// note: 'computation' is not a reference, it's copied so that we
// can modify it internally.
static void CheckComputationOnline(const Nnet &nnet,
                                   NnetComputation computation,
                                   bool check_rewrite) {
  int32 num_commands = computation.commands.size();
  KALDI_ASSERT(computation.commands[num_commands-1].command_type == kGotoLabel);
  for (int32 c = num_commands - 2;
       c >= 0 && computation.commands[c].command_type == kSwapMatrix;
       c--) {
    // this command can be interpreted as "initialize matrix referred to by
    // c.arg2 with the matrix referred to by c.arg2".
    // Because this would be interpreted by the analysis code as initializing a
    // matrix that has already been initialized, we turn this into a command
    // that just deallocates the matrix in c.arg2. [note: all these indexes
    // are actually submatrix indexes].
    computation.commands[c].command_type = kDeallocMatrix;
    std::swap(computation.commands[c].arg1, computation.commands[c].arg2);
  }

  CheckComputationOptions opts;
  opts.check_rewrite = check_rewrite;
  opts.check_unused_variables = false;
  // We can always do this check with online computations, since they do not
  // have the RemoveUnnecessaryAllocation() optimization applied.
  ComputationChecker checker(opts, nnet, computation);
  checker.Check();
}

void CheckComputation(const Nnet &nnet,
                      const NnetComputation &computation,
                      bool check_rewrite) {
  try {
    if (!computation.commands.empty() &&
        computation.commands.back().command_type == kGotoLabel) {
      // Online computations need to be treated specially.
      CheckComputationOnline(nnet, computation, check_rewrite);
    } else {
      CheckComputationOptions opts;
      opts.check_rewrite = check_rewrite;
      ComputationChecker checker(opts, nnet, computation);
      checker.Check();
    }
  } catch (...) {
    computation.Print(std::cerr, nnet);
    KALDI_ERR << "Computation check failed for computation printed above "
        "(actual error message is above computation)";
  }
}

void ComputeMatrixToSubmatrix(
    const NnetComputation &computation,
    std::vector<std::vector<int32> > *mat_to_submat) {
  int32 num_matrices = computation.matrices.size(),
      num_submatrices = computation.submatrices.size();
  mat_to_submat->clear();
  mat_to_submat->resize(num_matrices);
  for (int32 submatrix_index = 1;
       submatrix_index < num_submatrices;
       submatrix_index++) {
    int32 matrix_index = computation.submatrices[submatrix_index].matrix_index;
    KALDI_ASSERT(matrix_index > 0 && matrix_index < num_matrices);
    (*mat_to_submat)[matrix_index].push_back(submatrix_index);
  }
}

int32 ComputationAnalysis::FirstNontrivialAccess(int32 s) const {
  KALDI_ASSERT(static_cast<size_t>(s) < computation_.submatrices.size() && s>0);
  int32 ans = computation_.commands.size();
  std::vector<int32> variable_indexes;
  analyzer_.variables.AppendVariablesForSubmatrix(s, &variable_indexes);
  std::vector<int32>::const_iterator iter = variable_indexes.begin(),
          end = variable_indexes.end();
  for (; iter != end; ++iter) {
    int32 v = *iter;
    const std::vector<Access> &accesses = analyzer_.variable_accesses[v];
    std::vector<Access>::const_iterator access_iter = accesses.begin(),
        access_end = accesses.end();
    for (; access_iter != access_end; ++access_iter) {
      int32 command_index = access_iter->command_index;
      const NnetComputation::Command &command = computation_.commands[
          command_index];
      if (!(command.command_type == kSetConst &&
            command.alpha == 0.0)) {  // if it's not a zeroing command..
        ans = std::min(ans, command_index);
        break;  // break from access_iter loop (an optimization)
      }
    }
  }
  return ans;
}


int32 ComputationAnalysis::FirstAccess(int32 s) const {
  KALDI_ASSERT(static_cast<size_t>(s) < computation_.submatrices.size() && s>0);
  int32 ans = computation_.commands.size();
  std::vector<int32> variable_indexes;
  analyzer_.variables.AppendVariablesForSubmatrix(s, &variable_indexes);
  std::vector<int32>::const_iterator iter = variable_indexes.begin(),
          end = variable_indexes.end();
  for (; iter != end; ++iter) {
    int32 v = *iter;
    const std::vector<Access> &accesses = analyzer_.variable_accesses[v];
    if (!accesses.empty())
      ans = std::min(ans, accesses[0].command_index);
  }
  return ans;
}


int32 ComputationAnalysis::FirstNontrivialMatrixAccess(int32 m) const {
  KALDI_ASSERT(static_cast<size_t>(m) < computation_.matrices.size() && m > 0);
  int32 ans = computation_.commands.size();
  const std::vector<Access> &accesses =
      analyzer_.matrix_accesses[m].accesses;
  std::vector<Access>::const_iterator access_iter = accesses.begin(),
      access_end = accesses.end();
  for (; access_iter != access_end; ++access_iter) {
    int32 command_index = access_iter->command_index;
    const NnetComputation::Command command =
        computation_.commands[command_index];
    if (!(command.command_type == kSetConst &&
          command.alpha == 0.0)) {  // except for zeroing commands..
      ans = std::min(ans, command_index);
      break;  // break from access_iter loop (an optimization; note, the
              // list 'accesses' is sorted.)
    }
  }
  return ans;
}


int32 ComputationAnalysis::LastMatrixAccess(int32 m) const {
  KALDI_ASSERT(static_cast<size_t>(m) < computation_.matrices.size() && m > 0);
  int32 ans = -1;
  const std::vector<Access> &accesses =
      analyzer_.matrix_accesses[m].accesses;
  std::vector<Access>::const_reverse_iterator access_iter = accesses.rbegin(),
      access_end = accesses.rend();
  for (; access_iter != access_end; ++access_iter) {
    int32 command_index = access_iter->command_index;
    ans = std::max(ans, command_index);
    break;  // break from access_iter loop (an optimization)
  }
  return ans;
}


int32 ComputationAnalysis::LastAccess(int32 s) const {
  KALDI_ASSERT(static_cast<size_t>(s) < computation_.submatrices.size() && s>0);
  int32 ans = -1;
  std::vector<int32> variable_indexes;
  analyzer_.variables.AppendVariablesForSubmatrix(s, &variable_indexes);
  std::vector<int32>::const_iterator iter = variable_indexes.begin(),
      end = variable_indexes.end();
  for (; iter != end; ++iter) {
    int32 v = *iter;
    const std::vector<Access> &accesses = analyzer_.variable_accesses[v];
    // Go through the variable accesses in reverse order (of command index)
    std::vector<Access>::const_reverse_iterator access_iter = accesses.rbegin(),
        access_end = accesses.rend();
    for (; access_iter != access_end; ++access_iter) {
      int32 command_index = access_iter->command_index;
      CommandType command_type =
          computation_.commands[command_index].command_type;
      // deallocation command should not be listed here.
      KALDI_ASSERT(command_type != kDeallocMatrix);
      ans = std::max(ans, command_index);
      break;  // break from access_iter loop (an optimization)
    }
  }
  return ans;
}


int32 ComputationAnalysis::LastWriteAccess(int32 s) const {
  KALDI_ASSERT(static_cast<size_t>(s) < computation_.submatrices.size() && s>0);
  int32 matrix_index = computation_.submatrices[s].matrix_index;
  if (analyzer_.matrix_accesses[matrix_index].is_output)
    return computation_.commands.size();
  int32 ans = -1;
  std::vector<int32> variable_indexes;
  analyzer_.variables.AppendVariablesForSubmatrix(s, &variable_indexes);
  std::vector<int32>::const_iterator iter = variable_indexes.begin(),
      end = variable_indexes.end();
  for (; iter != end; ++iter) {
    int32 v = *iter;
    const std::vector<Access> &accesses = analyzer_.variable_accesses[v];
    // Go through the variable accesses in reverse order (of command index)
    std::vector<Access>::const_reverse_iterator access_iter = accesses.rbegin(),
        access_end = accesses.rend();
    for (; access_iter != access_end; ++access_iter) {
      int32 command_index = access_iter->command_index;
      CommandType command_type =
          computation_.commands[command_index].command_type;
      // deallocation command should not be listed here.
      KALDI_ASSERT(command_type != kDeallocMatrix);
      if (access_iter->access_type != kReadAccess) {
        // If this operation is of type kWriteAccess or kReadWriteAccess
        ans = std::max(ans, command_index);
        break;  // break from access_iter loop (an optimization)
      }
    }
  }
  return ans;
}

int32 ComputationAnalysis::DataInvalidatedCommand(int32 c, int32 s) const {
  KALDI_ASSERT(static_cast<size_t>(c) < computation_.commands.size());
  KALDI_ASSERT(static_cast<size_t>(s) < computation_.submatrices.size() && s>0);
  int32 matrix_index = computation_.submatrices[s].matrix_index;
  int32 ans = analyzer_.matrix_accesses[matrix_index].deallocate_command;
  if (ans == -1)
    ans = static_cast<int32>(computation_.commands.size());
  std::vector<int32> variable_indexes;
  analyzer_.variables.AppendVariablesForSubmatrix(s, &variable_indexes);
  std::vector<int32>::const_iterator iter = variable_indexes.begin(),
          end = variable_indexes.end();
  for (; iter != end; ++iter) {
    int32 v = *iter;
    const std::vector<Access> &accesses = analyzer_.variable_accesses[v];
    std::vector<Access>::const_iterator access_iter = accesses.begin(),
        access_end = accesses.end();
    for (; access_iter != access_end; ++access_iter) {
      int32 command_index = access_iter->command_index;
      if (command_index > c &&
          access_iter->access_type != kReadAccess) {
        ans = std::min(ans, command_index);
      }
    }
  }
  return ans;
}

void PrintMatrixAccesses(std::ostream &os,
                         const std::vector<MatrixAccesses> &matrix_accesses) {
  int32 num_matrices = matrix_accesses.size();
  for (int32 m = 1; m < num_matrices; m++) {
    const MatrixAccesses &a = matrix_accesses[m];
    os << "m" << m << ": init-command=" << a.allocate_command
       << ", destroy-command=" << a.deallocate_command
       << ", accesses=";
    std::vector<Access>::const_iterator iter = a.accesses.begin(),
        end = a.accesses.end();
    for (; iter != end; ++iter)
      os << 'c' << iter->command_index << "("
         << (iter->access_type == kReadAccess ? "r" :
             (iter->access_type == kWriteAccess ? "w" : "rw")) << ") ";
    os << "\n";
  }
}

void PrintCommandAttributes(std::ostream &os,
                            const std::vector<CommandAttributes> &attributes) {
  int32 num_commands = attributes.size();
  for (int32 c = 0; c < num_commands; c++) {
    const CommandAttributes &this_attr = attributes[c];
    os << "c" << c << ": ";
    if (!this_attr.variables_read.empty()) {
      os << "r(";
      std::vector<int32>::const_iterator iter = this_attr.variables_read.begin(),
          end = this_attr.variables_read.end();
      for (; iter != end; ++iter) {
        os << "v" << *iter;
        if (iter+1 != end) os << ",";
      }
      os << ") ";
    }
    if (!this_attr.variables_written.empty()) {
      os << "w(";
      std::vector<int32>::const_iterator
          iter = this_attr.variables_written.begin(),
          end = this_attr.variables_written.end();
      for (; iter != end; ++iter) {
        os << "v" << *iter;
        if (iter+1 != end) os << ",";
      }
      os << ") ";
    }
    if (!this_attr.matrices_read.empty()) {
      os << "r(";
      std::vector<int32>::const_iterator iter = this_attr.matrices_read.begin(),
          end = this_attr.matrices_read.end();
      for (; iter != end; ++iter) {
        os << "m" << *iter;
        if (iter+1 != end) os << ",";
      }
      os << ") ";
    }
    if (!this_attr.matrices_written.empty()) {
      os << "w(";
      std::vector<int32>::const_iterator
          iter = this_attr.matrices_written.begin(),
          end = this_attr.matrices_written.end();
      for (; iter != end; ++iter) {
        os << "m" << *iter;
        if (iter+1 != end) os << ",";
      }
      os << ")";
    }
    os << "\n";
  }
}


void Analyzer::Init(const Nnet &nnet, const NnetComputation &computation) {
  variables.Init(computation);
  ComputeCommandAttributes(nnet, computation, variables, &command_attributes);
  ComputeVariableAccesses(variables, command_attributes, &variable_accesses);
  ComputeMatrixAccesses(nnet, computation, variables, command_attributes,
                        &matrix_accesses);
}

void GetCommandsOfType(const NnetComputation &computation,
                       CommandType t,
                       std::vector<int32> *command_indexes) {
  int32 num_commands = computation.commands.size();
  command_indexes->clear();
  for (int32 c = 0; c < num_commands; c++)
    if (computation.commands[c].command_type == t)
      command_indexes->push_back(c);
}

int64 GetMaxMemoryUse(const NnetComputation &computation) {
  int64 cur_memory_use = 0,
      max_memory_use = 0;
  int32 num_commands = computation.commands.size(),
      num_submatrices = computation.submatrices.size();
  // the vector 'num_compressed_bytes' is used to remember the number of bytes
  // in the compressed matrices for each submatrix (this will only be used for
  // those that correspond to a 'whole matrix).  It's needed because the
  // decompression command doesn't tell us what compression type was used for
  // that matrix.
  std::vector<int32> num_compressed_bytes(num_submatrices, -100000000);
  for (int32 command_index = 0; command_index < num_commands; ++command_index) {
    const NnetComputation::Command &c = computation.commands[command_index];
    int64 this_num_bytes = -100000000,
        this_compressed_num_bytes = -10000000;


    if (c.arg1 >= 0 && c.arg1 < num_submatrices) {
      // if arg1 could plausibly be a sub-matrix index...
      const NnetComputation::SubMatrixInfo &submat_info =
          computation.submatrices[c.arg1];
      this_num_bytes = static_cast<int64>(sizeof(BaseFloat)) *
          submat_info.num_rows * submat_info.num_cols;

      if (c.command_type == kCompressMatrix) {
        this_compressed_num_bytes =
            ((c.arg2 == static_cast<int32>(kCompressedMatrixInt8) ||
              c.arg2 == static_cast<int32>(kCompressedMatrixUint8)) ?
             1 : 2) * static_cast<int64>(submat_info.num_rows) *
            submat_info.num_cols;
        num_compressed_bytes[c.arg1] = this_compressed_num_bytes;
      } else if (c.command_type == kDecompressMatrix) {
        this_compressed_num_bytes = num_compressed_bytes[c.arg1];
      }
    }
    switch (c.command_type) {
      case kAllocMatrix:
      case kAcceptInput:
        cur_memory_use += this_num_bytes;
        break;
      case kDeallocMatrix:
        cur_memory_use -= this_num_bytes;
        break;
      case kCompressMatrix:
        cur_memory_use += this_compressed_num_bytes - this_num_bytes;
        break;
      case kDecompressMatrix:
        cur_memory_use += this_num_bytes - this_compressed_num_bytes;
        break;
      default:
        break;
    }
    KALDI_ASSERT(cur_memory_use >= 0);
    if (cur_memory_use > max_memory_use)
      max_memory_use = cur_memory_use;
  }
  return max_memory_use;
}

} // namespace nnet3
} // namespace kaldi
