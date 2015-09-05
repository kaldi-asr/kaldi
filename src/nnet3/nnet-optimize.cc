// nnet3/nnet-optimize.cc

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

#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace nnet3 {


void IdentifySubmatrixArgs(NnetComputation::Command *c,
                           std::vector<int32*> *submatrix_args) {
  submatrix_args->clear();
  switch (c->command_type) {
    case NnetComputation::kAllocMatrixZeroed:
    case NnetComputation::kAllocMatrixUndefined:
    case NnetComputation::kDeallocMatrix:
    case NnetComputation::kAllocMatrixFromOther:
    case NnetComputation::kAllocMatrixFromOtherZeroed:
      break;
    case NnetComputation::kPropagate:
      submatrix_args->push_back(&c->arg3);
      submatrix_args->push_back(&c->arg4);
      break;
    case NnetComputation::kStoreStats:
      submatrix_args->push_back(&c->arg2);
      break;
    case NnetComputation::kBackprop:
      submatrix_args->push_back(&c->arg3);
      submatrix_args->push_back(&c->arg4);
      submatrix_args->push_back(&c->arg5);
      submatrix_args->push_back(&c->arg6);
      break;
    case NnetComputation::kMatrixCopy:
    case NnetComputation::kMatrixAdd:
    case NnetComputation::kAddRows:
    case NnetComputation::kCopyRows:
    case NnetComputation::kAddRowRanges:
      submatrix_args->push_back(&c->arg1);
      submatrix_args->push_back(&c->arg2);
      break;
    case NnetComputation::kAddRowsMulti:
    case NnetComputation::kCopyRowsMulti:
    case NnetComputation::kAddToRowsMulti:
    case NnetComputation::kCopyToRowsMulti:
      submatrix_args->push_back(&c->arg1);
      break;
    case NnetComputation::kNoOperation:
    case NnetComputation::kNoOperationMarker:
      break;
    default:
      KALDI_ERR << "Unknown command type.";
  }
}

void IdentifyMatrixArgs(NnetComputation::Command *c,
                        std::vector<int32*> *matrix_args) {
  matrix_args->clear();
  switch (c->command_type) {
    case NnetComputation::kAllocMatrixZeroed:
    case NnetComputation::kAllocMatrixUndefined:
    case NnetComputation::kDeallocMatrix:
      matrix_args->push_back(&c->arg1);
      break;
    case NnetComputation::kAllocMatrixFromOther:
    case NnetComputation::kAllocMatrixFromOtherZeroed:
      matrix_args->push_back(&c->arg1);
      matrix_args->push_back(&c->arg2);
      break;
    default:
      break;
  }
}

// We declare this class in the .cc file, we don't need to export it.
// It's used inside RemoveSomeMatrices.  matrices_to_remove must be
// sorted and uniq.
class ComputationRenumberer {
 public:
  ComputationRenumberer(NnetComputation *computation):
      computation_(computation) { }

  void Renumber() {
    SetUpMappings();
    RenumberCommands();
    RenumberMatrices();
    RenumberSubmatrices();
    RenumberIndexesMulti();
    RenumberDebugInfo();
    RenumberIo();
  }
 private:
  void SetUpMappings();
  void RenumberCommands();
  void RenumberMatrices();
  void RenumberSubmatrices();
  void RenumberIndexesMulti();
  void RenumberDebugInfo();
  void RenumberIo();

  struct SubMatrixHasher {
    SubMatrixHasher() { }
    size_t operator () (const NnetComputation::SubMatrixInfo &submat) const {
      // these numbers are arbitrarily chosen primes.
      return submat.matrix_index +
          19553 * submat.row_offset +
          29297 * submat.num_rows +
          42209 * submat.col_offset +
          56527 * submat.num_cols;
    }
  };

  /// creates a renumbering that removes the elements in "to_remove",
  /// e.g. if old_num_elements = 3 and to_remove = [1], would output
  /// the vector [ 0, -1, 1 ].
  static void CreateRenumbering(int32 old_num_elements,
                                const std::vector<int32> &to_remove,
                                std::vector<int32> *renumbering);

  std::vector<int32> matrices_to_remove_;
  NnetComputation *computation_;
  int32 num_matrices_orig_;
  int32 num_submatrices_orig_;
  int32 num_matrices_new_;
  int32 num_submatrices_new_;
  std::vector<int32> old_to_new_matrix_; // numbered by orig-matrix-index, gives
                                         // new-matrix-index.  -1 for removed
                                         // ones.
  std::vector<int32> old_to_new_submatrix_; // numbered by orig-submatrix-index,
                                            // gives new-submatrix-index.  -1
                                            // for removed ones.

};

//static
void ComputationRenumberer::CreateRenumbering(
    int32 old_num_elements,
    const std::vector<int32> &to_remove,
    std::vector<int32> *renumbering) {
  KALDI_ASSERT(IsSortedAndUniq(to_remove) && old_num_elements > 0);
  renumbering->clear();
  renumbering->resize(old_num_elements, 0);
  int32 num_remove = to_remove.size();
  for (int32 r = 0; r < num_remove; r++) {
    int32 this_remove = to_remove[r];
    // the "> 0" would be ">= 0" in a more generic context, but zero is
    // not valid in this particular application.
    KALDI_ASSERT(this_remove > 0 && this_remove < old_num_elements);
    (*renumbering)[this_remove] = -1;
  }
  int32 cur_number = 0;
  for (int32 i = 0; i < old_num_elements; i++) {
    if ((*renumbering)[i] != -1)
      (*renumbering)[i] = cur_number++;
  }
  KALDI_ASSERT(cur_number == old_num_elements -
               static_cast<int32>(to_remove.size()));
}


void ComputationRenumberer::SetUpMappings() {
  KALDI_ASSERT(matrices_to_remove_.empty());
  num_matrices_orig_ = computation_->matrices.size();
  num_submatrices_orig_ = computation_->submatrices.size();

  // list of submats per matrix.
  std::vector<std::vector<int32> > submatrix_lists;
  ComputeSubmatLists(*computation_, &submatrix_lists);

  for (int32 m = 1; m < num_matrices_orig_; m++)
    if (submatrix_lists[m].empty())
      matrices_to_remove_.push_back(m);

  CreateRenumbering(num_matrices_orig_, matrices_to_remove_,
                    &old_to_new_matrix_);

  num_matrices_new_ = num_matrices_orig_ -
      static_cast<int32>(matrices_to_remove_.size());

  unordered_map<NnetComputation::SubMatrixInfo, int32,
                SubMatrixHasher> submat_map;
  int32 cur_index = 1;
  // the old_to_new_submatrix_ map will remove duplicates.
  old_to_new_submatrix_.resize(num_submatrices_orig_);
  old_to_new_submatrix_[0] = 0;
  for (int32 s = 1; s < num_submatrices_orig_; s++) {
    const NnetComputation::SubMatrixInfo &info =
        computation_->submatrices[s];
    if (submat_map.count(info) > 0)
      old_to_new_submatrix_[s] = submat_map[info];
    else
      old_to_new_submatrix_[s] = (submat_map[info] = cur_index++);
  }
  num_submatrices_new_ = cur_index;
}

void ComputationRenumberer::RenumberCommands() {
  // renumbers matrices and submatrices in commands.
  const int32 num_matrices_old = num_matrices_orig_,
      num_submatrices_old = num_submatrices_orig_;
  int32 num_commands = computation_->commands.size();
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    NnetComputation::Command &c = computation_->commands[command_index];
    {
      std::vector<int32*> submatrix_args;
      IdentifySubmatrixArgs(&c, &submatrix_args);
      std::vector<int32*>::const_iterator iter = submatrix_args.begin(),
          end = submatrix_args.end();
      for (; iter != end; ++iter) {
        int32 *submatrix_arg = *iter;
        int32 submatrix_index = *submatrix_arg,
            new_submatrix_index = old_to_new_submatrix_[submatrix_index];
        KALDI_ASSERT(submatrix_index >= 0 &&
                     submatrix_index < num_submatrices_old);
        // renumber the argument of the command.
        *submatrix_arg = new_submatrix_index;
      }
    }
    {
      std::vector<int32*> matrix_args;
      IdentifyMatrixArgs(&c, &matrix_args);
      std::vector<int32*>::const_iterator iter = matrix_args.begin(),
          end = matrix_args.end();
      for (; iter != end; ++iter) {
        int32 *matrix_arg = *iter;
        int32 matrix_index = *matrix_arg,
            new_matrix_index = old_to_new_matrix_[matrix_index];
        KALDI_ASSERT(matrix_index >= 0 && matrix_index < num_matrices_old &&
                     new_matrix_index >= 0);
        // renumber the argument of the command.
        *matrix_arg = new_matrix_index;
      }
    }
  }
}

void ComputationRenumberer::RenumberMatrices() {
  std::vector<NnetComputation::MatrixInfo> new_matrices(num_matrices_new_);
  for (int32 m = 0; m < num_matrices_orig_; m++) {
    int32 m_new = old_to_new_matrix_[m];
    if (m_new != -1)
      new_matrices[m_new] = computation_->matrices[m];
  }
  computation_->matrices = new_matrices;
}



void ComputationRenumberer::RenumberSubmatrices() {
  std::vector<NnetComputation::SubMatrixInfo> new_submatrices(
      num_submatrices_new_);
  for (int32 s = 0; s < num_submatrices_orig_; s++) {
    int32 s_new = old_to_new_submatrix_[s];
    if (s_new != -1) {
      NnetComputation::SubMatrixInfo &dest = new_submatrices[s_new];
      dest = computation_->submatrices[s];
      dest.matrix_index = old_to_new_matrix_[dest.matrix_index];
      KALDI_ASSERT(dest.matrix_index >= 0);
    }
  }
  computation_->submatrices = new_submatrices;
}

void ComputationRenumberer::RenumberIndexesMulti() {
  std::vector<std::vector<std::pair<int32,int32> > >::iterator
      iter = computation_->indexes_multi.begin(),
      end = computation_->indexes_multi.end();
  for (; iter != end; ++iter) {
    std::vector<std::pair<int32,int32> >::iterator
        iter2 = iter->begin(), end2 = iter->end();
    for (; iter2 != end2; ++iter2) {
      int32 &submatrix_index = iter2->first;
      if (submatrix_index > 0) {
        KALDI_ASSERT(submatrix_index < num_submatrices_orig_);
        submatrix_index = old_to_new_submatrix_[submatrix_index];
      }
    }
  }
}

void ComputationRenumberer::RenumberDebugInfo() {
  if (computation_->matrix_debug_info.empty())
    return;
  KALDI_ASSERT(static_cast<int32>(computation_->matrix_debug_info.size()) ==
               num_matrices_orig_);
  // we arbitrarily keep the matrix debug info from the earliest numbered matrix
  // when constructing the new debug info.  The info may sometimes differ and
  // we'll just choose to identify the matrix with one or other of the nodes.
  // this information is only consumed by human readers anyway, while debugging.
  std::vector<NnetComputation::MatrixDebugInfo> matrix_debug_info(
      num_matrices_new_);
  for (int32 m = 0; m < num_matrices_orig_; m++) {
    int32 m_new = old_to_new_matrix_[m];
    if (m_new != -1 && matrix_debug_info[m_new].indexes.empty())
      matrix_debug_info[m_new] = computation_->matrix_debug_info[m];
  }
  computation_->matrix_debug_info = matrix_debug_info;
}

void ComputationRenumberer::RenumberIo() {
  unordered_map<int32, std::pair<int32, int32> >::iterator
      iter = computation_->input_output_info.begin(),
      end = computation_->input_output_info.end();
  for (; iter != end; ++iter) {
    int32 &value_matrix_index = iter->second.first,
        &deriv_matrix_index = iter->second.second;
    KALDI_ASSERT(value_matrix_index > 0 && value_matrix_index <
                 num_matrices_orig_);
    value_matrix_index = old_to_new_matrix_[value_matrix_index];
    KALDI_ASSERT(value_matrix_index != -1);
    if (deriv_matrix_index != 0) {
      KALDI_ASSERT(deriv_matrix_index > 0 && deriv_matrix_index <
                   num_matrices_orig_);
      deriv_matrix_index = old_to_new_matrix_[deriv_matrix_index];
      KALDI_ASSERT(deriv_matrix_index > 0);
    }
  }
}


/// This function detects matrices that have no submatrices corresponding to them (due,
/// to changes made in other optimization code), and removes them from the computation.
/// It also renumbers the submatrix indexes to remove duplicates.
void RemoveOrphanMatrices(NnetComputation *computation) {
  ComputationRenumberer renumberer(computation);
  renumberer.Renumber();
}

void RemoveNoOps(NnetComputation *computation) {
  std::vector<NnetComputation::Command>::iterator
      input_iter = computation->commands.begin(),
      input_end = computation->commands.end(),
      output_iter = computation->commands.begin();
  for (; input_iter != input_end; ++input_iter) {
    if (input_iter->command_type != NnetComputation::kNoOperation) {
      *output_iter = *input_iter;
      ++output_iter;
    }
  }
  computation->commands.resize(output_iter - computation->commands.begin());
}

/// Wherever matrix orig_matrix_index appears in the input of the network
/// (i.e. in computation->input_output_info), replaces it with new_matrix_index.
/// Returns true if it did replace it.
bool ReplaceInInput(
    const Nnet &nnet,
    int32 orig_matrix_index, int32 new_matrix_index,
    NnetComputation *computation) {
  bool ans = false;
  int32 num_matrices = computation->matrices.size();
  KALDI_ASSERT(orig_matrix_index > 0 && orig_matrix_index < num_matrices &&
               new_matrix_index > 0 && new_matrix_index < num_matrices);
  unordered_map<int32, std::pair<int32, int32> >::iterator
      iter = computation->input_output_info.begin(),
      end = computation->input_output_info.end();
  for (; iter != end; ++iter) {
    int32 network_node = iter->first,
        &value_matrix_index = iter->second.first,
        &deriv_matrix_index = iter->second.second;
    if (!nnet.IsOutputNode(network_node)) {
      // value_matrix_index would be an input to the computation.
      if (value_matrix_index == orig_matrix_index) {
        value_matrix_index = new_matrix_index;
        ans = true;
      }
    } else {
      // deriv_matrix_index would be an input to the computation.
      if (deriv_matrix_index == orig_matrix_index) {
        deriv_matrix_index = new_matrix_index;
        ans = true;
      }
    }
  }
  return ans;
}

VariableMergingOptimizer::VariableMergingOptimizer(
    const NnetOptimizeOptions &config,
    const Nnet &nnet,
    const ComputationRequest &request,
    NnetComputation *computation):
    config_(config), nnet_(nnet), request_(request),
    computation_(computation) {
  analyzer_.Init(nnet, *computation);
}

bool VariableMergingOptimizer::MergeVariables() {
  if (!config_.optimize)
    return false;
  bool merged = false;
  Initialize();
  int32 num_commands = computation_->commands.size();
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    const NnetComputation::Command &c =
        computation_->commands[command_index];
    int32 s1 = -1, s2 = -1;
    if (c.command_type == NnetComputation::kMatrixCopy &&
        config_.remove_assignments) {
      s2 = c.arg1;  // s2 is the written-to matrix.
      s1 = c.arg2;
    } else if (c.command_type == NnetComputation::kPropagate &&
               config_.propagate_in_place) {
      const Component *component = nnet_.GetComponent(c.arg1);
      if (component->Properties() & kPropagateInPlace) {
        s1 = c.arg3;
        s2 = c.arg4;  // s2 is the written-to matrix.
      }
    } else if (c.command_type == NnetComputation::kBackprop &&
               config_.backprop_in_place) {
      const Component *component = nnet_.GetComponentForNode(c.arg1);
      if (component->Properties() & kBackpropInPlace) {
        s1 = c.arg5;
        s2 = c.arg6;  // s2 is the written-to matrix.
        if (s1 == c.arg3 || s2 == c.arg3 || s1 == c.arg4 || s2 == c.arg4) {
          // we don't think this should ever happen, but just out of an
          // abundance of caution: if either of these submatrix indexes are the
          // input-value or output-value args to Backprop, don't do the optimization.
          s1 = -1;
          s2 = -1;
        }
      }
    }
    if (s1 != -1 && IsCandidate(command_index, s1, s2)) {
      merged = true;
      DoMerge(command_index, s1, s2);
    }
  }
  if (merged) {
    RemoveOrphanMatrices(computation_);
    RemoveNoOps(computation_);
  }
  return merged;
}

/**
   This static function returns a SubMatrixInfo corresponding to
   replacing the matrix-index in a's "matrix_index" with, essentially, sub-matrix b.
   Of course the matrix_index will be b's "matrix_index", but we may
   have to modify the row and column offsets.  The idea is that sub-matrix
   b should have the same dimensions as
 */
static NnetComputation::SubMatrixInfo GetSubMatrixOfSubMatrix(
    const NnetComputation &computation, int32 submat_a, int32 submat_b) {
  KALDI_ASSERT(static_cast<size_t>(submat_a) < computation.submatrices.size());
  KALDI_ASSERT(static_cast<size_t>(submat_b) < computation.submatrices.size());
  const NnetComputation::SubMatrixInfo &a = computation.submatrices[submat_a],
                                       &b = computation.submatrices[submat_b];
  const NnetComputation::MatrixInfo &a_mat =
      computation.matrices[a.matrix_index];
  KALDI_ASSERT(a_mat.num_rows == b.num_rows && a_mat.num_cols == b.num_cols);
  NnetComputation::SubMatrixInfo ans;
  ans.matrix_index = b.matrix_index;
  ans.row_offset = a.row_offset + b.row_offset;
  ans.num_rows = a.num_rows;
  ans.col_offset = a.col_offset + b.col_offset;
  ans.num_cols = a.num_cols;
  return ans;
}

void VariableMergingOptimizer::DoMerge(int32 command_index,
                                       int32 s1, int32 s2) {
  NnetComputation::Command &c = computation_->commands[command_index];
  int32 m1 = computation_->submatrices[s1].matrix_index,
      m2 = computation_->submatrices[s2].matrix_index;
  KALDI_ASSERT(m1 != m2 && m1 > 0 && m2 > 0);
  { // modify submatrices for submatrices of m1 to effectively be sub-matrices of
    // s2 instead (they will refer to m2 as the matrix_index).
    std::vector<int32>::const_iterator iter = submatrix_lists_[m1].begin(),
        end = submatrix_lists_[m1].end();
    for (; iter != end; ++iter) {
      int32 submatrix_index = *iter;
      KALDI_ASSERT(computation_->submatrices[submatrix_index].matrix_index==m1);
      computation_->submatrices[submatrix_index] =
          GetSubMatrixOfSubMatrix(*computation_, submatrix_index, s2);
    }
  }
  const std::vector<MatrixAccesses> &matrix_accesses = analyzer_.matrix_accesses;
  // - If m1 was an input, replace it as an input with m2
  bool replaced = ReplaceInInput(nnet_, m1, m2, computation_);
  KALDI_ASSERT(replaced == matrix_accesses[m1].is_input);
  if (replaced) {  // Remove the command that initializes m2.
    int32 alloc_command = matrix_accesses[m2].allocate_command;
    KALDI_ASSERT(alloc_command != -1);
    computation_->commands[alloc_command].command_type =
        NnetComputation::kNoOperation;

  }
  //  - If it was case (a), replace the assignment command with a no-op.
  if (c.command_type == NnetComputation::kMatrixCopy) {
    // remove the command.
    c.command_type = NnetComputation::kNoOperation;
    c.arg1 = -1;
    c.arg2 = -1;
  }
  //   - If both m2 and m1 have commands that deallocate them, keep only the
  //    later of the two and make it refer to m2 (otherwise delete any
  //     deallocation command).

  int32 dealloc1 = matrix_accesses[m1].deallocate_command,
      dealloc2 = matrix_accesses[m2].deallocate_command;
  if (dealloc1 != -1 && dealloc2 != -1) {
    int32 earlier_index = std::min(dealloc1, dealloc2),
            later_index = std::max(dealloc1, dealloc2);
    NnetComputation::Command
        &earlier_command = computation_->commands[earlier_index],
        &later_command = computation_->commands[later_index];
    earlier_command.command_type = NnetComputation::kNoOperation;
    later_command.arg1 = m2;
  } else {
    if (dealloc1 != -1)
      computation_->commands[dealloc1].command_type =
          NnetComputation::kNoOperation;
    if (dealloc2 != -1)
      computation_->commands[dealloc2].command_type =
          NnetComputation::kNoOperation;
  }

  // Remove the original command that allocated m1, if it exists.
  if (matrix_accesses[m1].allocate_command != -1) {
    NnetComputation::Command &allocate_command = computation_->commands[
        matrix_accesses[m1].allocate_command];
    KALDI_ASSERT((allocate_command.command_type ==
                  NnetComputation::kAllocMatrixZeroed ||
                  allocate_command.command_type ==
                  NnetComputation::kAllocMatrixUndefined) &&
                 allocate_command.arg1 == m1);
    allocate_command.command_type = NnetComputation::kNoOperation;
  }
  // Prevent further optimizations touching m1 or m2 (we can
  // try again in a later round of optimization, with a new
  // instance of this class).
  matrix_already_optimized_[m1] = true;
  matrix_already_optimized_[m2] = true;
}

// see comment by declaration of this function in nnet-optimize.h.
bool VariableMergingOptimizer::IsCandidate(int32 command_index,
                                           int32 s1, int32 s2) const {
  bool is_assignment = (computation_->commands[command_index].command_type ==
                        NnetComputation::kMatrixCopy);
  if (s1 == s2) return false;
  if (!computation_->IsWholeMatrix(s1))
    return false;
  int32 m1 = computation_->submatrices[s1].matrix_index,
      m2 = computation_->submatrices[s2].matrix_index;
  if (matrix_already_optimized_[m1] || matrix_already_optimized_[m2])
    return false;
  const std::vector<MatrixAccesses> &matrix_accesses = analyzer_.matrix_accesses;
  const MatrixAccesses &m1_access = matrix_accesses[m1],
      &m2_access = matrix_accesses[m2];
  if (m1_access.is_input && !computation_->IsWholeMatrix(s2))
    return false;
  if (m1_access.is_output && !is_assignment) return false;
  if (m2_access.is_input) return false;
  // the following check would probably indicate a coding error- this
  // function should never be called if those things are empty.
  if (m1_access.accesses.empty() || m2_access.accesses.empty())
    KALDI_ERR << "Matrices never accessed [confusing].";

  if (is_assignment) {
    // check that:  m1 is never written to after command C, and
    if (MatrixIsWrittenToAfterCommand(matrix_accesses, m1, command_index))
      return false;  // m1, or equivalently s1 (since it's all of m1) is written
                     // to after command C
    // check that:
    //        - If s2 is written to after command C, then m1 is never read or written
    //          to at time >= (the first time s2 is written to after command C)
    int32 s2_write_index =
        FirstTimeSubmatrixIsWrittenToAfterCommand(analyzer_, s2, command_index);
    // the -1 in "s2_write_index - 1" is because we want to ensure that
    // m1 is never written to a time ">=" (not ">") that command.
    if (s2_write_index != -1 &&
        MatrixIsAccessedAfterCommand(matrix_accesses, m1, s2_write_index - 1))
      return false;
  } else {
    if (MatrixIsAccessedAfterCommand(matrix_accesses, m1, command_index))
      return false;
  }
  if (MatrixIsAccessedBeforeCommand(matrix_accesses, m2, command_index))
    return false;

  return true;
}



void VariableMergingOptimizer::Initialize() {
  KALDI_ASSERT(matrix_already_optimized_.empty() &&
               "You cannot call Merge twice on the same object.");
  ComputeSubmatLists(*computation_, &submatrix_lists_);
  matrix_already_optimized_.resize(computation_->matrices.size(), false);
}

// move commands that resize matrices to as late/early as possible.
void MoveSizingCommands(const Nnet &nnet, NnetComputation *computation) {
  ComputationVariables variables;
  variables.Init(*computation);
  std::vector<CommandAttributes> attributes;
  ComputeCommandAttributes(nnet, *computation, variables, &attributes);
  std::vector<std::vector<Access> > variable_accesses;
  ComputeVariableAccesses(variables, attributes, &variable_accesses);
  std::vector<MatrixAccesses> matrix_accesses;
  ComputeMatrixAccesses(nnet, *computation, variables, attributes,
                        &matrix_accesses);

  // The way we will renumber the commands is, we will first set this vector up
  // with pairs (command-index * 3, pointer-to-command), and we will then modify
  // the command-indexes in this vector to the numbers that we want, and sort
  // it.  The reason for the * 3 is so that we can number commands "just-after"
  // existing indexes (by adding 1) and "just-before" (by subtracting 1).
  int32 num_commands = computation->commands.size(),
      num_matrices = matrix_accesses.size();
  std::vector<std::pair<int32,NnetComputation::Command*> >
      commands(num_commands);
  for (int32 c = 0; c < num_commands; c++) {
    commands[c].first = c * 3;
    commands[c].second = &(computation->commands[c]);
  }
  for (int32 m = 1; m < num_matrices; m++) {
    const MatrixAccesses &ma = matrix_accesses[m];
    if (ma.allocate_command != -1) {
      // first_access_command will be index of first non-initializing access.
      int32 first_access_command = -1;
      if (!ma.accesses.empty()) {
        first_access_command = ma.accesses[0].command_index;
        if (first_access_command == ma.allocate_command) {
          if (ma.accesses.size() > 1)
            first_access_command = ma.accesses[1].command_index;
          else
            first_access_command = -1;
        }
      }
      if (first_access_command != -1) {
        KALDI_ASSERT(first_access_command > ma.allocate_command);
        // move the initialization command to just before the first access.
        commands[ma.allocate_command].first = first_access_command * 3 - 1;
      }
    }
    if (ma.deallocate_command != -1) {
      if (!ma.accesses.empty()) {
        int32 last_access_command = ma.accesses.back().command_index;
        // move the destruction command to just after the last access.
        commands[ma.deallocate_command].first = last_access_command * 3 + 1;
      }
    }
  }
  std::sort(commands.begin(), commands.end());
  std::vector<NnetComputation::Command> reordered_commands(num_commands);
  for (int32 c = 0; c < num_commands; c++)
    reordered_commands[c] = *(commands[c].second);
  computation->commands = reordered_commands;
}


// This command replaces commands of type kAllocMatrixZeroed with commands of
// type kAllocMatrixUndefined, where possible.
void RemoveUnnecessaryZeroing(const Nnet &nnet,
                              NnetComputation *computation) {
  Analyzer a;
  a.Init(nnet, *computation);

  // OK, now we'll work out which matrices have all their pieces (i.e. all the
  // variables belonging to that matrix) written to as the first instruction
  // apart from the initial zeroing.  These matrices can have the initial
  // zeroing replaced by a sizing operation that leaves the data undefined.

  int32 num_matrices = a.matrix_accesses.size();
  for (int32 matrix_index = 0; matrix_index < num_matrices; matrix_index++) {
    const MatrixAccesses &accesses = a.matrix_accesses[matrix_index];
    int32 allocate_command = accesses.allocate_command;
    if (allocate_command == -1)  // an input
      continue;  // nothing to do.
    if (computation->commands[allocate_command].command_type !=
        NnetComputation::kAllocMatrixZeroed) {
      KALDI_ASSERT(computation->commands[allocate_command].command_type ==
                   NnetComputation::kAllocMatrixUndefined);
      continue;  // already leaving it undefined, so nothing to do.
    }
    std::vector<int32> variables_for_matrix;
    a.variables.AppendVariablesForMatrix(matrix_index, &variables_for_matrix);
    bool all_variables_ok = true;  // if this stays true, it means we don't need
                                   // the initial zeroing.
    for (size_t i = 0; i < variables_for_matrix.size(); i++) {
      int32 variable_index = variables_for_matrix[i];
      const std::vector<Access> &v_accesses =
          a.variable_accesses[variable_index];
      KALDI_ASSERT(v_accesses.size() > 0 &&
                   v_accesses[0].command_index == allocate_command &&
                   v_accesses[0].access_type == kWriteAccess);
      if (v_accesses.size() > 1 &&
          v_accesses[1].access_type != kWriteAccess)
        all_variables_ok = false;  // first access after zeroing was not a write
    }
    if (all_variables_ok) {
      // Here is where the change actually happens.
      computation->commands[allocate_command].command_type =
          NnetComputation::kAllocMatrixUndefined;
    }
  }
}


/*
  This function is called from RemoveUnnecessaryAllocation.  The input is two
  sorted, unique lists, of (deallocation-commands, allocation-commands)
  e.g. (d1, d2, d3.. ), (a1, a2, a3..); and to the output is *appended* a list
  of pairs (d, a).  Each output pair must satisfy the property that d < a, and
  no member of the input lists may appear more than once in the output pairs
  (although it's OK for input a and d values not to appear in any output pairs).

  The goal of the implementation is to output as many pairs as possible, and
  secondarily for the pairs to be as close as possible to each other (to avoid
  wasting too much memory).  I'm not sure if this implementation achieves that.
*/
static void ComputeCommandPairs(
    const std::pair<std::vector<int32>, std::vector<int32> > &lists,
    std::vector<std::pair<int32,int32> > *pairs) {
  std::vector<int32> d_list = lists.first;

  std::set<int32> a_set;
  CopyVectorToSet(lists.second, &a_set);

  std::vector<int32>::reverse_iterator iter = d_list.rbegin(),
      end = d_list.rend();

  // from the latest to the earliest deallocation command...
  for (; iter != end; ++iter) {
    int32 d = *iter;
    std::set<int32>::iterator a_iter = a_set.upper_bound(d);
    // a_iter is an iterator to the first element a of the set 'a_set' such
    // that a > d, or a_set.end() if no such element exists.
    if (a_iter == a_set.end())
      continue;  // we will output no pair for this d.
    int32 a = *a_iter;
    KALDI_PARANOID_ASSERT(a > d);  // or code error
    a_set.erase(a_iter);  // remove this a from 'a_set' so it doesn't get used
                          // twice
    pairs->push_back(std::pair<int32,int32>(d, a));
  }
}

void RemoveUnnecessaryAllocation(const Nnet &nnet,
                                 NnetComputation *computation) {
  // For each size of matrix (i.e. each pair<int32,int32>), we
  // accumulate a list of indexes of deallocation commands that
  // are for that size, and a list of indexes of allocation commands
  // for that size.
  // For each distinct matrix size, we then call ComputeCommandPairs on those
  // two lists, to get pairs of (deallocation, allocation) command-indexes that
  // we can optimize out to a single command.

  // The map is from a (num-rows,num-columns) to two lists, of
  // (deallocation-commands, allocation-commands).  The order may seem
  // backwards, but that's the order of the pairs we are looking for.
  typedef unordered_map<std::pair<int32,int32>,
      std::pair<std::vector<int32>,std::vector<int32> >,
      PairHasher<int32> > MapType;
  MapType pair_map;
  int32 num_commands = computation->commands.size();
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    NnetComputation::Command &command = computation->commands[command_index];
    if (command.command_type == NnetComputation::kAllocMatrixZeroed ||
        command.command_type == NnetComputation::kAllocMatrixUndefined ||
        command.command_type == NnetComputation::kDeallocMatrix) {
      int32 m = command.arg1, num_rows = computation->matrices[m].num_rows,
          num_cols = computation->matrices[m].num_cols;
      std::pair<int32,int32> p(num_rows, num_cols);
      std::pair<std::vector<int32>,std::vector<int32> > &lists = pair_map[p];
      if (command.command_type == NnetComputation::kDeallocMatrix)
        lists.first.push_back(command_index);
      else
        lists.second.push_back(command_index);
    }
  }

  MapType::const_iterator iter = pair_map.begin(), end = pair_map.end();
  std::vector<std::pair<int32,int32> > command_pairs;
  for (; iter != end; ++iter)
    ComputeCommandPairs(iter->second, &command_pairs);

  for (size_t i = 0; i < command_pairs.size(); i++) {
    int32 dealloc_index = command_pairs[i].first,
        alloc_index = command_pairs[i].second;
    NnetComputation::Command
        &dealloc_command = computation->commands[dealloc_index],
        &alloc_command = computation->commands[alloc_index];
    KALDI_ASSERT(dealloc_command.command_type ==
                 NnetComputation::kDeallocMatrix);
    KALDI_ASSERT(alloc_command.command_type ==
                 NnetComputation::kAllocMatrixUndefined ||
                 alloc_command.command_type==
                 NnetComputation::kAllocMatrixZeroed);
    // remove the deallocation command.
    dealloc_command.command_type =  NnetComputation::kNoOperation;
    alloc_command.arg2 = dealloc_command.arg1;
    if (alloc_command.command_type ==
        NnetComputation::kAllocMatrixUndefined)
      alloc_command.command_type =
          NnetComputation::kAllocMatrixFromOther;
    else
      alloc_command.command_type =
          NnetComputation::kAllocMatrixFromOtherZeroed;
  }
  RemoveNoOps(computation);
}

void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              const ComputationRequest &request,
              NnetComputation *computation) {
  if (!config.optimize)
    return;
  bool changed = true;
  while (changed) {
    changed = false;
    VariableMergingOptimizer opt(config, nnet, request, computation);
    if (opt.MergeVariables())
      changed = true;
  }
  if (config.initialize_undefined)
    RemoveUnnecessaryZeroing(nnet, computation);

  if (config.move_sizing_commands)
    MoveSizingCommands(nnet, computation);

  if (config.allocate_from_other)
    RemoveUnnecessaryAllocation(nnet, computation);

}

// ComputationRequests are distinguished by the names and indexes
// of inputs and outputs
size_t ComputationRequestHasher::operator() (const ComputationRequest *cr) const {
  size_t ans = 0;
  std::vector<IoSpecification>::const_iterator itr = cr->inputs.begin(),
                                               end = cr->inputs.end();
  for (; itr != end; ++itr) {
    ans += IoSpecificationToInt(*itr);
  }
  itr = cr->outputs.begin();
  end = cr->outputs.end();
  for (; itr != end; ++itr) {
    ans += IoSpecificationToInt(*itr);
  }
  return ans;
}

size_t ComputationRequestHasher::IoSpecificationToInt(const IoSpecification& spec) const {
  size_t ans;
  StringHasher string_hasher;
  ans = string_hasher(spec.name);
  std::vector<Index>::const_iterator itr = spec.indexes.begin(),
                                     end = spec.indexes.end();
  for (; itr != end; ++itr) {
    ans += (*itr).n * 1619;
    ans += (*itr).t * 15649;
    ans += (*itr).x * 89809;
  }
  return ans;
}

void CachingOptimizingCompiler::UpdateCache(bool insert_new_computation) {
  if (insert_new_computation) {
    // not exist, insert
    if (computation_cache_.size() == capacity_) {
      // full, locate the least-recently-accessed request
      const typename CacheType::iterator it
         = computation_cache_.find(access_queue_.front());
      KALDI_ASSERT(it != computation_cache_.end());
      // purge the least-recently-accessed request
      computation_cache_.erase(it);
      access_queue_.pop_front();
    }
    typename AqType::iterator ait
      = access_queue_.insert(access_queue_.end(), &request_);
    computation_cache_.insert(std::make_pair(&request_,
                              std::make_pair(&computation_, ait)));
  } else {
    // exist, update access record by moving the accessed
    // request to the end of the access queue
    typename CacheType::iterator cit = computation_cache_.find(&request_);
    KALDI_ASSERT(cit != computation_cache_.end());
    access_queue_.splice(access_queue_.end(), access_queue_,
                         cit->second.second);
  }
}

const NnetComputation* CachingOptimizingCompiler::Compile(
    const ComputationRequest  &request) {
  // find computation in the cache
  typename CacheType::iterator cit = computation_cache_.find(&request_);
  if (cit == computation_cache_.end()) {
    // if not found, compile and update cache
    request_ = request;
    Compiler compiler(request_, nnet_);
    CompilerOptions opts;

    compiler.CreateComputation(opts, &computation_);

    int32 verbose_level = 4;
    if (GetVerboseLevel() >= verbose_level) {
      std::ostringstream os1;
      request.Print(os1);
      KALDI_LOG << "Computation request is " << os1.str();
      std::ostringstream os2;
      computation_.Print(os2, nnet_);
      KALDI_LOG << "Generated computation is: " << os2.str();
    }
    { // some checking.
      CheckComputationOptions check_config;
      // we can do the rewrite check since it's before optimization.
      check_config.check_rewrite = true;
      ComputationChecker checker(check_config, nnet_, request_,
                                 computation_);
      checker.Check();
    }
    Optimize(opt_config_, nnet_, request_, &computation_);
    { // check the computation again.
      CheckComputationOptions check_config;
      ComputationChecker checker(check_config, nnet_, request_, computation_);
      checker.Check();
    }
    computation_.ComputeCudaIndexes();
    UpdateCache(true);
  } else {
    // if found, get computation and update access queue
    request_ = *(cit->first);
    computation_ = *(cit->second.first);
    UpdateCache(false);
  }
  return &computation_;
}


} // namespace nnet3
} // namespace kaldi
