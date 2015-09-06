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
    case kAllocMatrixZeroed:
    case kAllocMatrixUndefined:
    case kDeallocMatrix:
    case kAllocMatrixFromOther:
    case kAllocMatrixFromOtherZeroed:
      break;
    case kPropagate:
      submatrix_args->push_back(&c->arg3);
      submatrix_args->push_back(&c->arg4);
      break;
    case kStoreStats:
      submatrix_args->push_back(&c->arg2);
      break;
    case kBackprop:
    case kBackpropNoModelUpdate:
      submatrix_args->push_back(&c->arg3);
      submatrix_args->push_back(&c->arg4);
      submatrix_args->push_back(&c->arg5);
      submatrix_args->push_back(&c->arg6);
      break;
    case kMatrixCopy:
    case kMatrixAdd:
    case kAddRows:
    case kCopyRows:
    case kAddRowRanges:
      submatrix_args->push_back(&c->arg1);
      submatrix_args->push_back(&c->arg2);
      break;
    case kAddRowsMulti:
    case kCopyRowsMulti:
    case kAddToRowsMulti:
    case kCopyToRowsMulti:
      submatrix_args->push_back(&c->arg1);
      break;
    case kNoOperation:
    case kNoOperationMarker:
      break;
    default:
      KALDI_ERR << "Unknown command type.";
  }
}

void IdentifyMatrixArgs(NnetComputation::Command *c,
                        std::vector<int32*> *matrix_args) {
  matrix_args->clear();
  switch (c->command_type) {
    case kAllocMatrixZeroed:
    case kAllocMatrixUndefined:
    case kDeallocMatrix:
      matrix_args->push_back(&c->arg1);
      break;
    case kAllocMatrixFromOther:
    case kAllocMatrixFromOtherZeroed:
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
    if (input_iter->command_type != kNoOperation) {
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
    if (nnet.IsOutputNode(network_node)) {
      // deriv_matrix_index would be an input to the computation.
      if (deriv_matrix_index == orig_matrix_index) {
        deriv_matrix_index = new_matrix_index;
        ans = true;
      }
    } else {
      // value_matrix_index would be an input to the computation.
      if (value_matrix_index == orig_matrix_index) {
        value_matrix_index = new_matrix_index;
        ans = true;
      }
    }
  }
  return ans;
}


/// Wherever matrix orig_matrix_index appears in the output of the network
/// (i.e. in computation->input_output_info), replaces it with new_matrix_index.
/// Returns true if it did replace it.
bool ReplaceInOutput(
    const Nnet &nnet, int32 orig_matrix_index, int32 new_matrix_index,
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
    if (nnet.IsOutputNode(network_node)) {
      // value_matrix_index would be an output of the computation.
      if (value_matrix_index == orig_matrix_index) {
        value_matrix_index = new_matrix_index;
        ans = true;
      }
    } else {
      // deriv_matrix_index would be an output of the computation.
      if (deriv_matrix_index == orig_matrix_index) {
        // we'd only have derivatives for actual inputs. [note: we also allow
        // users to provide inputs for component nodes, but these would not have
        // derivatives.]
        KALDI_ASSERT(nnet.IsInputNode(network_node));
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
    computation_(computation),
    already_called_merge_variables_(false) {
  analyzer_.Init(nnet, *computation);
  ComputeSubmatLists(*computation_, &submatrix_lists_);
  variable_dirty_.resize(analyzer_.variables.NumVariables(), false);
}

bool VariableMergingOptimizer::MergeVariables() {
  KALDI_ASSERT(!already_called_merge_variables_);
  already_called_merge_variables_ = true;
  if (!config_.optimize)
    return false;
  bool merged = false;
  int32 num_commands = computation_->commands.size();
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    // This loop looks for pairs of sub-matrix indexes s1,s2 that we could
    // potentially merge into a single variable.
    const NnetComputation::Command &c =
        computation_->commands[command_index];
    int32 s1 = -1, s2 = -1;
    if (c.command_type == kMatrixCopy &&
        config_.remove_assignments) {
      s2 = c.arg1;  // s2 is the written-to matrix.
      s1 = c.arg2;
    } else if (c.command_type == kPropagate &&
               config_.propagate_in_place) {
      const Component *component = nnet_.GetComponent(c.arg1);
      if (component->Properties() & kPropagateInPlace) {
        s1 = c.arg3;
        s2 = c.arg4;  // s2 is the written-to matrix.
      }
    } else if ((c.command_type == kBackprop ||
                c.command_type == kBackpropNoModelUpdate) &&
               config_.backprop_in_place) {
      const Component *component = nnet_.GetComponent(c.arg1);
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
    if (s1 > 0 && s2 > 0) {
      std::pair<bool,bool> p = MayBeMerged(command_index, s1, s2);
      if (p.first) {
        DoLeftMerge(command_index, s1, s2);
        merged = true;
      } else if (p.second) {
        DoRightMerge(command_index, s1, s2);
        merged = true;
      }
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

void VariableMergingOptimizer::MarkAsDirty(int32 s) {
  std::vector<int32> variable_indexes;
  analyzer_.variables.AppendVariablesForSubmatrix(s, &variable_indexes);
  std::vector<int32>::const_iterator iter = variable_indexes.begin(),
      end = variable_indexes.end();
  for (; iter != end; ++iter) {
    int32 v = *iter;
    KALDI_ASSERT(static_cast<size_t>(v) < variable_dirty_.size());
    variable_dirty_[v] = true;
  }
}

void VariableMergingOptimizer::DoRightMerge(int32 command_index,
                                            int32 s1, int32 s2) {
  // Prevent further optimizations touching s1 or s2 (we can
  // try again in a later round of optimization, with a new
  // instance of this class).
  MarkAsDirty(s1);
  MarkAsDirty(s2);

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
  if (replaced) {  // Remove the command that allocates m2.
    int32 alloc_command = matrix_accesses[m2].allocate_command;
    KALDI_ASSERT(alloc_command != -1);
    computation_->commands[alloc_command].command_type =
        kNoOperation;
  }
  // we keep matrix m2 (so m2 is m_to_keep, m1 is m_to_discard).
  DoMergeCommon(command_index, m2, m1);
}

void VariableMergingOptimizer::DoMergeCommon(int32 command_index,
                                             int32 m_to_keep,
                                             int32 m_to_discard) {
  NnetComputation::Command &c = computation_->commands[command_index];
  const std::vector<MatrixAccesses> &matrix_accesses =
      analyzer_.matrix_accesses;

  //  - If it was case (a), replace the assignment command with a no-op.
  if (c.command_type == kMatrixCopy) {
    // remove the command.
    c.command_type = kNoOperation;
    c.arg1 = -1;
    c.arg2 = -1;
  }

  //   - If both m_to_keep and m_to_discard have commands that deallocate them, keep only the
  //    later of the two and make it refer to m_to_keep (otherwise delete any
  //     deallocation command).
  int32 dealloc1 = matrix_accesses[m_to_keep].deallocate_command,
      dealloc2 = matrix_accesses[m_to_discard].deallocate_command;
  if (dealloc1 != -1 && dealloc2 != -1) {
    int32 earlier_index = std::min(dealloc1, dealloc2),
            later_index = std::max(dealloc1, dealloc2);
    NnetComputation::Command
        &earlier_command = computation_->commands[earlier_index],
        &later_command = computation_->commands[later_index];
    earlier_command.command_type = kNoOperation;
    later_command.arg1 = m_to_keep;
  } else {
    if (dealloc1 != -1)
      computation_->commands[dealloc1].command_type =
          kNoOperation;
    if (dealloc2 != -1)
      computation_->commands[dealloc2].command_type =
          kNoOperation;
  }

  //   - If both m_to_keep and m_to_discard have commands that allocate them,
  //     keep only the earlier of the two and make it refer to m_to_keep
  //     (otherwise delete any allocation command).
  int32 alloc1 = matrix_accesses[m_to_keep].allocate_command,
      alloc2 = matrix_accesses[m_to_discard].allocate_command;
  if (alloc1 != -1 && alloc2 != -1) {
    int32 earlier_index = std::min(alloc1, alloc2),
        later_index = std::max(alloc1, alloc2);
    NnetComputation::Command
        &earlier_command = computation_->commands[earlier_index],
        &later_command = computation_->commands[later_index];
    later_command.command_type = kNoOperation;
    earlier_command.arg1 = m_to_keep;
    // Make sure we don't initialize as undefined- checking that
    // that is correct would require some analysis.  We'll deal with
    // that in a later optimization pass.
    if (earlier_command.command_type == kAllocMatrixUndefined) {
      earlier_command.command_type = kAllocMatrixZeroed;
    } else if (earlier_command.command_type == kAllocMatrixFromOther) {
      earlier_command.command_type = kAllocMatrixFromOtherZeroed;
    }
  } else {
    if (alloc1 != -1)
      computation_->commands[alloc1].command_type =
          kNoOperation;
    if (alloc2 != -1)
      computation_->commands[alloc2].command_type =
          kNoOperation;
  }
}

void VariableMergingOptimizer::DoLeftMerge(int32 command_index,
                                           int32 s1, int32 s2) {
  // Prevent further optimizations touching s1 or s2 (we can
  // try again in a later round of optimization, with a new
  // instance of this class).
  MarkAsDirty(s1);
  MarkAsDirty(s2);

  int32 m1 = computation_->submatrices[s1].matrix_index,
      m2 = computation_->submatrices[s2].matrix_index;
  KALDI_ASSERT(m1 != m2 && m1 > 0 && m2 > 0);
  { // modify submatrices for submatrices of m2 to effectively be sub-matrices of
    // s1 instead (they will refer to m1 as the matrix_index).
    std::vector<int32>::const_iterator iter = submatrix_lists_[m2].begin(),
        end = submatrix_lists_[m2].end();
    for (; iter != end; ++iter) {
      int32 submatrix_index = *iter;
      KALDI_ASSERT(computation_->submatrices[submatrix_index].matrix_index==m2);
      computation_->submatrices[submatrix_index] =
          GetSubMatrixOfSubMatrix(*computation_, submatrix_index, s1);
    }
  }
  const std::vector<MatrixAccesses> &matrix_accesses = analyzer_.matrix_accesses;
  // - If m2 was an output, replace it as an input with m1.
  bool replaced = ReplaceInOutput(nnet_, m2, m1, computation_);
  KALDI_ASSERT(replaced == matrix_accesses[m2].is_output);
  if (replaced) {  // Remove the command that deallocates m1.
    int32 dealloc_command = matrix_accesses[m1].deallocate_command;
    KALDI_ASSERT(dealloc_command != -1);
    computation_->commands[dealloc_command].command_type =
        kNoOperation;
  }
  // we keep matrix m1 (so m1 is m_to_keep, m2 is m_to_discard).
  DoMergeCommon(command_index, m1, m2);
}




std::pair<bool,bool> VariableMergingOptimizer::MayBeMerged(
    int32 command_index, int32 s1, int32 s2) const {
  KALDI_ASSERT(s1 > 0 && s2 > 0 && static_cast<size_t>(command_index) <
               computation_->commands.size());
  if (!config_.allow_left_merge && !config_.allow_right_merge)
    return std::pair<bool,bool>(false,false);
  int32 m1 = computation_->submatrices[s1].matrix_index,
      m2 = computation_->submatrices[s2].matrix_index;
  // we can't merge two different submatrices of the same matrix.
  if (m1 == m2) return std::pair<bool,bool>(false,false);
  std::vector<int32> variable_indexes;
  analyzer_.variables.AppendVariablesForSubmatrix(s1, &variable_indexes);
  analyzer_.variables.AppendVariablesForSubmatrix(s2, &variable_indexes);
  std::vector<int32>::iterator iter = variable_indexes.begin(),
      end = variable_indexes.end();
  // condition c5:
  for (; iter != end; ++iter)
    if (variable_dirty_[*iter])
      return std::pair<bool,bool>(false,false);
  const std::vector<MatrixAccesses> &matrix_accesses = analyzer_.matrix_accesses;
  const MatrixAccesses &m1_access = matrix_accesses[m1],
      &m2_access = matrix_accesses[m2];
  // condition c1:
  if ((m1_access.is_input && m2_access.is_input) ||
      (m1_access.is_output && m2_access.is_output))
    return std::pair<bool,bool>(false,false);
  // condition c2:
  if ((m1_access.is_input || m1_access.is_output ||
       m2_access.is_input || m2_access.is_output) &&
      (!computation_->IsWholeMatrix(s1) ||
       !computation_->IsWholeMatrix(s2)))
    return std::pair<bool,bool>(false,false);
  bool left = config_.allow_left_merge,
      right = config_.allow_right_merge;
  // condition c3:
  if (!computation_->IsWholeMatrix(s2)) left = false;
  if (!computation_->IsWholeMatrix(s1)) right = false;
  if (!left && !right)  // save some time.
    return std::pair<bool,bool>(false,false);
  bool is_assignment = (computation_->commands[command_index].command_type ==
                        kMatrixCopy);
  ComputationAnalysis analysis(*computation_, analyzer_);
  if (is_assignment) {
    if (analysis.FirstAccess(s2) == command_index &&
        analysis.LastWriteAccess(s1) < command_index &&
        analysis.LastAccess(s1) <
        analysis.DataInvalidatedCommand(command_index, s2)) {
      return std::pair<bool,bool>(left, right);  // possible success.
    }
  } else {
    if (analysis.FirstAccess(s2) == command_index &&
        analysis.LastAccess(s1) == command_index) {
      return std::pair<bool,bool>(left, right);  // possible success.
    }
  }
  // failure.
  return std::pair<bool,bool>(false,false);
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
        kAllocMatrixZeroed) {
      KALDI_ASSERT(computation->commands[allocate_command].command_type ==
                   kAllocMatrixUndefined);
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
          kAllocMatrixUndefined;
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
    if (command.command_type == kAllocMatrixZeroed ||
        command.command_type == kAllocMatrixUndefined ||
        command.command_type == kDeallocMatrix) {
      int32 m = command.arg1, num_rows = computation->matrices[m].num_rows,
          num_cols = computation->matrices[m].num_cols;
      std::pair<int32,int32> p(num_rows, num_cols);
      std::pair<std::vector<int32>,std::vector<int32> > &lists = pair_map[p];
      if (command.command_type == kDeallocMatrix)
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
                 kDeallocMatrix);
    KALDI_ASSERT(alloc_command.command_type ==
                 kAllocMatrixUndefined ||
                 alloc_command.command_type==
                 kAllocMatrixZeroed);
    // remove the deallocation command.
    dealloc_command.command_type =  kNoOperation;
    alloc_command.arg2 = dealloc_command.arg1;
    if (alloc_command.command_type ==
        kAllocMatrixUndefined)
      alloc_command.command_type =
          kAllocMatrixFromOther;
    else
      alloc_command.command_type =
          kAllocMatrixFromOtherZeroed;
  }
  RemoveNoOps(computation);
}

void VariableMergingOptimization(const NnetOptimizeOptions &config,
                                 const Nnet &nnet,
                                 const ComputationRequest &request,
                                 NnetComputation *computation) {
  bool changed = true;
  while (changed) {
    changed = false;
    VariableMergingOptimizer opt(config, nnet, request, computation);
    if (opt.MergeVariables())
      changed = true;
  }
}

void ModelUpdateConsolidator::AppendDebugInfoForSubmatrix(
    int32 submatrix_index,
    NnetComputation::MatrixDebugInfo *debug_info) const {
  KALDI_ASSERT(!computation_->matrix_debug_info.empty());
  KALDI_ASSERT(static_cast<size_t>(submatrix_index) <
               computation_->submatrices.size());
  NnetComputation::SubMatrixInfo submatrix_info =
      computation_->submatrices[submatrix_index];
  int32 matrix_index = submatrix_info.matrix_index;
  KALDI_ASSERT(matrix_index > 0 && static_cast<size_t>(matrix_index) <
               computation_->matrix_debug_info.size());
  const NnetComputation::MatrixDebugInfo &src_info =
      computation_->matrix_debug_info[matrix_index];
  debug_info->is_deriv = src_info.is_deriv;
  if (debug_info->node_index == -1) {
    debug_info->node_index = src_info.node_index;
  } else if (debug_info->node_index != src_info.node_index) {
    // I'd make this an error, but if we end up calling this optimization
    // after other optimizations such as variable-merging optimizations,
    // it's possible the debug info could get mixed up.
    KALDI_WARN << "Unexpected node-index mismatch in debug info: "
               << nnet_.GetNodeName(debug_info->node_index) << " vs. "
               << nnet_.GetNodeName(src_info.node_index);
  }
  KALDI_ASSERT(src_info.indexes.size() ==
               computation_->matrices[matrix_index].num_rows);
  int32 row_begin = submatrix_info.row_offset,
      row_end = row_begin + submatrix_info.num_rows;
  debug_info->indexes.insert(debug_info->indexes.end(),
                             src_info.indexes.begin() + row_begin,
                             src_info.indexes.begin() + row_end);
}


int32 ModelUpdateConsolidator::ConsolidateSubmatrices(
    const std::vector<int32> &commands,
    const std::vector<int32> &submatrices) {
  int32 num_submatrices = submatrices.size();
  KALDI_ASSERT(num_submatrices > 1 && commands.size() == submatrices.size());
  int32 first_submatrix = submatrices[0];
  int32 num_cols = computation_->submatrices[first_submatrix].num_cols,
      num_rows = 0;
  NnetComputation::MatrixDebugInfo debug_info;
  for (int32 i = 0; i < num_submatrices; i++) {
    int32 submatrix = submatrices[i];
    num_rows += computation_->submatrices[submatrix].num_rows;
    KALDI_ASSERT(computation_->submatrices[submatrix].num_cols == num_cols);
    if (!computation_->matrix_debug_info.empty())
      AppendDebugInfoForSubmatrix(submatrix, &debug_info);
  }
  // new_whole_submatrix is a new submatrix index corresponding to the whole
  // of a new matrix that we are creating.
  int32 new_whole_submatrix = computation_->NewMatrix(num_rows, num_cols);
  // Add a command at the very start, to initialize this new matrix.
  int32 new_matrix_index =
      computation_->submatrices[new_whole_submatrix].matrix_index;
  extra_commands_[0].push_back(
      NnetComputation::Command(kAllocMatrixUndefined, new_matrix_index));
  final_deallocate_commands_.push_back(
      NnetComputation::Command(kDeallocMatrix, new_matrix_index));
  if (!computation_->matrix_debug_info.empty())
    computation_->matrix_debug_info[new_matrix_index].Swap(&debug_info);

  int32 row_offset = 0;
  for (int32 i = 0; i < num_submatrices; i++) {
    int32 submatrix_index = submatrices[i];
    int32 this_num_rows = computation_->submatrices[submatrix_index].num_rows;
    // submatrix corresponding to the part of the new matrix corresponding
    // to 'submatrices[i]'.
    int32 new_submatrix = computation_->NewSubMatrix(new_whole_submatrix,
                                                     row_offset, this_num_rows,
                                                     0, num_cols);
    // Just before command 'commands[i]', add a command that assigns to the
    // submatrix numbered 'new_submatrix' the contents of the submatrix numbered
    // 'submatrices[i]'.  Note: we hope that a later pass of optimization
    // (VariableMergingOptimization) will remove this redundant copy by
    // having the operation that created it right directly to the location
    // we want it to be.
    NnetComputation::Command c(kMatrixCopy, new_submatrix, submatrices[i]);
    extra_commands_[commands[i]].push_back(c);
    row_offset += this_num_rows;
  }
  KALDI_ASSERT(row_offset == num_rows);
  return new_whole_submatrix;
}

void ModelUpdateConsolidator::AddCommandsToComputation() {
  KALDI_ASSERT(computation_->commands.size() == extra_commands_.size());
  int32 old_num_commands = computation_->commands.size(),
      new_num_commands = old_num_commands +
      static_cast<int32>(final_commands_.size() +
                         final_deallocate_commands_.size());
  for (size_t i = 0; i < extra_commands_.size(); i++)
    new_num_commands += static_cast<int32>(extra_commands_[i].size());
  std::vector<NnetComputation::Command> new_commands;
  new_commands.reserve(new_num_commands);
  for (int32 c = 0; c < old_num_commands; c++) {
    new_commands.insert(new_commands.end(),
                        extra_commands_[c].begin(), extra_commands_[c].end());
    new_commands.push_back(computation_->commands[c]);
  }
  new_commands.insert(new_commands.end(),
                      final_commands_.begin(), final_commands_.end());
  new_commands.insert(new_commands.end(),
                      final_deallocate_commands_.begin(),
                      final_deallocate_commands_.end());
  computation_->commands.swap(new_commands);
}

/** This function, called from ConsolidateModelUpdate, is passed a list of
    commands that are all backprops for the same component, and it consolidates
    them into a single model-update command. */
void ModelUpdateConsolidator::ConsolidateUpdateForComponent(
    int32 component_index,
    const std::vector<int32> &backprop_commands) {
  const Component *component = nnet_.GetComponent(component_index);
  int32 num_backprop_commands = backprop_commands.size();

  bool need_input = (component->Properties() & kBackpropNeedsInput) != 0,
      need_output = (component->Properties() & kBackpropNeedsOutput) != 0;

  std::vector<int32>  input_submatrices(num_backprop_commands),
      output_submatrices(num_backprop_commands),
      output_deriv_submatrices(num_backprop_commands);

  for (int32 i = 0; i < num_backprop_commands; i++) {
    int32 command_index = backprop_commands[i];
    NnetComputation::Command &command =
        computation_->commands[command_index];
    // arg2 must be 0 because simple components don't use precomputed indexes.
    KALDI_ASSERT(command.command_type == kBackprop && command.arg2 == 0);
    command.command_type = kBackpropNoModelUpdate;
    int32 input_submatrix = command.arg3,
        output_submatrix = command.arg4,
        output_deriv_submatrix = command.arg5;
    KALDI_ASSERT((input_submatrix != 0) == need_input &&
                 (output_submatrix != 0) == need_output);
    input_submatrices[i] = input_submatrix;
    output_submatrices[i] = output_submatrix;
    output_deriv_submatrices[i] = output_deriv_submatrix;
  }
  // Get the sub-matrix indexes of whichever of the consolidated matrices we
  // need (will usually be input_submatrix and output_deriv_submatrix).
  int32 input_submatrix = (need_input ?
                           ConsolidateSubmatrices(backprop_commands,
                                                  input_submatrices) : 0),
      output_submatrix = (need_output ?
                         ConsolidateSubmatrices(backprop_commands,
                                                output_submatrices) : 0),
      output_deriv_submatrix = ConsolidateSubmatrices(backprop_commands,
                                                      output_deriv_submatrices);
  int32 precomputed_indexes_index = 0,  // unused since simple component
      input_deriv_submatrix = 0;  // we don't need the input-deriv, so this is
                                  // zero.
  NnetComputation::Command c(kBackprop, component_index, precomputed_indexes_index,
                             input_submatrix, output_submatrix,
                             output_deriv_submatrix, input_deriv_submatrix);
  final_commands_.push_back(c);
}

ModelUpdateConsolidator::ModelUpdateConsolidator(
    const Nnet &nnet,
    NnetComputation *computation):
    nnet_(nnet), computation_(computation),
    extra_commands_(computation->commands.size()) { }

void ModelUpdateConsolidator::ConsolidateModelUpdate() {
  int32 num_components = nnet_.NumComponents(),
      num_commands = computation_->commands.size();
  // 'backprop_commands' is a list, for each component (but nonempty only for
  // updatable components), of the command indexes for the backprop commands.
  std::vector<std::vector<int32> > backprop_commands(num_components);
  std::vector<NnetComputation::Command>::const_iterator iter =
      computation_->commands.begin(), end = computation_->commands.end();
  for (int32 command_index = 0;
       command_index < num_commands; command_index++) {
    const NnetComputation::Command &c = computation_->commands[command_index];
    if (c.command_type == kBackprop) {
      int32 component_index = c.arg1;
      const Component *component = nnet_.GetComponent(component_index);
      if (component->Properties() & kUpdatableComponent)
        backprop_commands[component_index].push_back(command_index);
    }
  }
  bool consolidated = false;
  for (int32 component = 0; component < num_components; component++) {
    if (backprop_commands[component].size() > 1) {
      ConsolidateUpdateForComponent(component,
                                    backprop_commands[component]);
      consolidated = true;
    }
  }
  if (!consolidated)  // This is an optimization to avoid redundant computation
    return;           // if there is nothing to do.
  // the following function call commits all the commands we stored in member
  // variables, to computation_->commands.
  AddCommandsToComputation();
}


// This is a simplified top-level interface to the model-update consolidation
// code from class ModelUpdateConsolidator.
void ConsolidateModelUpdate(const Nnet &nnet,
                            const ComputationRequest &request,
                            NnetComputation *computation) {
  if (!request.need_model_derivative)
    return;   // An optimization; there would be nothing to do in this case.
  ModelUpdateConsolidator consolidator(nnet, computation);
  consolidator.ConsolidateModelUpdate();
}


void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              const ComputationRequest &request,
              NnetComputation *computation) {
  if (!config.optimize)
    return;

  if (config.consolidate_model_update)
    ConsolidateModelUpdate(nnet, request, computation);

  if (config.remove_assignments || config.backprop_in_place ||
      config.propagate_in_place)
    VariableMergingOptimization(config, nnet, request, computation);

  if (config.initialize_undefined)
    RemoveUnnecessaryZeroing(nnet, computation);

  if (config.move_sizing_commands)
    MoveSizingCommands(nnet, computation);

  if (config.allocate_from_other)
    RemoveUnnecessaryAllocation(nnet, computation);

}

const NnetComputation* CachingOptimizingCompiler::Compile(
    const ComputationRequest  &request) {
  if (!(request == request_)) {
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
  }
  return &computation_;
}


} // namespace nnet3
} // namespace kaldi
