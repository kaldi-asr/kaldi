// nnet3/nnet-optimize-utils.cc

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

#include <map>
#include "nnet3/nnet-optimize-utils.h"
#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace nnet3 {


void IdentifySubmatrixArgs(NnetComputation::Command *c,
                           std::vector<int32*> *submatrix_args) {
  submatrix_args->clear();
  switch (c->command_type) {
    case kAllocMatrix:
    case kDeallocMatrix:
    case kSetConst:
      submatrix_args->push_back(&c->arg1);
      break;
    case kSwapMatrix:
      submatrix_args->push_back(&c->arg1);
      submatrix_args->push_back(&c->arg2);
      break;
    case kPropagate:
      submatrix_args->push_back(&c->arg3);
      submatrix_args->push_back(&c->arg4);
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
    case kAcceptInput: case kProvideOutput:
      submatrix_args->push_back(&c->arg1);
      break;
    case kNoOperation:
    case kNoOperationPermanent:
    case kNoOperationMarker:
    case kNoOperationLabel:
    case kGotoLabel:
      break;
    default:
      KALDI_ERR << "Unknown command type.";
  }
}

void IdentifySubmatrixArgs(std::vector<NnetComputation::Command> *commands,
                           std::vector<int32*> *submatrix_args) {
  submatrix_args->clear();
  std::vector<NnetComputation::Command>::iterator iter = commands->begin(),
      end = commands->end();
  std::vector<int32*> this_submatrix_args;
  for (; iter != end; ++iter) {
    IdentifySubmatrixArgs(&(*iter), &this_submatrix_args);
    submatrix_args->insert(submatrix_args->end(),
                           this_submatrix_args.begin(),
                           this_submatrix_args.end());
  }
}



void IdentifyMatrixArgsInComputation(NnetComputation *computation,
                                     std::vector<int32*> *matrix_args) {
  int32 num_submatrices = computation->submatrices.size();
  matrix_args->reserve(computation->submatrices.size());
  for (int32 s = 1; s < num_submatrices; s++)
    matrix_args->push_back(&(computation->submatrices[s].matrix_index));
}


void IdentifyIndexesMultiArgs(std::vector<NnetComputation::Command> *commands,
                              std::vector<int32*> *indexes_multi_args) {
  indexes_multi_args->clear();
  std::vector<NnetComputation::Command>::iterator iter = commands->begin(),
      end = commands->end();
  for (; iter != end; ++iter) {
    NnetComputation::Command &command = *iter;
    if (command.command_type == kAddRowsMulti ||
        command.command_type == kAddToRowsMulti ||
        command.command_type == kCopyRowsMulti ||
        command.command_type == kCopyToRowsMulti)
      indexes_multi_args->push_back(&(command.arg2));
  }
}


void IdentifyIndexesRangesArgs(std::vector<NnetComputation::Command> *commands,
                               std::vector<int32*> *indexes_ranges_args) {
  indexes_ranges_args->clear();
  std::vector<NnetComputation::Command>::iterator iter = commands->begin(),
      end = commands->end();
  for (; iter != end; ++iter) {
    NnetComputation::Command &command = *iter;
    if (command.command_type == kAddRowRanges)
      indexes_ranges_args->push_back(&(command.arg3));
  }
}

void IdentifyIndexesArgs(std::vector<NnetComputation::Command> *commands,
                         std::vector<int32*> *indexes_args) {
  indexes_args->clear();
  std::vector<NnetComputation::Command>::iterator iter = commands->begin(),
      end = commands->end();
  for (; iter != end; ++iter) {
    NnetComputation::Command &command = *iter;
    if (command.command_type == kCopyRows ||
        command.command_type == kAddRows)
      indexes_args->push_back(&(command.arg3));
  }
}

// We declare this class in the .cc file, we don't need to export it.
// It's used inside RenumberComputation.
class ComputationRenumberer {
 public:
  ComputationRenumberer(NnetComputation *computation):
      computation_(computation) { }

  void Renumber();
 private:
  // this function removes unused vectors within the indexes_multi_ array, i.e.
  // ones that are not referenced in the computation.
  void RemoveUnusedIndexesMulti();
  // this function computes the submatrix_is_used_ vector, saying whether each
  // of the original submatrices is referenced somewhere.
  void ComputeSubmatrixIsUsed();
  // this function computes the matrix_is_used_ vector (from the
  // submatrix_is_used_ vector, from computation_->input_output_info, and from
  // computation_->commands, saying whether each of the original matrices is
  // referenced somewhere, directly or indirectly.
  void ComputeMatrixIsUsed();
  // This function sets up mappings from old to new matrix and submatrix indexes,
  // writing to num_{,sub}matrices_new_ and old_to_new_{,sub}matrix_.
  void SetUpMappings();
  // This function renumbers submatrix indexes appearing within commands and
  // indexes_multi_, and then removes unused submatrices from the list of
  // submatrices while leaving the matrix-indexes at their old values (they will
  // be mapped by RenumberMatrices()).
  void RenumberSubmatrices();
  // renumber matrix indexes appearing within 'commmands', within 'submatrices'
  // and 'input_output_info'; renumber 'matrices' and if applicable
  // 'debug_info'.
  void RenumberMatrices();
  // removes duplicates within the indexes_multi array itself.
  void RemoveIndexesMultiDuplicates();
  // removes unused elements and duplicates within 'computation->indexes'
  void RenumberIndexes();
  // removes unused elements and duplicates within 'computation->indexes_ranges'
  void RenumberIndexesRanges();
  // renumbers memos, removing any gaps between memo indexes.
  void RenumberMemos();

  struct SubMatrixHasher {
    SubMatrixHasher() { }
    size_t operator () (const NnetComputation::SubMatrixInfo &submat) const noexcept {
      // these numbers are arbitrarily chosen primes.
      return submat.matrix_index +
          19553 * submat.row_offset +
          29297 * submat.num_rows +
          42209 * submat.col_offset +
          56527 * submat.num_cols;
    }
  };


  // Here, T will be int32 or std::pair<int32,int32>
  template <class T>
  struct PointerCompare {
    // This provides an operator < on two vectors of ints or pairs of ints.  It
    // is designed to provide a total order on the vectors while accessing as
    // small a portion of the vectors' data as possible.  It's used in removing
    // duplicates from computation_->indexes_multi and computation_->indexes.
    // First it compares the length, then it does lexicographical compare.
    bool operator ()(const std::vector<T> *ptr1,
                     const std::vector<T> *ptr2) const {
      size_t size1 = ptr1->size(), size2 = ptr2->size();
      if (size1 < size2) return true;
      else if (size1 > size2) return false;
      else return (*ptr1 < *ptr2);  // use the std::vector operator <, which is
                                    // lexicographical comparison.
    }
  };

  /// creates a renumbering that removes the elements in "to_remove",
  /// e.g. if old_num_elements = 3 and to_remove = [1], would output
  /// the vector [ 0, -1, 1 ].
  static void CreateRenumbering(int32 old_num_elements,
                                const std::vector<int32> &to_remove,
                                std::vector<int32> *renumbering);

  /// creates a renumbering from old to new index that removes the unused
  /// elements, e.g. if used == [ true, false, true, true], would output the
  /// vector [ 0, -1, 1, 2 ].  Returns number of new elements, i.e. the
  /// number of elements of 'used' that were true.
  static int32 CreateRenumbering(const std::vector<bool> &used,
                                 std::vector<int32> *renumbering);

  // vector of bool indexed by original submatrix-index, that is true if a
  // submatrix-index is used somewhere in the computation (always true for
  // the zeroth element).
  std::vector<bool> submatrix_is_used_;
  // vector of bool indexed by original submatrix-index, that is true if a
  // submatrix-index will be kept; this is like submatrix_is_used_; but for
  // duplicate submatrices, all but the first duplicate will be marked false).
  std::vector<bool> submatrix_is_kept_;
  // vector of bool indexed by original-matrix-index > 0, that is true if a
  // matrix-index is used somewhere in the computation, directly or indirectly.
  // always true for the zeroth element.
  std::vector<bool> matrix_is_used_;
  NnetComputation *computation_;
  int32 num_matrices_new_;
  int32 num_submatrices_new_;
  std::vector<int32> old_to_new_matrix_; // numbered by orig-matrix-index, gives
                                         // new-matrix-index.  -1 for removed
                                         // ones.
  std::vector<int32> old_to_new_submatrix_; // numbered by orig-submatrix-index,
                                            // gives new-submatrix-index.  -1
                                            // for removed ones.
};

// static
int32 ComputationRenumberer::CreateRenumbering(
    const std::vector<bool> &used,
    std::vector<int32> *renumbering) {
  renumbering->clear();
  renumbering->reserve(used.size());
  std::vector<bool>::const_iterator iter = used.begin(), end = used.end();
  int32 cur_index = 0;
  for (; iter != end; ++iter) {
    if (*iter) renumbering->push_back(cur_index++);
    else renumbering->push_back(-1);
  }
  return cur_index;
}

// static
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


void ComputationRenumberer::RenumberMemos() {
  // this is indexed by memo-index, and maps to the
  // (propagate, backprop) commands that use that memo-index, or
  // (-1, -1) if there are no such commands.
  std::vector<std::pair<int32, int32> > memo_to_commands;
  std::vector<int32> memo_indexes_used;
  std::pair<int32, int32> blank(-1, -1);
  int32 num_commands = computation_->commands.size();
  for (int32 c = 0; c < num_commands; c++) {
    NnetComputation::Command &command = computation_->commands[c];
    if (command.command_type == kPropagate) {
      int32 memo_index = command.arg5;
      if (memo_index > 0) {
        if (memo_to_commands.size() <= static_cast<size_t>(memo_index))
          memo_to_commands.resize(memo_index + 1, blank);
        KALDI_ASSERT(memo_to_commands[memo_index].first == -1);
        memo_to_commands[memo_index].first = c;
        memo_indexes_used.push_back(memo_index);
      }
    } else if (command.command_type == kBackprop) {
      int32 memo_index = command.arg7;
      if (memo_index > 0) {
        if (memo_to_commands.size() <= static_cast<size_t>(memo_index))
          memo_to_commands.resize(memo_index + 1, blank);
        KALDI_ASSERT(memo_to_commands[memo_index].first > 0 &&
                     memo_to_commands[memo_index].second == -1);
        memo_to_commands[memo_index].second = c;
      }
    }
  }
  int32 new_memo_index = 1;
  for (std::vector<int32>::iterator iter = memo_indexes_used.begin();
       iter != memo_indexes_used.end(); ++iter) {
    int32 memo_index = *iter;
    int32 propagate_command = memo_to_commands[memo_index].first,
        backprop_command = memo_to_commands[memo_index].second;
    KALDI_ASSERT(backprop_command > 0 &&
                 "Propagate generates memo but backprop doesn't use it.");
    computation_->commands[propagate_command].arg5 = new_memo_index;
    computation_->commands[backprop_command].arg7 = new_memo_index;
    new_memo_index++;
  }
}

void IdentifySubmatrixArgsInComputation(NnetComputation *computation,
                                        std::vector<int32*> *submatrix_args) {
  IdentifySubmatrixArgs(&(computation->commands), submatrix_args);

  size_t extra_size = 0;
  for (size_t i = 0; i < computation->indexes_multi.size(); i++)
    extra_size += computation->indexes_multi[i].size();
  submatrix_args->reserve(submatrix_args->size() + extra_size);

  for (size_t i = 0; i < computation->indexes_multi.size(); i++) {
    std::vector<std::pair<int32, int32> > &indexes_multi =
        computation->indexes_multi[i];
    std::vector<std::pair<int32, int32> >::iterator
        iter = indexes_multi.begin(), end = indexes_multi.end();
    for (; iter != end; ++iter)
      if (iter->first != -1)
        submatrix_args->push_back(&(iter->first));
  }
}


void ComputationRenumberer::ComputeSubmatrixIsUsed() {
  int32 num_submatrices = computation_->submatrices.size();
  submatrix_is_used_.clear();
  submatrix_is_used_.resize(num_submatrices, false);
  submatrix_is_used_[0] = true;
  // the zeroth element of the array is 'special', it refers to the
  // zero submatrix, and we don't want to renumber it.
  std::vector<int32*> submatrix_args;
  IdentifySubmatrixArgsInComputation(computation_, &submatrix_args);
  std::vector<int32*>::iterator iter = submatrix_args.begin(),
      end = submatrix_args.end();
  int32 cur_submatrix_index = -1;  // an optimization to avoid too many
                                   // indexings of the bool vector
                                   // submatrix_is_used_.
  for (; iter != end; ++iter) {
    int32 submatrix_index = **iter;
    if (submatrix_index > 0 && submatrix_index != cur_submatrix_index) {
      cur_submatrix_index = submatrix_index;
      KALDI_ASSERT(submatrix_index < num_submatrices);
      submatrix_is_used_[submatrix_index] = true;
    }
  }
}

void ComputationRenumberer::ComputeMatrixIsUsed() {
  matrix_is_used_.clear();
  matrix_is_used_.resize(computation_->matrices.size(), false);
  matrix_is_used_[0] = true;
  // We also need to take into account when matrices are used indirectly via
  // submatrices (which is actually the main way they are accessed).
  int32 num_submatrices = computation_->submatrices.size();
  for (int32 s = 1; s < num_submatrices; s++) {
    int32 matrix_index = computation_->submatrices[s].matrix_index;
    if (submatrix_is_used_[s])
      matrix_is_used_[matrix_index] = true;
  }
}



void ComputationRenumberer::SetUpMappings() {
  num_matrices_new_ = CreateRenumbering(matrix_is_used_, &old_to_new_matrix_);

  unordered_map<NnetComputation::SubMatrixInfo, int32,
                SubMatrixHasher> submat_map;
  int32 cur_index = 1, num_submatrices_orig =
      computation_->submatrices.size();
  // the old_to_new_submatrix_ map will remove duplicates.
  // -1's will appear wherever a particular submatrix was never used.
  submatrix_is_kept_ = submatrix_is_used_;
  old_to_new_submatrix_.resize(num_submatrices_orig, -1);
  old_to_new_submatrix_[0] = 0;
  for (int32 s = 1; s < num_submatrices_orig; s++) {
    if (submatrix_is_used_[s]) {
      const NnetComputation::SubMatrixInfo &info =
          computation_->submatrices[s];
      if (submat_map.count(info) > 0) {  // a duplicate...
        old_to_new_submatrix_[s] = submat_map[info];
        submatrix_is_kept_[s] = false;
      } else {
        old_to_new_submatrix_[s] = (submat_map[info] = cur_index++);
      }
    }
  }
  num_submatrices_new_ = cur_index;
}

void ComputationRenumberer::RenumberSubmatrices() {
  std::vector<int32*> submatrix_args;
  IdentifySubmatrixArgsInComputation(computation_, &submatrix_args);
  std::vector<int32*>::iterator iter = submatrix_args.begin(),
      end = submatrix_args.end();
  for (; iter != end; ++iter) {
    if (**iter > 0) {
      int32 new_submatrix_index = old_to_new_submatrix_[**iter];
      // old_to_new_submatrix_[s] for s > 0 is only <= 0 (actually, -1) for
      // submatrices that are never accessed, and these should never appear
      // in this list.
      KALDI_ASSERT(new_submatrix_index > 0);
      **iter = new_submatrix_index;
    }
  }
  std::vector<NnetComputation::SubMatrixInfo> new_submatrices;
  int32 num_submatrices_old = computation_->submatrices.size();
  new_submatrices.reserve(num_submatrices_old);
  for (int32 s = 0; s < num_submatrices_old; s++)
    if (submatrix_is_kept_[s])
      new_submatrices.push_back(computation_->submatrices[s]);
  computation_->submatrices.swap(new_submatrices);
  // We'll map the matrix indexes inside computation_->submatrices
  // when we call RenumberMatrices().
}

void ComputationRenumberer::RenumberMatrices() {
  std::vector<int32*> matrix_args;
  int32 num_submatrices = computation_->submatrices.size();
  for (int32 s = 1; s < num_submatrices; s++) {
    int32 *matrix_index = &(computation_->submatrices[s].matrix_index);
    // old_to_new_matrix_[s] for s > 0 is only <= 0 (actually, -1) for
    // submatrices that are never accessed, and these should never appear
    // in this list.  (presumably because we renumber the submatrices first).
    int32 new_matrix_index = old_to_new_matrix_[*matrix_index];
    KALDI_ASSERT(new_matrix_index > 0);
    *matrix_index = new_matrix_index;
  }

  std::vector<NnetComputation::MatrixInfo> new_matrices;
  int32 num_matrices_old = computation_->matrices.size();
  new_matrices.reserve(num_matrices_old);
  for (int32 m = 0; m < num_matrices_old; m++)
    if (matrix_is_used_[m])
      new_matrices.push_back(computation_->matrices[m]);
  computation_->matrices.swap(new_matrices);

  std::vector<NnetComputation::MatrixDebugInfo> new_debug_info;
  int32 debug_info_size = computation_->matrix_debug_info.size();
  KALDI_ASSERT(debug_info_size == 0 || debug_info_size == num_matrices_old);
  new_debug_info.reserve(debug_info_size);
  for (int32 m = 0; m < debug_info_size; m++) {
    if (matrix_is_used_[m]) {
      new_debug_info.push_back(NnetComputation::MatrixDebugInfo());
      new_debug_info.back().Swap(&(computation_->matrix_debug_info[m]));
    }
  }
  computation_->matrix_debug_info.swap(new_debug_info);
}


void ComputationRenumberer::Renumber() {
  RemoveUnusedIndexesMulti();
  ComputeSubmatrixIsUsed();
  ComputeMatrixIsUsed();
  SetUpMappings();
  RenumberSubmatrices();
  RenumberMatrices();
  RemoveIndexesMultiDuplicates();
  RenumberIndexes();
  RenumberIndexesRanges();
  RenumberMemos();
}

void ComputationRenumberer::RemoveUnusedIndexesMulti() {
  int32 num_indexes_multi = computation_->indexes_multi.size();
  if (num_indexes_multi == 0)
    return;  // Nothing to do.  An optimization.
  std::vector<bool> indexes_multi_used(num_indexes_multi, false);
  std::vector<int32*> indexes_multi_args;
  IdentifyIndexesMultiArgs(&(computation_->commands), &indexes_multi_args);
  std::vector<int32*>::iterator iter = indexes_multi_args.begin(),
      end = indexes_multi_args.end();
  for (; iter != end; ++iter) {
    int32 indexes_multi_index = **iter;
    KALDI_ASSERT(indexes_multi_index >= 0 &&
                 indexes_multi_index < num_indexes_multi);
    indexes_multi_used[indexes_multi_index] = 1;
  }
  // old->new mapping for the indexes_multi arrays.  will remain -1 for
  // ones that are unused.
  std::vector<int32> old_to_new(num_indexes_multi, -1);
  int32 new_num_indexes_multi = CreateRenumbering(indexes_multi_used,
                                                  &old_to_new);
  if (new_num_indexes_multi == num_indexes_multi)
    return;  // Nothing to do.  An optimization.
  std::vector<std::vector<std::pair<int32, int32> > >
      new_indexes_multi(new_num_indexes_multi);
  for (int32 i = 0; i < num_indexes_multi; i++) {
    if (old_to_new[i] != -1)
      new_indexes_multi[old_to_new[i]].swap(computation_->indexes_multi[i]);
  }
  computation_->indexes_multi.swap(new_indexes_multi);
  // renumber within the commands.
  for (iter = indexes_multi_args.begin(); iter != end; ++iter)
    **iter = old_to_new[**iter];
}


// removes duplicates within the indexes_multi_ array itself.
void ComputationRenumberer::RemoveIndexesMultiDuplicates() {
  int32 cur_index = 0,
      old_indexes_multi_size = computation_->indexes_multi.size();
  if (old_indexes_multi_size == 0)
    return;
  // create index mapping from old to new.  the use of map is generally not that
  // efficient, but the idea here is that we can do most of the comparisons just
  // based on the size of the vectors, and avoid even visiting most of their
  // contents.
  std::vector<int32> indexes_multi_old_to_new(old_indexes_multi_size);
  typedef std::vector<std::pair<int32,int32> > PairVectorType;
  typedef std::map<const PairVectorType*, int32,
                   PointerCompare<std::pair<int32,int32> > > MapType;
  MapType indexes_multi_map;
  for (int32 i = 0; i < computation_->indexes_multi.size(); i++) {
    std::pair<MapType::iterator, bool> p =
        indexes_multi_map.insert(std::pair<const PairVectorType*, int32>(
            &(computation_->indexes_multi[i]), cur_index));
    if (p.second) {  // was inserted-- was not there already.
      indexes_multi_old_to_new[i] = cur_index++;
    } else {
      int32 index_from_map = p.first->second;
      indexes_multi_old_to_new[i] = index_from_map;
    }
  }
  if (cur_index == old_indexes_multi_size)
    return;  // An optimization.  No duplicates were found.
  std::vector<PairVectorType> new_indexes_multi(cur_index);
  for (int32 i = 0; i < old_indexes_multi_size; i++) {
    int32 new_index = indexes_multi_old_to_new[i];
    computation_->indexes_multi[i].swap(new_indexes_multi[new_index]);
  }
  computation_->indexes_multi.swap(new_indexes_multi);

  std::vector<int32*> indexes_multi_args;
  IdentifyIndexesMultiArgs(&(computation_->commands), &indexes_multi_args);
  std::vector<int32*>::const_iterator iter = indexes_multi_args.begin(),
      end = indexes_multi_args.end();
  for (; iter != end; ++iter)
    **iter = indexes_multi_old_to_new[**iter];
}


void ComputationRenumberer::RenumberIndexes() {
  int32 old_num_indexes = computation_->indexes.size();
  if (old_num_indexes == 0)
    return;
  std::vector<int32*> indexes_args;
  IdentifyIndexesArgs(&(computation_->commands), &indexes_args);

  std::vector<bool> indexes_seen(old_num_indexes, false);
  std::vector<int32*>::const_iterator iter = indexes_args.begin(),
      end = indexes_args.end();
  for (; iter != end; ++iter)
    indexes_seen[**iter] = true;

  std::vector<int32> old_to_new_index(old_num_indexes);
  typedef std::map<const std::vector<int32>*, int32,
                   PointerCompare<int32> > MapType;
  MapType indexes_map;

  int32 cur_index = 0;
  for (int32 i = 0; i < old_num_indexes; i++) {
    if (!indexes_seen[i]) {
      old_to_new_index[i] = -1;
    } else {
      std::pair<MapType::iterator, bool> p =
          indexes_map.insert(std::pair<const std::vector<int32>*, int32>(
              &(computation_->indexes[i]), cur_index));
      if (p.second) {  // was inserted-- was not there already.
        old_to_new_index[i] = cur_index++;
      } else {
        int32 index_from_map = p.first->second;
        old_to_new_index[i] = index_from_map;
      }
    }
  }
  if (cur_index == old_num_indexes)
    return;  // An optimization.  No changes to the numbering are made.
  std::vector<std::vector<int32> > new_indexes(cur_index);
  for (int32 i = 0; i < old_num_indexes; i++) {
    int32 new_index = old_to_new_index[i];
    if (new_index != -1)
      computation_->indexes[i].swap(new_indexes[new_index]);
  }
  computation_->indexes.swap(new_indexes);

  // renumber the indexes inside the commmands.
  for (iter = indexes_args.begin(); iter != end; ++iter) {
    int32 old_index = **iter;
    KALDI_ASSERT(old_index >= 0 && old_index < old_num_indexes);
    int32 new_index = old_to_new_index[old_index];
    KALDI_ASSERT(new_index >= 0);
    **iter = new_index;
  }
}

void ComputationRenumberer::RenumberIndexesRanges() {
  int32 old_num_indexes_ranges = computation_->indexes_ranges.size();
  if (old_num_indexes_ranges == 0)
    return;
  std::vector<int32*> indexes_ranges_args;
  IdentifyIndexesRangesArgs(&(computation_->commands), &indexes_ranges_args);

  std::vector<bool> is_seen(old_num_indexes_ranges, false);
  std::vector<int32*>::const_iterator iter = indexes_ranges_args.begin(),
      end = indexes_ranges_args.end();
  for (; iter != end; ++iter)
    is_seen[**iter] = true;

  std::vector<int32> old_to_new_index(old_num_indexes_ranges);
  typedef std::map<const std::vector<std::pair<int32, int32> >*, int32,
                   PointerCompare<std::pair<int32, int32> > > MapType;
  MapType indexes_map;
  int32 cur_index = 0;
  for (int32 i = 0; i < old_num_indexes_ranges; i++) {
    if (!is_seen[i]) {
      old_to_new_index[i] = -1;
    } else {
      std::pair<MapType::iterator, bool> p =
          indexes_map.insert(
              std::pair<const std::vector<std::pair<int32, int32> >*, int32>(
                  &(computation_->indexes_ranges[i]), cur_index));
      if (p.second) {  // was inserted-- was not there already.
        old_to_new_index[i] = cur_index++;
      } else {
        int32 index_from_map = p.first->second;
        old_to_new_index[i] = index_from_map;
      }
    }
  }
  if (cur_index == old_num_indexes_ranges)
    return;  // An optimization.  No changes to the numbering are made.
  std::vector<std::vector<std::pair<int32, int32> > > new_indexes_ranges(
      cur_index);
  for (int32 i = 0; i < old_num_indexes_ranges; i++) {
    int32 new_index = old_to_new_index[i];
    if (new_index != -1)
      computation_->indexes_ranges[i].swap(new_indexes_ranges[new_index]);
  }
  computation_->indexes_ranges.swap(new_indexes_ranges);

  // renumber the indexes inside the commmands.
  for (iter = indexes_ranges_args.begin(); iter != end; ++iter) {
    int32 old_index = **iter;
    KALDI_ASSERT(old_index >= 0 && old_index < old_num_indexes_ranges);
    int32 new_index = old_to_new_index[old_index];
    KALDI_ASSERT(new_index >= 0);
    **iter = new_index;
  }
}




void RenumberComputation(NnetComputation *computation) {
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


VariableMergingOptimizer::VariableMergingOptimizer(
    const NnetOptimizeOptions &config,
    const Nnet &nnet,
    NnetComputation *computation):
    config_(config), nnet_(nnet),
    computation_(computation),
    already_called_merge_variables_(false) {
  analyzer_.Init(nnet, *computation);
  ComputeMatrixToSubmatrix(*computation_, &matrix_to_submatrix_);
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
    const NnetComputation::Command &c = computation_->commands[command_index];
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
        DoMerge(command_index, s1, s2);
        merged = true;
      } else if (p.second) {
        DoMerge(command_index, s2, s1);
        merged = true;
      }
    }
  }
  if (merged) {
    RenumberComputation(computation_);
    RemoveNoOps(computation_);
  }
  return merged;
}

/**
   This static function returns a SubMatrixInfo corresponding to
   replacing the matrix-index in a's "matrix_index" with, essentially, sub-matrix b.
   Of course the matrix_index will be b's "matrix_index", but we may
   have to modify the row and column offsets.  The idea is that sub-matrix
   submat_b should have the same dimensions as the matrix underlying
   submat_a.
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

void VariableMergingOptimizer::DoMerge(int32 command_index,
                                       int32 s_to_keep,
                                       int32 s_to_discard) {
  // Prevent further optimizations touching either submatrix (we can try again
  // in a later round of optimization, with a new instance of this class).
  MarkAsDirty(s_to_keep);
  MarkAsDirty(s_to_discard);

  int32 m_to_keep = computation_->submatrices[s_to_keep].matrix_index,
      m_to_discard = computation_->submatrices[s_to_discard].matrix_index;
  KALDI_ASSERT(m_to_keep != m_to_discard && m_to_keep > 0 && m_to_discard > 0);

  { // modify submatrices of m_to_discard to effectively be sub-matrices of
    // s_to_keep instead (they will refer to m_to_keep as the matrix_index).
    std::vector<int32>::const_iterator iter =
        matrix_to_submatrix_[m_to_discard].begin(),
        end = matrix_to_submatrix_[m_to_discard].end();
    for (; iter != end; ++iter) {
      int32 submatrix_index = *iter;
      KALDI_ASSERT(computation_->submatrices[submatrix_index].matrix_index
                   == m_to_discard);
      computation_->submatrices[submatrix_index] =
          GetSubMatrixOfSubMatrix(*computation_, submatrix_index,
                                  s_to_keep);
    }
  }

  ComputationAnalysis analysis(*computation_, analyzer_);
  NnetComputation::Command &c = computation_->commands[command_index];
  const std::vector<MatrixAccesses> &matrix_accesses =
      analyzer_.matrix_accesses;

  //  - If it was a matrix-copy (assignment) with scale 1.0, replace the
  //    assignment command with a no-op.
  //    If it was matrix-copy with a scale, leave the command there;
  //    it will have the effect of scaling the matrix (it will be
  //    mapped so that arg1 == arg2, but that is OK).
  if (c.command_type == kMatrixCopy && c.alpha == 1.0) {
    // remove the command.
    c.command_type = kNoOperation;
    c.arg1 = -1;
    c.arg2 = -1;
  }

  //   We want to ensure that there is only one deallocation command.
  //   If neither matrix is an output, then there will be 2 deallocation
  //   commands and we keep the one for m_to_keep (which, if the sizes
  //   differ, will be the larger of the two, so it's the one whose
  //   submatrix index refers to the entirety of the matrix).
  //   If one of them is an output, then remove the deallocation command
  //   of whichever one is not an output.
  //   As a simplification to the logic above: if the 'discard' matrix
  //   has a deallocation command (i.e. if that matrix was not an output)
  //   then remove it; otherwise remove the deallocation command of
  //   the 'keep' matrix.

  int32 dealloc_keep = matrix_accesses[m_to_keep].deallocate_command,
      dealloc_discard = matrix_accesses[m_to_discard].deallocate_command;
  if (dealloc_discard != -1) {
    computation_->commands[dealloc_discard].command_type = kNoOperation;
  } else {
    KALDI_ASSERT(dealloc_keep != -1);
    computation_->commands[dealloc_keep].command_type = kNoOperation;
  }

  {
    //   - Both m_to_keep and m_to_discard will have commands that allocate
    //     them, as all matrices do (note, kAcceptInput counts as an allocation
    //     command).  If one of them is kAcceptInput, then delete the other one,
    //     because the position of the kAcceptInput commands is important.
    //     Otherwise delete the "discard" one.  As a simplification of the logic
    //     of the previous sentence: if the "discard" allocate command is
    //     kAcceptInput then delete the "keep" allocate command, else delete
    //     the "discard" allocate command.
    //     Note: after we renumber the submatrices, they both refer to the
    //     same underlying matrix, but we need to refer to them using a
    //     submatrix that refers to the entire matrix.  The one we keep will
    //     always refer to the entire matrix.  (In the case where one of
    //     them is an input, both submatrices are guaranteed to refer to the
    //     entire matrix, this is guaranteed by the logic we use to decide
    //     which matrices we can merge).
    int32 alloc_keep = matrix_accesses[m_to_keep].allocate_command,
        alloc_discard = matrix_accesses[m_to_discard].allocate_command;

    KALDI_ASSERT(alloc_keep != -1 && alloc_discard != -1);
    KALDI_ASSERT(analysis.FirstNontrivialMatrixAccess(m_to_discard) >
                 alloc_keep);

    NnetComputation::Command
        &keep_alloc_command = computation_->commands[alloc_keep],
        &discard_alloc_command = computation_->commands[alloc_discard];
    int32 matrix_whose_zeroing_to_discard;
    if (discard_alloc_command.command_type == kAcceptInput) {
      keep_alloc_command.command_type = kNoOperation;
      matrix_whose_zeroing_to_discard = m_to_keep;
    } else {
      discard_alloc_command.command_type = kNoOperation;
      matrix_whose_zeroing_to_discard = m_to_discard;
    }
    // Now remove the command that zeroed one of the matrices
    // (the one whose allocation command we just discarded).
    int32 zeroing_command_to_discard =
     matrix_accesses[matrix_whose_zeroing_to_discard].accesses[0].command_index;
    NnetComputation::Command &zeroing_command =
        computation_->commands[zeroing_command_to_discard];
    if (zeroing_command.command_type == kSetConst &&
        zeroing_command.alpha == 0.0) {
      // if 'zeroing_command' actually *was* a zeroing command, then remove it.
      zeroing_command.command_type = kNoOperation;
    }
  }

  //  If the matrix to discard had stride_type == kStrideEqualNumCols, set the
  //  stride type of the matrix we're keeping to kStrideEqualNumCols.
  if (computation_->matrices[m_to_discard].stride_type == kStrideEqualNumCols) {
    computation_->matrices[m_to_keep].stride_type = kStrideEqualNumCols;
    // ... and perform an additional check.
    KALDI_ASSERT(computation_->matrices[m_to_discard].num_rows ==
                 computation_->matrices[m_to_keep].num_rows &&
                 computation_->matrices[m_to_discard].num_cols ==
                 computation_->matrices[m_to_keep].num_cols);
  }
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
  // condition c4:
  if (!computation_->IsWholeMatrix(s1)) right = false;
  // condition c6:
  if (computation_->matrices[m2].stride_type == kStrideEqualNumCols &&
      !computation_->IsWholeMatrix(s1)) left = false;
  // condition c7:
  if (computation_->matrices[m1].stride_type == kStrideEqualNumCols &&
      !computation_->IsWholeMatrix(s2)) right = false;


  if (!left && !right)  // save some time.
    return std::pair<bool,bool>(false,false);
  bool is_assignment = (computation_->commands[command_index].command_type ==
                        kMatrixCopy &&
                        computation_->commands[command_index].alpha == 1.0);
  ComputationAnalysis analysis(*computation_, analyzer_);
  if (is_assignment) {
    if (analysis.FirstNontrivialAccess(s2) == command_index &&
        analysis.LastWriteAccess(s1) < command_index &&
        analysis.LastAccess(s1) <
        analysis.DataInvalidatedCommand(command_index, s2)) {
      return std::pair<bool,bool>(left, right);  // possible success.
    }
  } else {
    if (analysis.FirstNontrivialAccess(s2) == command_index &&
        analysis.LastAccess(s1) == command_index) {
      return std::pair<bool,bool>(left, right);  // possible success.
    }
  }
  // failure.
  return std::pair<bool,bool>(false,false);
}


// This class is used inside the function
// `void ExtendMatrices(NnetComputation *computation)`;
// see that function's declaration in nnet-optimize-utils.h for
// a summary of what this class does.
class MatrixExtender {
 public:
  typedef NnetComputation::SubMatrixInfo SubMatrixInfo;
  typedef NnetComputation::MatrixInfo MatrixInfo;

  MatrixExtender(NnetComputation *computation);

  void ExtendMatrices();

 private:
  // This function returns true if a copy command from 'src_submatrix'
  // to 'dest_submatrix' has the properties we need to be able to
  // extend its rows to cover all of the source matrix.
  bool CanBeExtended(int32 dest_submatrix_index,
                     int32 src_submatrix_index);

  // This actually extends the matrices... it's called only if CanBeExtended()
  // with the same args returned true.  It modifies 'dest_submatrix_index'
  // and 'src_submatrix_index'.
  void Extend(int32 *dest_submatrix_index, int32 *src_submatrix_index);

  // This function modifies the computation to fix certain problems
  // that might have been introduced by Extend()... allocation, deallocation,
  void FixComputation();

  // This function modifies the computation to fix the debug info; if needed,
  // it's called from FixComputation().
  void FixDebugInfo();

  // don't extend a destination matrix if it wasn't already
  // at least 'min_proportion' (80%) big enough to store the source.
  BaseFloat min_proportion_;

  NnetComputation *computation_;

  // Indexed by matrix-index m, orig_num_rows_[m] is the value of
  // computation_->matrices[m].num_rows when this class was initialized,
  // i.e. before we changed anything.
  std::vector<int32> orig_num_rows_;

  // Indexed by matrix-index m, this vector contains true if matrix
  // m is involved in any AcceptInput() or ProvideOutput() operations.
  std::vector<bool> is_input_or_output_;
};

// note: the initializer for min_proportion_ below needs to be kept in sync with
// the min_proportion variable in
// ComputationChecker::CheckComputationUndefined() in nnet-analyze.cc.
MatrixExtender::MatrixExtender(NnetComputation *computation):
    min_proportion_(0.8),
    computation_(computation) {
  int32 num_matrices = computation_->matrices.size();

  { // set up orig_num_rows_.
    orig_num_rows_.resize(num_matrices);
    // matrix 0 is not a real matrix so skip that index.
    for (int32 m = 1; m < num_matrices; m++)
      orig_num_rows_[m] = computation_->matrices[m].num_rows;
  }
  { // set up is_input_or_output_.
    is_input_or_output_.resize(num_matrices, false);
    std::vector<NnetComputation::Command>::iterator
      command_iter = computation_->commands.begin(),
      command_end = computation_->commands.end();
    for (; command_iter != command_end; ++command_iter) {
      const NnetComputation::Command &command = *command_iter;
      // make sure there are no kSwapMatrix commands; they should not be present
      // at this stage of optimization.
      KALDI_ASSERT(command.command_type != kSwapMatrix);
      if (command.command_type == kProvideOutput ||
          command.command_type == kAcceptInput) {
        int32 s = command.arg1,
            m = computation_->submatrices[s].matrix_index;
        is_input_or_output_[m] = true;
      }
    }
  }
}


bool MatrixExtender::CanBeExtended(int32 dest_submatrix_index,
                                   int32 src_submatrix_index) {
  const SubMatrixInfo
      &src_submatrix = computation_->submatrices[src_submatrix_index],
      &dest_submatrix = computation_->submatrices[dest_submatrix_index];
  if (src_submatrix.matrix_index == dest_submatrix.matrix_index)
    return false;

  // we can't resize the destination matrix if it's involved in input or output.
  if (is_input_or_output_[dest_submatrix.matrix_index])
    return false;

  const MatrixInfo
      &src_matrix = computation_->matrices[src_submatrix.matrix_index];

  int32 dest_matrix_orig_num_rows = orig_num_rows_[dest_submatrix.matrix_index],
      src_matrix_orig_num_rows = orig_num_rows_[src_submatrix.matrix_index];

  if (src_submatrix.num_rows < min_proportion_ * src_matrix_orig_num_rows)
    return false;

  // The following checks that the source submatrix covers be all of the
  // source matrix except a few final rows, and the destination submatrix goes
  // to the final row of its matrix.
  return (src_submatrix.col_offset == 0 &&
          src_submatrix.num_cols == src_matrix.num_cols &&
          src_submatrix.row_offset == 0 &&
          src_submatrix.num_rows < src_matrix.num_rows &&
          dest_submatrix.row_offset + dest_submatrix.num_rows ==
          dest_matrix_orig_num_rows);
}


void MatrixExtender::Extend(int32 *dest_submatrix_index,
                            int32 *src_submatrix_index) {
  // copy the SubMatrixInfo to avoid iterator invalidation.
  SubMatrixInfo
      src_submatrix = computation_->submatrices[*src_submatrix_index],
      dest_submatrix = computation_->submatrices[*dest_submatrix_index];

  MatrixInfo  &src_matrix = computation_->matrices[src_submatrix.matrix_index],
      &dest_matrix = computation_->matrices[dest_submatrix.matrix_index];

  int32 new_dest_num_rows = dest_submatrix.row_offset + src_matrix.num_rows;

  // extend the destination matrix so it has enough rows to fit the entire
  // source matrix.  Note: doing this will break certain invariances in the
  // computation, principally with allocation and deallocation commands, which
  // we'll later fix up by calling FixComputation().
  if (new_dest_num_rows > dest_matrix.num_rows) {
    dest_matrix.num_rows = new_dest_num_rows;
    // make sure there's a submatrix index covering the whole of the dest matrix.
    computation_->submatrices.push_back(
        SubMatrixInfo(dest_submatrix.matrix_index, 0, new_dest_num_rows,
                      0, dest_matrix.num_cols));
  }

  // The following 3 statements create a new submatrix that will be
  // the destination submatrix; it's the same as the original destination
  // submatrix, but with a few extra rows.
  *dest_submatrix_index = computation_->submatrices.size();
  dest_submatrix.num_rows = src_matrix.num_rows;
  computation_->submatrices.push_back(
      SubMatrixInfo(dest_submatrix));

  // The following 3 statements create a new submatrix that will be
  // the source submatrix; it's the same as the original source
  // submatrix, but with a few extra rows, and actually will cover
  // the entire source matrix.
  *src_submatrix_index = computation_->submatrices.size();
  computation_->submatrices.push_back(
      SubMatrixInfo(src_submatrix.matrix_index, 0, src_matrix.num_rows,
                    0, src_matrix.num_cols));
}

void MatrixExtender::ExtendMatrices() {
  std::vector<NnetComputation::Command>::iterator
      command_iter = computation_->commands.begin(),
      command_end = computation_->commands.end();
  bool changed = false;
  for (; command_iter != command_end; ++command_iter) {
    NnetComputation::Command &command = *command_iter;
    if (command.command_type == kMatrixCopy &&
        command.alpha == 1.0) {
      int32 dest_submatrix_index = command.arg1,
          src_submatrix_index = command.arg2;
      if (CanBeExtended(dest_submatrix_index, src_submatrix_index)) {
        Extend(&command.arg1, &command.arg2);
        changed = true;
      }
    }
  }
  if (changed)
    FixComputation();
}

void MatrixExtender::FixComputation() {
  // make sure that allocation and deallocation commands
  // operate on whole matrix.
  std::vector<NnetComputation::Command>::iterator
      command_iter = computation_->commands.begin(),
      command_end = computation_->commands.end();
  std::vector<int32> whole_submatrices;
  computation_->GetWholeSubmatrices(&whole_submatrices);
  for (; command_iter != command_end; ++command_iter) {
    NnetComputation::Command &command = *command_iter;
    if (command.command_type == kAllocMatrix ||
        command.command_type == kDeallocMatrix) {
      int32 s = command.arg1,
          m = computation_->submatrices[s].matrix_index,
          new_s = whole_submatrices[m];
      if (new_s != s) {
        KALDI_ASSERT(
            computation_->submatrices[s] == computation_->submatrices[new_s] ||
            orig_num_rows_[m] != computation_->matrices[m].num_rows);
        command.arg1 = new_s;
      }
    }
    if (command.command_type == kSetConst && command.alpha == 0.0) {
      int32 s = command.arg1,
          m = computation_->submatrices[s].matrix_index,
          new_s = whole_submatrices[m];
      if (new_s != s) {
        {
          const NnetComputation::SubMatrixInfo &info = computation_->submatrices[
              command.arg1];
          const NnetComputation::MatrixInfo &mat_info = computation_->matrices[
              info.matrix_index];
          // If this command wasn't zeroing the the entirety of a matrix,
          // (before we extended the matrix), we don't need to extend it.
          if (!(info.row_offset == 0 && info.col_offset == 0 &&
                info.num_cols == mat_info.num_cols &&
                info.num_rows == orig_num_rows_[info.matrix_index]))
            continue;
          // I know doing this via 'continue' is odd, but it's done this way to
          // avoid invalid iterators still being in scope; I think some runtimes
          // check for it.
        }
        command.arg1 = new_s;
      }
    }
  }
  if (!computation_->matrix_debug_info.empty())
    FixDebugInfo();
  RenumberComputation(computation_);
}

void MatrixExtender::FixDebugInfo() {
  int32 num_matrices = computation_->matrices.size();
  // matrix zero is not a 'real' matrix.
  for (int32 m = 1; m < num_matrices; m++) {
    NnetComputation::MatrixDebugInfo &debug_info =
        computation_->matrix_debug_info[m];
    int32 new_num_rows = computation_->matrices[m].num_rows,
        old_num_rows = debug_info.cindexes.size();
    if (new_num_rows != old_num_rows) {
      debug_info.cindexes.resize(new_num_rows);
      int32 num_extra_rows = new_num_rows - old_num_rows;
      // the following should be true because min_proportion_ > 0.5.
      KALDI_ASSERT(num_extra_rows <= old_num_rows);
      for (int32 r = old_num_rows; r < new_num_rows; r++) {
        Cindex cindex = debug_info.cindexes[r - num_extra_rows];
        // set the 't' value to kNoTime which indicates that it's not a 'real'
        // time step, and may avoid errors in checking code.
        cindex.second.t = kNoTime;
        debug_info.cindexes[r] = cindex;
      }
    }
  }
}

void ExtendMatrices(NnetComputation *computation) {
  MatrixExtender ext(computation);
  ext.ExtendMatrices();
}



/** This class is responsible for consolidating the model-update part of
    backprop commands, for components in (e.g.) recurrent networks that need to
    have many separate backprop commands, into more efficient single commands
    operating on consolidated data in larger matrices.  This is useful for
    recurrent networks.  */
class ModelUpdateConsolidator {
 public:
  ModelUpdateConsolidator(const Nnet &nnet,
                          NnetComputation *computation);
  void ConsolidateModelUpdate();
 private:
  void ConsolidateUpdateForComponent(
      int32 component,
      const std::vector<int32> &backprop_commands);

  /// This function, called at the end of ConsolidateModelUpdate(), takes the
  /// commands that we have put in extra_commands_, final_commands_ and
  /// final_deallocate_commands_, and puts them in the appropriate place in
  /// computation->commands_.
  void AddCommandsToComputation();

  /// You call this function when you want to consolidate the values of a list
  /// of submatrices taken just prior to particular commands.  The input
  /// 'commands' and 'submatrices' lists must be the same size, and size must be
  /// > 1.  This function will create a new matrix that is the row-wise
  /// concatentation of all these submatrices, with values taken just prior to
  /// the respective command indexes.  This function will will add to
  /// extra_commands_ the commands to do the copying at the appropriate places
  /// (at the supplied command indexes; they will be inserted just before).  The
  /// return value is the submatrix index of a submatrix that represents the
  /// whole of the consolidated matrix.  This command will insert, at the
  /// beginning of the computation (in extra_commands_[0]), a command to
  /// initialize the matrix; and will append to final_deallocate_commands_ the
  /// commands to deallocate the matrix.  If computation_->matrix_debug_info is
  /// nonempty, this function will also update computation_->matrix_debug_info
  /// with suitable values for the newly added matrix
  int32 ConsolidateSubmatrices(
      const std::vector<int32> &commands,
      const std::vector<int32> &submatrices);

  /// This function, called from ConsolidateSubmatrices, will
  /// update 'debug_info' by appending the corresponding 'indexes' from
  /// the existing debug info for this submatrix.  It will also set
  /// the 'is_deriv' of '*debug_info' to the same value as the
  /// debug info for 'submatrix_index', and set the 'node_index' to the
  /// 'node_index' in the debug info for that submatrix-index.
  /// It requires that computation_->matrix_debug_info be nonempty.
  void AppendDebugInfoForSubmatrix(
      int32 submatrix_index,
      NnetComputation::MatrixDebugInfo *debug_info) const;

  const Nnet &nnet_;
  NnetComputation *computation_;

  // Indexed by the original command index in *computation_ (and sized to the
  // original number of commands in *computation_ before we added anything),
  // extra_commands_[c] contains a list of commands that need to be inserted
  // just before command c in the previously existing computation.
  std::vector<std::vector<NnetComputation::Command> > extra_commands_;

  // This is as list of kBackprop commands that will be placed after the
  // commands in 'computation_->commands' and 'extra_commands_', but before
  // the 'final_deallocate_commands_'.
  std::vector<NnetComputation::Command> final_commands_;
  // This is a list of commands to deallocate our 'consolidated' matrices; the
  // commands will be placed after the commands in 'final_commands_'.
  std::vector<NnetComputation::Command> final_deallocate_commands_;
};


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
  KALDI_ASSERT(src_info.cindexes.size() ==
               computation_->matrices[matrix_index].num_rows);
  int32 row_begin = submatrix_info.row_offset,
      row_end = row_begin + submatrix_info.num_rows;
  debug_info->cindexes.insert(debug_info->cindexes.end(),
                             src_info.cindexes.begin() + row_begin,
                             src_info.cindexes.begin() + row_end);
}

// see comment by declaration in header.
int32 ModelUpdateConsolidator::ConsolidateSubmatrices(
    const std::vector<int32> &commands,
    const std::vector<int32> &submatrices) {
  int32 num_submatrices = submatrices.size();
  KALDI_ASSERT(num_submatrices > 1 && commands.size() == submatrices.size());
  int32 first_submatrix = submatrices[0];
  int32 num_cols = computation_->submatrices[first_submatrix].num_cols,
      num_rows = 0;
  MatrixStrideType stride_type = kDefaultStride;
  NnetComputation::MatrixDebugInfo debug_info;
  for (int32 i = 0; i < num_submatrices; i++) {
    int32 submatrix = submatrices[i];
    num_rows += computation_->submatrices[submatrix].num_rows;
    KALDI_ASSERT(computation_->submatrices[submatrix].num_cols == num_cols);
    if (!computation_->matrix_debug_info.empty())
      AppendDebugInfoForSubmatrix(submatrix, &debug_info);
    if (computation_->IsWholeMatrix(submatrix)) {
      int32 matrix = computation_->submatrices[submatrix].matrix_index;
      if (computation_->matrices[matrix].stride_type == kStrideEqualNumCols)
        stride_type = kStrideEqualNumCols;
    }
  }
  // new_whole_submatrix is a new submatrix index corresponding to the whole
  // of a new matrix that we are creating.
  int32 new_whole_submatrix = computation_->NewMatrix(num_rows, num_cols,
                                                      stride_type);
  // Add commands at the very start, to initialize and then zero this new
  // matrix.  we can later on remove the zeroing if it is not necessary.
  extra_commands_[0].push_back(
      NnetComputation::Command(kAllocMatrix, new_whole_submatrix));
  extra_commands_[0].push_back(
      NnetComputation::Command(0.0, kSetConst, new_whole_submatrix));

  final_deallocate_commands_.push_back(
      NnetComputation::Command(kDeallocMatrix, new_whole_submatrix));
  int32 new_matrix_index =
      computation_->submatrices[new_whole_submatrix].matrix_index;
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
    // having the operation that created it write directly to the location
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
      input_deriv_submatrix = 0,  // we don't need the input-deriv.
      memo_index = 0;  // we checked that no memos were used.
  NnetComputation::Command c(kBackprop, component_index, precomputed_indexes_index,
                             input_submatrix, output_submatrix,
                             output_deriv_submatrix, input_deriv_submatrix,
                             memo_index);
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
  // updatable simple components), of the command indexes for the backprop
  // commands.
  std::vector<std::vector<int32> > backprop_commands(num_components);
  for (int32 command_index = 0;
       command_index < num_commands; command_index++) {
    const NnetComputation::Command &c = computation_->commands[command_index];
    if (c.command_type == kBackprop) {
      int32 component_index = c.arg1;
      const Component *component = nnet_.GetComponent(component_index);
      int32 properties = component->Properties();
      if ((properties & kUpdatableComponent) &&
          (properties & kSimpleComponent) &&
          !(properties & kUsesMemo))
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


void ConsolidateModelUpdate(const Nnet &nnet,
                            NnetComputation *computation) {
  // This following if-statement is an optimization: if the computation
  // request(s) had need_model_derivative == false, there would be nothing to
  // optimize, so don't bother trying.
  if (!computation->need_model_derivative)
    return;
  ModelUpdateConsolidator consolidator(nnet, computation);
  consolidator.ConsolidateModelUpdate();
}


// inline
void DerivativeTimeLimiter::GetPruneValues(int32 initial_submatrix,
                                           int32 new_submatrix,
                                           int32 *left_prune,
                                           int32 *right_prune) const {
  KALDI_ASSERT(initial_submatrix > 0 && new_submatrix > 0);
  const NnetComputation::SubMatrixInfo
      initial_info = computation_->submatrices[initial_submatrix],
      new_info = computation_->submatrices[new_submatrix];
  KALDI_ASSERT(initial_info.matrix_index == new_info.matrix_index);
  *left_prune = new_info.row_offset - initial_info.row_offset;
  if (right_prune != NULL) {
    *right_prune = initial_info.num_rows - new_info.num_rows - *left_prune;
    KALDI_ASSERT(*left_prune >= 0 && *right_prune >= 0);
  }
}

bool DerivativeTimeLimiter::RowIsKept(
    int32 submatrix,
    int32 row_index) const {
  KALDI_ASSERT(submatrix > 0 && submatrix < computation_->submatrices.size());
  const NnetComputation::SubMatrixInfo &info =
      computation_->submatrices[submatrix];
  KALDI_ASSERT(row_index >= 0 &&
               row_index < computation_->submatrices[submatrix].num_rows);
  int32 matrix_index = info.matrix_index;
  const NnetComputation::MatrixDebugInfo
      &debug_info = computation_->matrix_debug_info[matrix_index];
  if (!debug_info.is_deriv) {
    // the derivative time limitation doesn't apply to things that aren't
    // derivatives.
    return true;
  }
  int32 t = debug_info.cindexes[row_index + info.row_offset].second.t;
  return (t >= min_deriv_time_ && t <= max_deriv_time_);
}


// modify commands to take into account the fact that some matrices are zero or
// partially zero.  Allocation commands and sizes of underlying matrices are not
// affected-- we'll work out later on, what to do with them.
void DerivativeTimeLimiter::ModifyCommand(NnetComputation::Command *command) {
  CommandType command_type = command->command_type;
  switch (command_type) {
    case kAllocMatrix:
    case kDeallocMatrix:
    case kSetConst:
    case kSwapMatrix:
      break;  // we'll deal with allocation and deallocation later on.
    case kPropagate:
      // Propagate commands are unchanged, except that if the output of the
      // propagate is completely outside the accepted time-range (only likely if
      // we're inside a recurrency), then we don't store stats; this is not
      // really important, and is mostly done to minimize the difference from an
      // older version of the code, to reduce the need for testing.
      if (submatrix_map_[command->arg4] == 0)
        command->arg6 = 0;
      break;
    case kBackpropNoModelUpdate:  // we actually don't expect to encounter this,
                                  // but it's trivial to support as it's the
                                  // same as backprop.
    case kBackprop: {
      const Component *component = nnet_.GetComponent(command->arg1);
      int32 properties = component->Properties();
      if (!(properties & kSimpleComponent)) {
        // we don't (yet) do this optimization for non-simple Components...
        // it would be a bit more complicated as we'd have to recompute the
        // PrecomputedIndexes.
        break;
      }
      int32 input_submatrix = command->arg3,
          output_submatrix = command->arg4,
          output_deriv_submatrix = command->arg5,
          input_deriv_submatrix = command->arg6;
      int32 mapped_input_submatrix = submatrix_map_[input_submatrix],
           mapped_output_submatrix =  submatrix_map_[output_submatrix],
     mapped_output_deriv_submatrix = submatrix_map_[output_deriv_submatrix],
      mapped_input_deriv_submatrix = submatrix_map_[input_deriv_submatrix];

      if (mapped_output_deriv_submatrix == 0) {
        // completely outside range..
        KALDI_ASSERT(mapped_input_deriv_submatrix == 0 &&
                     mapped_input_submatrix == 0 &&
                     mapped_output_submatrix == 0);
        // just delete the command.
        command->command_type = kNoOperation;
        if (command->arg7 > 0)
          memos_to_delete_.insert(command->arg7);
      } else if (mapped_output_deriv_submatrix !=
                 output_deriv_submatrix &&
                 !(properties & kUsesMemo)) {
        // we're operating on a range of the input or output.
        // we can't do this type of mapping of the component uses
        // a memo, though.
        command->arg3 = mapped_input_submatrix;
        command->arg4 = mapped_output_submatrix;
        command->arg5 = mapped_output_deriv_submatrix;
        command->arg6 = mapped_input_deriv_submatrix;
      }
      break;
    }
    case kMatrixCopy: case kMatrixAdd:
      MapSimpleMatrixCommand(command);
      break;
    case kCopyRows: case kAddRows:
      MapIndexesCommand(command);
      break;
    case kCopyRowsMulti: case kCopyToRowsMulti:
    case kAddRowsMulti: case kAddToRowsMulti:
      MapIndexesMultiCommand(command);
      break;
    case kAddRowRanges: {
      MapAddRowRangesCommand(command);
      break;
    }
    case kAcceptInput: case kProvideOutput:
    case kNoOperation: case kNoOperationPermanent: case kNoOperationMarker:
      break;
    default:
      KALDI_ERR << "Un-handled command type.";
  }
}

void DerivativeTimeLimiter::MapSimpleMatrixCommand(NnetComputation::Command *c) {
  int32 submatrix1 = c->arg1,
      submatrix2 = c->arg2;
  int32 submatrix1_mapped = submatrix_map_if_deriv_[submatrix1],
      submatrix2_mapped = submatrix_map_if_deriv_[submatrix2];
  if (submatrix1_mapped == submatrix1 &&
      submatrix2_mapped == submatrix2) {
    // nothing to do.
    return;
  }
  if (submatrix1_mapped == 0 || submatrix2_mapped == 0) {
    // remove the operation-- it has nothing to do.
    c->command_type = kNoOperation;
    return;
  }
  // left_prune1 is the number of rows pruned away on the left for submatrix1.
  int32 orig_num_rows = computation_->submatrices[submatrix1].num_rows,
      left_prune1, left_prune2, right_prune1, right_prune2;
  GetPruneValues(submatrix1, submatrix1_mapped, &left_prune1, &right_prune1);
  GetPruneValues(submatrix2, submatrix2_mapped, &left_prune2, &right_prune2);
  if (left_prune1 == left_prune2 && right_prune1 == right_prune2) {
    // we took the same number of rows away from the left and right for
    // both arguments; the normal mapped values will work in this case
    c->arg1 = submatrix1_mapped;
    c->arg2 = submatrix2_mapped;
  } else {
    // there is some kind of mismatch- we'll prune back to what remains
    // after applying the maximum pruning on the left and right.
    int32 left_prune = std::max(left_prune1, left_prune2),
        right_prune = std::max(right_prune1, right_prune2);
    if (left_prune + right_prune >= orig_num_rows) {
      // everything was pruned away; remove the operation.
      c->command_type = kNoOperation;
      return;
    } else {
      int32 num_rows = orig_num_rows - left_prune - right_prune;
      // note: the call NewSubMatrix effectively gives us a sub-matrix of a
      // sub-matrix.
      c->arg1 = computation_->NewSubMatrix(submatrix1,
                                           left_prune, num_rows, 0, -1);
      c->arg2 = computation_->NewSubMatrix(submatrix2,
                                           left_prune, num_rows, 0, -1);
    }
  }
}

// does the processing for a command of type kCopyRows or kAddRows, where
// 1st and 2nd args are submatrix indexes and the 3rd arg is a vector of
// row-indexes.
void DerivativeTimeLimiter::MapIndexesCommand(NnetComputation::Command *c) {
  int32 output_submatrix = c->arg1,
      input_submatrix = c->arg2;
  int32 input_submatrix_mapped = submatrix_map_if_deriv_[input_submatrix],
      output_submatrix_mapped = submatrix_map_if_deriv_[output_submatrix];
  // input_submatrix_mapped and output_submatrix_mapped map both submatrices to
  // just the portion that we are treating as nonzero.

  if (input_submatrix_mapped == 0 ||
      output_submatrix_mapped == 0) {
    // Either input or output is all zeros; make the command a no-op.
    // It may not be obvious that in the case of kCopyRows it would
    // be valid to make this a no-op (because what if the existing
    // contents were nonzero?), but we insist that this optimization
    // come before optimizations, and we know that the originally
    // generated computation would not overwrite a nonzero value
    // (and there are no undefined values because we make sure to
    // initialize everything with zeros; ununitialized values are
    // allowed only at a later optimization stage.
    c->command_type = kNoOperation;
    return;
  }
  const std::vector<int32> &old_indexes = computation_->indexes[c->arg3];

  int32 left_prune_input, left_prune_output;
  GetPruneValues(input_submatrix, input_submatrix_mapped,
                 &left_prune_input, NULL);
  GetPruneValues(output_submatrix, output_submatrix_mapped,
                 &left_prune_output, NULL);
  int32 new_num_input_rows =
      computation_->submatrices[input_submatrix_mapped].num_rows,
      new_num_output_rows =
      computation_->submatrices[output_submatrix_mapped].num_rows;
  std::vector<int32> new_indexes(new_num_output_rows);
  bool must_keep_command = false;
  for (int32 i = 0; i < new_num_output_rows; i++) {
    // the index into the 'new_indexes' vector is the row of the output
    // submatrix; the value is the row of the input submatrix.
    int32 orig_index = old_indexes[i + left_prune_output];
    if (orig_index == -1 ||
        !RowIsKept(input_submatrix, orig_index) ||
        !RowIsKept(output_submatrix_mapped, i)) {
      new_indexes[i] = -1;
    } else {
      int32 mapped_index = orig_index - left_prune_input;
      // we can do the following assert because the RowIsKept command
      // would have turned it into a -1 if not.
      KALDI_ASSERT(mapped_index >= 0 && mapped_index < new_num_input_rows);
      new_indexes[i] = mapped_index;
      must_keep_command = true;
    }
  }
  if (!must_keep_command) {
    c->command_type = kNoOperation;
    return;
  }
  int32 new_indexes_index = computation_->indexes.size();
  computation_->indexes.push_back(new_indexes);
  c->arg1 = output_submatrix_mapped;
  c->arg2 = input_submatrix_mapped;
  c->arg3 = new_indexes_index;
}

void DerivativeTimeLimiter::MapIndexesMultiCommand(NnetComputation::Command *c) {
  int32 dest_submatrix = c->arg1,
      indexes_multi_arg = c->arg2;
  int32 dest_submatrix_mapped = submatrix_map_if_deriv_[dest_submatrix];
  if (dest_submatrix_mapped == 0) {
    // The destination matrix is completely outside the allowed time range.
    c->command_type = kNoOperation;
    return;
  }
  int32 left_prune;
  GetPruneValues(dest_submatrix, dest_submatrix_mapped, &left_prune, NULL);
  int32 new_num_rows = computation_->submatrices[dest_submatrix_mapped].num_rows;
  const std::vector<std::pair<int32, int32> > &old_indexes_multi(
      computation_->indexes_multi[indexes_multi_arg]);
  std::vector<std::pair<int32, int32> > new_indexes_multi(new_num_rows);
  bool must_keep_command = false;
  for (int32 i = 0; i < new_num_rows; i++) {
    std::pair<int32,int32> &this_pair = new_indexes_multi[i];
    this_pair = old_indexes_multi[i + left_prune];
    // note: 'this_submatrix' is the source submatrix, from where we copy or add
    // the the data; 'this_row' is the source row.
    int32 this_submatrix = this_pair.first,
        this_row = this_pair.second;
    if (this_submatrix == -1)  // don't map the (-1, -1) pairs.
      continue;
    if (!RowIsKept(this_submatrix, this_row) ||
        !RowIsKept(dest_submatrix_mapped, i)) {
      this_pair.first = -1;
      this_pair.second = -1;
      continue;
    }
    int32 this_submatrix_mapped = submatrix_map_if_deriv_[this_submatrix];

    // Reason for the assert below: if this_submatrix_mapped was 0, then all the
    // values in it should be not-kept, but RowIsKept above returned true, so
    // this would be a code error.
    KALDI_ASSERT(this_submatrix_mapped != 0);

    int32 this_left_prune, this_num_rows =
        computation_->submatrices[this_submatrix_mapped].num_rows;
    GetPruneValues(this_submatrix, this_submatrix_mapped,
                   &this_left_prune, NULL);
    int32 this_row_mapped = this_row - this_left_prune;
    // the above assert is there because if it was going to be outside the
    // kept range, RowIsKept should have returned false above.
    KALDI_ASSERT(this_row_mapped >= 0 && this_row_mapped < this_num_rows);
    this_pair.first = this_submatrix_mapped;
    this_pair.second = this_row_mapped;
    must_keep_command = true;
  }
  if (!must_keep_command) {
    c->command_type = kNoOperation;
    return;
  }
  if (dest_submatrix_mapped == dest_submatrix &&
      new_indexes_multi == old_indexes_multi)  // nothing changed.
    return;
  c->arg1 = dest_submatrix_mapped;
  c->arg2 = computation_->indexes_multi.size();
  computation_->indexes_multi.push_back(new_indexes_multi);
}

void DerivativeTimeLimiter::MapAddRowRangesCommand(
    NnetComputation::Command *c) {
  int32 dest_submatrix = c->arg1,
      src_submatrix = c->arg2,
      indexes_ranges_index = c->arg3;
  int32 dest_submatrix_mapped = submatrix_map_if_deriv_[dest_submatrix],
      src_submatrix_mapped = submatrix_map_if_deriv_[src_submatrix];
  if (dest_submatrix_mapped == dest_submatrix &&
      src_submatrix_mapped == src_submatrix)
    return;
  if (dest_submatrix_mapped == 0 || src_submatrix_mapped == 0) {
    c->command_type = kNoOperation;
    return;
  }
  int32 dest_num_rows = computation_->submatrices[dest_submatrix_mapped].num_rows,
      src_num_rows = computation_->submatrices[src_submatrix_mapped].num_rows,
      src_left_prune, dest_left_prune;
  GetPruneValues(dest_submatrix, dest_submatrix_mapped,
                 &dest_left_prune, NULL);
  GetPruneValues(src_submatrix, src_submatrix_mapped,
                 &src_left_prune, NULL);
  const std::vector<std::pair<int32,int32> > &old_indexes_ranges(
      computation_->indexes_ranges[indexes_ranges_index]);
  std::vector<std::pair<int32,int32> > new_indexes_ranges(dest_num_rows);

  bool must_keep_command = false;
  for (int32 i = 0; i < dest_num_rows; i++) {
    std::pair<int32, int32> &this_pair = new_indexes_ranges[i];
    this_pair = old_indexes_ranges[i + dest_left_prune];

    int32 start = this_pair.first, end = this_pair.second;
    if (!RowIsKept(dest_submatrix_mapped, i)) {
      start = -1;
      end = -1;
    } else if (start >= 0) {
      // no need to change start, end if they are (-1, -1).
      // Note: this code is not optimally efficient, as RowIsKept
      // has a bunch of statements that we could cache some variables
      // for, but this command is pretty rare so not worth to optimize
      // at this point.
      while (start < end && !RowIsKept(src_submatrix, start))
        start++;
      while (end > start && !RowIsKept(src_submatrix, end - 1))
        end--;
      if (start == end) {
        start = -1;
        end = -1;
      } else {
        start -= src_left_prune;
        end -= src_left_prune;
        must_keep_command = true;
        // the next assert is because if we were outside the 'kept' part of the
        // submatrix, RowIsKept() should have instructed us to modify the value.
        KALDI_ASSERT(start >= 0 && end <= src_num_rows && start < end);
      }
    }
    this_pair.first = start;
    this_pair.second = end;
  }
  if (must_keep_command) {
    c->arg1 = dest_submatrix_mapped;
    c->arg2 = src_submatrix_mapped;
    c->arg3 = computation_->indexes_ranges.size();
    computation_->indexes_ranges.push_back(new_indexes_ranges);
  } else {
    c->command_type = kNoOperation;
  }
}


DerivativeTimeLimiter::DerivativeTimeLimiter(const Nnet &nnet,
                                             int32 min_deriv_time,
                                             int32 max_deriv_time,
                                             NnetComputation *computation):
    nnet_(nnet),
    min_deriv_time_(min_deriv_time),
    max_deriv_time_(max_deriv_time),
    computation_(computation) { }

void DerivativeTimeLimiter::LimitDerivTimes() {
  KALDI_ASSERT(max_deriv_time_ >= min_deriv_time_);
  if (min_deriv_time_ == std::numeric_limits<int32>::min() &&
      max_deriv_time_ == std::numeric_limits<int32>::max())
    return;  // nothing to do.

  computation_->GetWholeSubmatrices(&whole_submatrices_);
  ComputeMatrixPruneInfo();
  ComputeSubmatrixMaps();
  ModifyCommands();
  PruneMatrices();
  RemoveNoOps(computation_);
  RemoveUnusedMemos();
  RenumberComputation(computation_);
}

void DerivativeTimeLimiter::RemoveUnusedMemos() {
  if (memos_to_delete_.empty())
    return;
  size_t num_commands = computation_->commands.size(),
      num_memos_removed = 0;
  for (size_t command_index = 0; command_index < num_commands;
       command_index++) {
    NnetComputation::Command &c = computation_->commands[command_index];
    if (c.command_type == kPropagate &&
        memos_to_delete_.count(c.arg5) != 0) {
      c.arg5 = 0;
      num_memos_removed++;
    }
  }
  KALDI_ASSERT(num_memos_removed == memos_to_delete_.size());
}

void DerivativeTimeLimiter::ComputeMatrixPruneInfo() {
  KALDI_ASSERT(computation_->matrix_debug_info.size() ==
               computation_->matrices.size() &&
               "Limiting derivative times requires debug info.");
  const int32 num_matrices = computation_->matrices.size(),
      min_deriv_time = min_deriv_time_,
      max_deriv_time = max_deriv_time_;
  matrix_prune_info_.resize(num_matrices);
  // matrix_prune_info_[0] will remain undefined.
  for (int32 matrix_index = 1; matrix_index < num_matrices; matrix_index++) {
    NnetComputation::MatrixDebugInfo &debug_info =
        computation_->matrix_debug_info[matrix_index];
    MatrixPruneInfo &prune_info = matrix_prune_info_[matrix_index];
    const std::vector<Cindex> &cindexes = debug_info.cindexes;
    int32 num_rows = computation_->matrices[matrix_index].num_rows;
    KALDI_ASSERT(num_rows == static_cast<int32>(cindexes.size()));
    int32 first_row_within_range = num_rows,
        last_row_within_range = -1;
    for (int32 i = 0; i < num_rows; i++) {
      int32 t = cindexes[i].second.t;
      if (t >= min_deriv_time && t <= max_deriv_time) {
        if (i < first_row_within_range) first_row_within_range = i;
        if (i > last_row_within_range) last_row_within_range = i;
      }
    }
    if (last_row_within_range == -1) {
      prune_info.fully_inside_range = false;
      prune_info.partly_inside_range = false;
    } else if (last_row_within_range == num_rows - 1 &&
               first_row_within_range == 0) {
      prune_info.fully_inside_range = true;
      prune_info.partly_inside_range = false;
    } else {
      prune_info.fully_inside_range = false;
      prune_info.partly_inside_range = true;
      prune_info.row_begin = first_row_within_range;
      prune_info.row_end = last_row_within_range + 1;
    }
  }
}

void DerivativeTimeLimiter::ComputeSubmatrixMaps() {
  int32 num_submatrices = computation_->submatrices.size();
  submatrix_map_.resize(num_submatrices);
  submatrix_map_if_deriv_.resize(num_submatrices);
  // index zero is for the empty submatrix.
  submatrix_map_[0] = 0;
  submatrix_map_if_deriv_[0] = 0;
  for (int32 s = 1; s < num_submatrices; s++) {
    NnetComputation::SubMatrixInfo &submatrix_info(computation_->submatrices[s]);
    int32 matrix_index = submatrix_info.matrix_index;
    int32 row_offset = submatrix_info.row_offset,
        num_rows = submatrix_info.num_rows;
    const MatrixPruneInfo &matrix_prune_info = matrix_prune_info_[matrix_index];
    if (matrix_prune_info.fully_inside_range) {
      submatrix_map_[s] = s;
    } else if (!matrix_prune_info.partly_inside_range) {
      // completely outside time range.
      submatrix_map_[s] = 0;
    } else {
      // the matrix is partly inside the time range.
      int32 pruned_row_begin = std::max(matrix_prune_info.row_begin,
                                        row_offset),
          pruned_row_end = std::min(matrix_prune_info.row_end,
                                    row_offset + num_rows);
      if (pruned_row_end <= pruned_row_begin) {
        // there was no overlap between the submatrix and the part
        // of the matrix that was inside the time range.
        submatrix_map_[s] = 0;
      } else {
        // caution: this invalidates the reference 'submatrix_info'.
        int32 row_offset_within_submatrix =
            pruned_row_begin - row_offset,
            new_num_rows = pruned_row_end - pruned_row_begin;
        submatrix_map_[s] =
            computation_->NewSubMatrix(s, row_offset_within_submatrix,
                                       new_num_rows, 0, -1);
      }
    }
    bool is_deriv = computation_->matrix_debug_info[matrix_index].is_deriv;
    submatrix_map_if_deriv_[s] = (is_deriv ?
                                  submatrix_map_[s] : s);
  }
}

void DerivativeTimeLimiter::ModifyCommands() {
  std::vector<NnetComputation::Command>::iterator
      iter = computation_->commands.begin(),
      end =  computation_->commands.end();
  for (; iter != end; ++iter)
    ModifyCommand(&(*iter));
}

// called from PruneMatrices only for matrices that are derivatives,
// not inputs or outputs of the computation, and which are partly
// inside the time range, this function returns true if we can
// limit the size of the matrix (because variables outside the
// desired range are never accessed), and false otherwise.
bool DerivativeTimeLimiter::CanLimitMatrix(const Analyzer &analyzer,
                                           int32 m) const {
  int32 s_whole = whole_submatrices_[m];  // submatrix consisting of
                                          // all of the matrix m
  int32 s_mapped = submatrix_map_[s_whole];  // submatrix consisting of the time
                                             // range of the matrix m that we
                                             // plan to limit it to.
  KALDI_ASSERT(s_mapped != 0 && s_mapped != s_whole);
  std::vector<int32> whole_variables, mapped_variables;
  analyzer.variables.AppendVariablesForSubmatrix(s_whole,
                                                 &whole_variables);
  analyzer.variables.AppendVariablesForSubmatrix(s_mapped,
                                                 &mapped_variables);
  KALDI_ASSERT(whole_variables.size() > mapped_variables.size());
  std::vector<int32> excluded_variables(whole_variables.size() -
                                        mapped_variables.size());
  std::vector<int32>::iterator end_iter =
      std::set_difference(whole_variables.begin(), whole_variables.end(),
                          mapped_variables.begin(), mapped_variables.end(),
                          excluded_variables.begin());
  KALDI_ASSERT(end_iter == excluded_variables.end());
  // We want to make sure that none of the excluded variables are ever accessed,
  // except possibly for zeroing or setting to other constant value.  If they
  // are, we cannot prune the matrix.
  for (std::vector<int32>::iterator iter = excluded_variables.begin();
       iter != end_iter; ++iter) {
    int32 variable_index = *iter;
    const std::vector<Access> &variable_accesses =
        analyzer.variable_accesses[variable_index];
    std::vector<Access>::const_iterator viter = variable_accesses.begin(),
        vend = variable_accesses.end();
    for (; viter != vend; ++viter) {
      // if a variable outside the pruned range of the matrix is ever accessed
      // apart from on allocation, we cannot prune.
      int32 command_index = viter->command_index;
      NnetComputation::Command &command = computation_->commands[command_index];
      if (command.command_type != kSetConst) {
        // we may one day want to look at this.. it's not really expected.
        KALDI_VLOG(3) << "Cannot prune matrix " << m;
        return false;
      }
    }
  }
  return true;
}

void DerivativeTimeLimiter::LimitMatrices(const std::vector<bool> &will_limit) {
  // first modify 'submatrices'.
  int32 num_submatrices = computation_->submatrices.size(),
      num_matrices = computation_->matrices.size();
  for (int32 s = 1; s < num_submatrices; s++) {
    NnetComputation::SubMatrixInfo &submat_info = computation_->submatrices[s];
    int32 m = submat_info.matrix_index;
    if (will_limit[m]) {
      // we need to do something...
      const MatrixPruneInfo &prune_info = matrix_prune_info_[m];
      int32 matrix_num_rows = prune_info.row_end - prune_info.row_begin;
      KALDI_ASSERT(matrix_num_rows > 0 &&
                   matrix_num_rows < computation_->matrices[m].num_rows);
      KALDI_ASSERT(prune_info.partly_inside_range);
      int32 new_row_begin = submat_info.row_offset - prune_info.row_begin;
      if (new_row_begin >= 0 &&
          submat_info.num_rows + new_row_begin <= matrix_num_rows) {
        // If this submatrix is entirely inside the limited range of the matrix,
        // then we modify its row_offset to account for the truncation of
        // rows to the left.
        submat_info.row_offset = new_row_begin;
      } else {
        // This submatrix is not entirely inside the kept range of the matrix.
        // We assume that this submatrix is never accessed directly except (if
        // it was the whole matrix) for in allocation and deallocation commands,
        // since when we modified the computation we ensured this.
        if (computation_->IsWholeMatrix(s)) {
          // If it was the whole matrix then it may be used in allocation and
          // deallocation commands, so we should modify it to be the whole of the
          // new matrix, which will have fewer rows than before.
          submat_info.num_rows = matrix_num_rows;
        } else {
          // We believe this matrix should never be used.  We give it a valid
          // but stupid size of num-rows=1, num-cols=1, so that if it ever does
          // get accessed it should produce an error.
          submat_info.row_offset = 0;
          submat_info.num_rows = 1;
          submat_info.col_offset = 0;
          submat_info.num_cols = 1;
        }
      }
    }
  }
  // next modify 'matrices'
  for (int32 m = 1; m < num_matrices; m++) {
    if (will_limit[m]) {
      const MatrixPruneInfo &prune_info = matrix_prune_info_[m];
      NnetComputation::MatrixInfo &matrix_info = computation_->matrices[m];
      if (!computation_->matrix_debug_info.empty()) {
        NnetComputation::MatrixDebugInfo &debug_info =
            computation_->matrix_debug_info[m];
        std::vector<Cindex> &cindexes = debug_info.cindexes;
        KALDI_ASSERT(cindexes.size() == static_cast<size_t>(matrix_info.num_rows));
        cindexes.erase(cindexes.begin() + prune_info.row_end, cindexes.end());
        cindexes.erase(cindexes.begin(),
                       cindexes.begin() + prune_info.row_begin);
      }
      matrix_info.num_rows = prune_info.row_end - prune_info.row_begin;
      // num_cols stays the same.
    }
  }
}

void DerivativeTimeLimiter::PruneMatrices() {
  Analyzer analyzer;
  analyzer.Init(nnet_, *computation_);
  KALDI_ASSERT(computation_->matrices.size() == whole_submatrices_.size());
  int32 num_matrices = computation_->matrices.size();
  std::vector<bool> will_limit(num_matrices, false);
  bool will_limit_at_least_one = false;
  for (int32 m = 1; m < num_matrices; m++) {
    const MatrixAccesses &accesses = analyzer.matrix_accesses[m];
    const MatrixPruneInfo &matrix_prune_info = matrix_prune_info_[m];
    if (matrix_prune_info.fully_inside_range ||
        accesses.is_input || accesses.is_output ||
        !computation_->matrix_debug_info[m].is_deriv)
      continue;  // nothing to do: it's inside the time-range or not a
                 // derivative.
    // if we got here it's not completely inside the time range, not an input or
    // an output, and it's a derivative.
    if (!matrix_prune_info.partly_inside_range) {
      // completely outside time range.  we can prune the matrix if it is not an
      // input or output, and is never accessed apart from allocation.
      if (MatrixIsUnused(analyzer, *computation_, m))
          RemoveCommandsForUnusedMatrix(analyzer, m, computation_);
    } else {
      // the matrix is partly inside the time range, it's a derivative, and not
      // an input or an output.
      if (CanLimitMatrix(analyzer, m)) {
        will_limit[m] = true;
        will_limit_at_least_one = true;
      }
    }
  }
  if (will_limit_at_least_one)
    LimitMatrices(will_limit);
}


void LimitDerivativeTimes(const Nnet &nnet,
                          int32 min_deriv_time,
                          int32 max_deriv_time,
                          NnetComputation *computation) {
  DerivativeTimeLimiter limiter(nnet, min_deriv_time, max_deriv_time,
                                computation);
  limiter.LimitDerivTimes();
}


/*
  This helper function, used in ReplaceRowWithMatrixOps, detects
  when the vector 'indexes' has a 'special structure'.  The special structure
  is:
    zero or more -1's, then
    a consecutive nonempty sequence of nonnegative numbers, e.g. 6 7 8 9 10, then
    zero or more -1's.

  Note: this function assumes that any negative elements of 'indexes' are -1.
  If there are elements less than -1, then it is an error, but this function
  does not thoroughly check for that.  'indexes' is required to be a nonempty
  vector.

  If 'indexes' has the special structure then this function returns true
  and sets the following values, which will explain with the following
  example in mind: 'indexes = [ -1, -1, 5 6 7 8, -1 ]'.
     - '*first_nonnegative_pos' is set to the number of initial -1's (and also
       the location of the first nonnegative element): 2 in this case.
     - '*first_nonnegative_value' is set to the value of the first nonnegative
       element (5 in this case)
     - '*num_nonnegative_values' is set to the number of nonnegative values in
       the sequence (4 in this case).
  If 'indexes' does not have this special structure, then this function returns
  false, and the values of '*first_nonnegative_pos',
  '*first_nonnegative_value' and '*num_nonnegative_indexes' on exit are
  undefined.
*/
static bool IndexesHaveSpecialStructure(const std::vector<int32> &indexes,
                                        int32 *first_nonnegative_pos,
                                        int32 *first_nonnegative_value,
                                        int32 *num_nonnegative_indexes) {
  KALDI_ASSERT(!indexes.empty());
  const int32 *indexes_ptr = &(indexes[0]);
  size_t pos = 0, size = indexes.size();

  // Find the first nonnegative element of 'indexes'.
  for (; pos < size; ++pos)
    if (indexes_ptr[pos] >= 0)
      break;
  if (pos == size)
    return false;  // all -1's... should not happen, but not our problem.
  *first_nonnegative_pos = static_cast<int32>(pos);
  int32 n = indexes_ptr[pos];
  *first_nonnegative_value = n;
  // Find the first element after '*first_nonnegative_index' that isn't
  // consecutive.
  for (; pos < size; ++pos,++n)
    if (indexes_ptr[pos] != n)
      break;

  *num_nonnegative_indexes = n - *first_nonnegative_value;

  // Check that the remaining values are all <0 (assumed equal to -1, but
  // checking <0 may be faster as just one instruction).
  for (; pos < size; ++pos)
    if (indexes_ptr[pos] >= 0)
      return false;  // does not have the special structure.

  return true;
}



bool ReplaceRowWithMatrixOps(NnetComputation *computation) {
  bool ans = false;
  int32 num_commands = computation->commands.size(),
      num_indexes = computation->indexes.size();
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    // non-const because we'll be changing it.
    NnetComputation::Command &c = computation->commands[command_index];

    int32 first_nonnegative_pos,
        first_nonnegative_value,
        num_nonnegative_indexes;
    switch (c.command_type) {
      case kCopyRows: case kAddRows: {
        int32 indexes_index = c.arg3;
        KALDI_ASSERT(indexes_index < num_indexes);
        const std::vector<int32> &indexes = computation->indexes[indexes_index];
        if (IndexesHaveSpecialStructure(indexes,
                                        &first_nonnegative_pos,
                                        &first_nonnegative_value,
                                        &num_nonnegative_indexes)) {
          ans = true;
          c.arg1 = computation->NewSubMatrix(c.arg1, first_nonnegative_pos,
                                             num_nonnegative_indexes,
                                             0, -1);
          c.arg2 = computation->NewSubMatrix(c.arg2, first_nonnegative_value,
                                             num_nonnegative_indexes,
                                             0, -1);
          c.command_type = (c.command_type == kCopyRows ? kMatrixCopy :
                            kMatrixAdd);
        }
        break;
      }
      default:
        break;
    }
  }
  return ans;
}



/*
  This function, used in SnipSingleRowOp(),
  finds the number of leading, and trailing, negative numbers
  in a vector of integers.  For instance, if vec is
    [ -1 -1 2 3 -1 4 5 -1 ]
  then '*num_leading_negatives' will be set to 2 and '*num_trailing_negatives'
  will be set to 1.  If all the numbers in 'vec' are all negative, or 'vec' is
  empty, it is an error and this function will invoke KALDI_ERR.
*/
static void FindNumLeadingAndTrailingNegatives(const std::vector<int32> &vec,
                                               int32 *num_leading_negatives,
                                               int32 *num_trailing_negatives) {
  KALDI_ASSERT(!vec.empty());
  const int32 *begin = &(vec[0]), *ptr = begin, *end = ptr + vec.size();
  while (ptr != end && *ptr < 0)
    ptr++;
  // note regarding error message: we assume all negative numbers are -1, due to
  // the way this is called, but it only affects how we describe the error.
  KALDI_ASSERT(ptr != end && "Vector consists entirely of -1's.");
  *num_leading_negatives = ptr - begin;
  const int32 *ptr2 = end - 1;
  // the following while loop should terminate before falling off the vector,
  // because we've established above (in the assertion) that the vector contains
  // at least one nonnegative number.
  while (*ptr2 < 0)
    ptr2--;
  KALDI_ASSERT(ptr2 >= begin);  // or would be code error.
  *num_trailing_negatives = end - 1 - ptr2;
}

// This function, called from SnipRowOps, is called when it encounters commands
// of type kAddRows; it modifies such commands when the indexes have leading or
// trailing -1's,h, to make them operate on a smaller submatrix.  It returns
// true if it made a change, and false otherwise.
static bool SnipSingleRowOp(NnetComputation *computation,
                            int32 command_index) {
  NnetComputation::Command &c = computation->commands[command_index];
  KALDI_ASSERT(static_cast<size_t>(c.arg3) < computation->indexes.size());
  const std::vector<int32> &indexes = computation->indexes[c.arg3];
  int32 num_leading_negatives, num_trailing_negatives;
  FindNumLeadingAndTrailingNegatives(indexes,
                                    &num_leading_negatives,
                                    &num_trailing_negatives);
  if (num_leading_negatives == 0 && num_trailing_negatives == 0)
    return false;

  int32 new_num_rows = static_cast<int32>(indexes.size()) -
      num_leading_negatives - num_trailing_negatives;
  KALDI_ASSERT(new_num_rows > 0);
  std::vector<int32> new_indexes(indexes.begin() + num_leading_negatives,
                                 indexes.begin() + num_leading_negatives +
                                 new_num_rows);
  c.arg3 = computation->indexes.size();
  computation->indexes.push_back(std::vector<int32>());
  computation->indexes.back().swap(new_indexes);
  c.arg1 = computation->NewSubMatrix(c.arg1,
                                     num_leading_negatives, new_num_rows,
                                     0, -1);
  return true;  // made a change.
}



/*
  This function, used in SnipSingleRowOp(), finds the number of leading, and
  trailing, negative values in a vector of pairs of integers.  In particular,
  it finds the number of leading and trailing pairs whose .first value is negative
  (in practice we'll only encounter either (-1,-1) pairs, or pairs of both
  nonnegative values).

  For instance, if vec is
    [ (-1,-1) (-1,-1) (80,2) (81,3) (-1,-1) (80,4) (81,5) (-1,-1) ]
  then '*num_leading_negatives' will be set to 2 and '*num_trailing_negatives'
  will be set to 1.  If all the .first numbers in 'vec' are all negative, or
  'vec' is empty, it is an error and this function will invoke KALDI_ERR.
*/
static void FindNumLeadingAndTrailingNegatives(
    const std::vector<std::pair<int32, int32> > &vec,
    int32 *num_leading_negatives,
    int32 *num_trailing_negatives) {
  KALDI_ASSERT(!vec.empty());
  const std::pair<int32, int32> *begin = &(vec[0]), *ptr = begin,
      *end = ptr + vec.size();
  while (ptr != end && ptr->first < 0)
    ptr++;
  // note regarding error message: we assume all negative numbers are -1, due to
  // the way this is called, but it only affects how we describe the error.
  KALDI_ASSERT(ptr != end && "Vector consists entirely of -1's.");
  *num_leading_negatives = ptr - begin;
  const std::pair<int32, int32> *ptr2 = end - 1;
  // the following while loop should terminate before falling off the vector,
  // because we've established above (in the assertion) that the vector contains
  // at least one nonnegative number.
  while (ptr2->first < 0)
    ptr2--;
  KALDI_ASSERT(ptr2 >= begin);  // would be code error.
  *num_trailing_negatives = end - 1 - ptr2;
}


// This function, called from SnipRowOps, is called when it encounters commands
// of type kAddRowsMulti, kAddToRowsMulti, or kCopyToRowsMulti; have leading or
// trailing (-1,-1) pairs, to make them operate on a smaller submatrix.  It
// returns true if it made a change, and false otherwise.
static bool SnipMultiRowOp(NnetComputation *computation,
                           int32 command_index) {
  NnetComputation::Command &c = computation->commands[command_index];
  KALDI_ASSERT(static_cast<size_t>(c.arg2) < computation->indexes_multi.size());
  const std::vector<std::pair<int32, int32> > &indexes_multi =
      computation->indexes_multi[c.arg2];
  int32 num_leading_negatives, num_trailing_negatives;
  FindNumLeadingAndTrailingNegatives(indexes_multi,
                                    &num_leading_negatives,
                                    &num_trailing_negatives);
  if (num_leading_negatives == 0 && num_trailing_negatives == 0)
    return false;

  int32 new_num_rows = static_cast<int32>(indexes_multi.size()) -
      num_leading_negatives - num_trailing_negatives;
  KALDI_ASSERT(new_num_rows > 0);
  std::vector<std::pair<int32, int32> > new_indexes_multi(
      indexes_multi.begin() + num_leading_negatives,
      indexes_multi.begin() + num_leading_negatives + new_num_rows);
  c.arg2 = computation->indexes_multi.size();
  computation->indexes_multi.push_back(std::vector<std::pair<int32, int32> >());
  computation->indexes_multi.back().swap(new_indexes_multi);
  c.arg1 = computation->NewSubMatrix(c.arg1,
                                     num_leading_negatives, new_num_rows,
                                     0, -1);
  return true;  // made a change.
}



/*
  This function, used in SnipRangeRowOp(), finds the number of leading and
  trailing values in a vector of pairs of integers, that are the same (i.e.
  pairs of the form (x, x) for any x.  [This is how we represent an empty
  range, which is a kind of no-op, in commands of kCopyRowRanges or
  kAddRowRanges.

  For instance, if vec is
    [ (0,0) (0,0) (4,5) (6,8) (0,0) (10,12) (14,20) (0,0) ]
  then '*num_leading_identicals' will be set to 2 and '*num_trailing_identicals'
  will be set to 1.  If all pairs in 'vec' are identical, or 'vec' is empty, it
  is an error and this function will invoke KALDI_ERR.
*/
static void FindNumLeadingAndTrailingIdenticals(
    const std::vector<std::pair<int32, int32> > &vec,
    int32 *num_leading_identicals,
    int32 *num_trailing_identicals) {
  KALDI_ASSERT(!vec.empty());
  const std::pair<int32, int32> *begin = &(vec[0]), *ptr = begin,
      *end = ptr + vec.size();
  while (ptr != end && ptr->first == ptr->second)
    ptr++;
  // note regarding error message: we assume all pairs of identical numbers are
  // -1, due to the way this is called, but it only affects how we describe the
  // error.
  KALDI_ASSERT(ptr != end && "Vector consists entirely of -1's.");
  *num_leading_identicals = ptr - begin;
  const std::pair<int32, int32> *ptr2 = end - 1;
  // the following while loop should terminate before falling off the vector,
  // because we've established above (in the assertion) that the vector contains
  // at least one nonnegative number.
  while (ptr2->first == ptr2->second)
    ptr2--;
  KALDI_ASSERT(ptr2 >= begin);  // would be code error.
  *num_trailing_identicals = end - 1 - ptr2;
}


// This function, called from SnipRowOps, is called when it encounters commands
// of type kAddRowRanges that have leading or trailing (x, x) pairs [i.e. pairs
// of identical values; these are how we represent empty ranges], to make them
// operate on a smaller submatrix.  It returns true if it made a change, and
// false otherwise.
static bool SnipRangesRowOp(NnetComputation *computation,
                            int32 command_index) {
  NnetComputation::Command &c = computation->commands[command_index];
  KALDI_ASSERT(static_cast<size_t>(c.arg3) < computation->indexes_ranges.size());
  const std::vector<std::pair<int32, int32> > &indexes_ranges =
      computation->indexes_ranges[c.arg3];
  int32 num_leading_identicals, num_trailing_identicals;
  FindNumLeadingAndTrailingIdenticals(indexes_ranges,
                                    &num_leading_identicals,
                                    &num_trailing_identicals);
  if (num_leading_identicals == 0 && num_trailing_identicals == 0)
    return false;

  int32 new_num_rows = static_cast<int32>(indexes_ranges.size()) -
      num_leading_identicals - num_trailing_identicals;
  KALDI_ASSERT(new_num_rows > 0);
  std::vector<std::pair<int32, int32> > new_indexes_ranges(
      indexes_ranges.begin() + num_leading_identicals,
      indexes_ranges.begin() + num_leading_identicals + new_num_rows);
  c.arg3 = computation->indexes_ranges.size();
  computation->indexes_ranges.push_back(std::vector<std::pair<int32, int32> >());
  computation->indexes_ranges.back().swap(new_indexes_ranges);
  c.arg1 = computation->NewSubMatrix(c.arg1,
                                     num_leading_identicals, new_num_rows,
                                     0, -1);
  return true;  // made a change.
}



bool SnipRowOps(NnetComputation *computation) {
  bool ans = false;
  int32 num_commands = computation->commands.size();
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    // non-const because we'll be changing it.
    NnetComputation::Command &c = computation->commands[command_index];

    // note: we can't do the snipping for commands of type case kCopyRows and case
    // kCopyRowsMulti, because the -1's aren't a pure no-op; they have the
    // meaning of setting the destination value to zero, so we can't prune
    // them away.

    switch (c.command_type) {
      case kAddRows: {
        if (SnipSingleRowOp(computation, command_index))
          ans = true;
        break;
      }
      case kAddRowsMulti: case kAddToRowsMulti:
      case kCopyToRowsMulti: {
        if (SnipMultiRowOp(computation, command_index))
          ans = true;
        break;
      }
      case kAddRowRanges: {
        if (SnipRangesRowOp(computation, command_index))
          ans = true;
        break;
      }
      default:
        break;
    }
  }
  return ans;
}



// This class implements the internals of the function SplitRowOps() which is
// declared in nnet-optimize-utils.h.
class RowOpsSplitter {
 public:
  RowOpsSplitter(NnetComputation *computation): computation_(computation) { }

  // Attempts to perform the optimization.  Returns true if it made any change
  // to the computation.
  bool Split() {
    return SplitIndexes() && SplitCommands();
  }

 private:

  // This function sets up split_info_, which describes how we can split up
  // the vectors that are elements of computation_->indexes_multi.
  // It will return true if it successfully split at least one of those
  // vectors, and false otherwise.
  bool SplitIndexes();

  // This function modifies the commands in the computation.  It returns
  // true if it made any change.
  bool SplitCommands();


  // This function attempts to optimize the command in
  // computation_->commands[command_index].  It returns true if it made any
  // change.  If we are going to have to insert an extra command into the
  // computation, this function will append an element to new_commands_.
  bool SplitCommand(int32 command_index);

  // Below, define a multi-index as an element of NnetComputation::indexes_multi,
  // for example,
  // const std::vector<std::pair<int32,int32> > &multi_index = computation_->indexes_multi[1];
  // It is a list of pairs.

  // This struct appears as an element of the list inside MultiIndexSplitInfo.
  // It helps us describe how we can split up a multi-index (a list of pairs)
  // into a sequence of ranges where the .first value is constant across the
  // range.
  struct SingleSplitInfo {
    // 'offset' is the index into the vector of pairs that forms the
    // start of this range.  In the example where we are splitting up
    // ((10,2), (10,3), (10,4), (15,3), (15,5), (15,7))
    // there would be two instances of struct SingleSplitInfo, with
    // offset = 0 and offset = 3.
    int32 offset;
    // 'size' is the number of pairs in this range; in the example
    // above, both 'size' elements would be 3.
    int32 size;
    // first_value is the value of the .first index throughout this range; in
    // the example above, it would be 10 and 15 respectively.  It represents a
    // submatrix index.
    int32 first_value;

    // initial_second_value is the minimum value of .second for any element in
    // this range: it would be 2 and 3 respectively in the example above.
    int32 min_second_value;

    // second_value_range is the highest value of .second for any element in
    // this range, plus one, minus min_second_value.  (It's the number of rows
    // in the other submatrix of the operation).
    int32 second_value_range;

    // If the .second values in the range are consecutive then
    // 'second_value_offsets' will be empty.  Otherwise it will
    // be a vector of size 'size', containing numbers in the
    // range 0 ... second_value_range - 1, such that
    // min_second_value + second_value_offsets[i] gives
    // the .second value at the corresponding position in the range.
    // In the second range of the example above, the range
    // consisting of ((15,3), (15,5), (15,7)), 'second_value_offsets
    // would be the vector (0, 2, 4).
    std::vector<int32> second_value_offsets;
  };

  // An instance of the struct MultiIndexSplitInfo will be created for each multi-index,
  // i.e. for each element of  NnetComputation::indexes_multi.
  struct MultiIndexSplitInfo {
    // If we can split this multi-index into at most two ranges, this
    // vector will be nonempty; otherwise it will be empty.
    std::vector<SingleSplitInfo> splits;
  };

  // GetSplitInfo() attempts to take a range of a
  // std::vector<std::pair<int32, int32> >, as represented by begin and end
  // iterators, and to extract its information into an object of type
  // SingleSplitInfo.  (all except for the .offset member, which will have
  // been set by calling code).
  // It return true if successful, and false otherwise.  The only reasons that
  // it might return false are that the range contains -1's or does not contain
  // all-identical .first members).
  bool GetSplitInfo(std::vector<std::pair<int32, int32> >::const_iterator begin,
                    std::vector<std::pair<int32, int32> >::const_iterator end,
                    SingleSplitInfo *info);

  // computation_ is the computation that we are modifying.
  NnetComputation *computation_;
  // split_info_ will contain information about how we can split up the members
  // of computation_->indexes_multi into ranges.
  std::vector<MultiIndexSplitInfo> split_info_;
  // The following is a list of additional commands that we are going to insert
  // into computation_, of the form (command-index, command) where command-index
  // is a command index just before which we will insert the new command.
  // (this is the format accepted by the function InsertCommands()).
  std::vector<std::pair<int32, NnetComputation::Command> > new_commands_;

};


bool RowOpsSplitter::GetSplitInfo(
    std::vector<std::pair<int32, int32> >::const_iterator begin,
    std::vector<std::pair<int32, int32> >::const_iterator end,
    SingleSplitInfo *info) {
  // max_size_ratio must be > 1.0, and could in principle be a float.  It is
  // there to prevent us from making changes to the computation which would end
  // up wastefully launching too many kernels that would do nothing.
  const int32 max_size_ratio = 2;

  int32 size = end - begin;
  KALDI_ASSERT(size != 0);
  int32 first = begin->first;
  if (first < 0)
    return false;
  info->size = size;
  info->first_value = first;
  int32 initial_second_value = begin->second,
      min_second_value = initial_second_value,
      max_second_value = initial_second_value;
  info->second_value_offsets.resize(size);
  bool is_consecutive = true;
  for (int32 i = 0; i < size; i++) {
    int32 second = begin[i].second;
    if (begin[i].first != first || second < 0) return false;
    info->second_value_offsets[i] = second;
    if (second != initial_second_value + i)
      is_consecutive = false;
    if (second < min_second_value) min_second_value = second;
    if (second > max_second_value) max_second_value = second;
  }
  info->min_second_value = min_second_value;
  info->second_value_range = max_second_value + 1 - min_second_value;
  if (info->second_value_range > size * max_size_ratio)
    return false;
  if (is_consecutive) {
    info->second_value_offsets.clear();
  } else {
    for (int32 i = 0; i < size; i++)
      info->second_value_offsets[i] -= min_second_value;
  }
  return true;
}


bool RowOpsSplitter::SplitIndexes() {
  bool ans = false;
  int32 num_indexes_multi = computation_->indexes_multi.size();
  split_info_.resize(num_indexes_multi);
  for (int32 i = 0; i < num_indexes_multi; i++) {
    const std::vector<std::pair<int32,int32> > &multi_index =
        computation_->indexes_multi[i];
    MultiIndexSplitInfo &split_info = split_info_[i];

    int32 num_pairs = multi_index.size();
    KALDI_ASSERT(num_pairs > 0);
    // 'split_point' will be set to the first index j for which
    // multi_index[j-1].first != multi_index[j].first, or -1
    // if no such j exists.
    int32 split_point = -1, initial_first = multi_index[0].first;
    for (int32 j = 1; j < num_pairs; j++) {
      if (multi_index[j].first != initial_first) {
        split_point = j;
        break;
      }
    }
    if (split_point == -1) {
      split_info.splits.resize(1);
      split_info.splits[0].offset = 0;
      if (!GetSplitInfo(multi_index.begin(), multi_index.end(),
                        &(split_info.splits[0]))) {
        split_info.splits.clear();
      } else {
        ans = true;
      }
    } else {
      split_info.splits.resize(2);
      split_info.splits[0].offset = 0;
      split_info.splits[1].offset = split_point;

      std::vector<std::pair<int32,int32> >::const_iterator mid_iter =
          multi_index.begin() + split_point;
      if (!GetSplitInfo(multi_index.begin(), mid_iter,
                        &(split_info.splits[0])) ||
          !GetSplitInfo(mid_iter, multi_index.end(),
                        &(split_info.splits[1]))) {
        split_info.splits.clear();
      } else {
        ans = true;
      }
    }
  }
  return ans;
}

bool RowOpsSplitter::SplitCommand(int32 c) {
  NnetComputation::Command &command = computation_->commands[c];
  CommandType command_type = command.command_type;
  // For commands that are not of the following four types, return false: we
  // won't be changing these commands.
  switch (command_type) {
    case kAddRowsMulti: case kCopyRowsMulti:
    case kAddToRowsMulti: case kCopyToRowsMulti: break;
    default: return false;
  }
  int32 indexes_multi_index = command.arg2;
  KALDI_ASSERT(indexes_multi_index <
               static_cast<int32>(split_info_.size()));
  const MultiIndexSplitInfo &split_info = split_info_[indexes_multi_index];
  if (split_info.splits.empty())
    return false;  // these indexes couldn't be split: e.g. they contained more
                   // than two distinct .first elements, or there were other
                   // reasons.

  // we'll be splitting the command into either one or two pieces.
  std::vector<NnetComputation::Command> split_commands(
      split_info.splits.size());
  for (size_t i = 0; i < split_info.splits.size(); i++) {
    const SingleSplitInfo &split = split_info.splits[i];
    NnetComputation::Command &command_out = split_commands[i];
    command_out.alpha = command.alpha;
    command_out.arg1 = computation_->NewSubMatrix(
        command.arg1, split.offset, split.size, 0, -1);
    command_out.arg2 = computation_->NewSubMatrix(
        split.first_value, split.min_second_value,
        split.second_value_range, 0, -1);

    if (split.second_value_offsets.empty()) {
      // The .second elements are consecutive.
      switch (command_type) {
        case kAddRowsMulti:
          command_out.command_type = kMatrixAdd;
          break;
        case kCopyRowsMulti:
          command_out.command_type = kMatrixCopy;
          break;
        case kAddToRowsMulti:
          command_out.command_type = kMatrixAdd;
          std::swap(command_out.arg1, command_out.arg2);
          break;
        case kCopyToRowsMulti:
          command_out.command_type = kMatrixCopy;
          std::swap(command_out.arg1, command_out.arg2);
          break;
        default:  // will never be reached.
          break;
      }
    } else {
      // Indexes are not consecutive: it needs to be a kAddRows or kCopyRows
      // command.
      command_out.arg3 = computation_->indexes.size();
      switch (command_type) {
        case kAddRowsMulti: case kCopyRowsMulti: {
          command_out.command_type = (command_type == kAddRowsMulti ?
                                      kAddRows : kCopyRows);
          computation_->indexes.push_back(split.second_value_offsets);
          break;
        }
        case kCopyToRowsMulti:  {
          // We can't operate on this command because of what would happen
          // with values of 'indexes' (see the variable in the block for
          // kAddToRowsMulti) which were -1.  Rows of the output would be
          // set to zero, which is not the behavior we want here; we'd want
          // them to be unaffected.
          return false;
        }
        case kAddToRowsMulti: {
          command_out.command_type = kAddRows;
          std::swap(command_out.arg1, command_out.arg2);
          // invert the indexes.
          std::vector<int32> indexes(split.second_value_range, -1);
          for (int32 i = 0; i < split.size; i++) {
            // the following assert should always succeed because the
            // AddToRowsMulti and CopyToRowsMulti should never have
            // duplicate destinations in their indexes.
            KALDI_ASSERT(indexes[split.second_value_offsets[i]] >= 0);
            indexes[split.second_value_offsets[i]] = i;
          }
          computation_->indexes.push_back(indexes);
          break;
        }
        default:
          KALDI_ERR << "Code error: un-handled case.";
      }
    }
  }
  command = split_commands[0];
  // note: for now, split_commands.size() will be 1 or 2.
  for (size_t i = 1; i < split_commands.size(); i++) {
    new_commands_.resize(new_commands_.size() + 1);
    // we'll want to insert this command right after command c,
    // which is the same as just before command c + 1.
    new_commands_.back().first = c + 1;
    new_commands_.back().second = split_commands[i];
  }
  return true;  // We made a change.
}

bool RowOpsSplitter::SplitCommands() {
  bool ans = false;
  int32 num_commands = computation_->commands.size();
  for (int32 c = 0; c < num_commands; c++)
    if (SplitCommand(c))
      ans = true;
  if (!new_commands_.empty())
    InsertCommands(&new_commands_, computation_);
  return ans;
}

bool SplitRowOps(NnetComputation *computation) {
  RowOpsSplitter splitter(computation);
  return splitter.Split();
}


/*
   This function finds and returns the 'n-stride' of the vector of Indexes, or
   returns 0 if it does not exist because the Indexes lack the required regular
   structure.  This function relates to 'shortcut' compilation and is used in
   class IoSpecificationIsDecomposable().  There is an overloaded version of
   this function that works with Cindex input, that has almost exactly
   the same code.

   It is used to discover regular structure in vectors of indexes.  We are
   interested in the structure on the 'n' index; in particular, the stride on
   the 'n' index.  We expect the vector 'indexes' to contain 'n' values of the
   form 0, 1, ... N-1 (where the value of N can be obtained easily by looking at
   the .n value of the last element of 'indexes').  And we expect the 'n' values
   of Indexes that are otherwise the same to be separated by a fixed stride,
   which we will return.

   If the stride is inconsistent or one of our other requirements (see below) is
   not fulfilled, we will return 0.  If it's always consistent and our
   requirements are fulfilled we'll return the stride.  If 'full_check' is true
   we do an exhaustive check for consistency; otherwise we do a randomized
   check.

   The full definition of 'consistency' is as follows:

   For some n_stride >= 1 (which we'll return), and with N as the number of
   'n' values (which should be numbered 0, 1, ... N-1):

     - For any Index with n < N-1 located at position i, an Index with one
       greater 'n' but otherwise the same must exist at position i + n_stride
     - For any Index with n > 0 located at position i, an Index with one
       smaller 'n' but otherwise the same must exist at position i - n_stride.
     - The input must be arranged in blocks of size block_size = n_stride * N,
       which these strides never cross.  "Strides never cross" is an informal
       definition: we can formalize this by saying that for an Index with n == 0
       at position i, we must have (i / block_size) == ((i + n_stride*(N-1)) /
       block_size), with integer division.
   The above conditions imply that the size of the input must be a multiple
   of the n-stride.

   Reminder: we return 0 if the regular structure is not found, and the n-stride
   if the regular structure is found.
*/
static int32 FindNStride(const std::vector<Index> &indexes,
                         bool full_check) {
  // First find candidate stride.  Later we'll check for consistency.
  int32 size = indexes.size();
  KALDI_ASSERT(size > 0);
  int32 N = indexes[size-1].n + 1,
      n_stride;
  if (N <= 1) {
    // we wouldn't be able to determine the stride if N <= 1.
    return 0;
  }
  Index index(indexes[0]);
  if (index.n != 0 || size % N != 0) {
    // for the n stride to be positive, we must start with an index with n == 0.
    // if indexes.size() is not divisible by N, we have no hope of finding the
    // regular structure.
    return 0;
  }
  index.n = 1;
  // First check the two most common strides, which are 1
  // and size / N.
  if (indexes[1] == index) {
    n_stride = 1;
  } else if (indexes[size / N] == index) {
    n_stride = size / N;
  } else {
    int32 stride;
    // try the other possible strides one by one (for subsampling
    // layers of convnets, we might see strides of 2, for instance).
    for (stride = 2; stride < size / N; stride++) {
      if (size % stride == 0 && indexes[stride] == index) {
        n_stride = stride;
        break;
      }
    }
    if (stride == size / N) {
      // if we fell off the loop then we found no candidates, which is strange
      // and means we did not find the expected structure; we'll return 0 as we
      // failed.
      return 0;
    }
  }
  // Now is the checking phase.

  // to understand block_size, see the comment above this functcion.
  int32 block_size = n_stride * N;

  std::vector<int32> indexes_to_check;
  if (full_check) {
    indexes_to_check.resize(size);
    for (int32 i = 0; i < size; i++)
      indexes_to_check[i] = i;
  } else {
    int32 num_to_check = std::min<int32>(5, size);
    indexes_to_check.resize(num_to_check);
    for (int32 j = 0; j < num_to_check; j++)
      indexes_to_check[j] = RandInt(0, size - 1);
    SortAndUniq(&indexes_to_check);
  }
  for (std::vector<int32>::iterator iter = indexes_to_check.begin();
       iter != indexes_to_check.end(); ++iter) {
    int32 i = *iter;
    Index index = indexes[i];
    int32 n = index.n;
    if (n < N - 1) {
      index.n = n + 1;
      if (i + n_stride >= size || indexes[i + n_stride] != index)
        return 0;
    }
    if (n == 0) {
      if (i / block_size != (i + n_stride * (N-1)) / block_size) {
        // this is a check that the input divides into blocks of size n_stride *
        // N and the N different versions of the same Index are always within a
        // block (i.e. that the n stride never crosses over the block; having
        // the same Index repeated within different blocks actually would not
        // matter).
        return 0;
      }
    } else { // n > 0
      index.n = n - 1;
      if (i - n_stride < 0 || indexes[i - n_stride] != index)
        return 0;
    }
  }
  return n_stride;
}


// This is almost exactly the same as the version of FindNStride declared above
// that takes a vector of Indexes as input.  Comments have been removed from
// this version; see the other version for documentation.
static int32 FindNStride(const std::vector<Cindex> &cindexes,
                         bool full_check) {
  int32 size = cindexes.size();
  KALDI_ASSERT(size > 0);
  int32 N = cindexes[size-1].second.n + 1,
      n_stride;
  if (N <= 1)
    return 0;
  Cindex cindex(cindexes[0]);
  if (cindex.second.n != 0 || size % N != 0)
    return 0;
  cindex.second.n = 1;
  if (cindexes[1] == cindex) {
    n_stride = 1;
  } else if (cindexes[size / N] == cindex) {
    n_stride = size / N;
  } else {
    int32 stride;
    for (stride = 2; stride < size / N; stride++) {
      if (size % stride == 0 && cindexes[stride] == cindex) {
        n_stride = stride;
        break;
      }
    }
    if (stride == size / N)
      return 0;
  }
  int32 block_size = n_stride * N;
  std::vector<int32> indexes_to_check;
  if (full_check) {
    indexes_to_check.resize(size);
    for (int32 i = 0; i < size; i++)
      indexes_to_check[i] = i;
  } else {
    int32 num_to_check = std::min<int32>(5, size);
    indexes_to_check.resize(num_to_check);
    for (int32 j = 0; j < num_to_check; j++)
      indexes_to_check[j] = RandInt(0, size - 1);
    SortAndUniq(&indexes_to_check);
  }
  for (std::vector<int32>::iterator iter = indexes_to_check.begin();
       iter != indexes_to_check.end(); ++iter) {
    int32 i = *iter;
    Cindex cindex = cindexes[i];
    int32 n = cindex.second.n;
    if (n < N - 1) {
      cindex.second.n = n + 1;
      if (i + n_stride >= size || cindexes[i + n_stride] != cindex)
        return 0;
    }
    if (n == 0) {
      if (i / block_size != (i + n_stride * (N-1)) / block_size)
        return 0;
    } else {
      cindex.second.n = n - 1;
      if (i - n_stride < 0 || cindexes[i - n_stride] != cindex)
        return 0;
    }
  }
  return n_stride;
}


/*
  This function, used in shortcut compilation, converts a vector of Indexes
  having a range of 'n' values (0, 1, ... old_N - 1), to a vector of
  Indexes that's otherwise the same, but has a different range of 'n' values
  (0, 1, ... new_N - 1).

  The input vector is expected to have a stride 'n_stride > 0', as
  would be returned by FindNStride, and the output vector will be given the
  same n-stride.
 */
static void ConvertNumNValues(int32 n_stride, int32 old_N, int32 new_N,
                              const std::vector<Index> &indexes_in,
                              std::vector<Index> *indexes_out) {
  int32 size_in = indexes_in.size();
  KALDI_ASSERT(size_in > 0 && indexes_in[size_in - 1].n == old_N - 1);
  int32 block_size_in = n_stride * old_N,
      block_size_out = n_stride * new_N;

  indexes_out->resize((size_in / old_N) * new_N);
  for (int32 i_in = 0; i_in < size_in; i_in++) {
    if (indexes_in[i_in].n != 0)
      continue;
    Index index(indexes_in[i_in]);
    int32 block_index = i_in / block_size_in,
        offset_within_block = i_in % block_size_in;


    int32 i_out = block_index * block_size_out +
        offset_within_block;
    for (int32 n = 0; n < new_N; n++, i_out += n_stride) {
      index.n = n;
      (*indexes_out)[i_out] = index;
    }
  }
}



// This class implements the internals of the ExpandComputation() function (used
// in shortcut compilation); see comment by the declaration of
// ExpandComputation() in nnet-optimize-utils.h for overview.  (It relates to
// shortcut compilation).
class ComputationExpander {
 public:
  ComputationExpander(const Nnet &nnet,
                      const MiscComputationInfo &misc_info,
                      const NnetComputation &computation,
                      bool need_debug_info,
                      int32 num_n_values,
                      NnetComputation *expanded_computation):
      nnet_(nnet), misc_info_(misc_info),
      computation_(computation),
      need_debug_info_(need_debug_info),
      num_n_values_(num_n_values),
      expanded_computation_(expanded_computation) {
    KALDI_ASSERT(num_n_values > 2);
  }

  // This function call implements the functionality of the class,
  // expanding the computation.
  void Expand();

 private:
  // This function sets up and computes the 'n_stride_' vector (see comment
  // by the declaration of 'n_stride_' for what this is.
  void InitStrideInfo();

  // This function sets up the 'matrices' vector in 'expanded_computation_'.
  // It's quite simple: it just multiplies all the num-rows by num_n_values_ and
  // divides by 2, and leaves the num-cols the same.
  void ComputeMatrixInfo();

  // This function, only called if need_debug_info_ is true, sets up
  // the 'matrix_debug_info' vector in 'expanded_computation_'.
  void ComputeDebugInfo();

  // This function sets up the 'submatrices' vector in 'expanded_computation_'.
  // Column ranges always stay the same, but for row ranges it's a little
  // more complicated.
  void ComputeSubmatrixInfo();

  // Expands a command of type kCopyRows or kAddRows; involves adding a new
  // element of 'indexes' to expanded_computation_.
  void ExpandRowsCommand(const NnetComputation::Command &c_in,
                         NnetComputation::Command *c_out);

  // Expands a command of type kCopyRowsMulti or kAddRowsMulti, kCopyToRowsMulti
  // or kAddToRowsMulti; involves adding a new element of 'indexes_multi' to
  // expanded_computation_.
  void ExpandRowsMultiCommand(const NnetComputation::Command &c_in,
                              NnetComputation::Command *c_out);


  // Expands a command of type kAddRowRanges; involves adding a new element of
  // 'indexes_ranges' to expanded_computation_.
  void ExpandRowRangesCommand(const NnetComputation::Command &c_in,
                              NnetComputation::Command *c_out);


  // This function computes all the PrecomputedIndexes in the
  // 'component_precomputed_indexes' member of 'expanded_computation_'.
  // They are all generated from scratch, by using the Component::PrecomputedIndexes()
  // member function.  The 'input_indexes' and 'output_indexes' arguments are worked
  // out from the 'debug_info' [if we're not generating debug_info we specially generate
  // it for the specific matrices in question], and the 'need_backprop'
  // argument is worked out by seeing whether there is a call to Backprop() with
  // the same precomputed-indexes element.
  void ComputePrecomputedIndexes();

  // Computes the 'commands' member of the output.  This function also adds as
  // needed to 'indexes', 'indexes_multi' and 'indexes_ranges' in the output.
  // Later on we can call RenumberComputation() to remove any duplicates that
  // might result from this.
  void ComputeCommands();


  // This command ensure that the debug-info in expanded_computation_ for the
  // matrix underlying the submatrix with index 'submatrix_index', exists and is
  // set up.  In some cases we need the debug info for some matrices in order to
  // do the expansion, even if debug info is not requested for the output; in
  // those cases we set it up temporarily and clear it before we finish.
  void EnsureDebugInfoExists(int32 submatrix_index);



  // This function is used in mapping row-indexes into sub-matrices from the
  // old to the new computation.  It is mostly a wrapper for
  // GetNewMatrixLocationInfo, but designed to give row indexes into
  // submatrices rather than matrices; see the documentation for
  // GetNewMatrixLocationinfo() for details and an explanation of the
  // interface.
  // This function assumes that ComputeSubmatrixInfo() has already
  // been called.
  // Note: it returns true if the index 'old_row_index' into submatrix
  // indexed 'submat_index' corresponds to an Index with n=0; otherwise
  // it returns false and does not set the output values.
  // If it returns true, it will set '*new_row_index' to be the row-index
  // into the new submatrix, that corresponds to the same Cindex that
  // 'old_row_index' points to in the old computation; and it will set
  // '*n_stride' to the n stride of the corresponding matrix (this is the
  // same in the old and new computations).
  bool GetNewSubmatLocationInfo(int32 submat_index,
                                int32 old_row_index,
                                int32 *new_row_index,
                                int32 *n_stride) const;


  /// This function is used in mapping row-indexes into matrices, from the
  /// old to the new computation.
  ///    @param [in] matrix_index  The matrix-index > 0, for which we
  ///                              are mapping row-indexes.  The
  ///                              matrix-indexes are the same in the old
  ///                              and new computations.
  ///    @param [in] old_row_index   The old row-index into the matrix.
  ///    @return                This function returns the row-index where the
  ///                           cindex referred to in 'old_matrix_index' will
  ///                           reside in the new, expanded computation, WITH
  ///                           THE CAVEAT THAT if the old cindex had n == 1,
  ///                           we'll output the location of the cindex with n
  ///                           == num_n_values_ - 1.  This happens to be what
  ///                           we want (it maps the last n value on the input
  ///                           to the last n value on the output.
  int32 GetNewMatrixLocationInfo(int32 old_matrix_index,
                                 int32 old_row_index) const;


  // This function 'expands' a set of indexes; it's called from
  // ComputePrecomputedIndexes().  The indexes are expected to
  // have the normal kind of regularity.
  void ExpandIndexes(const std::vector<Index> &indexes,
                     std::vector<Index> *indexes_expanded) const;



  // This 'n_stride_' vector is indexed by the matrix-index in the computation,
  // i.e. the same value that you would use to index computation_.matrix_info and
  // expanded_computation_->matrix_info.  For each matrix-index m > 0 it
  // contains the stride of the 'n' values in the Indexes.  This is worked out
  // from the debug_info of the input computation.  For example, if
  // the n stride is 3, and we have an Index (n, t, x) = (0, 50, 88) at the
  // 11'th row of the matrix, then we would expect to have an Index
  // (n, t, x) = (1, 50, 88) at the 11 + 3 = 14'th row of the matrix.
  // The input and output computations will always have the same n-stride, so
  // there is only one variable.
  //
  // Let's define num-n-in = 2, and num-n-out = num_n_values_, and suppose
  // we're dealing with a matrix that has an n stride of "n-stride".
  // We expect the (input, output) matrices to be arranged in blocks of num-rows
  // (n-stride * num-n-in), (n-stride * num-n-out) respectively, where
  // the n-stride never crosses the block boundaries.  We check this.
  std::vector<int32> n_stride_;

  const Nnet &nnet_;
  const MiscComputationInfo &misc_info_;
  const NnetComputation &computation_;
  bool need_debug_info_;
  int32 num_n_values_;
  NnetComputation *expanded_computation_;
};



void ComputationExpander::ExpandRowsCommand(
    const NnetComputation::Command &c_in,
    NnetComputation::Command *c_out) {
  // we need to expand the row-indexes in c_in.arg3, and put the index of the
  // resulting vector<int> in expanded_computation_->indexes, in 'c_out->arg3'.

  int32 s1 = c_in.arg1, s2 = c_in.arg2;

  // The command that gets called is something like
  // submat1.AddRows(submat2, indexes) if submat1 is the submatrix referred to in
  // 's1' and submat2 is the submatrix referred to in 's2'.
  // 'indexes' has the same size as the num-rows of submat1, and the values
  // in the vector are row-indexes into s2.
  int32 old_arg3 = c_out->arg3;
  c_out->arg3 = expanded_computation_->indexes.size();
  c_out->alpha = c_in.alpha;
  expanded_computation_->indexes.push_back(std::vector<int32>());
  std::vector<int32> &new_indexes = expanded_computation_->indexes.back();
  const std::vector<int32> &old_indexes = computation_.indexes[old_arg3];

  int32 old_size = old_indexes.size(),
      num_n_values = num_n_values_,
      new_s1_size = expanded_computation_->submatrices[s1].num_rows,
      new_s2_size = expanded_computation_->submatrices[s2].num_rows;

  KALDI_ASSERT(old_size == computation_.submatrices[s1].num_rows);

  new_indexes.resize(new_s1_size, -1);


  // A note on the variable names: i1 and i2 are indexes into the destination
  // submatrix and the source submatrix respectively, of the CopyRows or AddRows
  // command.
  // "n0" in the variable name means that this corresponds to an Index with n==0.
  // things without "new" in the name refer to the old computation; things with
  // "new" in the name refer to the computation that we are generating.
  for (int32 i1 = 0; i1 < old_size; i1++) {
    int32 new_i1_n0, n_stride1;
    if (GetNewSubmatLocationInfo(s1, i1, &new_i1_n0, &n_stride1)) {
      // GetNewSubmatLocationInfo() returns true if this corresponds to
      // a Cindex with n == 0.
      int32 i2 = old_indexes[i1];  // note: i2 is the row index into submatrix s2.
      int32 new_i2_n0, n_stride2;
      if (i2 < 0) {  // if i2 is -1, we'll just leave any relevant positions in
                     // 'new_indexes' with -1's in them.
        continue;
      } else {
        bool ans = GetNewSubmatLocationInfo(s2, i2, &new_i2_n0, &n_stride2);
        KALDI_ASSERT(ans);  // source should also be for n==0, because we don't
                            // (or at least shouldn't) create computations that
                            // mix up the 'n' values

        int32 new_i1 = new_i1_n0, new_i2 = new_i2_n0;
        for (int32 n = 0; n < num_n_values;
             ++n, new_i1 += n_stride1, new_i2 += n_stride2) {
          KALDI_ASSERT(new_i1 < new_s1_size && new_i2 < new_s2_size);
          new_indexes[new_i1] = new_i2;
        }
      }
    }
  }
}

void ComputationExpander::ExpandRowsMultiCommand(
    const NnetComputation::Command &c_in,
    NnetComputation::Command *c_out) {
  // we need to expand the (submatrix,row)-index pairs in c_in.arg2, and put the
  // index of the resulting vector<int> in expanded_computation_->indexes_multi,
  // in 'c_out->arg2'.

  int32 s1 = c_in.arg1,
      num_rows_old = computation_.submatrices[s1].num_rows,
      num_rows_new = expanded_computation_->submatrices[s1].num_rows;

  KALDI_ASSERT(num_rows_old % 2 == 0);
  int32 num_n_values = num_n_values_;

  int32 old_arg2 = c_out->arg2;
  c_out->arg2 = expanded_computation_->indexes_multi.size();
  expanded_computation_->indexes_multi.push_back(
      std::vector<std::pair<int32, int32> >());
  std::vector<std::pair<int32, int32> > &new_indexes_multi =
      expanded_computation_->indexes_multi.back();
  const std::vector<std::pair<int32, int32> > &old_indexes_multi =
      computation_.indexes_multi[old_arg2];
  // old_indexes_multi is a vector that has the same size as the num-rows
  // of submatrix s1.  It contains pairs that are either (-1, -1), or
  // pairs (submatrix-index, row-index) referring to other submatrices
  // in the computation.

  KALDI_ASSERT(static_cast<int32>(old_indexes_multi.size()) == num_rows_old);


  new_indexes_multi.resize(num_rows_new,
                           std::pair<int32,int32>(-1, -1));

  for (int32 i1 = 0; i1 < num_rows_old; i1++) {
    int32 new_i1_n0, n_stride1;
    if (GetNewSubmatLocationInfo(s1, i1, &new_i1_n0, &n_stride1)) {
      // GetNewSubmatLocationInfo() returns true if this corresponds to
      // a Cindex with n == 0.
      int32 s2 = old_indexes_multi[i1].first,
          i2 = old_indexes_multi[i1].second;
      int32 new_i2_n0, n_stride2;
      if (s2 < 0) {  // if s2 is -1, we don't have to do anything... we'd have
                     // to fill any relevant positions in 'new_indexes_multi'
                     // with (-1,-1)'s, but it's filled with that by default.
        continue;
      } else {
        bool ans = GetNewSubmatLocationInfo(s2, i2, &new_i2_n0, &n_stride2);
        KALDI_ASSERT(ans);  // source should also be for n==0, because we don't
                            // (or at least shouldn't) create computations that
                            // mix up the 'n' values

        int32 new_i1 = new_i1_n0, new_i2 = new_i2_n0;

        for (int32 n = 0; n < num_n_values;
             n++, new_i1 += n_stride1, new_i2 += n_stride2) {
          new_indexes_multi[new_i1].first = s2;
          new_indexes_multi[new_i1].second = new_i2;
        }
      }
    }
  }
}



void ComputationExpander::ExpandRowRangesCommand(
    const NnetComputation::Command &c_in,
    NnetComputation::Command *c_out) {
  // we need to expand the pairs of row-indexes in c_in.arg2, and put the index
  // of the resulting vector<int> in expanded_computation_->indexes_ranges, in
  // 'c_out->arg2'.

  int32 s1 = c_in.arg1, s2 = c_in.arg2,
      num_rows_old = computation_.submatrices[s1].num_rows,
      num_rows_new = expanded_computation_->submatrices[s1].num_rows;
  KALDI_ASSERT(static_cast<size_t>(c_in.arg3) <
               computation_.indexes_ranges.size());
  int32 num_n_values = num_n_values_;

  int32 old_arg3 = c_out->arg3;
  c_out->arg3 = expanded_computation_->indexes_ranges.size();
  expanded_computation_->indexes_ranges.push_back(
      std::vector<std::pair<int32, int32> >());
  std::vector<std::pair<int32, int32> > &new_indexes_ranges =
      expanded_computation_->indexes_ranges.back();
  const std::vector<std::pair<int32, int32> > &old_indexes_ranges =
      computation_.indexes_ranges[old_arg3];
  // old_indexes_ranges is a vector that has the same size as the num-rows of
  // submatrix s1.  It contains pairs that are either two copies of the same
  // value (in practice the pair (-1, -1)), or pairs (begin-row-index,
  // end-row-index) representing the (begin,end) of a range in submatrix s2.
  // Note: end-row-index is one past the end of the range, as for C++ iterators.

  KALDI_ASSERT(static_cast<int32>(old_indexes_ranges.size()) == num_rows_old);

  new_indexes_ranges.resize(num_rows_new,
                           std::pair<int32,int32>(-1, -1));

  for (int32 i1 = 0; i1 < num_rows_old; i1++) {
    int32 new_i1_n0, n_stride1;
    if (GetNewSubmatLocationInfo(s1, i1, &new_i1_n0, &n_stride1)) {
      // GetNewSubmatLocationInfo() returns true if this corresponds to
      // a Cindex with n == 0.
      int32 i2_begin = old_indexes_ranges[i1].first,
          i2_end = old_indexes_ranges[i1].second;
      if (i2_end == i2_begin)
        continue;  // (-1, -1) pair, meaning an empty range.
                   // 'new_indexes_ranges' is filled with (-1, -1) pairs as a
                   // default so we don't have to do anything for these
                   // elements.
      int32 i2_last = i2_end - 1;
      int32 new_i2_n0_begin, new_i2_n0_last,
          n_stride2;  // only 1 stride variable; both calls will output
                          // the same value.

      bool ans1 = GetNewSubmatLocationInfo(s2, i2_begin, &new_i2_n0_begin,
                                           &n_stride2),
          ans2 = GetNewSubmatLocationInfo(s2, i2_last, &new_i2_n0_last,
                                          &n_stride2);
      KALDI_ASSERT(ans1 && ans2 && new_i2_n0_last >= new_i2_n0_begin &&
                   new_i2_n0_begin >= 0 && n_stride1 > 0 && n_stride2 > 0);
      // source should also be for n==0, because we don't (or at least
      // shouldn't) create computations that mix up the 'n' values


      int32 new_i1 = new_i1_n0,
          new_i2_begin = new_i2_n0_begin,
          new_i2_end = new_i2_n0_last + 1;
      for (int32 n = 0; n < num_n_values;
           n++, new_i1 += n_stride1, new_i2_begin += n_stride2,
               new_i2_end += n_stride2) {
        new_indexes_ranges[new_i1].first = new_i2_begin;
        new_indexes_ranges[new_i1].second = new_i2_end;
      }
    }
  }
}



void ComputationExpander::ComputeCommands() {
  int32 num_commands = computation_.commands.size();
  expanded_computation_->commands.resize(num_commands);
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    const NnetComputation::Command &c = computation_.commands[command_index];
    NnetComputation::Command &c_out =
        expanded_computation_->commands[command_index];
    c_out = c;
    // Commands that only operate on submatrices, components and
    // precomputed-indexes do not have to be changed because we'll take care of
    // the expansion by suitably redefining the matrices and submatrices, and
    // recreating the precomputed-indexes.
    // However, commands that require, 'indexes', 'indexes_multi' or
    // 'indexes_ranges' do need to be modified.
    switch (c.command_type) {
      case kAllocMatrix:
      case kDeallocMatrix:
      case kSetConst:
      case kSwapMatrix:
      case kPropagate: case kBackprop:
      case kBackpropNoModelUpdate: case kMatrixCopy: case kMatrixAdd:
        break;
      case kCopyRows: case kAddRows:
        ExpandRowsCommand(c, &c_out);
        break;
      case kCopyRowsMulti: case kAddRowsMulti:
      case kCopyToRowsMulti: case kAddToRowsMulti:
        ExpandRowsMultiCommand(c, &c_out);
        break;
      case kAddRowRanges:
        ExpandRowRangesCommand(c, &c_out);
        break;
      case kCompressMatrix: case kDecompressMatrix:
      case kAcceptInput: case kProvideOutput: case kNoOperation:
      case kNoOperationPermanent: case kNoOperationMarker:
      case kNoOperationLabel: case kGotoLabel:
        break;
      default:
        KALDI_ERR << "Un-handled command type";
    }
  }
}




void ComputationExpander::InitStrideInfo() {
  // note: the zeroth matrix is not a real matrix, it's the empty matrix.
  int32 num_matrices = computation_.matrices.size();
  n_stride_.resize(num_matrices);
  n_stride_[0] = 0;

  // the input computation to class ComputationExpander is required to
  // have its debug info set up.
  KALDI_ASSERT(!computation_.matrix_debug_info.empty());
  for (int32 m = 1; m < num_matrices; m++) {
    int32 num_rows = computation_.matrices[m].num_rows;
    const NnetComputation::MatrixDebugInfo &debug_info = computation_.matrix_debug_info[m];
    KALDI_ASSERT(debug_info.cindexes.size() == num_rows);
    bool full_check = true;  // TODO: eventually change this to false.
    int32 n_stride = FindNStride(debug_info.cindexes, full_check);
    if (n_stride == 0) {
      KALDI_ERR << "Problem encountered in 'shortcut' compilation: the computation "
                << "does not have the expected structure.  Try compiling with "
                << "--use-shortcut=false.";
    }
    n_stride_[m] = n_stride;
  }
}


void ComputationExpander::Expand() {
  InitStrideInfo();
  ComputeMatrixInfo();
  if (need_debug_info_)
    ComputeDebugInfo();
  else
    expanded_computation_->matrix_debug_info.clear();
  ComputeSubmatrixInfo();
  ComputePrecomputedIndexes();
  ComputeCommands();

  expanded_computation_->need_model_derivative =
      computation_.need_model_derivative;
}

void ComputationExpander::ComputeMatrixInfo() {
  int32 num_matrices = computation_.matrices.size();
  expanded_computation_->matrices.resize(num_matrices);
  // Matrix zero is a special case; it's the empty matrix.
  expanded_computation_->matrices[0] = computation_.matrices[0];
  int32 old_num_n_values = 2,
      new_num_n_values = num_n_values_;
  for (int32 m = 1; m < num_matrices; m++) {
    expanded_computation_->matrices[m] = computation_.matrices[m];
    expanded_computation_->matrices[m].num_rows =
        (computation_.matrices[m].num_rows / old_num_n_values) *
        new_num_n_values;
  }
}

void ComputationExpander::ComputeDebugInfo() {
  int32 num_matrices = computation_.matrices.size();
  KALDI_ASSERT(computation_.matrix_debug_info.size() == num_matrices);
  expanded_computation_->matrix_debug_info.resize(num_matrices);
  // Matrix zero is a special case; it's the empty matrix.
  expanded_computation_->matrix_debug_info[0] =
      computation_.matrix_debug_info[0];
  int32 num_n_values = num_n_values_;
  for (int32 m = 1; m < num_matrices; m++) {
    const NnetComputation::MatrixDebugInfo &info_in =
        computation_.matrix_debug_info[m];
    NnetComputation::MatrixDebugInfo &info_out =
        expanded_computation_->matrix_debug_info[m];
    info_out.is_deriv = info_in.is_deriv;
    int32 num_rows_in = computation_.matrices[m].num_rows,
        num_rows_out = expanded_computation_->matrices[m].num_rows;
    KALDI_ASSERT(num_rows_in == info_in.cindexes.size());
    info_out.cindexes.resize(num_rows_out);
    const Cindex *cindexes_in = &(info_in.cindexes[0]);
    Cindex *cindexes_out = &(info_out.cindexes[0]);
    for (int32 r = 0; r < num_rows_in; r++) {
      if (info_in.cindexes[r].second.n == 0) {
        int32 new_r = GetNewMatrixLocationInfo(m, r),
            n_stride = n_stride_[m];
        for (int32 n = 0; n < num_n_values; n++) {
          int32 r_out = new_r + n * n_stride;
          cindexes_out[r_out] = cindexes_in[r];
          cindexes_out[r_out].second.n = n;
        }
      }
    }
  }
}

void ComputationExpander::ComputeSubmatrixInfo() {
  int32 num_submatrices = computation_.submatrices.size();
  expanded_computation_->submatrices.resize(num_submatrices);
  // Sub-matrix zero is a special case; it's the empty submatrix.
  expanded_computation_->submatrices[0] = computation_.submatrices[0];
  for (int32 s = 1; s < num_submatrices; s++) {
    const NnetComputation::SubMatrixInfo &info_in = computation_.submatrices[s];
    int32 m = info_in.matrix_index;
    const NnetComputation::MatrixDebugInfo &debug_info_in =
        computation_.matrix_debug_info[m];

    // we may need to change the row_offset and num_rows.
    int32 first_row_in = info_in.row_offset,
        last_row_in = first_row_in + info_in.num_rows - 1;
    if (!(debug_info_in.cindexes[first_row_in].second.n == 0 &&
          debug_info_in.cindexes[last_row_in].second.n == 1)) {
      std::ostringstream computation_ss;
      std::vector<std::string> submat_strings;
      computation_.GetSubmatrixStrings(nnet_, &submat_strings);
      computation_.Print(computation_ss, nnet_);
      KALDI_ERR << "Submatrix s" << s << " = " << submat_strings[s]
                << " has strange dimensions.  Computation is: "
                << computation_ss.str();
    }

    int32 first_row_out = GetNewMatrixLocationInfo(m, first_row_in),
        last_row_out = GetNewMatrixLocationInfo(m, last_row_in),
        new_num_rows = (last_row_out + 1 - first_row_out);

    NnetComputation::SubMatrixInfo &info_out =
        expanded_computation_->submatrices[s];
    info_out.matrix_index = m;
    info_out.row_offset = first_row_out;
    info_out.num_rows = new_num_rows;
    info_out.col_offset = info_in.col_offset;
    info_out.num_cols = info_in.num_cols;
  }
}

void ComputationExpander::ComputePrecomputedIndexes() {
  // for each element of 'component_precomputed_indexes',
  // we will try to work out the command-index of the associated
  // Propagate() command and of the associated Backprop() command,
  // if it exists.
  // We expect that each such element will be associated with
  // exactly one Propagate() command and at most one Backprop() command.
  int32 num_commands = computation_.commands.size(),
    num_precomputed_indexes = computation_.component_precomputed_indexes.size();

  std::vector<bool> need_backprop(num_precomputed_indexes, false);

  std::vector<int32> component_index(num_precomputed_indexes, -1);

  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    const NnetComputation::Command &c = computation_.commands[command_index];

    if (c.command_type == kPropagate && c.arg2 > 0) {
      KALDI_ASSERT(c.arg2 < num_precomputed_indexes);
      component_index[c.arg2] = c.arg1;
    }
    if ((c.command_type == kBackprop ||
         c.command_type == kBackpropNoModelUpdate) && c.arg2 > 0) {
      KALDI_ASSERT(c.arg2 < num_precomputed_indexes);
      need_backprop[c.arg2] = true;
    }
  }

  for (size_t p = 1;
       p < expanded_computation_->component_precomputed_indexes.size();
       ++p)
    delete expanded_computation_->component_precomputed_indexes[p].data;
  expanded_computation_->component_precomputed_indexes.clear();
  expanded_computation_->component_precomputed_indexes.resize(
      num_precomputed_indexes);

  for (int32 p = 1; p < num_precomputed_indexes; ++p) {
    const NnetComputation::PrecomputedIndexesInfo &old_info =
        computation_.component_precomputed_indexes[p];
    NnetComputation::PrecomputedIndexesInfo &new_info =
        expanded_computation_->component_precomputed_indexes[p];
    KALDI_ASSERT(!old_info.input_indexes.empty() &&
                 !old_info.output_indexes.empty() &&
                 "Input/output indexes not present in precomputed info of "
                 "computation to be expanded.");
    // note: we could place these expanded indexes into 'new_info.input_indexes'
    // and 'new_info.output_indexes', but we actually don't need to keep them
    // there, because they are only required to be kept in computations where
    // the n indexes consist of the set (0, 1), and the computation we're
    // creating has more distinct n indexes than that.
    std::vector<Index> input_indexes, output_indexes;
    ExpandIndexes(old_info.input_indexes, &input_indexes);
    ExpandIndexes(old_info.output_indexes, &output_indexes);
    KALDI_ASSERT(component_index[p] >= 0);
    const Component *component = nnet_.GetComponent(component_index[p]);
    ComponentPrecomputedIndexes *expanded_precomputed_indexes =
        component->PrecomputeIndexes(misc_info_, input_indexes,
                                     output_indexes, need_backprop[p]);
    // this object should not be null because it was not NULL the
    // last time we generated it from the same component, for the
    // same computation.
    KALDI_ASSERT(expanded_precomputed_indexes != NULL);
    new_info.data = expanded_precomputed_indexes;
  }
}


bool ComputationExpander::GetNewSubmatLocationInfo(
    int32 submat_index, int32 old_row_index,
    int32 *new_row_index, int32 *n_stride) const {
  int32 matrix_index = computation_.submatrices[submat_index].matrix_index,
   old_row_offset = computation_.submatrices[submat_index].row_offset,
   new_row_offset = expanded_computation_->submatrices[submat_index].row_offset;

  const NnetComputation::MatrixDebugInfo &debug_info_in =
      computation_.matrix_debug_info[matrix_index];
  if (debug_info_in.cindexes[old_row_index + old_row_offset].second.n != 0)
    return false;
  *new_row_index = (GetNewMatrixLocationInfo(matrix_index,
                                             old_row_index + old_row_offset) -
                    new_row_offset);
  *n_stride = n_stride_[matrix_index];
  return true;
}

int32 ComputationExpander::GetNewMatrixLocationInfo(
    int32 matrix_index, int32 old_row_index) const {
  // to understand 'block_size', read the comment for FindNStride().
  int32 n_stride = n_stride_[matrix_index],
      old_num_n_values = 2, new_num_n_values = num_n_values_,
      old_block_size = old_num_n_values * n_stride,
      new_block_size = new_num_n_values * n_stride,
      block_index = old_row_index / old_block_size,
      offset_within_block = old_row_index % old_block_size;

  // within each block, we can show, given our assumptions, that
  // we must first have a sub-block of 'n_stride' values all with
  // n == 0, then another sub-clock of 'n_stride' values all with
  // n == 1, and so on.  [except there is no 'and so on' for the
  // input computation, where we expect the 'n' values to be the
  // set {0, 1}.]
  int32 old_n_value = offset_within_block / n_stride,
      index_within_subblock = offset_within_block % n_stride;
  const std::vector<Cindex> &cindexes =
      computation_.matrix_debug_info[matrix_index].cindexes;
  KALDI_ASSERT(old_n_value == cindexes[old_row_index].second.n &&
               (old_n_value == 0 || old_n_value == 1));
  // Search for CAVEAT in the comment for this function to see what this is
  // about.  Mapping old_n_value == 1 -> new_n_value == new_num_n_values - 1
  // just happens to be useful for the way we use this function... it maps the
  // end of an old submatrix to the end of a new submatrix.
  int32 new_n_value = (old_n_value == 0 ? 0 : new_num_n_values - 1);

  return block_index * new_block_size + index_within_subblock +
      new_n_value * n_stride;
}


void ComputationExpander::ExpandIndexes(
    const std::vector<Index> &indexes,
    std::vector<Index> *indexes_expanded) const {
  bool full_check = false;
  int32 n_stride = FindNStride(indexes, full_check);
  KALDI_ASSERT(n_stride > 0);
  ConvertNumNValues(n_stride, 2, num_n_values_,
                    indexes, indexes_expanded);
}

void ExpandComputation(const Nnet &nnet,
                       const MiscComputationInfo &misc_info,
                       const NnetComputation &computation,
                       bool need_debug_info,
                       int32 num_n_values,
                       NnetComputation *expanded_computation) {
  ComputationExpander expander(nnet, misc_info, computation,
                               need_debug_info, num_n_values,
                               expanded_computation);
  expander.Expand();
}



// This helper function is used in RequestIsDecomposable(); you can work out
// what it does, and why, from the documentation of RequestIsDecomposable() in
// the header.  This function does basically the same thing, except
// at a lower level, for an IoSpecification rather than a ComputationRequest.
static bool IoSpecificationIsDecomposable(const IoSpecification &io_spec,
                                          IoSpecification *mini_io_spec,
                                          int32 *num_n_values_out) {
  mini_io_spec->name = io_spec.name;
  mini_io_spec->has_deriv = io_spec.has_deriv;
  const std::vector<Index> &indexes = io_spec.indexes;
  KALDI_ASSERT(!indexes.empty() && "Empty Indexes in computation request");

  bool full_check = true;  // We might eventually change this to false, for
                           // efficiency.
  int32 num_n_values = indexes.back().n + 1;
  if (num_n_values <= 2) {
    // Computations with 2 or fewer 'n' values are not decomposable, as there
    // would be no speed benefit in shortcut compilation (which relies on
    // compiling an otherwise similar computation with n == 2).
    return false;
  }
  *num_n_values_out = num_n_values;

  int32 n_stride = FindNStride(indexes, full_check);

  if (n_stride == 0)
    return false;

  ConvertNumNValues(n_stride, num_n_values, 2,
                    indexes, &(mini_io_spec->indexes));

  return true;
}

bool RequestIsDecomposable(const ComputationRequest &request,
                           ComputationRequest *mini_request,
                           int32 *num_n_values) {
  size_t num_inputs = request.inputs.size(),
      num_outputs = request.outputs.size();
  mini_request->inputs.resize(num_inputs);
  mini_request->outputs.resize(num_outputs);
  mini_request->need_model_derivative = request.need_model_derivative;
  mini_request->store_component_stats = request.store_component_stats;
  mini_request->misc_info = request.misc_info;

  KALDI_ASSERT(num_inputs != 0 && num_outputs != 0);
  for (size_t i = 0; i < num_inputs; i++) {
    int32 this_num_n_values = 0;
    if (!IoSpecificationIsDecomposable(request.inputs[i],
                                       &(mini_request->inputs[i]),
                                       &this_num_n_values))
      return false;
    if (i == 0) {
      *num_n_values = this_num_n_values;
    } else {
      if (this_num_n_values != *num_n_values)
        return false;  // .. which would be odd.
    }
  }
  for (size_t i = 0; i < num_outputs; i++) {
    int32 this_num_n_values = 0;
    if (!IoSpecificationIsDecomposable(request.outputs[i],
                                       &(mini_request->outputs[i]),
                                       &this_num_n_values))
      return false;
    if (this_num_n_values != *num_n_values)
      return false;  // .. which would be odd.
  }
  return true;
}


class ComputationLoopedOptimizer {
 public:
  ComputationLoopedOptimizer(const Nnet &nnet,
                             NnetComputation *computation):
      nnet_(nnet), computation_(computation) { }
  bool Optimize();

 private:

  // Figures out the time shift between the successive computation requests.
  static int32 FindTimeShift(const NnetComputation &computation);

  // This function creates a mapping from a matrix-index > 0,
  // to a pair (unique_id, time_offset) that represents the debug-info
  // for that matrix-id in computation.debug_info (these terms are explained
  // below).
  //
  // The output vector 'matrix_to_pair' is indexed by the matrix-index in the
  // computation (the zeroth member is not valid).
  //
  // The 'time_offset' is equal to the 't' value of the first member of the
  // cindexes vector for with t != kNoTime.  The 'unique_id' is an integer that
  // uniquely identifies what we get from subtracting the 'time_offset' from
  // each 't' value of that 'cindexes' vector for which t != kNoTime, and then
  // pairing it up with the 'is_deriv' value of the DebugInfo.  That is, if two
  // 'cindexes' vectors differ only by a time offset, and the 'is_deriv' values
  // are the same they will map to the same unique_id.
  static void CreateMatrixPairs(const NnetComputation &computation,
                                std::vector<std::pair<int32, int32> > *matrix_to_pair);

  // This helper function, used in CreateMatrixPairs, find the value 't' which
  // is the first (*cindexes)[i].second.t that is not kNoTime; it then subtracts
  // that 't' value from all (*cindexes)[i].second.t that are not kNoTime.  If
  // all the 't' values are kNoTime, which we don't expect to happen, we throw
  // an error.
  static inline int32 NormalizeCindexes(std::vector<Cindex> *cindexes);


  // This very simple helper function reverses the map 'matrix_to_pair' so we can
  // do the reverse lookup.  It outputs a map from pair to matrix index m, where
  // 1 <= m < matrix_to_pair.size().
  static void GetPairToMatrixMap(
      std::vector<std::pair<int32, int32> > &matrix_to_pair,
      unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > *pair_to_matrix);


  // Given a vector of lists, one list for each segment, of the active matrices
  // at the end of that segment, this function converts those lists into a
  // different representation where each matrix is represented as a pair instead
  // of as a single int32.  'active_pairs' will have the same dimensions as
  // 'active_matrices'.
  static void ConvertListsToPairLists(
      const std::vector<std::vector<int32> > &active_matrices,
      const std::vector<std::pair<int32, int32> > &matrix_to_pair,
      std::vector<std::vector<std::pair<int32, int32> > > *active_pairs);

  // This function, used in FindFirstRepeat, tells us whether the two lists a
  // and b are the same except for a possible time-shift.
  // Each element of a or b is of the form (matrix-unique-index, time-offset).
  // Let's suppose we have two pairs p1=(m1, o1) and p2=(m2, o2).
  // For p2 to be equal to p1 except for a possible shift of value 'shift', we
  // require m2 == m1 and either o2 == o1 + 'shift' or o2 == o1.
  // This function returns true if a.size() == b.size() and for each
  // i, b[i].first == a[i].first and b[i].second is either
  // a[i].second or a[i].second + shift.
  static bool ListsAreEqualExceptForPossibleShift(
      const std::vector<std::pair<int32, int32> > &a,
      const std::vector<std::pair<int32, int32> > &b,
      int32 shift);

  // This function looks in the matrix 'active_pairs' for the first pair of
  // identical values, i.e. it is looking for i < j for which
  // normalized_active_pairs[i] == normalized_active_pairs[j].  (However, the
  // pair i,j must satisfy an extra condition, see below).  If a pair
  // i,j exists satisfying these conditions, this function outputs them to *seg1
  // and *seg2, and returns true; otherwise it returns false.
  //
  // Extra condition:
  // It turns out that under some circumstances, we can
  // fine repeats that were not "really" repeats (the matrices were not time
  // shifted) The situation was a bit obscure (it was a non-recurrent setup with
  // a lot of extra-right-context, where some inputs were never used), but to
  // prevent it happening again we are now checking in addition to the above,
  // that the time-shift between the segments (i.e. time_offsets[j] -
  // time_offsets[i]), has the "expected value" based on the assumption that
  // each segment should be shifted relative to the previous segment, by
  // 'time_shift_per_segment'.
  static bool FindFirstRepeat(
      const std::vector<std::vector<std::pair<int32, int32> > > &active_pairs,
      int32 time_shift_per_segment,
      int32 *seg1, int32 *seg2);


  // 'pair_list1' is the list of active (unique-id, time-offset) pairs for one
  // segment of the computation and 'pair_list2' is the same list for a later
  // segment.  The map 'pair_to_matrix' can convert these back into matrix
  // indexes.  This function will output two lists of matrices.  These will just
  // be 'pair_list1' and 'pair_list2' converted back into matrix indexes,
  // except we omit pairs which are identical (i.e. the time-offset was zero).
  static void GetIdentifiedMatrices(
      const std::vector<std::pair<int32, int32> > &pair_list1,
      const std::vector<std::pair<int32, int32> > &pair_list2,
      const unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > &pair_to_matrix,
      std::vector<int32> *matrix_list1,
      std::vector<int32> *matrix_list2);


  // This function just does some checking (via asserts), that
  // the lists of matrices 'list1' and 'list2' are of the same length,
  // that time_difference > 0, that each matrix with index m = list2[i] is of the
  // same dimension as the list1[i], with Cindexes that are the same except for
  // the time index being greater by 'time_difference'
  static void CheckIdentifiedMatrices(
      const NnetComputation &computation,
      const std::vector<int32> &list1,
      const std::vector<int32> &list2,
      int32 time_difference);


  // Given two command indexes command1 < command2 pointing to commands of type
  // kNoOperationMarker, this function modifies the computation by
  // removing all commands after command2, replacing command2 with a kGotoLabel
  // command pointing to command1  and then inserting just before command1
  // a marker of type kNoOperationLabel.
  static void FormInfiniteLoop(int32 command1, int32 command2,
                               NnetComputation *computation);

  // This is to be called after FormInfiniteLoop.  It inserts, just before
  // the final kGotoLabel command, commands that initialize
  // each of the matrices in list 'matrices1' from the corresponding
  // matrix in 'matrices2', using the kAllocMatrixFromOther command.
  // This effectively does, for example, matrices1[i] = matrices2[i],
  // and it's implemented as a shallow swap.
  // It does this in such an order that even if the two lists are
  // not disjoint, the right thing happens.
  static void AddMatrixSwapCommands(
      const std::vector<int32> &matrices1,
      const std::vector<int32> &matrices2,
      NnetComputation *computation);


  // Called from AddMatrixSwapCommands, this function figures out for us
  // an acceptable order in which to execute the kAllocMatrixFromOther
  // commands.  This is easy to do if matrices1 and matrices2 are disjoint
  // sets, but has to be done more carefully if they overlap.
  // The output is a list of pairs where each pair (a, b) comes from
  // from matrices1 and matrices2 in the same position, i.e.
  // a = matrices1[i] and b = matrices2[i].
  static void GetMatrixSwapOrder(
      const std::vector<int32> &matrices1,
      const std::vector<int32> &matrices2,
      std::vector<std::pair<int32, int32> > *swaps);



  /// Given a list of command indexes ('splice_point_commands') which are
  /// expected to be command indexes of the kNoOperationMarker at segment
  /// boundaries, this function outputs for each of these command indexes a list
  /// of matrices which are 'active' at that point in time.  By 'active' we mean
  /// that the matrix has been written to before that time (including zeroing),
  /// and will be read after that time.  These is the list of matrices that
  /// 'need to be in scope' at those points in time.  '*active_matrices' is
  /// indexed by the same index as 'splice_point_commands', and is then a list
  /// of active matrices, in numerical order of matrix index.  Note: for each i,
  /// (*active_matrices)[i] will be sorted and unique.
  static void FindActiveMatrices(const NnetComputation &computation,
                                 const Analyzer &analyzer,
                                 const std::vector<int32> &splice_point_commands,
                                 std::vector<std::vector<int32> > *active_matrices);


  const Nnet &nnet_;
  NnetComputation *computation_;
  Analyzer analyzer_;
  std::vector<std::pair<int32, int32> > matrix_to_pair_;

  std::vector<int32> splice_point_commands_;
};

// static
int32 ComputationLoopedOptimizer::FindTimeShift(
    const NnetComputation &computation) {
  std::vector<int32> segment_ends;
  GetCommandsOfType(computation, kNoOperationMarker, &segment_ends);
  KALDI_ASSERT(segment_ends.size() >= 3);
  // Ignore the first segment as it tends to be a special case
  // (it has more left context),
  int32 second_segment_begin = segment_ends[0],
      third_segment_begin = segment_ends[1],
      fourth_segment_begin = segment_ends[2];
  int32 first_output_command_seg2 = -1,
      first_output_command_seg3 = -1;
  for (int32 c = second_segment_begin; c < third_segment_begin; c++)
    if (computation.commands[c].command_type == kProvideOutput &&
        first_output_command_seg2 < 0)
      first_output_command_seg2 = c;
  for (int32 c = third_segment_begin; c < fourth_segment_begin; c++)
    if (computation.commands[c].command_type == kProvideOutput &&
        first_output_command_seg3 < 0)
      first_output_command_seg3 = c;
  if (first_output_command_seg2 < 0 ||
      first_output_command_seg3 < 0)
    KALDI_ERR << "Could not locate output commands for segments 2 and 3.";
  const NnetComputation::Command
      &command2 = computation.commands[first_output_command_seg2],
      &command3 = computation.commands[first_output_command_seg3];
  int32 seg2_node = command2.arg2, seg3_node = command3.arg2;
  KALDI_ASSERT(seg2_node == seg3_node);
  int32 seg2_submatrix = command2.arg1,
      seg3_submatrix = command3.arg1;
  KALDI_ASSERT(computation.IsWholeMatrix(seg2_submatrix) &&
               computation.IsWholeMatrix(seg3_submatrix));
  int32 seg2_matrix = computation.submatrices[seg2_submatrix].matrix_index,
      seg3_matrix = computation.submatrices[seg3_submatrix].matrix_index;
  KALDI_ASSERT(computation.matrices[seg2_matrix].num_rows ==
               computation.matrices[seg3_matrix].num_rows);
  KALDI_ASSERT(!computation.matrix_debug_info.empty());
  const NnetComputation::MatrixDebugInfo
      &debug_info2 = computation.matrix_debug_info[seg2_matrix],
      &debug_info3 = computation.matrix_debug_info[seg3_matrix];
  int32 t_offset = debug_info3.cindexes[0].second.t -
      debug_info2.cindexes[0].second.t;
  int32 num_rows = debug_info2.cindexes.size();
  for (int32 r = 0; r < num_rows; r++) {
    KALDI_ASSERT(debug_info3.cindexes[r].second.t ==
                 debug_info2.cindexes[r].second.t + t_offset);
  }
  return t_offset;
}

// static inline
int32 ComputationLoopedOptimizer::NormalizeCindexes(
    std::vector<Cindex> *cindexes) {
  std::vector<Cindex>::iterator iter = cindexes->begin(),
      end = cindexes->end();
  int32 ans;
  for (; iter != end; iter++) {
    if (iter->second.t != kNoTime) {
      ans = iter->second.t;
      break;
    }
  }
  if (iter == end) {
    // this should not happen.
    KALDI_ERR << "All t values are kNoTime in matrix.";
  }
  iter = cindexes->begin();
  for (; iter != end; iter++)
    if (iter->second.t != kNoTime)
      iter->second.t -= ans;
  return ans;
}

// static
void ComputationLoopedOptimizer::CreateMatrixPairs(
    const NnetComputation &computation,
    std::vector<std::pair<int32, int32> > *matrix_to_pair) {
  typedef unordered_map<std::vector<Cindex>, int32,
                        CindexVectorHasher> MapType;
  int32 cur_vector_id = 1;
  // Note: cindex_map just maps the vector<Cindex> to a unique value,
  // and then we manually work out a unique id that takes into
  // account the 'is_deriv' values.
  MapType cindex_map;
  int32 num_matrices = computation.matrices.size();
  matrix_to_pair->resize(num_matrices);
  KALDI_ASSERT(computation.matrix_debug_info.size() == num_matrices);
  for (int32 m = 1; m < num_matrices; m++) {
    KALDI_ASSERT(!computation.matrix_debug_info[m].cindexes.empty());
    std::vector<Cindex> cindexes = computation.matrix_debug_info[m].cindexes;
    int32 t_offset = NormalizeCindexes(&cindexes);
    MapType::const_iterator iter = cindex_map.find(cindexes);
    int32 vector_id;
    if (iter != cindex_map.end()) {
      vector_id = iter->second;
    } else {
      vector_id = cur_vector_id++;
      cindex_map[cindexes] = vector_id;
    }
    bool is_deriv = computation.matrix_debug_info[m].is_deriv;
    int32 unique_id = 2 * vector_id + (is_deriv ? 1 : 0);
    (*matrix_to_pair)[m].first = unique_id;
    (*matrix_to_pair)[m].second = t_offset;
  }
}

// static
void ComputationLoopedOptimizer::GetPairToMatrixMap(
      std::vector<std::pair<int32, int32> > &matrix_to_pair,
      unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > *pair_to_matrix) {
  int32 num_matrices = matrix_to_pair.size();
  // actually there are one fewer matrices than num_matrices.
  pair_to_matrix->clear();
  for (int32 m = 1; m < num_matrices; m++)
    (*pair_to_matrix)[matrix_to_pair[m]] = m;
}


// static
void ComputationLoopedOptimizer::ConvertListsToPairLists(
      const std::vector<std::vector<int32> > &active_matrices,
      const std::vector<std::pair<int32, int32> > &matrix_to_pair,
      std::vector<std::vector<std::pair<int32, int32> > > *active_pairs) {
  active_pairs->clear();
  active_pairs->resize(active_matrices.size());
  int32 num_matrices = matrix_to_pair.size();
  for (size_t seg = 0; seg < active_matrices.size(); seg++) {
    const std::vector<int32> &this_active_matrix_list = active_matrices[seg];
    std::vector<std::pair<int32, int32> > &this_active_pair_list =
        (*active_pairs)[seg];
    this_active_pair_list.resize(this_active_matrix_list.size());
    std::vector<int32>::const_iterator iter = this_active_matrix_list.begin(),
        end = this_active_matrix_list.end();
    std::vector<std::pair<int32, int32> >::iterator
        out_iter = this_active_pair_list.begin();
    for (; iter != end; ++iter, ++out_iter) {
      KALDI_ASSERT(*iter > 0 && *iter < num_matrices);
      *out_iter = matrix_to_pair[*iter];
    }
  }
}

// static
bool ComputationLoopedOptimizer::ListsAreEqualExceptForPossibleShift(
    const std::vector<std::pair<int32, int32> > &a,
    const std::vector<std::pair<int32, int32> > &b,
    int32 shift) {
  size_t size = a.size();
  if (b.size() != size)
    return false;
  for (size_t i = 0; i < size; i++) {
    const std::pair<int32, int32> &p1 = a[i],
        &p2 = b[i];
    if (p1.first != p2.first)
      return false;
    if (p2.second != p1.second + shift && p2.second != p1.second)
      return false;
  }
  return true;
}

// static
bool ComputationLoopedOptimizer::FindFirstRepeat(
    const std::vector<std::vector<std::pair<int32, int32> > > &active_pairs,
    int32 time_shift_per_segment,
    int32 *seg1, int32 *seg2) {
  int32 num_segments = active_pairs.size();
  // This algorithm may seem like it would be very slow, but the number of
  // segments will normally be quite small (e.g. 10), and the comparison of
  // elements of 'active_pairs' should be fast in cases where they
  // differ.
  KALDI_ASSERT(num_segments >= 2);

  for (int32 s = 0; s < num_segments; s++) {
    for (int32 t = s + 1; t < num_segments; t++) {
      if (ListsAreEqualExceptForPossibleShift(active_pairs[s],
                                              active_pairs[t],
                                              (t - s) * time_shift_per_segment)) {
        *seg1 = s;
        *seg2 = t;
        return true;
      }
    }
  }
  return false;
}

// static
void ComputationLoopedOptimizer::GetIdentifiedMatrices(
    const std::vector<std::pair<int32, int32> > &pair_list1,
    const std::vector<std::pair<int32, int32> > &pair_list2,
    const unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > &pair_to_matrix,
    std::vector<int32> *matrix_list1,
    std::vector<int32> *matrix_list2) {
  size_t size = pair_list1.size();
  KALDI_ASSERT(pair_list2.size() == size);
  matrix_list1->clear();
  matrix_list2->clear();
  matrix_list1->reserve(size);
  matrix_list2->reserve(size);
  std::vector<std::pair<int32, int32> >::const_iterator
      iter1 = pair_list1.begin(), end1 = pair_list1.end(),
      iter2 = pair_list2.begin();
  for (; iter1 != end1; ++iter1, ++iter2) {
    if (iter1->second == iter2->second)
      continue;
    // skip those that have no time shift, we won't have to do any swapping for
    // those.
    unordered_map<std::pair<int32, int32>, int32,
                  PairHasher<int32> >::const_iterator
        map_iter1 = pair_to_matrix.find(*iter1),
        map_iter2 = pair_to_matrix.find(*iter2);
    if (map_iter1 == pair_to_matrix.end() ||
        map_iter2 == pair_to_matrix.end())
      KALDI_ERR << "Could not find pair in map (code error)";
    matrix_list1->push_back(map_iter1->second);
    matrix_list2->push_back(map_iter2->second);
  }
}



// static
void ComputationLoopedOptimizer::FindActiveMatrices(
    const NnetComputation &computation,
    const Analyzer &analyzer,
    const std::vector<int32> &splice_point_commands,
    std::vector<std::vector<int32> > *active_matrices) {
  int32 num_matrices = computation.matrices.size();
  int32 num_splice_points = splice_point_commands.size();
  active_matrices->clear();
  active_matrices->resize(num_splice_points);
  // this object just makes available some extra functions, vs. the Analyzer
  // object.
  ComputationAnalysis analysis(computation, analyzer);
  KALDI_ASSERT(IsSortedAndUniq(splice_point_commands));

  // the following vector gives us, for each matrix index, a submatrix index
  // that covers the whole of that matrix (needed by interface of 'analysis' object).
  std::vector<int32> whole_submatrices;
  computation.GetWholeSubmatrices(&whole_submatrices);
  for (int32 m = 1; m < num_matrices; m++) {
    // the following are command indexes, comparable with the indexes
    // in 'splice_point_commands'.
    int32 s = whole_submatrices[m],  // submatrix consisting of the whole of
                                     // 'm'.
        first_access = analysis.FirstNontrivialAccess(s),
        last_access = analysis.LastAccess(s);
    for (int32 i = 0; i < num_splice_points; i++) {
      int32 splice_point = splice_point_commands[i];
      if (first_access < splice_point && last_access > splice_point) {
        // If the block of time during which the matrix is accessed, includes
        // this command index, then the matrix is considered 'active' at this
        // time.
        (*active_matrices)[i].push_back(m);
      }
    }
  }
}

// static
void ComputationLoopedOptimizer::CheckIdentifiedMatrices(
    const NnetComputation &computation,
    const std::vector<int32> &list1,
    const std::vector<int32> &list2,
    int32 time_difference) {
  KALDI_ASSERT(time_difference > 0);
  KALDI_ASSERT(list1.size() == list2.size());
  KALDI_ASSERT(!computation.matrix_debug_info.empty());
  for (size_t i = 0; i < list1.size(); i++) {
    int32 m1 = list1[i], m2 = list2[i];
    const NnetComputation::MatrixInfo
        &matrix_info1 = computation.matrices[m1],
        &matrix_info2 = computation.matrices[m2];
    KALDI_ASSERT(matrix_info1.num_rows == matrix_info2.num_rows &&
                 matrix_info1.num_cols == matrix_info2.num_cols &&
                 matrix_info1.stride_type == matrix_info2.stride_type);
    const NnetComputation::MatrixDebugInfo
        &debug_info1 = computation.matrix_debug_info[m1],
        &debug_info2 = computation.matrix_debug_info[m2];
    KALDI_ASSERT(debug_info1.is_deriv == debug_info2.is_deriv);
    KALDI_ASSERT(debug_info1.cindexes.size() == debug_info2.cindexes.size());
    std::vector<Cindex>::const_iterator iter1 = debug_info1.cindexes.begin(),
        end1 = debug_info1.cindexes.end(),
        iter2 = debug_info2.cindexes.begin();
    for (; iter1 != end1; iter1++,iter2++) {
      KALDI_ASSERT(iter2->first == iter1->first &&
                   iter2->second.n == iter1->second.n &&
                   ((iter1->second.t == kNoTime && iter2->second.t == kNoTime) ||
                    iter2->second.t == iter1->second.t + time_difference) &&
                   iter2->second.x == iter1->second.x);
    }
  }
}


// static
void ComputationLoopedOptimizer::GetMatrixSwapOrder(
    const std::vector<int32> &matrices1,
    const std::vector<int32> &matrices2,
    std::vector<std::pair<int32, int32> > *swaps) {
  KALDI_ASSERT(matrices1.size() == matrices2.size());
  swaps->clear();
  int32 num_matrices = matrices1.size();
  std::vector<bool> processed(num_matrices, false);
  std::vector<int32> queue;

  // num_loops is just for infinite-loop detection.
  int32 num_loops = 0;
  for (; static_cast<int32>(swaps->size()) < num_matrices; num_loops++) {
    for (int32 i = 0; i < num_matrices; i++) {
      if (processed[i])
        continue;
      int32 m1 = matrices1[i], m2 = matrices2[i];
      std::vector<int32>::const_iterator iter =
          std::lower_bound(matrices2.begin(), matrices2.end(), m1);
      if (iter == matrices2.end() || *iter != m1) {
        // Matrix m1 does not appear in the list 'matrices2', so
        // we are safe to process it at any time.
        swaps->push_back(std::pair<int32,int32>(m1, m2));
        processed[i] = true;
      } else {
        int32 m1_pos_in_matrices2 = iter - matrices2.begin();
        if (processed[m1_pos_in_matrices2]) {
          // We're safe to do this swap now, because the matrix m1 has already
          // appeared on the RHS of a swap, and by this point has been
          // deallocated, in effect.
          swaps->push_back(std::pair<int32,int32>(m1, m2));
          processed[i] = true;
        }
        // else do nothing, we cannot process m1 yet because
        // at this point in the computation it is still allocated.
      }
    }
    // The following assert is to check that we don't loop infinitely.  We can
    // prove that infinite looping won't happen, after on proving that there can
    // be no cycles like (m1, m2), (m2, m3), (m3, m1) (the length of 3 is chosen
    // arbitrarily as an example).  If such a cycle existed, we can reach a
    // contradiction based on the time-index (t) of the first cindex in m1.
    // Define t1 = that time index, t2 the same for m2, t3 the same for m3.  The
    // existence of the three pairs [as pairs like (matrices1[i], matrices2[i])]
    // implies that t2 > t1, t3 > t2, and t1 > t3 respectively, but this is
    // impossible.
    // This shows that all chains of dependencies must terminate.
    KALDI_ASSERT(num_loops <= num_matrices);
  }
}

// static
void ComputationLoopedOptimizer::AddMatrixSwapCommands(
    const std::vector<int32> &matrices1,
    const std::vector<int32> &matrices2,
    NnetComputation *computation) {
  std::vector<std::pair<int32, int32> > swaps;
  // Note: in 'easy' cases where matrices1 and matrices2 are disjoint,
  // 'swaps' will just be the vector { (matrices1[0],matrices2[0]),
  // (matrices1[1],matrices2[1]), ... },
  // but in some cases these may need to get reordered.
  GetMatrixSwapOrder(matrices1, matrices2, &swaps);

  NnetComputation::Command goto_label_command = computation->commands.back();
  KALDI_ASSERT(goto_label_command.command_type == kGotoLabel);
  computation->commands.pop_back();

  // the following vector gives us, for each matrix index, a submatrix index
  // that covers the whole of that matrix (needed because the commands
  // require submatrix indexes)
  std::vector<int32> whole_submatrices;
  computation->GetWholeSubmatrices(&whole_submatrices);
  size_t num_matrices = whole_submatrices.size();

  for (size_t i = 0; i < swaps.size(); i++) {
    int32 m1 = swaps[i].first, m2 = swaps[i].second;
    KALDI_ASSERT(static_cast<size_t>(m1) < num_matrices &&
                 static_cast<size_t>(m2) < num_matrices);
    int32 s1 = whole_submatrices[m1], s2 = whole_submatrices[m2];
    computation->commands.push_back(
        NnetComputation::Command(kSwapMatrix, s1, s2));
  }
  computation->commands.push_back(goto_label_command);
}

// static
void ComputationLoopedOptimizer::FormInfiniteLoop(
    int32 command1, int32 command2,
    NnetComputation *computation) {
  KALDI_ASSERT(static_cast<int32>(computation->commands.size()) >=
               command2 + 1 && command1 < command2);
  KALDI_ASSERT(
      computation->commands[command1].command_type == kNoOperationPermanent &&
      computation->commands[command2].command_type == kNoOperationPermanent);
  // Remove any commands after 'command2'.
  computation->commands.resize(command2 + 1);
  computation->commands[command2].command_type = kGotoLabel;
  computation->commands[command2].arg1 = command1;
  NnetComputation::Command c(kNoOperationLabel);
  computation->commands.insert(computation->commands.begin() + command1,
                               c);
  // Now the kNoOperationLabel command is at position 'command1'.
}



bool ComputationLoopedOptimizer::Optimize() {
  analyzer_.Init(nnet_, *computation_);
  KALDI_ASSERT(!computation_->matrix_debug_info.empty() &&
               "You must request matrix debug info when compiling "
               "looped computations.");

  // get the indexes of potential splice points, one per segment of the
  // computation.  We locate the splice points where kNoOperationPermanent is.
  // This is guaranteed to be after the inputs have been received, and before
  // the bulk of the computation in the segment, and of course before we provide
  // the output.  It happens that by choosing this as the splice point we avoid
  // certain problems that would arise, for instance, if we chose the segment
  // boundaries (kNoOperationMarker).
  std::vector<int32> splice_points;
  GetCommandsOfType(*computation_, kNoOperationPermanent,
                    &splice_points);
  int32 time_shift_per_segment = FindTimeShift(*computation_);


  std::vector<std::vector<int32> > active_matrices;
  // Find the list of matrices active at each of the potential splice points.
  FindActiveMatrices(*computation_, analyzer_, splice_points,
                     &active_matrices);

  // Find a representation of the matrices of the computation as pairs
  // (unique_id, time_offset) that are more amenable to finding
  // matrices that represet lists of Cindexes that differ only by
  // a time offset.
  std::vector<std::pair<int32, int32> > matrix_to_pair;
  CreateMatrixPairs(*computation_, &matrix_to_pair);

  // Create the reverse map from pair to matrix index; we'll need it later.
  unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > pair_to_matrix;
  GetPairToMatrixMap(matrix_to_pair, &pair_to_matrix);

  // get lists of matrix per splice-point in the pair representation.
  std::vector<std::vector<std::pair<int32, int32> > > pair_lists;
  ConvertListsToPairLists(active_matrices, matrix_to_pair,
                          &pair_lists);

  // Note: seg1 and seg2 are indexes into 'splice_points', representing
  // potential splice points (located near the beginnings of segments).
  int32 seg1, seg2;
  if (!FindFirstRepeat(pair_lists,
                       time_shift_per_segment,
                       &seg1, &seg2)) {
    KALDI_VLOG(2) << "Could not find repeats of variables.";
    return false;
  }

  std::vector<int32> seg1_matrices, seg2_matrices;
  GetIdentifiedMatrices(pair_lists[seg1], pair_lists[seg2],
                        pair_to_matrix,
                        &seg1_matrices, &seg2_matrices);

  int32 time_difference = time_shift_per_segment * (seg2 - seg1);
  CheckIdentifiedMatrices(*computation_, seg1_matrices, seg2_matrices,
                          time_difference);

  FormInfiniteLoop(splice_points[seg1], splice_points[seg2], computation_);

  AddMatrixSwapCommands(seg1_matrices, seg2_matrices, computation_);

  RenumberComputation(computation_);

  FixGotoLabel(computation_);

  return true;
}


void OptimizeLoopedComputation(const Nnet &nnet,
                               NnetComputation *computation) {
  ComputationLoopedOptimizer optimizer(nnet, computation);
  optimizer.Optimize();
}



void FixGotoLabel(NnetComputation *computation) {
  int32 num_commands = computation->commands.size();
  if (num_commands == 0)
    return;
  for (int32 c = num_commands - 1; c >= 0; c--) {
    if (computation->commands[c].command_type == kGotoLabel) {
      int32 dest_command = computation->commands[c].arg1;
      if (static_cast<size_t>(dest_command) <  computation->commands.size() &&
          computation->commands[dest_command].command_type == kNoOperationLabel)
        return;  // nothing to fix.
      for (int32 d = 0; d + 1 < num_commands; d++) {
        if (computation->commands[d].command_type == kNoOperationLabel) {
          computation->commands[c].arg1 = d;
          return;
        }
      }
      KALDI_ERR << "Label not found.";
    } else if (computation->commands[c].command_type == kProvideOutput) {
      // sometimes kProvideOutput commands are temporarily ordered after
      // the kGotoLabel command, and we need to work in that case.
      continue;
    } else {
      // it loks like there is no 'goto' command in this computation-
      // if there were, it would be right at the end, possibly followed by
      // kProvideOutput commands.
      break;
    }
  }
}

bool MatrixIsUnused(const Analyzer &analyzer,
                    const NnetComputation &computation,
                    int32 m) {
  const MatrixAccesses &accesses = analyzer.matrix_accesses[m];
  if (accesses.is_input || accesses.is_output)
    return false;
  for (size_t i = 0; i < accesses.accesses.size(); i++) {
    int32 command_index = accesses.accesses[i].command_index;
    const NnetComputation::Command &command =
        computation.commands[command_index];
    if (command.command_type != kNoOperation &&
        command.command_type != kSetConst) {
      return false;
    }
  }
  return true;
}

void RemoveCommandsForUnusedMatrix(const Analyzer &analyzer,
                                   int32 m,
                                   NnetComputation *computation) {
  const MatrixAccesses &accesses = analyzer.matrix_accesses[m];
  if (accesses.allocate_command >= 0) {
    NnetComputation::Command &command = computation->commands[
        accesses.allocate_command];
    KALDI_ASSERT(command.command_type == kNoOperation ||
                 command.command_type == kAllocMatrix);
    command.command_type = kNoOperation;
  }
  if (accesses.deallocate_command >= 0) {
    NnetComputation::Command &command = computation->commands[
        accesses.deallocate_command];
    KALDI_ASSERT(command.command_type == kNoOperation ||
                 command.command_type == kDeallocMatrix);
    command.command_type = kNoOperation;
  }
  for (size_t i = 0; i < accesses.accesses.size(); i++) {
    int32 command_index = accesses.accesses[i].command_index;
    NnetComputation::Command &command = computation->commands[command_index];
    KALDI_ASSERT(command.command_type == kNoOperation ||
                 command.command_type == kSetConst);
    command.command_type = kNoOperation;
  }
}



// This comparison operator is used in the function InsertCommands()
// to sort a list of these pairs by the .first element.
struct CommandPairComparator {
  // operator () should be viewed as a '<' operator that only looks at
  // the .first element, treating the .second elements as equal.
  bool operator () (const std::pair<int32, NnetComputation::Command> &p1,
                    const std::pair<int32, NnetComputation::Command> &p2) const {
    return p1.first < p2.first;
  }
};

void InsertCommands(
    std::vector<std::pair<int32, NnetComputation::Command> > *new_commands,
    NnetComputation *computation) {
  int32 num_new_commands = new_commands->size(),
      num_old_commands = computation->commands.size();
  if (num_new_commands == 0)
    return;
  CommandPairComparator comparison_operator;
  // use std::stable_sort so that for entries in 'new_commands' that
  // have the same .first value, they stay in the same order they were
  // in before sorting.
  std::stable_sort(new_commands->begin(), new_commands->end(),
                   comparison_operator);

  if (RandInt(0, 3) == 0) {   // check 'new_commands'
    for (int32 i = 0; i + 1 < num_new_commands; i++) {
      KALDI_ASSERT((*new_commands)[i].first <= (*new_commands)[i+1].first &&
                   (*new_commands)[i].first >= 0 &&
                   (*new_commands)[i+1].first <= num_old_commands);
    }
  }
  std::vector<NnetComputation::Command> merged_commands;
  merged_commands.reserve(num_old_commands + num_new_commands);

  std::vector<std::pair<int32, NnetComputation::Command> >::const_iterator
      new_commands_iter = new_commands->begin(),
      new_commands_end = new_commands->end();

  for (int32 old_command_index = 0; old_command_index <= num_old_commands;
       old_command_index++) {
    while (new_commands_iter != new_commands_end &&
           new_commands_iter->first <= old_command_index) {
      merged_commands.push_back(new_commands_iter->second);
      ++new_commands_iter;
    }
    if (old_command_index < num_old_commands)
      merged_commands.push_back(computation->commands[old_command_index]);
  }
  KALDI_ASSERT(merged_commands.size() == num_old_commands +
               num_new_commands);
  // copy to 'computation->commands' via shallow swap.
  computation->commands.swap(merged_commands);
  FixGotoLabel(computation);
}

/**
   This class is used in the function OptimizeMemoryCompression(),
   once we determine that there is some potential to do memory compression
   for this computation.
 */
class MemoryCompressionOptimizer {
 public:

  /** @param [in] nnet         The neural net the computation is for.
      @param [in] memory_compression_level.  The level of compression:
         0 = no compression (the constructor should not be calle with this value).
         1 = compression that doesn't affect the results (but still takes time).
         2 = compression that affects the results only very slightly
         3 = compression that affects the results a little more.
      @param [in] middle_command  Must be the command-index of the
          command of type kNoOperationMarker in 'computation'.
      @param [in,out] computation  The computation we're optimizing.
  */
  MemoryCompressionOptimizer(const Nnet &nnet,
                             int32 memory_compression_level,
                             int32 middle_command,
                             NnetComputation *computation):
      nnet_(nnet), memory_compression_level_(memory_compression_level),
      middle_command_(middle_command), computation_(computation) { }

  void Optimize();
 private:

  // This function, called from Compress(), figures out whether we can compress
  // matrix m, and if so, adds an entry to compress_info_.
  void ProcessMatrix(int32 m);

  // This function modifies the commands in '*computation_', taking
  // as input the commands in compress_info_.
  void ModifyComputation();

  // While deciding what matrices to compress we will create a list of structs
  // of type MatrixCompressInfo.  Later we copy-and-modify the commands in the
  // computation, putting the compression commands into their appropriate place.
  struct MatrixCompressInfo {
    // m is the matrix-index of the matrix we're going to compress.
    int32 m;
    // compression_command_index is the command-index of the command
    // *after* which we will place the compression command.  Normally
    // this will be some type of propagation.
    int32 compression_command_index;
    // compression_command_index is the command-index of the command
    // *before* which we will place the uncompression command.  Normally
    // this will be some type of backprop.
    int32 uncompression_command_index;
    // 'compression_type' (e.g. kCompressedMatrixInt8) determines the type
    // we compress the BaseFloats to.
    CuCompressedMatrixType compression_type;
    // 'range' determines range of values that the compressed values can
    // be in: for signed types they are in [-range, range], for unsigned
    // types, in [0, range].
    // As a special case, range = 0 means that the compression just stores the
    // sign (-1, 0 or 1) of the input, and decompresses it to -1, 0 or 1; this
    // is useful for ReLUs.
    BaseFloat range;
    // this is provided to the initializer of CuCompressedMatrix; it should
    // be true if the values being compressed are potentially outside of
    // the representable range.
    bool truncate;
    MatrixCompressInfo(int32 m, int32 forward_command_index,
                       int32 backward_command_index,
                       CuCompressedMatrixType compression_type,
                       BaseFloat range, bool truncate):
        m(m), compression_command_index(forward_command_index),
        uncompression_command_index(backward_command_index),
        compression_type(compression_type), range(range),
        truncate(truncate) { }

  };
  std::vector<MatrixCompressInfo> compress_info_;

  const Nnet &nnet_;
  int32 memory_compression_level_;
  int32 middle_command_;
  NnetComputation *computation_;
  Analyzer analyzer_;
};


void MemoryCompressionOptimizer::ModifyComputation() {
  // whole_submatrices[m] is the submatrix-index of the submatrix that
  // represents the whole of matrix m.
  std::vector<int32> whole_submatrices;
  computation_->GetWholeSubmatrices(&whole_submatrices);

  // 'pairs_to_insert' will be a list of pairs (command-index, command),
  // meaning: (command-index just before which to insert this command; command
  // to insert).
  std::vector<std::pair<int32, NnetComputation::Command> >
      pairs_to_insert;
  pairs_to_insert.reserve(compress_info_.size() * 2);
  for (size_t i = 0; i < compress_info_.size(); i++) {
    const MatrixCompressInfo &info = compress_info_[i];
    int32 s = whole_submatrices[info.m];
    // below we use compression_command_index + 1 because we want the
    // compression to go after the command in 'info.compression_command_index'
    // (which might be, for instance, a forward propagation command).
    std::pair<int32, NnetComputation::Command> p1(
        info.compression_command_index + 1,
        NnetComputation::Command(info.range, kCompressMatrix,
                                 s, static_cast<int32>(info.compression_type),
                                 info.truncate ? 1 : 0));
    pairs_to_insert.push_back(p1);
    std::pair<int32, NnetComputation::Command> p2(
        info.uncompression_command_index,
        NnetComputation::Command(1.0, kDecompressMatrix, s));
    pairs_to_insert.push_back(p2);
  }
  InsertCommands(&pairs_to_insert,
                 computation_);
}


void MemoryCompressionOptimizer::Optimize() {
  analyzer_.Init(nnet_, *computation_);
  // note: matrix zero is not really a matrix.
  int32 num_matrices = computation_->matrices.size();
  for (int32 m = 1; m < num_matrices; m++)
    ProcessMatrix(m);
  if (!compress_info_.empty())
    ModifyComputation();
}

void MemoryCompressionOptimizer::ProcessMatrix(int32 m) {
  if (analyzer_.matrix_accesses[m].is_output) {
    return;  // We can't do this optimization for matrices that are going to be
             // output to the user.
  }

  // 'accesses' list the commands that access this matrix.
  const std::vector<Access> &accesses = analyzer_.matrix_accesses[m].accesses;
  // the 'kReadAccess' below is actually a don't-care  This is just
  // to find the position in 'accesses' that corresponds to command-index
  // 'middle_command'.
  Access middle_access(middle_command_, kReadAccess);
  std::vector<Access>::const_iterator iter = std::lower_bound(accesses.begin(),
                                                              accesses.end(),
                                                              middle_access);
  // At this point, 'iter' points to the first access in 'accesses'
  // whose command index is >= 'middle_command_' (which separates the forward
  // and backward passes), or accesses.end() if this matrix was not
  // accessed during the backward pass.
  if (iter == accesses.end()) {
    return;  // There is nothing to do: this matrix was not accessed during the
             // backward pass.
  }
  if (iter == accesses.begin()) {
    return;  // There is nothing to do: this matrix was not accessed during the
             // forward pass.
  }
  // 'backward_access' is the first access of the matrix in the backward
  // pass of the computation, and
  // 'forward_access' is the last access of the matrix in the forward pass
  // of the computation.
  const Access &backward_access = iter[0],
      &forward_access = iter[-1];
  KALDI_ASSERT(forward_access.command_index < middle_command_ &&
               backward_access.command_index > middle_command_);

  // 'backward_access_is_last_access' is going to be set to true if
  // 'backward_access' is the last command to access the matrix (apart from
  // deallocation or matrix-swap commands, which don't show up in the list of
  // accesses).
  bool backward_access_is_last_access = (accesses.end() == iter + 1);

  int32 backward_command_index = backward_access.command_index,
      forward_command_index = forward_access.command_index;
  NnetComputation::Command
      &backward_command = computation_->commands[backward_command_index];

  if (memory_compression_level_ >= 1 &&
      backward_access_is_last_access &&
      backward_access.access_type == kReadAccess &&
      backward_command.command_type == kBackprop) {
    int32 component_index = backward_command.arg1;
    const Component *component = nnet_.GetComponent(component_index);
    // this is potentially a candidate for our optimization for ReLU units,
    // where we only need to store the sign.
    if (component->Type() == "RectifiedLinearComponent") {
      compress_info_.push_back(
          MatrixCompressInfo(m, forward_command_index,
                             backward_command_index,
                             kCompressedMatrixUint8, 0.0,
                             true));
      return;
    }
  }

  // If memory_compression_level >= 2 (an "intermediate" level of compression),
  // then we'll consider compressing quantities using 16 bits in the range
  // [-10, 10].  Because of the way this compression works, exact zero will
  // still be uncompressed as exact zero, so even if this is the output
  // of a ReLU, it's OK.  (Having a few derivatives zero for ReLU outputs
  // that were very close to zero is OK.)
  if (memory_compression_level_ >= 2) {
    compress_info_.push_back(
        MatrixCompressInfo(m, forward_command_index,
                           backward_command_index,
                           kCompressedMatrixInt16, 10.0,
                           true));
    return;
  }

  // TODO: later maybe implement something for memory compression level = 3.
}




void OptimizeMemoryCompression(const Nnet &nnet,
                               int32 memory_compression_level,
                               NnetComputation *computation) {
  if (memory_compression_level == 0 || computation->commands.empty())
    return;
  // don't apply this optimization to looped computations.
  if (computation->commands.back().command_type == kGotoLabel)
    return;

  // 'middle_command' will be the index of the command of type
  // 'kNoOperationMarker' that separates the forward and backward
  // passes.  If it doesn't exist, it means this computation doesn't
  // include
  int32 middle_command = -1;
  for (size_t i = 0; i < computation->commands.size(); i++) {
    if (computation->commands[i].command_type == kNoOperationMarker) {
      if (middle_command < 0) {
        middle_command = static_cast<int32>(i);
      } else {
        KALDI_WARN << "Found more than one command of type kNoOperationMarker "
            "in non-looped computation.";
        // there are more than one command of this type... this wasn't expected.
        // return (i.e. do nothing).
        return;
      }
    }
  }
  if (middle_command == -1) {
    return;  // This computation doesn't have a backprop pass.
  }
  if (memory_compression_level >= 1) {
    int64 bytes_used_initial, bytes_used_final;
    if (GetVerboseLevel() >= 2)
      bytes_used_initial = GetMaxMemoryUse(*computation);

    MemoryCompressionOptimizer opt(nnet, memory_compression_level,
                                   middle_command, computation);
    opt.Optimize();

    if (GetVerboseLevel() >= 2) {
      bytes_used_final = GetMaxMemoryUse(*computation);
      if (bytes_used_final != bytes_used_initial) {
        KALDI_VLOG(2) << "Memory compression reduced  memory use from "
                      << bytes_used_initial << " to "
                      << bytes_used_final << " bytes.";
      }
    }
  }
}


std::shared_ptr<const NnetComputation> ComputationCache::Find(
    const ComputationRequest &in_request) {
  std::lock_guard<std::mutex> lock(mutex_);

  CacheType::iterator iter = computation_cache_.find(&in_request);
  if (iter == computation_cache_.end()) {
    return NULL;
  } else {
    std::shared_ptr<const NnetComputation> ans = iter->second.first;
    // Update access record by moving the accessed request to the end of the
    // access queue, which declares that it's the most recently used.
    access_queue_.splice(access_queue_.end(), access_queue_,
                         iter->second.second);
    return ans;
  }
}


ComputationCache::ComputationCache(int32 cache_capacity):
    cache_capacity_(cache_capacity) {
  KALDI_ASSERT(cache_capacity > 0);
}

std::shared_ptr<const NnetComputation> ComputationCache::Insert(
    const ComputationRequest &request_in,
    const NnetComputation *computation_in) {

  std::lock_guard<std::mutex> lock(mutex_);
  if (static_cast<int32>(computation_cache_.size()) >= cache_capacity_) {
    //  Cache has reached capacity; purge the least-recently-accessed request
    const CacheType::iterator iter =
        computation_cache_.find(access_queue_.front());
    KALDI_ASSERT(iter != computation_cache_.end());
    const ComputationRequest *request = iter->first;
    computation_cache_.erase(iter);
    delete request;
    // we don't need to delete the computation in iter->second.first, as the
    // shared_ptr takes care of that automatically.
    access_queue_.pop_front();
  }

  // Now insert the thing we need to insert.  We'll own the pointer 'request' in
  // 'computation_cache_', so we need to allocate our own version.
  ComputationRequest *request = new ComputationRequest(request_in);
  // When we construct this shared_ptr, it takes ownership of the pointer
  // 'computation_in'.
  std::shared_ptr<const NnetComputation> computation(computation_in);

  AqType::iterator ait = access_queue_.insert(access_queue_.end(), request);

  std::pair<CacheType::iterator, bool> p = computation_cache_.insert(
      std::make_pair(request, std::make_pair(computation, ait)));
  if (!p.second) {
    // if p.second is false, this pair was not inserted because
    // a computation for the same computation-request already existed in
    // the map. This is possible in multi-threaded operations, if two
    // threads try to compile the same computation at the same time (only
    // one of them will successfully add it).
    // We need to erase the access-queue element that we just added, it's
    // no longer going to be needed.
    access_queue_.erase(ait);
    delete request;
  }
  return computation;
}


void ComputationCache::Read(std::istream &is, bool binary) {
  // Note: the object on disk doesn't have tokens like "<ComputationCache>"
  // and "</ComputationCache>" for back-compatibility reasons.
  int32 computation_cache_size;
  ExpectToken(is, binary, "<ComputationCacheSize>");
  ReadBasicType(is, binary, &computation_cache_size);
  KALDI_ASSERT(computation_cache_size >= 0);
  computation_cache_.clear();
  access_queue_.clear();
  ExpectToken(is, binary, "<ComputationCache>");
  for (size_t c = 0; c < computation_cache_size; c++) {
    ComputationRequest request;
    request.Read(is, binary);
    NnetComputation *computation = new NnetComputation();
    computation->Read(is, binary);
    Insert(request, computation);
  }
}

void ComputationCache::Check(const Nnet &nnet) const {
  CacheType::const_iterator iter = computation_cache_.begin(),
      end = computation_cache_.end();
  // We only need to explicitly delete the pointer to the ComputationRequest.
  // The pointers to Computation are deleted automatically by std::shared_ptr
  // when the reference count goes to zero.
  for (; iter != end; ++iter) {
    const NnetComputation &computation = *(iter->second.first);
    CheckComputationOptions check_config;
    ComputationChecker checker(check_config, nnet, computation);
    checker.Check();
  }
}

void ComputationCache::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ComputationCacheSize>");
  WriteBasicType(os, binary, static_cast<int32>(computation_cache_.size()));
  WriteToken(os, binary, "<ComputationCache>");
  for (CacheType::const_iterator iter = computation_cache_.begin();
           iter != computation_cache_.end(); ++iter) {
    iter->first->Write(os, binary);
    iter->second.first->Write(os, binary);
  }
}

ComputationCache::~ComputationCache() {
  CacheType::const_iterator iter = computation_cache_.begin(),
      end = computation_cache_.end();
  // We only need to explicitly delete the pointer to the ComputationRequest.
  // The pointers to Computation are deleted automatically by std::shared_ptr
  // when the reference count goes to zero.
  for (; iter != end; ++iter)
    delete iter->first;
}

} // namespace nnet3
} // namespace kaldi
