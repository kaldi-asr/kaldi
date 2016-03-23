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


void IdentifyMatrixArgs(std::vector<NnetComputation::Command> *commands,
                        std::vector<int32*> *matrix_args) {
  matrix_args->clear();
  std::vector<NnetComputation::Command>::iterator iter = commands->begin(),
      end = commands->end();
  std::vector<int32*> this_matrix_args;
  for (; iter != end; ++iter) {
    IdentifyMatrixArgs(&(*iter), &this_matrix_args);
    matrix_args->insert(matrix_args->end(),
                        this_matrix_args.begin(),
                        this_matrix_args.end());
  }
}


void IdentifyMatrixArgsInComputation(bool include_in_submatrices,
                                     NnetComputation *computation,
                                     std::vector<int32*> *matrix_args) {
  IdentifyMatrixArgs(&(computation->commands), matrix_args);
  int32 num_submatrices = computation->submatrices.size();
  matrix_args->reserve(matrix_args->size() +
                       (include_in_submatrices ?
                        computation->submatrices.size() : 0) +
                       2 * computation->input_output_info.size());
  if (include_in_submatrices)
    for (int32 s = 1; s < num_submatrices; s++)
      matrix_args->push_back(&(computation->submatrices[s].matrix_index));
  unordered_map<int32, std::pair<int32, int32> >::iterator
      iter = computation->input_output_info.begin(),
      end = computation->input_output_info.end();
  for (; iter != end; ++iter) {
    matrix_args->push_back(&(iter->second.first));
    matrix_args->push_back(&(iter->second.second));
  }
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

  std::vector<int32*> matrix_args;
  bool include_in_submatrices = false;
  IdentifyMatrixArgsInComputation(include_in_submatrices,
                                  computation_, &matrix_args);
  std::vector<int32*>::iterator iter = matrix_args.begin(),
      end = matrix_args.end();
  for (; iter != end; ++iter) {
    int32 matrix_index = **iter;
    if (matrix_index > 0)
      matrix_is_used_[matrix_index] = true;
  }
  // We also need to take into account when matrices are used indirectly via
  // submatrices (which is actually the main way they are accessed).
  int32 num_submatrices_orig = computation_->submatrices.size();
  for (int32 s = 1; s < num_submatrices_orig; s++) {
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
  bool include_in_submatrices = true;
  IdentifyMatrixArgsInComputation(include_in_submatrices,
                                  computation_, &matrix_args);
  std::vector<int32*>::iterator iter = matrix_args.begin(),
      end = matrix_args.end();
  for (; iter != end; ++iter) {
    if (**iter > 0) {
      int32 new_matrix_index = old_to_new_matrix_[**iter];
      // old_to_new_matrix_[s] for s > 0 is only <= 0 (actually, -1) for
      // submatrices that are never accessed, and these should never appear
      // in this list.
      KALDI_ASSERT(new_matrix_index > 0);
      **iter = new_matrix_index;
    }
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
    std::vector<int32>::const_iterator iter = matrix_to_submatrix_[m1].begin(),
        end = matrix_to_submatrix_[m1].end();
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
  ComputationAnalysis analysis(*computation_, analyzer_);
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

  //   - If both m_to_keep and m_to_discard have commands that deallocate them,
  //    keep only the allocation command for m_to_keep, and make sure it's after
  //    the last access of m_to_discard (otherwise delete any deallocation
  //    command).
  int32 dealloc_keep = matrix_accesses[m_to_keep].deallocate_command,
      dealloc_discard = matrix_accesses[m_to_discard].deallocate_command;
  if (dealloc_keep != -1 && dealloc_discard != -1) {
    KALDI_ASSERT(analysis.LastMatrixAccess(m_to_discard) < dealloc_keep);
    computation_->commands[dealloc_discard].command_type = kNoOperation;
  } else {
    if (dealloc_keep != -1)
      computation_->commands[dealloc_keep].command_type =
          kNoOperation;
    if (dealloc_discard != -1)
      computation_->commands[dealloc_discard].command_type =
          kNoOperation;
  }

  //   - If both m_to_keep and m_to_discard have commands that allocate them,
  //     keep only the allocation command for m_to_keep and make sure it's
  //     before the first access of m_to_discard.
  //     (otherwise delete any allocation command).
  int32 alloc_keep = matrix_accesses[m_to_keep].allocate_command,
      alloc_discard = matrix_accesses[m_to_discard].allocate_command;
  if (alloc_keep != -1 && alloc_discard != -1) {
    KALDI_ASSERT(analysis.FirstMatrixAccess(m_to_discard) > alloc_keep);
    NnetComputation::Command
        &keep_alloc_command = computation_->commands[alloc_keep],
        &discard_alloc_command = computation_->commands[alloc_discard];
    discard_alloc_command.command_type = kNoOperation;
    if (keep_alloc_command.command_type == kAllocMatrixUndefined) {
      keep_alloc_command.command_type = kAllocMatrixZeroed;
    } else if (keep_alloc_command.command_type == kAllocMatrixFromOther) {
      keep_alloc_command.command_type = kAllocMatrixFromOtherZeroed;
    }
  } else {
    if (alloc_keep != -1)
      computation_->commands[alloc_keep].command_type =
          kNoOperation;
    if (alloc_discard != -1)
      computation_->commands[alloc_discard].command_type =
          kNoOperation;
  }

  //  If the matrix to discard had stride_type == kStrideEqualNumCols, set the
  //  matrix to keep's stride_type to kStrideEqualNuMCols.
  if (computation_->matrices[m_to_discard].stride_type == kStrideEqualNumCols) {
    computation_->matrices[m_to_keep].stride_type = kStrideEqualNumCols;
    // ... and perform an additional check.
    KALDI_ASSERT(computation_->matrices[m_to_discard].num_rows ==
                 computation_->matrices[m_to_keep].num_rows &&
                 computation_->matrices[m_to_discard].num_cols ==
                 computation_->matrices[m_to_keep].num_cols);
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
    std::vector<int32>::const_iterator iter = matrix_to_submatrix_[m2].begin(),
        end = matrix_to_submatrix_[m2].end();
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
  // Add a command at the very start, to initialize this new matrix.
  int32 new_matrix_index =
      computation_->submatrices[new_whole_submatrix].matrix_index;
  // we can later on optimize this zeroed initialization to an undefined
  // initialization.
  extra_commands_[0].push_back(
      NnetComputation::Command(kAllocMatrixZeroed, new_matrix_index));
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

// modify commands to take into account the fact that some matrices are zero or
// partially zero.  Allocation commands and sizes of underlying matrices are not
// affected-- we'll work out later on, what to do with them.
void DerivativeTimeLimiter::ModifyCommand(NnetComputation::Command *command) {
  CommandType command_type = command->command_type;
  switch (command_type) {
    case kAllocMatrixUndefined:
    case kAllocMatrixFromOther:
    case kAllocMatrixFromOtherZeroed:
      KALDI_ERR << "No undefined initialization or initialization-from-other "
                << "is allowed before LimitDerivativeTimes";
      break;
    case kAllocMatrixZeroed:
    case kDeallocMatrix:
      break;  // we'll deal with allocation and deallocation later on.
    case kPropagate:
      // Propagate commands are unchanged.
      break;
    case kStoreStats: {
      const Component *component = nnet_.GetComponent(command->arg1);
      if ((component->Properties() & kSimpleComponent)) {
        // We choose to apply the time-limitation here, as it will save time and
        // is probably what the user wants.
        int32 submatrix_mapped = submatrix_map_[command->arg2];
        if (submatrix_mapped == 0)
          command->command_type = kNoOperation;
        else
          command->arg2 = submatrix_mapped;
      }
      break;
    }
    case kBackpropNoModelUpdate:  // we actually don't expect to encounter this,
                                  // but it's trivial to support as it's the
                                  // same as backprop.
    case kBackprop: {
      const Component *component = nnet_.GetComponent(command->arg1);
      if (!(component->Properties() & kSimpleComponent)) {
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
      } else if (mapped_output_deriv_submatrix !=
                 output_deriv_submatrix) {
        // we're operating on a range of the input or output.
        command->arg3 = mapped_input_submatrix;
        command->arg4 = mapped_output_submatrix;
        command->arg5 = mapped_output_deriv_submatrix;
        command->arg6 = mapped_input_deriv_submatrix;
      }
    }
      break;
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
    case kNoOperation: case kNoOperationMarker:
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
  // left_prune1 is the nmber of rows pruned away on the left for submatrix1.
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
      // subm-matrix.
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

  if (input_submatrix_mapped == input_submatrix &&
      output_submatrix_mapped == output_submatrix) {
    return;  // nothing is changed.
  }
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
    if (orig_index == -1) {
      new_indexes[i] = -1;
    } else {
      int32 mapped_index = orig_index - left_prune_input;
      if (mapped_index >= 0 && mapped_index < new_num_input_rows) {
        new_indexes[i] = mapped_index;
        must_keep_command = true;
      } else {
        // input was out of range (i.e. it takes a value that we are asserting
        // is zero)-- use -1 as the index.
        new_indexes[i] = -1;
      }
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
  int32 submatrix_arg = c->arg1,
      indexes_multi_arg = c->arg2;
  int32 submatrix_mapped = submatrix_map_if_deriv_[submatrix_arg];
  if (submatrix_mapped == 0) {
    c->command_type = kNoOperation;
    return;
  }
  int32 left_prune;
  GetPruneValues(submatrix_arg, submatrix_mapped, &left_prune, NULL);
  int32 new_num_rows = computation_->submatrices[submatrix_mapped].num_rows;
  const std::vector<std::pair<int32, int32> > &old_indexes_multi(
      computation_->indexes_multi[indexes_multi_arg]);
  std::vector<std::pair<int32, int32> > new_indexes_multi(new_num_rows);
  for (int32 i = 0; i < new_num_rows; i++) {
    std::pair<int32,int32> &this_pair = new_indexes_multi[i];
    this_pair = old_indexes_multi[i + left_prune];
    int32 this_submatrix = this_pair.first,
        this_row = this_pair.second;
    if (this_submatrix == -1)  // don't map the (-1, -1) pairs.
      continue;
    int32 this_submatrix_mapped = submatrix_map_if_deriv_[this_submatrix];
    if (this_submatrix_mapped == this_submatrix) {
      continue;
    } else if (this_submatrix_mapped == 0) {  // was completely out of range.
      this_pair.first = -1;
      this_pair.second = -1;
    } else {
      int32 this_left_prune, this_num_rows =
          computation_->submatrices[this_submatrix_mapped].num_rows;
      GetPruneValues(this_submatrix, this_submatrix_mapped,
                     &this_left_prune, NULL);
      int32 this_row_mapped = this_row - this_left_prune;
      if (this_row_mapped >= 0 && this_row_mapped < this_num_rows) {
        this_pair.first = this_submatrix_mapped;
        this_pair.second = this_row_mapped;
      } else {
        this_pair.first = -1;
        this_pair.second = -1;
      }
    }
  }
  if (submatrix_mapped == submatrix_arg &&
      new_indexes_multi == old_indexes_multi)  // nothing changed.
    return;
  bool command_can_be_deleted = true;
  std::vector<std::pair<int32, int32> >::iterator
      iter = new_indexes_multi.begin(),
      end = new_indexes_multi.end();
  for (; iter != end; ++iter) {
    if (iter->first != -1) {
      command_can_be_deleted = false;
      break;
    }
  }
  if (command_can_be_deleted) {
    c->command_type = kNoOperation;
    return;
  }
  c->arg1 = submatrix_mapped;
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
  for (int32 i = 0; i < dest_num_rows; i++) {
    std::pair<int32, int32> &this_pair = new_indexes_ranges[i];
    this_pair = old_indexes_ranges[i + dest_left_prune];
    // note: the .first is a start-index in the src matrix, and the .second is
    // an end-index in the src matrix.
    int32 new_first = this_pair.first - src_left_prune,
        new_second = this_pair.second - src_left_prune;
    if (new_first < 0) new_first = 0;
    if (new_first >= src_num_rows) new_first = src_num_rows - 1;
    if (new_second < 0) new_second = 0;
    if (new_second >= src_num_rows) new_second = src_num_rows - 1;
    if (new_first == new_second) {
      // for clarity, represent empty ranges as (-1, -1).
      new_first = -1;
      new_second = -1;
    }
    KALDI_ASSERT(new_second >= new_first);
    this_pair.first = new_first;
    this_pair.second = new_second;
  }
  c->arg1 = dest_submatrix_mapped;
  c->arg2 = src_submatrix_mapped;
  c->arg3 = computation_->indexes_ranges.size();
  computation_->indexes_ranges.push_back(new_indexes_ranges);
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
  if (min_deriv_time_ == std::numeric_limits<BaseFloat>::min() &&
      max_deriv_time_ == std::numeric_limits<BaseFloat>::max())
    return;  // nothing to do.

  EnsureMatricesHaveEntireSubmatrices();
  ComputeMatrixPruneInfo();
  ComputeSubmatrixMaps();
  ModifyCommands();
  PruneMatrices();
  RemoveNoOps(computation_);
  RenumberComputation(computation_);
}

void DerivativeTimeLimiter::EnsureMatricesHaveEntireSubmatrices() {
  int32 num_matrices = computation_->matrices.size(),
      num_submatrices = computation_->submatrices.size();
  entire_submatrix_.clear();
  entire_submatrix_.resize(num_matrices, -1);
  entire_submatrix_[0] = 0;
  for (int32 s = 1; s < num_submatrices; s++)
    if (computation_->IsWholeMatrix(s))
      entire_submatrix_[computation_->submatrices[s].matrix_index] = s;
  for (int32 m = 1; m < num_matrices; m++)
    if (entire_submatrix_[m] == -1)
      entire_submatrix_[m] = computation_->NewSubMatrix(m, 0, -1, 0, -1);
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
  int32 s_entire = entire_submatrix_[m];  // submatrix consisting of
                                                     // all of the matrix.
  int32 s_mapped = submatrix_map_[s_entire];  // the matrix limited in time.
  KALDI_ASSERT(s_mapped != 0 && s_mapped != s_entire);
  std::vector<int32> entire_variables, mapped_variables;
  analyzer.variables.AppendVariablesForSubmatrix(s_entire,
                                                 &entire_variables);
  analyzer.variables.AppendVariablesForSubmatrix(s_mapped,
                                                 &mapped_variables);
  KALDI_ASSERT(entire_variables.size() > mapped_variables.size());
  std::vector<int32> excluded_variables(entire_variables.size() -
                                        mapped_variables.size());
  std::vector<int32>::iterator end_iter =
      std::set_difference(entire_variables.begin(), entire_variables.end(),
                          mapped_variables.begin(), mapped_variables.end(),
                          excluded_variables.begin());
  KALDI_ASSERT(end_iter == excluded_variables.end());
  // We want to make sure that none of the excluded variables are
  // ever accessed.  If they are, we cannot prune the matrix.
  int32 allocate_command = analyzer.matrix_accesses[m].allocate_command;
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
      if (viter->command_index != allocate_command) {
        // we may one day want to look at this.. it's not really expected.
        KALDI_VLOG(4) << "Cannot prune matrix " << m;
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
        // This submatrix is not entirely the kept range of the matrix.
        // We assume that this submatrix is never accessed directly (as when
        // we modified the computation we ensured this).  We
        // give it a valid but stupid size of num-rows=1, num-cols=1, so
        // that if it ever does get accessed it should produce an error.
        submat_info.row_offset = 0;
        submat_info.num_rows = 1;
        submat_info.col_offset = 0;
        submat_info.num_cols = 1;
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
  KALDI_ASSERT(computation_->matrices.size() == entire_submatrix_.size());
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
      if (accesses.accesses.empty() ||
          (accesses.accesses.size() == 1 &&
           accesses.accesses[0].command_index == accesses.allocate_command)) {
        // we prune the matrix away.  the only thing we need to do here is
        // to remove the allocation and deallocation commands.
        // they should exist, because we just checked that it's not an input
        // or an output.
        KALDI_ASSERT(accesses.allocate_command >= 0 &&
                     accesses.deallocate_command >= 0);
        computation_->commands[accesses.allocate_command].command_type =
            kNoOperation;
        computation_->commands[accesses.deallocate_command].command_type =
            kNoOperation;
      }
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

} // namespace nnet3
} // namespace kaldi
