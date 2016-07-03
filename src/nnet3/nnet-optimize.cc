// nnet3/nnet-optimize.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Xiaohui Zhang

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
#include "nnet3/nnet-optimize-utils.h"

namespace kaldi {
namespace nnet3 {

void NnetOptimizeOptions::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetOptimizeOptions>");
  ExpectToken(is, binary, "<Optimize>");
  ReadBasicType(is, binary, &optimize);
  ExpectToken(is, binary, "<ConsolidateModelUpdate>");
  ReadBasicType(is, binary, &consolidate_model_update);
  ExpectToken(is, binary, "<PropagateInPlace>");
  ReadBasicType(is, binary, &propagate_in_place);
  ExpectToken(is, binary, "<BackpropInPlace>");
  ReadBasicType(is, binary, &backprop_in_place);
  ExpectToken(is, binary, "<ConvertAddition>");
  ReadBasicType(is, binary, &convert_addition);
  ExpectToken(is, binary, "<RemoveAssignments>");
  ReadBasicType(is, binary, &remove_assignments);
  ExpectToken(is, binary, "<AllowLeftMerge>");
  ReadBasicType(is, binary, &allow_left_merge);
  ExpectToken(is, binary, "<AllowRightMerge>");
  ReadBasicType(is, binary, &allow_right_merge);
  ExpectToken(is, binary, "<InitializeUndefined>");
  ReadBasicType(is, binary, &initialize_undefined);
  ExpectToken(is, binary, "<MoveSizingCommands>");
  ReadBasicType(is, binary, &move_sizing_commands);
  ExpectToken(is, binary, "<AllocateFromOther>");
  ReadBasicType(is, binary, &allocate_from_other);
  ExpectToken(is, binary, "<MinDerivTime>");
  ReadBasicType(is, binary, &min_deriv_time);
  ExpectToken(is, binary, "<MaxDerivTime>");
  ReadBasicType(is, binary, &max_deriv_time);
  ExpectToken(is, binary, "</NnetOptimizeOptions>");
}

void NnetOptimizeOptions::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetOptimizeOptions>");
  WriteToken(os, binary, "<Optimize>");
  WriteBasicType(os, binary, optimize);
  WriteToken(os, binary, "<ConsolidateModelUpdate>");
  WriteBasicType(os, binary, consolidate_model_update);
  WriteToken(os, binary, "<PropagateInPlace>");
  WriteBasicType(os, binary, propagate_in_place);
  WriteToken(os, binary, "<BackpropInPlace>");
  WriteBasicType(os, binary, backprop_in_place);
  WriteToken(os, binary, "<ConvertAddition>");
  WriteBasicType(os, binary, convert_addition);
  WriteToken(os, binary, "<RemoveAssignments>");
  WriteBasicType(os, binary, remove_assignments);
  WriteToken(os, binary, "<AllowLeftMerge>");
  WriteBasicType(os, binary, allow_left_merge);
  WriteToken(os, binary, "<AllowRightMerge>");
  WriteBasicType(os, binary, allow_right_merge);
  WriteToken(os, binary, "<InitializeUndefined>");
  WriteBasicType(os, binary, initialize_undefined);
  WriteToken(os, binary, "<MoveSizingCommands>");
  WriteBasicType(os, binary, move_sizing_commands);
  WriteToken(os, binary, "<AllocateFromOther>");
  WriteBasicType(os, binary, allocate_from_other);
  WriteToken(os, binary, "<MinDerivTime>");
  WriteBasicType(os, binary, min_deriv_time);
  WriteToken(os, binary, "<MaxDerivTime>");
  WriteBasicType(os, binary, max_deriv_time);
  WriteToken(os, binary, "</NnetOptimizeOptions>");
}

bool NnetOptimizeOptions::operator == (const NnetOptimizeOptions &other) const {
  return (other.propagate_in_place == propagate_in_place &&
          other.optimize == optimize &&
          other.consolidate_model_update == consolidate_model_update &&
          other.backprop_in_place == backprop_in_place &&
          other.convert_addition == convert_addition &&
          other.remove_assignments == remove_assignments &&
          other.allow_left_merge == allow_left_merge &&
          other.allow_right_merge == allow_right_merge &&
          other.initialize_undefined == initialize_undefined &&
          other.move_sizing_commands == move_sizing_commands &&
          other.allocate_from_other == allocate_from_other &&
          other.min_deriv_time == min_deriv_time &&
          other.max_deriv_time == max_deriv_time);
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
          v_accesses[1].access_type != kWriteAccess) {
        all_variables_ok = false;  // first access after zeroing was not a write
        break;
      }
      if (v_accesses.size() == 1 &&
          accesses.is_output) {
        // the only command that touches this variable is the allocation, and it
        // is an output variable.  (this is unusual, but can happen e.g. if it's
        // a derivative, but due to min_deriv_time and max_deriv_time it ends up
        // always being zero.
        all_variables_ok = false;
        break;
      }
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
  // For each size of matrix and stride-type, represented as a pair<int32,int32>
  // (the num-rows, and the num-cols * (stride-type == kDefaultStride ? 1 : -1), we
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
          num_cols = computation->matrices[m].num_cols,
          num_cols_mod = num_cols * (
              computation->matrices[m].stride_type == kDefaultStride ? 1 : -1);
      std::pair<int32,int32> p(num_rows, num_cols_mod);
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


void ConvertAdditionToAssignment(const Nnet &nnet,
                                 NnetComputation *computation) {
  Analyzer analyzer;
  analyzer.Init(nnet, *computation);
  ComputationAnalysis analysis(*computation, analyzer);
  int32 num_commands = computation->commands.size();
  for (int32 command = 0; command < num_commands; command++) {
    NnetComputation::Command &c = computation->commands[command];
    switch (c.command_type) {
      case kAllocMatrixUndefined: case kAllocMatrixFromOther:
        KALDI_ERR << "Cannot call ConvertAdditionToAssignment after "
                  << "allowing undefined initialization.";
      case kMatrixAdd: case kAddRows: case kAddRowsMulti:
      case kAddToRowsMulti: {
        const std::vector<int32> &submatrices_written =
            analyzer.command_attributes[command].submatrices_written;
        KALDI_ASSERT(!submatrices_written.empty());
        std::vector<int32>::const_iterator iter = submatrices_written.begin(),
            end = submatrices_written.end();
        bool can_convert = true;
        for (; iter != end; ++iter) {
          int32 submatrix_written = *iter;
          int32 first_access_command = analysis.FirstAccess(submatrix_written);
          // first_access_command is first non-initialization command that
          // accesses this submatrix.  It can be assumed to be a write command,
          // since it makes no sense to read a variable before it's written to.
          // If it's before this command then we need to add rather than copy,
          // we can't do the conversion to a copy command.
          if (first_access_command != command) {
            can_convert = false;
            break;
          }
        }
        if (can_convert) {  // convert to a copy command.
          switch (c.command_type) {
            case kMatrixAdd: c.command_type = kMatrixCopy; break;
            case kAddRows: c.command_type = kCopyRows; break;
            case kAddRowsMulti: c.command_type = kCopyRowsMulti; break;
            case kAddToRowsMulti: c.command_type = kCopyToRowsMulti; break;
            default: KALDI_ERR << "Unexpected command type.";
          }
        }
        break;
      }
      default:
        break;
    }
  }
}

void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              const ComputationRequest &request,
              NnetComputation *computation) {
  if (!config.optimize)
    return;

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, true);

  // this will do nothing unless --min-deriv-time or --max-deriv-time was
  // set.
  LimitDerivativeTimes(nnet, config.min_deriv_time, config.max_deriv_time,
                       computation);

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, true);

  if (config.consolidate_model_update)
    ConsolidateModelUpdate(nnet, request, computation);

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, true);

  if (config.convert_addition)
    ConvertAdditionToAssignment(nnet, computation);

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, true);

  if (config.remove_assignments || config.backprop_in_place ||
      config.propagate_in_place)
    VariableMergingOptimization(config, nnet, request, computation);

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, false);

  if (config.initialize_undefined)
    RemoveUnnecessaryZeroing(nnet, computation);

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, false);

  if (config.move_sizing_commands)
    MoveSizingCommands(nnet, computation);

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, false);

  if (config.allocate_from_other)
    RemoveUnnecessaryAllocation(nnet, computation);

  if (GetVerboseLevel() >= 4)
    CheckComputation(nnet, request, *computation, false);
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

void CachingOptimizingCompiler::UpdateCache(const ComputationRequest *request,
                                            NnetComputation *computation) {
  if (computation_cache_.size() == cache_capacity_) {
    // full, locate the least-recently-accessed request
    const CacheType::iterator it =
        computation_cache_.find(access_queue_.front());
    KALDI_ASSERT(it != computation_cache_.end());
    // purge the least-recently-accessed request
    delete it->first;
    delete it->second.first;
    computation_cache_.erase(it);
    access_queue_.pop_front();
  }
  AqType::iterator ait = access_queue_.insert(access_queue_.end(), request);
  computation_cache_.insert(std::make_pair(request,
                            std::make_pair(computation, ait)));
}

void CachingOptimizingCompiler::ReadCache(std::istream &is, bool binary) {
  NnetOptimizeOptions opt_config_cached;
  opt_config_cached.Read(is, binary);
  // we won't read cached computations if any optimize option has been changed.
  bool read_cache = (opt_config_ == opt_config_cached);

  if (read_cache) {
    int32 computation_cache_size;
    ExpectToken(is, binary, "<ComputationCacheSize>");
    ReadBasicType(is, binary, &computation_cache_size);
    KALDI_ASSERT(computation_cache_size >= 0);
    computation_cache_.clear();
    access_queue_.clear();
    ExpectToken(is, binary, "<ComputationCache>");
    for (size_t c = 0; c < computation_cache_size; c++) {
      ComputationRequest *request = new ComputationRequest();
      request->Read(is, binary);
      NnetComputation *computation = new NnetComputation();
      computation->Read(is, binary);
      UpdateCache(request, computation);
    }
  }
}

void CachingOptimizingCompiler::WriteCache(std::ostream &os, bool binary) const {
  opt_config_.Write(os, binary);
  WriteToken(os, binary, "<ComputationCacheSize>");
  WriteBasicType(os, binary, static_cast<int32>(computation_cache_.size()));
  WriteToken(os, binary, "<ComputationCache>");
  for (CacheType::const_iterator iter = computation_cache_.begin();
           iter != computation_cache_.end(); ++iter) {
    iter->first->Write(os, binary);
    iter->second.first->Write(os, binary);
  }
}

void CachingOptimizingCompiler::UpdateAccessQueue(CacheType::iterator &cit) {
  // exist, update access record by moving the accessed
  // request to the end of the access queue
  KALDI_ASSERT(cit != computation_cache_.end());
  access_queue_.splice(access_queue_.end(), access_queue_,
                       cit->second.second);
}

CachingOptimizingCompiler::~CachingOptimizingCompiler() {
  CacheType::const_iterator itr = computation_cache_.begin(),
      end = computation_cache_.end();
  for (; itr !=end; ++itr) {
    delete itr->first;
    delete itr->second.first;
  }
}

const NnetComputation* CachingOptimizingCompiler::Compile(
    const ComputationRequest  &in_request) {
  NnetComputation *computation;
  ComputationRequest *request;
  // find computation in the cache
  CacheType::iterator cit = computation_cache_.find(&in_request);
  if (cit == computation_cache_.end()) {
    // if not found, compile and update cache
    request = new ComputationRequest;
    *request = in_request;
    Compiler compiler(*request, nnet_);
    CompilerOptions opts;
    computation = new NnetComputation;
    compiler.CreateComputation(opts, computation);

    int32 verbose_cutoff = 4;
    if (GetVerboseLevel() >= verbose_cutoff) {
      std::ostringstream os1;
      request->Print(os1);
      KALDI_LOG << "Computation request is " << os1.str();
      std::ostringstream os2;
      computation->Print(os2, nnet_);
      KALDI_LOG << "Generated computation is: " << os2.str();
    }
    { // some checking.
      CheckComputationOptions check_config;
      // we can do the rewrite check since it's before optimization.
      check_config.check_rewrite = true;
      ComputationChecker checker(check_config, nnet_, *computation);
      checker.Check();
    }
    Optimize(opt_config_, nnet_, *request, computation);
    if (GetVerboseLevel() >= verbose_cutoff) {
      std::ostringstream os;
      computation->Print(os, nnet_);
      KALDI_LOG << "Optimized computation is: " << os.str();
    }
    {  // check the computation again.
      CheckComputationOptions check_config;
      ComputationChecker checker(check_config, nnet_, *computation);
      checker.Check();
    }
    computation->ComputeCudaIndexes();
    UpdateCache(request, computation);
  } else {
    // if found, update access queue
    computation = cit->second.first;
    UpdateAccessQueue(cit);
  }
  return computation;
}


} // namespace nnet3
} // namespace kaldi
