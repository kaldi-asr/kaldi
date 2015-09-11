// nnet3/nnet-optimize.h

// Copyright 2015    Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_OPTIMIZE_H_
#define KALDI_NNET3_NNET_OPTIMIZE_H_

#include "nnet3/nnet-compile.h"
#include "nnet3/nnet-analyze.h"

#include <list>

namespace kaldi {
namespace nnet3 {

// Options class for optimizing a NnetComputation The main projected use for
// this is in debugging the optimization code itself, so that if an error is
// detected, we can work out which optimization was responsible for the error.
struct NnetOptimizeOptions {
  bool optimize;  // setting this false disallow all optimization.
  bool consolidate_model_update;
  bool propagate_in_place;
  bool backprop_in_place;
  bool remove_assignments;
  bool allow_left_merge;
  bool allow_right_merge;
  bool initialize_undefined;
  bool move_sizing_commands;
  bool allocate_from_other;

  NnetOptimizeOptions(): optimize(true),
                         consolidate_model_update(true),
                         propagate_in_place(true),
                         backprop_in_place(true),
                         remove_assignments(true),
                         allow_left_merge(true),
                         allow_right_merge(true),
                         initialize_undefined(true),
                         move_sizing_commands(true),
                         allocate_from_other(true) { }

  void Register(OptionsItf *opts) {
    opts->Register("optimize", &optimize, "Set this to false to turn off all "
                 "optimizations");
    opts->Register("consolidate-model-update", &consolidate_model_update,
                   "Set to false to disable optimization that consolidates "
                   "the model-update phase of backprop (e.g. for recurrent "
                   "architectures");
    opts->Register("propagate-in-place", &propagate_in_place, "Set to false to "
                   "disable optimization that allows in-place propagation");
    opts->Register("backprop-in-place", &backprop_in_place, "Set to false to "
                   "disable optimization that allows in-place backprop");
    opts->Register("remove-assignments", &remove_assignments, "Set to false to "
                   "disable optimization that removes redundant assignments");
    opts->Register("allow-left-merge", &allow_left_merge, "Set to false to "
                   "disable left-merging of variables (obscure option)");
    opts->Register("allow-right-merge", &allow_right_merge, "Set to false to "
                   "disable right-merging of variables (obscure option)");
    opts->Register("initialize-undefined", &initialize_undefined, "Set to false "
                   "to disable optimization that avoids redundant zeroing");
    opts->Register("move-sizing-commands", &move_sizing_commands, "Set to false "
                   "to disable optimization that moves matrix allocation and "
                   "deallocation commands to conserve memory.");
    opts->Register("allocate-from-other", &allocate_from_other, "Instead of "
                   "deleting a matrix of a given size and then allocating "
                   "a matrix of the same size, allow re-use of that memory");
  }
};

/// This is the top-level function for optimizing a computation.
void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              const ComputationRequest &request,
              NnetComputation *computation);

// Hash function for ComputationRequest. It converts
// ComputationRequest to hash code by looking at input
// and output IoSpecifications vectors.
struct ComputationRequestHasher {
  size_t operator()(const ComputationRequest *cr) const;
 private:
  size_t IoSpecificationToInt(const IoSpecification& spec) const;
  static const int kPrime = 7853;
};

// Equality function for ComputationRequest pointer
struct ComputationRequestPtrEqual {
 public:
  bool operator() (const ComputationRequest* cr1,
                   const ComputationRequest* cr2) const {
    return (*cr1) == (*cr2);
  }
};

/// This class enables you to do the compilation and optimization in one call,
/// and also ensures that if the ComputationRequest is identical to the previous
/// one, the compilation process is not repeated.
class CachingOptimizingCompiler {
 public:
  CachingOptimizingCompiler(const Nnet &nnet,
                           const int32 capacity = 20):
      nnet_(nnet), cache_capacity_(capacity) { }

  /// Note: nnet is retained as a const reference but opt_config is copied.
  CachingOptimizingCompiler(const Nnet &nnet,
                            const NnetOptimizeOptions &opt_config,
                            const int32 capacity = 20):
      nnet_(nnet), opt_config_(opt_config), cache_capacity_(capacity) { }

  ~CachingOptimizingCompiler();
  /// Does the compilation and returns a const pointer to
  /// the result, which is owned by this class, not the caller.
  /// It calls ComputeCudaIndexes() for you, because you wouldn't
  /// be able to do this on a const object.
  const NnetComputation* Compile(const ComputationRequest &request);
 private:
  const Nnet &nnet_;
  NnetOptimizeOptions opt_config_;

  // The access queue for keeping track of the freshness of computation.
  // Most-recently-accessed computation is at the end, and
  // least-recently-accessed computaiton is at the beginning.
  // Together with computation_cache_, this forms a most-recently-used (MRU)
  // cache for Computations, indexed by ComputationRequest. Pointers
  // are owned in computation_cache_.
  typedef std::list<const ComputationRequest*> AqType;
  AqType access_queue_;

  // Map from computation-request to pair of (computation, and position in
  // access_queue_). Used for fast lookup of previously compiled computations.
  // All pointers are owned here.
  typedef unordered_map<const ComputationRequest*, std::pair<NnetComputation*,
    AqType::iterator>, ComputationRequestHasher,
    ComputationRequestPtrEqual> CacheType;
  CacheType computation_cache_;

  // This function updates the computation cache. It is called within
  // Compile(). It insert the request to the end of the queue, and purge
  // the least-recently-accessed request from the queue and the cache
  // if the capacity is reached.
  void UpdateCache(const ComputationRequest *request,
                   NnetComputation *computation);
  // This function updates the recently accessed queue.
  void UpdateAccessQueue(CacheType::iterator &cit);
  // This configuration value determines how many unique Computations
  // to cache in our most-recently-used cache.
  int32 cache_capacity_;
};


/// This wraps class VariableMergingOptimizer in a simplified interface.
void VariableMergingOptimization(const NnetOptimizeOptions &config,
                                 const Nnet &nnet,
                                 const ComputationRequest &request,
                                 NnetComputation *computation);


/// This consolidates the model-update parts of the backprop into larger
/// operations (applicable mostly to recurrent setups)-- internally it uses
/// class ModelUpdateConsolidator.  Will fail if called a
/// second time.
void ConsolidateModelUpdate(const Nnet &nnet,
                            const ComputationRequest &request,
                            NnetComputation *computation);

/// This optimization function changes, where possible, matrix initializations
/// of type kAllocMatrixZeroed to kAllocMatrixUndefined.
void RemoveUnnecessaryZeroing(const Nnet &nnet, NnetComputation *computation);


/// This optimization moves commands that initialize matrices to as late as
/// possible, and commands that empty matrices to as early as possible.
void MoveSizingCommands(const Nnet &nnet, NnetComputation *computation);

/// This optimization detects cases where we deallocate a matrix, and then
/// later allocate another matrix of the same size; and replaces them
/// with commands of type kAllocFromOther or kAllocFromOtherZeroed.
void RemoveUnnecessaryAllocation(const Nnet &nnet,
                                 NnetComputation *computation);



} // namespace nnet3
} // namespace kaldi


#endif
