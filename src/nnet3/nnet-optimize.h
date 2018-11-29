// nnet3/nnet-optimize.h

// Copyright      2015-2016  Johns Hopkins University (author: Daniel Povey)
//                2015       Xiaohui Zhang

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
#include "nnet3/nnet-optimize-utils.h"

namespace kaldi {
namespace nnet3 {

// Options class for optimizing a NnetComputation.  The main projected use for
// this is in debugging the optimization code itself, so that if an error is
// detected, we can work out which optimization was responsible for the error.
// See the Register() function below for option-specific documentation.
struct NnetOptimizeOptions {
  // Caution: if adding or removing members, the Read and Write functions and
  // the == operator should be modified.  This relates to computation caching.
  bool optimize;  // setting this false disallow all optimization.
  bool consolidate_model_update;
  bool propagate_in_place;
  bool backprop_in_place;
  bool optimize_row_ops;
  bool split_row_ops;
  bool extend_matrices;
  bool convert_addition;
  bool remove_assignments;
  bool allow_left_merge;
  bool allow_right_merge;
  bool initialize_undefined;
  bool move_sizing_commands;
  bool allocate_from_other;
  int32 min_deriv_time;
  int32 max_deriv_time;
  int32 max_deriv_time_relative;
  bool snip_row_ops;
  int32 memory_compression_level;
  // optimize_looped_computation is a 'hidden config' not available from
  // the command line; it's set to true to enable the optimization for
  // looped computation that turns a linear computation into a loop.
  bool optimize_looped_computation;

  NnetOptimizeOptions():
      optimize(true),
      consolidate_model_update(true),
      propagate_in_place(true),
      backprop_in_place(true),
      optimize_row_ops(true),
      split_row_ops(true),
      extend_matrices(true),
      convert_addition(true),
      remove_assignments(true),
      allow_left_merge(true),
      allow_right_merge(true),
      initialize_undefined(true),
      move_sizing_commands(true),
      allocate_from_other(true),
      min_deriv_time(std::numeric_limits<int32>::min()),
      max_deriv_time(std::numeric_limits<int32>::max()),
      max_deriv_time_relative(std::numeric_limits<int32>::max()),
      snip_row_ops(true),
      memory_compression_level(1),
      optimize_looped_computation(false) { }

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
    opts->Register("extend-matrices", &extend_matrices, "This optimization "
                   "can reduce memory requirements for TDNNs when applied "
                   "together with --convert-addition=true");
    opts->Register("optimize-row-ops", &optimize_row_ops, "Set to false to "
                   "disable certain optimizations that act on operations of "
                   "type *Row*.");
    opts->Register("split-row-ops", &split_row_ops, "Set to false to disable "
                   "an optimization that may replace some operations of type "
                   "kCopyRowsMulti or kAddRowsMulti with up to two simpler "
                   "operations.");
    opts->Register("convert-addition", &convert_addition, "Set to false to "
                   "disable the optimization that converts Add commands into "
                   "Copy commands wherever possible.");
    opts->Register("remove-assignments", &remove_assignments, "Set to false to "
                   "disable optimization that removes redundant assignments");
    opts->Register("allow-left-merge", &allow_left_merge, "Set to false to "
                   "disable left-merging of variables in remove-assignments "
                   "(obscure option)");
    opts->Register("allow-right-merge", &allow_right_merge, "Set to false to "
                   "disable right-merging of variables in remove-assignments "
                   "(obscure option)");
    opts->Register("initialize-undefined", &initialize_undefined, "Set to false "
                   "to disable optimization that avoids redundant zeroing");
    opts->Register("move-sizing-commands", &move_sizing_commands, "Set to false "
                   "to disable optimization that moves matrix allocation and "
                   "deallocation commands to conserve memory.");
    opts->Register("allocate-from-other", &allocate_from_other, "Instead of "
                   "deleting a matrix of a given size and then allocating "
                   "a matrix of the same size, allow re-use of that memory");
    opts->Register("min-deriv-time", &min_deriv_time, "You can set this to "
                   "the minimum t value that you want derivatives to be computed "
                   "at when updating the model.  This is an optimization that "
                   "saves time in the backprop phase for recurrent frameworks");
    opts->Register("max-deriv-time", &max_deriv_time, "You can set this to "
                   "the maximum t value that you want derivatives to be computed "
                   "at when updating the model.  This is an optimization that "
                   "saves time in the backprop phase for recurrent frameworks");
    opts->Register("max-deriv-time-relative", &max_deriv_time_relative,
                   "An alternative mechanism for setting the --max-deriv-time, "
                   "suitable for situations where the length of the egs is "
                   "variable.  If set, it is equivalent to setting the "
                   "--max-deriv-time to this value plus the largest 't' value "
                   "in any 'output' node of the computation request.");
    opts->Register("snip-row-ops", &snip_row_ops, "Set this to false to "
                   "disable an optimization that reduces the size of certain "
                   "per-row operations");
    opts->Register("memory-compression-level", &memory_compression_level,
                   "This is only relevant to training, not decoding.  Set this "
                   "to 0,1,2; higher levels are more aggressive at reducing "
                   "memory by compressing quantities needed for backprop, "
                   "potentially at the expense of speed and the accuracy "
                   "of derivatives.  0 means no compression at all; 1 means "
                   "compression that shouldn't affect results at all.");

  }
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;
  bool operator == (const NnetOptimizeOptions &other) const;
};


/* This utility function, used in code that calls LimitDerivativeTimes() (and
   required in code that calls Optimize(), returns the largest time
   't' in any of the 'outputs' in the computation request, or crashes if there
   are no outputs (or no cindexes in those outputs). */
int32 MaxOutputTimeInRequest(const ComputationRequest &request);


/** This is the top-level function for optimizing a computation.  Note: it
    should really be called OptimizeAndPostprocess(), because there is at least
    one thing it does (reordering I/O commands) that is necessary for a
    computation to be run.

    @param [in] config   The options that control, among other things,
                         which optimizations to apply.
    @param [in] nnet     The neural net for which the computation is being built
    @param [in] max_output_time_in_request  This value is only needed when the
                         max-deriv-time-relative config value is set in
                         'config'.  It should be set to the largest 't' value
                         encountered in any of the indexes in the 'output'
                         IoSpecifications in the ComputationRequests used to
                         compile the computation.  However if there are multiple
                         ComputationRequests (i.e. it was an online computation)
                         you can just set it to any value you want, because
                         backpropagation is not supported so the
                         max-deriv-time-relative configuration value would not
                         have any effect.
    @param [in,out] computation  The computation to be optimized; this function
                         modifies it in-place.
 */
void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              int32 max_output_time_in_request,
              NnetComputation *computation);



struct CachingOptimizingCompilerOptions {
  bool use_shortcut;
  int32 cache_capacity;

  CachingOptimizingCompilerOptions():
      use_shortcut(true),
      cache_capacity(64) { }

  void Register(OptionsItf *opts) {
    opts->Register("use-shortcut", &use_shortcut,
                   "If true, use the 'shortcut' in compilation whereby "
                   "computation requests with regular structure are identified "
                   "as such, a computation with a smaller number of distinct "
                   "values of 'n' is compiled (e.g. 2), and the compiled "
                   "computation is expanded to match the size of the real "
                   "computation request.");
    opts->Register("cache-capacity", &cache_capacity,
                   "Determines how many computations the computation-cache will "
                   "store (most-recently-used).");
  }
};

/// This class enables you to do the compilation and optimization in one call,
/// and also ensures that if the ComputationRequest is identical to the previous
/// one, the compilation process is not repeated.
/// It is safe to call Compile() from multiple parallel threads without additional
/// synchronization; synchronization is managed internally by class ComputationCache.
class CachingOptimizingCompiler {
 public:
  CachingOptimizingCompiler(const Nnet &nnet,
                            const CachingOptimizingCompilerOptions config =
                            CachingOptimizingCompilerOptions());

  /// Note: nnet is retained as a const reference but opt_config is copied.
  CachingOptimizingCompiler(const Nnet &nnet,
                            const NnetOptimizeOptions &opt_config,
                            const CachingOptimizingCompilerOptions config =
                            CachingOptimizingCompilerOptions());

  ~CachingOptimizingCompiler();

  /// Does the compilation and returns a const pointer to the result, which is
  /// owned by this class, not the caller.  It calls ComputeCudaIndexes() for
  /// you, because you wouldn't be able to do this on a const object.
  ///
  /// Note: this used to return 'const NnetComputation*'.  If you get a
  /// compilation failure, just replace 'const NnetComputation*' with
  /// 'std::shared_ptr<const NnetComputation>' in the calling code.
  std::shared_ptr<const NnetComputation> Compile(
      const ComputationRequest &request);
  void ReadCache(std::istream &is, bool binary);
  void WriteCache(std::ostream &os, bool binary);


  // GetSimpleNnetContext() is equivalent to calling:
  // ComputeSimpleNnetContext(nnet_, &nnet_left_context,
  //                          &nnet_right_context)
  // but it caches it inside this class.  This functionality is independent of
  // the rest of the functionality of this class; it just happens to be a
  // convenient place to put this mechanism.
  void GetSimpleNnetContext(int32 *nnet_left_context,
                            int32 *nnet_right_context);

 private:

  // This function just implements the work of Compile(); it's made a separate
  // function for the convenience of the timer code, to avoid it being called
  // twice (we also call this function directly from inside the class).
  std::shared_ptr<const NnetComputation> CompileInternal(const ComputationRequest &request);

  // This function, called from CompileInternal(), is called when a
  // ComputationRequest has been determined not to have already been cached.  It
  // otherwise has the same interface as CompileInternal(), but assumes that
  // there is nothing cached for this computation as yet.  It compiles the
  // computation and takes care of caching it.
  std::shared_ptr<const NnetComputation> CompileAndCache(const ComputationRequest &request);


  // This function, called from CompileInternal(), tries to compile the
  // ComputationRequest 'request' via 'shortcut' compilation; if this is
  // possible, it returns a pointer to a newly allocated computation that it has
  // compiled this way (note: this computation will not yet have been placed in
  // the computation cache).  If this is not possible for some reason
  // (e.g. shortcut compilation is disabled in the config; or the computation
  // request was not decomposable because of too few n values or irregular or
  // unexpected structure), this function returns NULL and you should compile
  // via CompileNoShortcut.
  const NnetComputation *CompileViaShortcut(const ComputationRequest &request);

  // This function, called from CompileInternal(), tries to compile the
  // ComputationRequest 'request' via the regular (not shortcut) compilation
  // process; it returns a pointer to a newly allocated computation that it has
  // compiled this way (note: this computation will not yet have been placed in
  // the computation cache).
  const NnetComputation *CompileNoShortcut(const ComputationRequest &request);

  const Nnet &nnet_;
  CachingOptimizingCompilerOptions config_;
  NnetOptimizeOptions opt_config_;


  // seconds spent in various phases of compilation-- for diagnostic messages
  double seconds_taken_total_;
  double seconds_taken_compile_;
  double seconds_taken_optimize_;
  double seconds_taken_expand_;
  double seconds_taken_check_;
  double seconds_taken_indexes_;
  double seconds_taken_io_;

  ComputationCache cache_;

  // These following two variables are only used by the function GetSimpleNnetContext().
  int32 nnet_left_context_;
  int32 nnet_right_context_;
};


/// This optimization, which has no effect unless you set --min-deriv-time or
/// --max-deriv-time, modifies the backprop operations for efficiency based on
/// the assumption that derivatives for any Cindex with t < min_deriv_time or t
/// > max_deriv_time are zero.  (this is based on the fact that derivatives in
/// recurrent setups will either decay to zero over time, or will explode and
/// anyway become meaningless).  This is only applied if you are not comoputing
/// any input-derivatives).  The assumption, for simple Components, is that
/// backprop operations are no-ops as long as the input was zeroed, because the
/// back-propagated derivatives would be zero and the model would not be
/// updated.
///
/// The most important effect of this operation is to modify some operations of
/// type kBackprop and kBackpropNoModelUpdate for simple Components, to either
/// make them operate on row ranges of their original input (which in general
/// will be newly created submatrices), or to remove them altogether if they do
/// not operate on any 't' values within the correct range.
///
/// We assert as a requirement of this optimization that all allocation commands
/// must zero their matrices (this effectively means that you cannot apply this
/// optimization after RemoveUnnecessaryZeroing()).  This means that we don't
/// have to worry about leaving things undefined after removing backprop
/// operations.  We also assert that backprop commands that set instead of
/// adding to their input, must not be outputting to things that were
/// previously set to nonzero values.   (this shouldn't ever be a problem, but
/// we do check.
///
/// Note: after this optimization it will likely be beneficial to call
/// RemoveUnnecessaryOperations to remove operations not of type kBackprop that have
/// now become unnecessary-- e.g. operations that do the backprop through
/// Descriptors.
void LimitDerivativeTimes(const Nnet &nnet,
                          const ComputationRequest &request,
                          const NnetOptimizeOptions &opts,
                          NnetComputation *computation);

/// This consolidates the model-update parts of the backprop into larger
/// operations (applicable mostly to recurrent setups)-- internally it uses
/// class ModelUpdateConsolidator.  Will fail if called a
/// second time.
void ConsolidateModelUpdate(const Nnet &nnet,
                            NnetComputation *computation);

/// This converts addition operations (things with Add in their names) to
/// copy operations (things with Copy in their names).  This is slightly
/// more efficient, and it may later allow us to remove unnecessary zeroing.
void ConvertAdditionToAssignment(const Nnet &nnet,
                                 NnetComputation *computation);


/// This wraps class VariableMergingOptimizer in a simplified interface.
void VariableMergingOptimization(const NnetOptimizeOptions &config,
                                 const Nnet &nnet,
                                 NnetComputation *computation);


/// This optimization function removes, where possible, commands of type
/// type kSetConst.  (It can remove them where subsequent commands are
/// going to set the matrix without reading its previous value).
void RemoveUnnecessaryZeroing(const Nnet &nnet, NnetComputation *computation);


/// This optimization moves commands that allocate and zero matrices to as late as
/// possible, and moves commands that deallocate matrices to as early as possible.
void MoveSizingCommands(const Nnet &nnet, NnetComputation *computation);

/// This optimization detects cases where we deallocate a matrix, and then
/// later allocate another matrix of the same size; and replaces them
/// with commands of type kAllocFromOther or kAllocFromOtherZeroed.
void RemoveUnnecessaryAllocation(const Nnet &nnet,
                                 NnetComputation *computation);


/// This optimization puts the input operations (kAcceptInput) and output
/// operations (kProvideOutput) at the very beginning or end of segments of
/// computation, respectively.
///
/// This is actually necessary for computations to be run easily, because if these
/// commands were interspersed with the regular commands, you'd have to
/// call computer.Run() between the individual AcceptInput() and GetOutput()
/// function calls.
void ConsolidateIoOperations(const Nnet &nnet,
                             NnetComputation *computation);



} // namespace nnet3
} // namespace kaldi


#endif
