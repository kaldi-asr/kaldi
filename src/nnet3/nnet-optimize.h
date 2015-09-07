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
/// The rest of this file contains various things that are
/// called from this, and which you probably won't need to call
/// directly.
void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              const ComputationRequest &request,
              NnetComputation *computation);


/// This class enables you to do the compilation and optimization in one call,
/// and also ensures that if the ComputationRequest is identical to the previous
/// one, the compilation process is not repeated.
class CachingOptimizingCompiler {
 public:
  CachingOptimizingCompiler(const Nnet &nnet): nnet_(nnet) { }

  /// Note: nnet is retained as a const reference but opt_config is copied.
  CachingOptimizingCompiler(const Nnet &nnet,
                            const NnetOptimizeOptions &opt_config):
      nnet_(nnet), opt_config_(opt_config) { }

  /// Does the compilation and returns a const pointer to
  /// the result, which is owned by this class, not the caller.
  /// It calls ComputeCudaIndexes() for you, because you wouldn't
  /// be able to do this on a const object.
  const NnetComputation* Compile(const ComputationRequest  &request);
 private:
  const Nnet &nnet_;
  NnetOptimizeOptions opt_config_;
  ComputationRequest request_;
  NnetComputation computation_;
};


/**
   This class is responsible for merging matrices, although you probably want to
   access it via the the function VariableMergingOptimization().

   We identify pairs of submatrices which can potentially be merged into a single
   submatrix.

   Suppose there are two different submatrices s1 != s2 that are submatrices of
   different respective matrices m1 != m2, and somewhere in the computation we
   have a command C, which is one of:
      (a) the assignment command  "s2 = s1", or
      (b) a propagate command with s1 as input and s2 as output, with a component
          that supports propagate in place, or
      (c) a backprop command with s1 as output-deriv and s2 as input-deriv, with
          a component that supports backprop in place.

   Then the triple (C, s1, s2) is a candidate for merging.  We support two types
   of merging: 'right merging', in which we delete s1 and use s2 instead; and
   'left merging' in which we delete s2 and use s1 instead.  The two types of
   merging may seem to be the same thing, but remember that in general s1 and s2
   may be sub-matrices of larger matrices.

   Note: the following
     - Define last-access(submatrix) as:
       If matrix-of(submatrix) is an output, then num-commands, otherwise the
       last command that accesses that submatrix for either read or write.  [note:
       deallocation does not count as a read or write operation].
     - Define first-access(submatrix) as:
       If matrix-of(submatrix) is an input, then -1, otherwise the first command
       that is *not* an allocation command that accessed that submatrix for either
       read or write.
     - Define last-write-access(submatrix) as the last command-index that accessed
       the submatrix in a write operation, or -1 if there is no such command (this
       could happen for inputs).
     - Define data-invalidated-command(c, submatrix) as the first
       command-index after 'c' that 'submatrix' is written to; or if there is
       no such command, then the command index of the deallocation command
       for 'submatrix'; or if this does not exist, then num-commands.

   The conditions that must be satisfied for both left and right merges:
     - It cannot be the case that m1 and m2 are both inputs, or that they are
       both outputs.  [condition c1]
     - If either m1 or m2 is an input or an output, then s1 must be the entirety
       of m1 and s2 must be the entirety of m2 (this is because inputs and outputs
       must be whole matrices). [condition c2]
     - If we are left-merging (deleting s2,m2), then s2 must be the entirety of m2.
       [condition c3]
     - If we are right-merging (deleting s1,m1), then s1 must be the entirety of m1.
       [condition c4]
     - None of the the variables underlying s1 and s2 may be marked as 'dirty'
       (implying that they were the subjects of a previous merge during the lifetime of
       this class) [condition c5]

   If the command C is case (a), i.e. an assignment operation, then the following
   conditions must apply:
     - first-access(s2) == C
     - last-write-access(s1) < C
     - last-access(s1) < data-invalidated-command(C, s2)
   Otherwise (cases (b) and (c), in-place propagate or backprop), we insist that:
     - first-access(s2) == C
     - last-access(s1) == C


   The sequence of things we have to do for a right-merge (in which we delete
   s1,m1) is as follows:
     - All submatrices that reference m1, make them reference m2 instead.
       [later we'll renumber so that there are no duplicates.]
     - If m1 was an input, replace it as an input with m2 and remove the
       command that allocated m2.
     - If it was an assignment [case (a)], replace the assignment command with a
       no-op.
     - If both m1 and m2 have commands that allocate them, keep only the
       earlier of the two and make it refer to m1 (otherwise delete any
       allocation command, because m1 must be an input); and make sure
       it zeroes the new data (later we can change it to undefined
       initialization, if possible).
     - If both m1 and m2 have commands that deallocate them, keep only the
       later of the two and make it refer to m2 (otherwise delete any
       deallocation command, because m2 must be an output).


   The sequence of things we have to do for a right-merge (in which we delete
   s1,m1) is as follows:
     - All submatrices that reference m2, make them reference m1 instead.
       [later we'll renumber so that there are no duplicates.]
     - If m2 was an output, replace it as an output with m1 and remove the
       command that deallocated m1.
    ... the last three bullet-points, regarding removing the assignment
        command, and allocation and deallocation, are the same as for a
        left-merge.

   At the end when we call RemoveOrphanMatrices(), the renumbering code will
   automatically detect that there are duplicate submatrices, and will merge
   them, as well as removing the now-unused matrix indexes.  After merging, we
   will mark the variables (i.e. row-ranges) underlying s1 and s2 as being
   "dirty" so they can no longer be merged during the lifetime of this class.
 */
class VariableMergingOptimizer {
 public:
  VariableMergingOptimizer(const NnetOptimizeOptions &config,
                           const Nnet &nnet,
                           const ComputationRequest &request,
                           NnetComputation *computation);
  // Note: you can call this only once.  If it returns true, it means it has
  // merged variables.  In this case, you have the option to instantiate another
  // copy of the class and try again with that other copy.
  bool MergeVariables();

 private:
  /// @brief This function returns a pair of bools saying whether we can do a
  ///   (left and/or right) merge respectively, based on the conditions defined
  ///   in the header.
  ///
  /// Note: if one of the variables underlying s1 or s2 is marked as 'dirty' due
  /// to a previous merge, this function will return (false,false).  The terms
  /// left-merge and right-merge are defined in the extended comment above this
  /// class.  Note: left_merge will always be false if config.allow_left_merge
  /// == false, and the same respectively for right_merge.
  ///
  ///  @param command  [in] The command-index that assigns s2 := s1
  ///                        or does a forward or backprop with s1 as the
  ///                        input and s2 as the output
  ///  @param s1   [in]     A submatrix-index s1 > 0.
  ///  @param s2   [in]     A submatrix-index s2 > 0
  std::pair<bool,bool> MayBeMerged(int32 command, int32 s1, int32 s2) const;

  // performs the left merge.  Search for left-merge in the comment
  // above the class declaration for details.
  void DoLeftMerge(int32 command_index, int32 s1, int32 s2);

  // performs the right merge.  Search for right-merge in the comment
  // above the class declaration for details.
  void DoRightMerge(int32 command_index, int32 s1, int32 s2);

  // Performs the actions common to both left and right merges, regarding
  // removing the assignment command, and allocation and deallocation (called
  // from DoLeftMerge and DoRightMerge).  The m_to_keep and m_to_discard
  // are the matrix-indexes we will keep and discard respectively.
  void DoMergeCommon(int32 command_index, int32 m_to_keep,
                     int32 m_to_discard);

  /// Marks the variables underlying submatrix 's' as dirty
  void MarkAsDirty(int32 s);

  void Initialize();

  const NnetOptimizeOptions &config_;
  const Nnet &nnet_;
  const ComputationRequest &request_;
  NnetComputation *computation_;

  Analyzer analyzer_;

  // lists of submatrices that correspond to each matrix.
  std::vector<std::vector<int32> > submatrix_lists_;

  // for each variable (as defined by analyzer_.variables), true if
  // we have already performed a merge on it.
  std::vector<bool> variable_dirty_;

  bool already_called_merge_variables_;
};

/// This wraps class VariableMergingOptimizer in a simplified interface.
void VariableMergingOptimization(const NnetOptimizeOptions &config,
                                 const Nnet &nnet,
                                 const ComputationRequest &request,
                                 NnetComputation *computation);


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
  /// of submatrices taken just prior particular commands.  The input 'commands'
  /// and 'submatrices' lists must be the same size, and size must be > 1.  This
  /// function will create a new matrix that is the row-wise concatentation of
  /// all these submatrices, with values taken just prior to the respective
  /// command indexes.  This function will will add to extra_commands_ the
  /// commands to do the copying at the appropriate places (at the supplied
  /// command indexes; they will be inserted just before).  The return value is
  /// the submatrix index of a submatrix that represents the whole of the
  /// consolidated matrix.  This command will insert, at the beginning of
  /// the computation (in extra_commands_[0]), a command to initialize the matrix;
  /// and will append to final_deallocate_commands_ the commands to deallocate
  /// the matrix.
  /// If computation_->matrix_debug_info is nonempty, this function will
  /// also update computation_->matrix_debug_info with suitable values
  /// for the newly added matrix
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

/// This function detects matrices that have no submatrices corresponding to
/// them (due to changes made in other optimization code), and removes them
/// from the computation.  It also renumbers the submatrix indexes to remove
/// duplicates.
void RemoveOrphanMatrices(NnetComputation *computation);

/// Removes commands of type kNoOperation in the computation.
void RemoveNoOps(NnetComputation *computation);

/// Wherever matrix orig_matrix_index appears in the input of the network
/// (i.e. in computation->input_output_info), replaces it with new_matrix_index.
/// Returns true if it did replace it.
bool ReplaceInInput(
    const Nnet &nnet, int32 orig_matrix_index, int32 new_matrix_index,
    NnetComputation *computation);

/// A helper function used in some optimization functions.
/// Wherever matrix orig_matrix_index appears in the output of the network
/// (i.e. in computation->input_output_info), replaces it with new_matrix_index.
/// Returns true if it did replace it.
bool ReplaceInOutput(
    const Nnet &nnet, int32 orig_matrix_index, int32 new_matrix_index,
    NnetComputation *computation);

/// This function outputs to "submatrix_args" the addresses of a subset of
/// arguments arg1 through arg6 in "command", that correspond to the indexes of
/// submatrices.  This is useful in renumbering code.
void IdentifySubmatrixArgs(NnetComputation::Command *command,
                           std::vector<int32*> *submatrix_args);

/// This function outputs to "matrix_args" the addresses of a subset of the
/// arguments arg1 through arg6 in "command", that correspond to the indexes of
/// matrices.  This is useful in renumbering code.  (Note: only a few types of
/// command use matrix indexes).
void IdentifyMatrixArgs(NnetComputation::Command *command,
                        std::vector<int32*> *matrix_args);






/*
  Things we can do to optimize a computation...

  (1) replacing un-needed inputs to Backprop functions (if used)
      with the empty matrix

  (2) sharing of matrices that would otherwise just be copied.

    If the only input to a submatrix A (apart from zeroing) is copying or adding
    from another sub-matrix B, then

      - if A is a whole matrix we can remove submatrix A and let all references
        to it point to B instead, and remove the copy/add commands.  Otherwise,
      - if B is a whole matrix we can remove submatrix B and let all references
        to it point to A instead, and remove the copy/add commands.

  (3) sharing of matrices that are inputs and outputs of Propagate
     or Backprop functions that support in-place computation.
     If there are submatrices A and B that are also whole matrices,
     then

       - If there is a Propagate operation for which A is the input and B is the
         output, and the component supports in-place propagate, and there is no
         operation after that Propagate that reads A, and there is no operation
         prior to the Propagate that sets B (apart from sizing it and zeroing
         it) then make B point to A and replace all references to B with
         references to A.

       - If there is a Backprop operation for which A is the output-deriv and B
         is the input-deriv (note: Backprop reads A and sets B), and the
         component supports in-place backprop, and there is no operation prior
         to the Backprop that writes to B apart from sizing and zeroing,
         and there is no operation after the Backprop that reads A, then
         make B point to A and replace all references to B with references to
         A.

  (4) optimizations w.r.t. Propagate and Backprop functions that add to (rather
     than set) their output.
       TBD, but the basic idea is that if the output of, say, a Propagate function
      is added to another matrix, and that is the only time it is used,
      then we could just set the output location to that other matrix.

   (5) optimizations w.r.t. avoiding Backprop functions that are not needed.
      Basically, we need to keep track of what the outputs of each Backprop
      function are and whether they are used.  If we are are doing model
      update and this component is updatable then the Backprop function is
      considered to output to the model.  Also, it may output to the
      input-derivative of that component.  We have to keep track of which of
      these input-derivatives are actually used.

   (6) optimizations w.r.t. zeroing matrices.
      This optimization is to avoid unnecessarily zeroing matrices
      when we initialize them.  If the first time a matrix (or all the sub-parts
      thereof) is set, it is set in a copy operation, or in a Propagate or
      Backprop operation that sets (rather than adds to) its output, then
      we can initialize it with kUndefined rather than kZero.

  (7) optimizations for memory consumption.
      The idea here is to move the command to initialize a matrix to just
      before its first use, and to move the command to deinitialize a matrix
      to just after its last use.

  (8) renumbering optimizations.
       - renumber Matrices to get rid of zero-sized, un-needed ones, and a similar thing for Sub-matrices.
       - renumber Computations to get rid of no-ops introduced by earlier optimizations
         [also, modify forward_computation_end].
       - maybe renumber Indexes to get rid of duplicates.

  (9) optimizations to replace row-by-row copy and add commands with whole-matrix
      commands on smaller sub-matrices (if the row-by-row copy commands have certain
      regularities).  this is a minor issue, we can handle it later.  We have to be
      careful if this causes sub-matrices to overlap.

 */





} // namespace nnet3
} // namespace kaldi


#endif

