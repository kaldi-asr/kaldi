// nnet3/nnet-optimize-utils.h

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

#ifndef KALDI_NNET3_NNET_OPTIMIZE_UTILS_H_
#define KALDI_NNET3_NNET_OPTIMIZE_UTILS_H_

#include <mutex>
#include <list>
#include "nnet3/nnet-compile.h"
#include "nnet3/nnet-analyze.h"


namespace kaldi {
namespace nnet3 {


struct NnetOptimizeOptions;  // Forward declaration.

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
   merging may seem to be essentially equivalent, but they they are not because
   in general s1 and s2 may be sub-matrices of larger matrices.

   Note the following definitions:
     - Define last-access(submatrix) as the
       last command that accesses that submatrix for either read or write.  [note:
       deallocation does not count as a read or write operation].
     - Define first-nontrivial-access(submatrix) as the first command
       other than zeroing (kSetConst with 0.0) that accessed that submatrix for
       either read or write.
     - Define last-write-access(submatrix) as the last command-index that accessed
       the submatrix in a write operation, or -1 if there is no such command (this
       could happen for inputs).
     - Define data-invalidated-command(c, submatrix) as the first
       command-index after 'c' that 'submatrix' is written to; or if there is
       no such command, then the command index of the deallocation command
       for 'submatrix'; or if this does not exist, then num-commands.

   The conditions that must be satisfied for merges are as follows:
     - Condition c1: it cannot be the case that m1 and m2 are both inputs, or
       that they are both outputs.
     - Condition c2: If either m1 or m2 is an input or an output, then s1 must
       be the entirety of m1 and s2 must be the entirety of m2 (this is because
       inputs and outputs must be whole matrices).
     - Condition c3: if we are left-merging (deleting s2,m2), then s2 must be the
       entirety of m2.
     - Condition c4: If we are right-merging (deleting s1,m1), then s1 must be
       the entirety of m1.
     - Condition c5: None of the the variables underlying s1 and s2 may be
       marked as 'dirty' (implying that they were the subjects of a previous
       merge during the lifetime of this class).
     - Condition c6: if we are left-merging (deleting s2, m2) and m2 has
       stride_type == kStrideEqualNumCols, then s1 must be the entirety of m1.
       [note: because of condition c3, we can assume that s2 is the entirety of
       m2.]
     - Condition c7: if we are right-merging (deleting s1, m1) and m1 has
       stride_type == kStrideEqualNumCols, then s2 must be the entirety of m2.
       [note: because of condition c4, we can assume that s1 is the entirety of
       m1.]

   If the command C is case (a), i.e. an assignment operation, then the following
   conditions must apply:
     - first-nontrivial-access(s2) == C
     - last-write-access(s1) < C
     - last-access(s1) < data-invalidated-command(C, s2)
   Otherwise (cases (b) and (c), in-place propagate or backprop), we insist that:
     - first-nontrivial-access(s2) == C
     - last-access(s1) == C
   Note: in either case, these conditions imply that m2/s2 is not an input and m1/s1 is
   not an output.  [i.e. s1 *may* be an input and s2 *may* be an output].

   We can explain the procedure for both left-merge and right-merge in one, because
   it's the same.  Define s_to_keep and m_to_keep as s1 and m1 if we're left-merging
   and s2 and m2 if we're right-merging, and s_to_discard and m_to_discard the opposite
   way.

   The procedure to merge in general is as follows:

     - All submatrices that reference m1, make them reference m2 instead.
       [later we'll renumber so that there are no duplicates.]  This automatically
       takes care of making the input and output and allocation/deallocation
       commands refer to the right matrix, in most cases.
     - We need to get rid of duplicate or unnecessary allocation commands:
       If m_to_discard is an input then get rid of the allocation command for
       m_to_keep; otherwise get rid of the allocation command of m_to_discard.
     - We need to get rid of duplicate or unnecessary deallocation commands:
       If m_to_discard is an output then get rid of the deallocation command
       for m_to_keep; otherwise get rid of the deallocation command for
       m_to_discard.

   At the end when we call RemoveOrphanMatrices(), the renumbering code will
   automatically detect that there are duplicate submatrices, and will merge
   them, as well as removing the now-unused matrix indexes.  After merging, we
   will mark the variables (i.e. row-ranges) underlying s1 and s2 as being
   "dirty" so they can no longer be merged during the lifetime of this class--
   this is so we don't have to think to hard; we apply this optimization
   multiple times until it makes no change (see
   nnet-optimize.cc:VariableMerginOptimization()).
 */
class VariableMergingOptimizer {
 public:
  VariableMergingOptimizer(const NnetOptimizeOptions &config,
                           const Nnet &nnet,
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

  // Merges to matrices, whether left merge or right merge.  s_to_keep and
  // s_to_discard are the submatrix-indexes we will keep and discard
  // respectively (these are s1 and s2 in some order.
  void DoMerge(int32 command_index, int32 s_to_keep, int32 m_to_discard);

  /// Marks the variables underlying submatrix 's' as dirty
  void MarkAsDirty(int32 s);

  void Initialize();

  const NnetOptimizeOptions &config_;
  const Nnet &nnet_;
  NnetComputation *computation_;

  Analyzer analyzer_;

  // lists of submatrices that correspond to each matrix.
  std::vector<std::vector<int32> > matrix_to_submatrix_;

  // for each variable (as defined by analyzer_.variables), true if
  // we have already performed a merge on it.
  std::vector<bool> variable_dirty_;

  bool already_called_merge_variables_;
};

/**
   This is not really an optimization in itself but it can make things easier
   for class VariableMergingOptimizer (usually called by its wrapper
   VariableMergingOptimization()).  It looks for a case where most of a matrix
   (but not its final rows) are copied to some submatrix of another matrix,
   where the row-range of that submatrix extends to the last row of the other
   matrix; and it extends the other matrix with additional rows so that the
   entire source matrix can be copied to the destination.
 */
void ExtendMatrices(NnetComputation *computation);


/**
   This optimization consolidates
   the model-update part of
   backprop commands, for components in (e.g.) recurrent networks that need to
   have many separate backprop commands, into more efficient single commands
   operating on consolidated data in larger matrices.  This is useful for
   recurrent networks.  The resulting computation separates the backprop for
   data-derivatives from the model-update part of backprop.
 */
void ConsolidateModelUpdate(const Nnet &nnet,
                            NnetComputation *computation);




// Class DerivativeTimeLimiter is used inside LimitDerivativeTimes().
// Its function is to modify the computation so that we don't work
// with derivatives outside of a specified range of t values; this is
// useful, for instance, in BLSTMs where you might have a fair amount of
// left and right context in the training examples but don't want to
// propagate the derivatives to there.
//
// We require that the computation have debug info set up
// (!matrix_debug_info.empty()) and that this be the first
// optimization you perform.  This means that the debug_info will
// be accurate and that all matrices will be initialized with
// zero contents.
class DerivativeTimeLimiter {
 public:
  DerivativeTimeLimiter(const Nnet &nnet,
                        int32 min_deriv_time,
                        int32 max_deriv_time,
                        NnetComputation *computation);

  void LimitDerivTimes();

 private:

  // sets up matrix_prune_info_.
  void ComputeMatrixPruneInfo();

  // sets up subatrix_map_ and submatrix_map_if_deriv_.
  void ComputeSubmatrixMaps();

  // modifies all the commands as appropriate to reflect that some derivative
  // values are zero (i.e. save any computation we can, based on this
  // assumption).
  void ModifyCommands();

  // this function, called after we've modified the commands to operate on
  // submatrices of the original matrices, works out for which of the matrices
  // we can actually limit their extent in time, and changes the way the
  // matrices are allocated (it may remove some matrices entirely).
  void PruneMatrices();

  // this function modifies commands of type kPropagate to set the memo indexes
  // to zero if the memo indexes appear in the list memos_to_delete_.  It's
  // because if a backprop command has been deleted, the propagate command
  // should no longer store a memo.
  void RemoveUnusedMemos();


  // called from PruneMatrices only for matrices that are derivatives,
  // not inputs or outputs of the computation, and which are partly
  // inside the time range, this function returns true if we can
  // limit the size of the matrix (because variables outside the
  // desired range are never accessed), and false otherwise.
  inline bool CanLimitMatrix(const Analyzer &analyzer,
                             int32 matrix_index) const;

  // called from PruneMatrices after it has figured out which matrices we need
  // to limit to a row-range, this function changes computation->submatrices and
  // computation->matrices in the way required to do that.
  inline void LimitMatrices(const std::vector<bool> &will_limit);

  // does the processing for a command of type kMatrixCopy or kMatrixAdd.
  void MapSimpleMatrixCommand(NnetComputation::Command *c);

  // does the processing for a command of type kCopyRows or kAddRows, where
  // 1st and 2nd args are submatrix indexes and the 3rd arg is a vector of
  // row-indexes.
  void MapIndexesCommand(NnetComputation::Command *c);

  // does the processing for a command of type kAddRowsMulti, kAddToRowsMulti,
  // kCopyRowsMulti or kCopyToRowsMulti, 1st arg is submatrix index that the
  // command is called with, and 2nd arg is 'indexes_multi' index (which
  // contains pairs (source-submatrix, source-row).
  void MapIndexesMultiCommand(NnetComputation::Command *c);

  // does the processing for a command of type kAddRowRanges.
  void MapAddRowRangesCommand(NnetComputation::Command *c);

  // Modifies this command to take into account prune_info_.  At this point we
  // don't actually reduce the size of the matrices, we simply make the commands
  // operate on submatrices of the original matrices where possible- or
  // delete them completely if their output is all zeros or for other reasons
  // we detect that they would be no-ops.
  // Note: this calls computation_->NewSubMatrix, and will generate duplicates
  // of the same submatrix which we'll later remove in RemoveOrphanMatrices.
  void ModifyCommand(NnetComputation::Command *command);

  // this will detect which matrices we can reduce the allocated size of,
  // and reduce their size.
  void ResizeMatrices();

  // Requires that we have mapped 'initial_submatrix' to 'new_submatrix' in
  // an operation that may have removed some data on the left and/or the
  // right (but still they point to the same underlying matrix).  Outputs
  // to 'left_prune' and 'right_prune' the number of rows we have
  // removed on the left and on the right respectively.
  inline void GetPruneValues(int32 initial_submatrix,
                             int32 new_submatrix,
                             int32 *left_prune,
                             int32 *right_prune) const;

  // This helper function, used while mapping commands, returns true if the
  // Cindex represented by the pair (submatrix, row_index) has a 't' value
  // within the range [min_deriv_time_, max_deriv_time_].
  bool RowIsKept(int32 submatrix,
                 int32 row_index) const;


  struct MatrixPruneInfo {
    bool is_deriv;  // true if the matrix represents a derivative (copied from
                    // the debug-info; repeated here for convenience).
    bool fully_inside_range;  // True if the matrix is completely inside the time range
                             // specified.
    bool partly_inside_range;  // true if the matrix is partly (but not fully)
                               // inside the time range specified.
    int32 row_begin;  // if partly_inside_range, the first row that's within the time range (i.e. for which
                      // min_deriv_time_ <= t < max_deriv_time_.
    int32 row_end;    // if partly_inside_range, one plus the last row that's within
                      // the time range.
  };


  const Nnet &nnet_;

  int32 min_deriv_time_;
  int32 max_deriv_time_;

  // the computation; we require it to have debug info set up
  // (otherwise you shouldn't be instantiating this class).
  NnetComputation *computation_;

  // for each matrix index > 0, the index of a submatrix that consists of
  // the entirety of that matrix.
  std::vector<int32> whole_submatrices_;

  std::vector<MatrixPruneInfo> matrix_prune_info_;

  // for each submatrix in the original range of computation_->submatrices,
  // submatrix_map_ maps it to itself if the submatrix is completely inside the
  // time-range, or to zero if it's completely outside the time-range, or to a
  // newly created submatrix-index if it's partly inside the time-range.
  std::vector<int32> submatrix_map_;

  // submatrix_map_if_deriv_ contains the quantity:
  // IsDerivative(s) ? submatrix_map_[s] : s,
  // where IsDerivative(s) is true if s is part of a matrix that (according to its
  // debug info) represents a derivative.
  // this comes up so frequently that storing it separately seemed like a good idea.
  std::vector<int32> submatrix_map_if_deriv_;

  std::vector<MatrixPruneInfo> prune_info_;

  // List of indexes of memos that will no longer be stored because the backprop
  // commands using them were deleted.
  std::unordered_set<int32> memos_to_delete_;
};


// This utility function, used in code that calls LimitDerivativeTimes(), returns
// the largest time 't' in any of the 'outputs' in the computation request,
// or crashes if there are no outputs (or no cindexes in those outputs).
int32 MaxOutputTimeInRequest(const ComputationRequest &request);

// This is the top-level interface to limit the times on which derivatives are
// computed (e.g. for truncated BPTT); internally it uses class
// DerivativeLimiter.  Will do nothing if min_deriv_time and max_deriv_time are
// their default -inf,+inf values.
void LimitDerivativeTimes(const Nnet &nnet,
                          int32 min_deriv_time,
                          int32 max_deriv_time,
                          NnetComputation *computation);

/**  This function, used in 'shortcut' compilation where we first compile a
     smaller computation with the same structure but only 2 distinct 'n'
     values, works out whether a computation is 'decomposable'; if so,
     it returns true and outputs the 'mini_request' with the same structure,
     and the number of 'n' values.

     A computation is decomposable if the following conditions hold:

      - All of its inputs and outputs contain 'n' values for all 0 <= n < N,
        for some N > 2.  [we output this 'N' as 'num_n_values'].
      - All of its inputs and outputs have 'regular' structure: chiefly, that
        within vectors of Indexes, each (t, x) pair should be present for the
        same set of 'n' values (0, 1, ... N-1), and that we should be able to
        identify the stride of the 'n' index.  For more precise details on this
        regular structure, look at the comment for the function FindNStride(),
        in nnet-optimize-utils.cc.
 */
bool RequestIsDecomposable(const ComputationRequest &request,
                           ComputationRequest *mini_request,
                           int32 *num_n_values);


/**
  This function is used in 'shortcut' compilation to expand a computation
  that has been compiled for exactly 2 'n' values, to one that is suitable
  for some num_n_values > 2.
     @param [in] nnet         The neural network for which this computation
                              is being built.
     @param [in] misc_info    The same MiscComputationInfo object that was
                              present in the ComputationRequests that were
                              originally used to generate the computation
                              (required to generated the PrecomputedIndexes)
     @param [in] computation  The computation that was compiled for exactly
                              2 'n' values (n=0 and n=1)
     @param [in] need_debug_info True if we want to retain the 'debug_info'
                              in the output 'expanded_computation'.  In any
                              case, the 'debug_info' is required in the
                              input computation.
     @param [in] num_n_values The number of 'n' values we want in the output
                              computation
     @param [out] expanded_computation  The expanded computation.

 */
void ExpandComputation(const Nnet &nnet,
                       const MiscComputationInfo &misc_info,
                       const NnetComputation &computation,
                       bool need_debug_info,
                       int32 num_n_values,
                       NnetComputation *expanded_computation);



/// This function detects cases where commands of type kCopyRows, kAddRows or
/// kAddToRows can be converted to commands of type kMatrixCopy or kMatrixAdd,
/// and converts them (this may involve adding submatrices).
///
/// This function returns true if it made any changes to the computation; if it
/// returns true, then after doing this you should at some point do
/// RenumberComputation(), which will remove any now-unused members of
/// computation->indexes.
bool ReplaceRowWithMatrixOps(NnetComputation *computation);

/// This function detects cases where commands of type kCopyRows, kAddRows,
/// kAddRowsMulti, kAddToRowsMulti, kCopyRowsMulti, kCopyToRowsMulti or
/// kAddRowRanges use indexes that start or end with -1's or equivalents,
/// and replace them with similar commands that act on a sub-matrix of the
/// matrices they are currently acting on.  This will help efficiency by
/// avoiding launching unnecessary copies of the kernel (that don't really
/// have to do anything).
///
/// This function returns true if it made any changes to the computation; if it
/// returns true, then after doing this you should at some point do
/// RenumberComputation(), which will remove any now-unused members of
/// computation->indexes.
bool SnipRowOps(NnetComputation *computation);


/// This function detects cases where commands of type kAddRowsMulti,
/// kAddToRowsMulti, kCopyRowsMulti, kCopyToRowsMulti use indexes that
/// correspond to at most two submatrices, in two distinct ranges without gaps
/// filled by -1's, and could be converted to at most two commands of type
/// kMatrixAdd, kMatrixCopy, kAddRows or kCopyRows.  (Note: it's important that
/// this optimization takes place after SnipRowOps, because it doesn't remove
/// the -1's from the edges of the indexes, it relies on that operation doing
/// so).  The "without-gaps" stipulation is just for convenience of
/// implementation, to have fewer cases to worry about.
///
/// This function returns true if it made any changes to the computation; if it
/// returns true, then after calling this you should at some point do
/// RenumberComputation(), which will remove any now-unused members of
/// computation->indexes.
bool SplitRowOps(NnetComputation *computation);

/// This function detects submatrices and matrices that are never used (e.g. due
/// to changes made in other optimization code), and members of indexes,
/// indexes_multi and indexes_ranges that are unused or are duplicates, and memo
/// indexes that are unused; and it removes them from the computation by way of
/// suitable renumbering.  It does not remove no-ops from
/// computation->commands_; to do that, call RemoveNoOps(computation).
void RenumberComputation(NnetComputation *computation);


/// Removes commands of type kNoOperation in the computation.
void RemoveNoOps(NnetComputation *computation);

/// This function outputs to "submatrix_args" the addresses of a subset of
/// arguments arg1 through arg6 in "command", that correspond to the indexes of
/// submatrices.  This is useful in renumbering code.  Note: some of the
/// pointers may point to a zero value, for optional submatrix args.
void IdentifySubmatrixArgs(NnetComputation::Command *command,
                           std::vector<int32*> *submatrix_args);

/// This function returns true if matrix 1 <= m < computation->matrices.size()
/// is unused, defined as: it is not an input or an output, and is not
/// accessed other than via commands of type kAllocMatrix, kDeallocMatrix, and
/// kSetConst.
bool MatrixIsUnused(const Analyzer &analyzer,
                    const NnetComputation &computation,
                    int32 m);

/// This function removes from 'computation' the commands accessing matrix 'm',
/// which is assumed to be unused according to the MatrixIsUnused() command
/// above.  Specifically, it changes the types of the relevant commands in
/// 'computation' to kNoOperation.  (The commands changed in this way will be of
/// type kAllocMatrix, kDeallocMatrix and kSetConst).  The index for the matrix
/// may later be removed entirely by RenumberComputation().
void RemoveCommandsForUnusedMatrix(const Analyzer &analyzer,
                                   int32 m,
                                   NnetComputation *computation);


/// This function outputs to "submatrix_args" the addresses of the args
/// (arguments arg1 through arg6) in the vector "commands", that correspond to
/// the indexes of submatrices.  This is useful in renumbering code.  Note: some
/// of the pointers may point to a zero value, for optional submatrix args.
void IdentifySubmatrixArgs(std::vector<NnetComputation::Command> *commands,
                           std::vector<int32*> *submatrix_args);

/// This function outputs to "submatrix_args" the addresses of integers in
/// 'computation' that correspond to submatrices.  These may be present in
/// 'commands', and in 'indexes_multi'.  This is useful in renumbering code.
/// Note: some of the pointers may point to a zero value, for optional submatrix
/// args in commands, but for efficiency we don't provide pointers for the -1's
/// in 'indexes_multi'.
void IdentifySubmatrixArgsInComputation(NnetComputation *computation,
                                        std::vector<int32*> *submatrix_args);


/// Identifies in the vector of commands, arguments that correspond to indexes
/// into the computation's indexes_multi array, and outputs a list of pointers
/// to those arguments to 'indexes_multi_args'.  Useful in renumbering code.
void IdentifyIndexesMultiArgs(std::vector<NnetComputation::Command> *commands,
                              std::vector<int32*> *indexes_multi_args);

/// Identifies in the vector of commands, arguments that correspond to indexes
/// into the computation's 'indexes' array, and outputs a list of pointers
/// to those arguments to 'indexes_args'.  Useful in renumbering code.
void IdentifyIndexesArgs(std::vector<NnetComputation::Command> *commands,
                         std::vector<int32*> *indexes_args);

/// Identifies in the vector of commands, arguments that correspond to indexes
/// into the computation's 'indexes' array, and outputs a list of pointers
/// to those arguments to 'indexes_args'.  Useful in renumbering code.
void IdentifyIndexesArgs(std::vector<NnetComputation::Command> *commands,
                         std::vector<int32*> *indexes_args);

/// Identifies in the vector of commands, arguments that correspond to indexes
/// into the computation's 'indexes_ranges' array, and outputs a list of pointers
/// to those arguments to 'indexes_ranges_args'.  Useful in renumbering code.
void IdentifyIndexesRangesArgs(std::vector<NnetComputation::Command> *commands,
                               std::vector<int32*> *indexes_ranges_args);

/// Inserts commands into the computation at the requested places.  'commands'
/// is a list of pairs (command-index, command) that is expected to be sorted on
/// command-index.  For each entry (c, command) in 'commands', 'command' is
/// inserted into 'computation' just *before* the command that (at entry) is in
/// computation->commands[c].  If there are multiple pairs with the same index
/// c, they will remain in the same order in which they were present in
/// 'commands'; however, 'commands' does not have to be sorted on 'c'.  As a
/// special case, if c == computation->commands.size(), the corresponding
/// commands are inserted at the beginning of the computation.  This function
/// will appropriately renumber the argument of the kGotoLabel command of any
/// 'looped' computation.  Command indexes c in commands[*].first must be in the
/// range [0, computation->commands.size()].  This function may modify
/// 'commands' by sorting it.
void InsertCommands(
    std::vector<std::pair<int32, NnetComputation::Command> > *commands,
    NnetComputation *computation);

/// Performs optimization to reduce memory usage where possible,
/// making use of the kCompressMatrix and kDecompressMatrix commands.
/// Should only be done after most other optimizations, because some
/// optimizations (such as variable-merging) would not work correctly
/// after doing this optimization.  This does nothing for looped
/// computations.  It's OK, though, to expand a shortcut computation
/// (i.e. call ExpandComputation) after doing this.
///
/// memory_compression_level determines how aggressive the compression
/// is.  Allowed values:
///       0 = no compression at all
///       1 = compression that doesn't affect results (e.g. compress
///           ReLU outputs to 1 byte, as just the sign is needed).
///       2 = compression that may affect the results slightly (e.g. 16-bit
///           compression of the output of NormalizeComponent and the like),
///           but this is not implemented yet, so equivalent to 1.
///       3 = compression that may affect the results more than just
///           slightly.  Not implemented yet, so equivalent to 1.
void OptimizeMemoryCompression(const Nnet &nnet,
                               int32 memory_compression_level,
                               NnetComputation *computation);


/// This function tries to optimize computation 'computation' for an 'looped'
/// computation.  It expects as input a computation with no backprop but with
/// multiple 'segments' separated by command kNoOperationLabel, where each
/// segment corresponds to a new chunk of input and output.  It tries to locate
/// a pair of segment boundaries, with command indexes c1 and c2, where the
/// active matrices have the same debug-info other than a time offset and can be
/// identified with each other, and the no-op command at c2 can be replaced with
/// 'got c1', creating a computation that 'goes on forever'.
/// If it can't do this, it does nothing.  You can figure out that this is the
/// case by checking whether kGotoLabel is the last command in the computation.
/// [If this optimization fails, the whole computation may have to be
/// regenerated with more segments.]
void OptimizeLoopedComputation(const Nnet &nnet,
                               NnetComputation *computation);


/// This function ensures that the arg1 of a final command of type kGotoLabel is
/// the same as the command with type kNoOperationLabel.  This is necessary
/// if you do any other type of optimization after 'OptimizeLoopedComputation()'.
void FixGotoLabel(NnetComputation *computation);


/// Class ComputationCache is used inside class CachingOptimizingCompiler to
/// cache previously computed computations.  The code was moved from class
/// CachingOptimizingCompiler to this separate class for clarity when adding
/// thread-safety functionality.
/// It's OK to call Find() and Insert() from multiple threads without
/// additional synchronization.
class ComputationCache {
 public:
  ComputationCache(int32 cache_capacity);

  // Note: if something fails in Read(), or the written cache was from an older
  // format, it will just leave the cache empty.
  void Read(std::istream &is, bool binary);

  void Write(std::ostream &os, bool binary) const;


  // Searches for the computation corresponding to this computation, and returns
  // it if cached, or NULL (as std::shared_ptr) if not.  (We need shared_ptr to
  // handle multi-threaded operation, so that if the computation is ejected from
  // the cache by another thread, it won't be deleted while still in use).  This
  // function also moves this computation to the end of the
  // most-recently-accessed queue, which is why it's not const.
  std::shared_ptr<const NnetComputation> Find(const ComputationRequest &request);


  // Inserts the computation into the cache-- this is assumed to be the
  // computation for the computation-request 'request'.  Returns a shared_ptr
  // which can be used to access the object.  This function takes ownership of
  // 'computation'.
  std::shared_ptr<const NnetComputation> Insert(const ComputationRequest &request,
                                                const NnetComputation *computation);

  ~ComputationCache();

  // Checks the stored computation for correctness.
  void Check(const Nnet &nnet) const;
 private:

  std::mutex mutex_;  // Read/write mutex.

  int32 cache_capacity_;

  // The access queue for keeping track of the freshness of computation.
  // Most-recently-accessed computation is at the end, and
  // least-recently-accessed computaiton is at the beginning.  Together with
  // computation_cache_, this forms a most-recently-used (MRU) cache for
  // Computations, indexed by ComputationRequest. The pointers are owned in
  // computation_cache_.
  typedef std::list<const ComputationRequest*> AqType;
  AqType access_queue_;

  // Map from computation-request to pair of (computation, and position in
  // access_queue_). Used for fast lookup of previously compiled computations.
  // All pointers are owned here.
  typedef unordered_map<const ComputationRequest*,
                        std::pair<std::shared_ptr<const NnetComputation>, AqType::iterator>,
                        ComputationRequestHasher,
                        ComputationRequestPtrEqual> CacheType;
  CacheType computation_cache_;
};




} // namespace nnet3
} // namespace kaldi


#endif
