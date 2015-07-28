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
struct NnetOptimizeConfig {
  bool optimize;  // setting this false disallow all optimization.
  bool propagate_in_place;
  bool backprop_in_place;
  bool remove_assignments;
  bool initialize_undefined;
  bool move_sizing_commands;

  NnetOptimizeConfig(): optimize(true),
                        propagate_in_place(true),
                        backprop_in_place(true),
                        remove_assignments(true),
                        initialize_undefined(true),
                        move_sizing_commands(true) { }
  
  void Register(OptionsItf *po) {
  }
};


/// This is the top-level function for optimizing a computation.
/// The rest of this file contains various things that are
/// called from this, and which you probably won't need to call
/// directly.
void Optimize(const NnetOptimizeConfig &config,
              const Nnet &nnet,
              const ComputationRequest &request,
              NnetComputation *computation);


/// This class enables you to do the compilation and optimization in one call,
/// and also ensures that if the ComputationRequest is identical to the previous
/// one, the compilation process is not repeated.
class CachingOptimizingCompiler {
 public:
  CachingOptimizingCompiler(const Nnet &nnet): nnet_(nnet) { }
  
  CachingOptimizingCompiler(const Nnet &nnet,
                            const NnetOptimizeConfig &opt_config):
      nnet_(nnet), opt_config_(opt_config) { }

  /// Does the compilation and returns a const pointer to
  /// the result, which is owned by this class, not the caller.
  const NnetComputation* Compile(const ComputationRequest  &request);
 private:
  const Nnet &nnet_;
  NnetOptimizeConfig opt_config_;
  ComputationRequest request_;
  NnetComputation computation_;
};


/**
   This class is responsible for merging matrices.  Suppose there are m1 and s1
   on the one hand, where s1 is a submatrix consisting of the whole of m1, and
   s2 which is a submatrix of m2 on the other hand, and somewhere in the
   computation we have a command C, which is one of:
      (a) the assignment command  "s2 = s1", or
      (b) a propagate command with s1 as input and s2 as output, with a component
          that supports propagate in place, or
      (c) a backprop command with s1 as output-deriv and s2 as input-deriv, with
          a component that supports backprop in place.
   Suppose also that:
     - m1 is not an output (or it's case (a))
     - m1 is not an input, or s2 is the whole of m2.
     - after command C, s1 is never accessed [apart from deallocating its matrix]
       (or it's case (a) and s1 is never written to after command C).
     - before command C, s2 is never accessed, apart from initializing it and possibly
       zeroing it
     - m2 is not an input.
   Of course the matrices will have the same size.

   We merge the variables as follows:
     - All submatrices that reference m1, make them reference m2 instead.
       [later we'll renumber so that there are no duplicates.]
     - If m1 was an input, replace it as an input with m2.
     - If it was case (a), replace the assignment command with a no-op.
     - If both m1 and m2 have commands that deallocate them, keep only the
       later of the two and make it refer to m2 (otherwise delete any
       deallocation command).

    At the end when we call RemoveOrphanMatrices(), renumbering code will
    automatically detect that there are duplicate submatrices, and will merge
    them, as well as removing the now-unused matrix indexes.
 */
class VariableMergingOptimizer {
 public:
  VariableMergingOptimizer(const NnetOptimizeConfig &config,
                           const Nnet &nnet,
                           const ComputationRequest &request,
                           NnetComputation *computation);
  // Note: you can call this only once.  If it returns true, it means it has
  // merged variables.  In this case, you have the option to instantiate another
  // copy of the class and try again with that other copy.
  bool MergeVariables();

 private:
  // this function, called while testing whether the pair (s1,s2) is a candidate
  // for optimization, returns true if all the following conditions hold:
  //   - s1 != s2
  //   - s1 and s2 correspond to the whole of their corresponding matrices m1 and m2.
  //   - neither matrix_already_optimized_[m1] nor matrix_already_optimized_[m2] is true
  //   - m1 is not an output of the computation (or command command_index is an
  //     assignment).
  //   - m2 is not an input of the computation
  //   - after command "command_index", no part of m1 is ever accessed [apart
  //     from deallocating it] (or command "command_index" is an assignment and
  //     no part of m1 is written to after command "command_index"
  //   - before command C, no part of m2 is never accessed, apart from
  //     initializing it and possibly zeroing it.
  bool IsCandidate(int32 command_index, int32 s1, int32 s2) const;
  
  // performs the merge.
  // compute m1,m2 from s1,s2.
  //  - All submatrices that reference m2, make them reference m1 instead.
  //   [later we'll renumber so that there are no duplicates.]
  //  - If m1 was an input, replace it as an input with m2 and remove
  //    the command that initializes m2.
  //  - If it was case (a), replace the assignment command with a no-op.
  //  - If both m2 and m1 have commands that deallocate them, keep only the
  //    later of the two and make it refer to m1 (otherwise delete any
  //    deallocation command).
  //  - Remove the original command that allocated m1, if it exists.
  void DoMerge(int32 command_index, int32 s1, int32 s2);

  void Initialize();

  const NnetOptimizeConfig &config_;
  const Nnet &nnet_;
  const ComputationRequest &request_;
  NnetComputation *computation_;

  Analyzer analyzer_;
  
  // lists of submatrices that correspond to each matrix.
  std::vector<std::vector<int32> > submatrix_lists_;

  // true for each matrix that has already been part of
  // an optimization (either as m1 or m2), so we can
  // void potential
  std::vector<bool> matrix_already_optimized_;  
};

/// This optimization function changes, where possible, matrix initializations
/// of type kAllocMatrixZeroed to kAllocMatrixUndefined.
void RemoveUnnecessaryZeroing(const Nnet &nnet, NnetComputation *computation);


/// This optimization moves commands that initialize matrices to as late as
/// possible, and commands that empty matrices to as early as possible.
void MoveSizingCommands(const Nnet &nnet, NnetComputation *computation);

/// This function detects matrices that have no submatrices corresponding to
/// them (due, to changes made in other optimization code), and removes them
/// from the computation.  It also renumbers the submatrix indexes to remove
/// duplicates.
void RemoveOrphanMatrices(NnetComputation *computation);

/// Removes commands of type kNoOperation in the computation.
void RemoveNoOps(NnetComputation *computation);

/// Wherever matrix orig_matrix_index appears in the input of the network
/// (i.e. in computation->input_output_info), replaces it with new_matrix_index.
/// Returns true if it did replace it.
bool ReplaceInInput(
    const Nnet &nnet,
    int32 orig_matrix_index, int32 new_matrix_index,
    NnetComputation *computation);

/// This function outputs to "submatrix_args" the addresses of a subset of
/// arguments arg1 through arg6 in "command", that correspond to the indexes
/// of submatrices.  This is useful in renumbering code.
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

