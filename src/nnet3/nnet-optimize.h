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

#include <iostream>

namespace kaldi {
namespace nnet3 {

// Options class for optimizing a NnetComputation
struct NnetOptimizeConfig {
  // This will have options to disable various of the optimizations, which
  // might come in useful in debugging, to locate where a problematic
  // operation comes in.
  
  void Register(OptionsItf *po) {
  }
};


// this was a very early draft.  could end up completely changed.  I'll leave this till
// last as it's not essential to get the framework working.
class NnetOptimize {
 public:
  NnetOptimize(NnetComputation *computation);

  // Top-level optimization routine.
  void OptimizeComputation();

 private:

  // this is all provisional.
  struct MatrixOptInfo {
    // list of all sub-matrix indexes that point to this matrix.
    std::vector<int32> submatrices;
    // index of sub-matrix that is the whole of this matrix.
    int32 whole_submatrix;
  };

  // this is all provisional.
  struct SubmatrixOptInfo {
    // true if this sub-matrix is the whole of a matrix.
    bool is_whole_matrix;
    
    // list of other sub-matrix indexes that have some overlap with this one
    // (including this sub-matrix index).
    std::vector<int32> overlapping_submatrices;

    struct CommandInfo {
      bool writes;
      bool reads;
    };

    // list of commands that reference this index or others in
    // "overlapping_submatrices".
    std::vector<int32> commands;
    
    std::vector<int32> writing_commands;
    
    // list of sub-matrix indexes corresponding to this matrix.
    std::vector<int32> submatrices;
  };

  struct StepOptInfo {
  };
  
  NnetComputation *computation_;

  std::vector<MatrixOptInfo> matrix_info_;

  std::vector<SubmatrixOptInfo> submatrix_info_;
  
  std::vector<StepOptInfo> step_info_;

};

  
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

