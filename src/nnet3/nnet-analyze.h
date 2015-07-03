// nnet3/nnet-analyze.h

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

#ifndef KALDI_NNET3_NNET_ANALYZE_H_
#define KALDI_NNET3_NNET_ANALYZE_H_

#include "nnet3/nnet-compile.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {


/**
   @file This file contains utilities for analyzing and checking computations,
     which are used in the optimization code.
 */



// this struct contains the attributes for a single command.
// see class ComputationVariables for the meaning of a variable.
// note, variables may be both read and written, e.g. for
// operations that do += or that write to only some elements
// of a variable (we can think of these as, for purposes of
// analysis, reading the remaining elements and writing them
// back... a more exact analysis would have to define the
// variables on a more fine-grained level).
struct CommandAttributes {
  // variables read 
  std::vector<int32> variables_read;
  // variables written
  std::vector<int32> variables_written;
  // matrices accessed.  [computed for purposes of checking whether the matrices
  // are allocated at the time they are accessed].  Note, resizing is not
  // recorded here, we'll do that separately.
  std::vector<int32> matrices_accessed;
  
  // true if this command has side effects e.g. on the model (such as
  // Backprop on an updatable component, or StoreStats).
  bool has_side_effects;
  CommandAttributes(): has_side_effects(false) { }
};


enum AccessType {
  kReadAccess,
  kWriteAccess,
  kReadWriteAccess
};


/** This class relates the matrices and sub-matrices in the computation to
    imaginary "variables", such that we can think of the operations as
    operating on sets of individual variables, and we can then do analysis
    that lets us do optimization.  In principle it might make sense to have
    those variables correspond to the elements of the matrices, but that
    would be very inefficient.  On the other hand we could do a coarse-grained
    analysis making the variables correspond to the matrices, but that
    would cause the resulting analysis to be inaccurate.

    What we do instead, which is accurate enough in the cases we envisage, is to
    make the variables correspond to the most specific column ranges in the
    matrices that we ever access.  We do this as follows: for each matrix in the
    computation we get a list of all the "split points" at which column ranges
    ever start and end, and define a split_point_index as the index into the
    array.  The variable could be defined as the pair (matrix_index,
    split_point_index), but we map it to a single integer index called variable_index,
    which we compute from the pair using the expression
    (matrix_to_variable_index_[matrix_index] + split_point_index).
    Each sub-matrix in the computation will now correspond to a list of variables,
    and because these lists are always a contiguous range we can just store the
    start and end points.  In addition we note, for each submatrix, whether
    it spans the entire row range of the underlying matrix or just a part of
    the row range.  The reason we need to know this is that a write operation
    to just part of the row-range would have to be classed as a read-write
    operation because the final contents after the operation would in that
    case depend on the original contents.
 */
class ComputationVariables {
 public:
  
  ComputationVariables(const NnetComputation &computation);

  // This function updates the CommandAttributes object to record an access of
  // type read, write or read-write on the variables that this sub-matrix
  // corresponds to, and also updates the matrices_accessed variable by adding
  // the number of the underlying matrix.  The slightly non-obvious thing it
  // does is that if the access type is given as write, and the sub-matrix does
  // not span the full row range of the matrix it belongs to (and hence does not
  // span the full extent of the variables that we defined), the access is
  // recorded as both read and write (because the result of the operation on
  // those depends on the pre-existing contents, so it would not be correct to
  // consider it just a write operation).
  void RecordAccessForSubmatrix(
      int32 submatrix_index,
      AccessType access_type,
      CommandAttributes *ca) const;

  // Appends to variables_indexes the list of variables corresponding to a
  // matrix index.
  void AppendVariablesForMatrix(
      int32 matrix_index,
      std::vector<int32> *variable_indexes) const;

  
  int32 NumVariables() const { return num_variables_; }

 private:
  // Appends to variable_indexes the list of variables corresponding to a
  // submatrix index.  We might need to make this public at some point.
  void AppendVariablesForSubmatrix(
      int32 submatrix_index,
      std::vector<int32> *variable_indexes) const;

  
  // sets up split_points_ and matrix_to_variable_index_.  called from
  // constructor.
  void ComputeSplitPoints(const NnetComputation &computation);
  // sets up variable_ranges_, full_row_range_, and submatrix_to_matrix_.
  // called from constructor.
  void ComputeVariableRanges(const NnetComputation &computation);
  
  // Indexed first by matrix-index and then a list, this gives us all the split
  // points at which column ranges start and end.  For instance, if the 3'rd
  // matrix has 20 columns and is split into ranges 0:9 and 10:19,
  // split_points_[3] would equal [0, 10, 20].  zeroth one will be empty because
  // matrix-index zero is reserved for the empty matrix.
  std::vector<std::vector<int32> > split_points_;

  // maps from the matrix-index (note, zero is invalid as it corresponds to the
  // empty matrix) to the variable-index for its first split point.  for coding
  // convenience there is one extra element at the end, which is equal to the
  // total number of variables.
  std::vector<int32> matrix_to_variable_index_;

  // records the matrix index underlying each submatrix.
  std::vector<int32> submatrix_to_matrix_;

  int32 num_variables_;
  
  // maps each submatrix index to the start and end of the corresponding range
  // of variable indexes (note: the actual variable indexes spanned by
  // this submatrix can be expressed as start, start+1 ... end-1, i.e. they
  // don't include the end of the range.
  std::vector<std::pair<int32, int32> > variable_ranges_;

  // indexed by submatrix index, this is true if the submatrix spans the full
  // row range of the underlying matrix.  Affects whether write operations
  // should be classed as write operations or as read-write operations.
  std::vector<bool> full_row_range_;

};



struct VariableAccesses {
  struct Access {
    int32 command_index;
    AccessType access_type;
    bool operator < (const Access &other) const {
      return command_index < other.command_index;
    }
  };
  // the following vector will be sorted and unique.
  std::vector<Access> accesses;
};

// After the command-level attributes have been computed, this function
// organizes them per variable (see class ComputationVariables for how
// a variable is defined; it is part of a matrix).
void ComputeVariableAccesses(
    const std::vector<CommandAttributes> &command_attributes,
    std::vector<VariableAccesses> *variable_accesses);


struct MatrixAccesses {
  // initialize_command is the index of the command that initializes the
  // matrix, or -1 if it doesn't exist (e.g. it is an input).
  int32 initialize_command;
  // the index of the command that destroys the matrix (or -1 if never gets
  // destroyed.
  int32 destroy_command;
  // the indexes of commands that access the matrix for read/write.
  std::vector<int32> access_commands;
  // true if this matrix is an input to the computation.
  bool is_input;
  // true if this matrix is an output of the computation.  
  bool is_output;
};


void ComputeMatrixAccesses(
    const NnetComputation &computation,
    const std::vector<CommandAttributes> &command_attributes,
    std::vector<MatrixAccesses> *matrix_accesses);

void CheckMatrixAccesses(
    const NnetComputation &computation,
    const std::vector<CommandAttributes> &command_attributes,
    const std::vector<MatrixAccesses> &matrix_accesses);





// Returns the total memory, in bytes, used by the computation (just the
// temporary memory, not counting the memory used by the nnet itself).  This is
// defined as the maximum amount of memory used at any one instant.  The
// "assume_cuda" bool is used to determine the stride of matrices, which affects
// the memory used; in CUDA, the stride must be a multiple of ?? bytes.
int32 MaxMemoryUsage(bool assume_cuda,
                     const NnetComputation &computation);


// computes a vector of attributes, one for each Command in the computation.
void ComputeCommandAttributes(
    const Nnet &nnet,
    const NnetComputation &computation,
    const ComputationVariables &variables,
    std::vector<CommandAttributes> *attributes);


struct CheckComputationConfig {
  // do the check_rewrite check only for a non-optimized computation, it may
  // legitimately fail after optimization.  see code for details.
  bool check_rewrite;

  CheckComputationConfig(): check_rewrite(false) { }
};


class ComputationChecker {
 public:
  ComputationChecker(const CheckComputationConfig &config,
                     const Nnet &nnet,
                     const ComputationRequest &request,
                     const NnetComputation &computation);
  void Check();  // call this only once.
 private:
  // various dimension consistency checks and checks on properties.
  void CheckComputationIndexes() const;
  // make sure Propagate comes before kNoOpMarker and Backprop comes after it.
  void CheckComputationOrder() const;
  // checks we are never using matrices with zero size.
  void CheckComputationAllocation() const;
  // checks that all writes are done before reads. details with implementation.
  void CheckComputationRewrite() const;
  // check we never use undefined values.
  void CheckComputationUndefined() const;
  // check matrix accesses make sense.
  void CheckComputationMatrixAccesses() const;
  
  const CheckComputationConfig &config_;
  const Nnet &nnet_;
  const ComputationRequest &request_;
  const NnetComputation &computation_;
  ComputationVariables variables_;
  std::vector<CommandAttributes> attributes_;
};


                      

  





} // namespace nnet3
} // namespace kaldi


#endif

