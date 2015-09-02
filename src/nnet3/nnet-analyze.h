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



// this struct contains the attributes for a single command.  see class
// ComputationVariables for the meaning of a variable.  note, variables may be
// both read and written, e.g. for operations that do += or that write to only
// some elements of a variable (we can think of these as, for purposes of
// analysis, reading the remaining elements and writing them back... a more
// exact analysis would have to define the variables on a more fine-grained
// level).
struct CommandAttributes {
  // All of the vector variables below are made sorted and uniq by
  // ComputeCommandAttributes.

  // variables read
  std::vector<int32> variables_read;
  // variables written
  std::vector<int32> variables_written;

  // sub-matrices read (i.e. the submatrix appears in the command directly)
  std::vector<int32> submatrices_read;
  // sub-matrices written (i.e. the submatrix appears in the command directly)
  std::vector<int32> submatrices_written;

  // matrices read
  std::vector<int32> matrices_read;
  // matrices written
  std::vector<int32> matrices_written;

  // true if this command has side effects e.g. on the model (such as
  // Backprop on an updatable component, or StoreStats).
  bool has_side_effects;
  CommandAttributes(): has_side_effects(false) { }
};


/// This function is to be used in debugging; it produces human-readable output.
void PrintCommandAttributes(std::ostream &os,
                            const std::vector<CommandAttributes> &attributes);


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
  // This function must only be called once per object.
  void Init(const NnetComputation &computation);

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

  /// Appends to variables_indexes the list of variables corresponding to a
  /// matrix index.
  void AppendVariablesForMatrix(
      int32 matrix_index,
      std::vector<int32> *variable_indexes) const;

  // Appends to variable_indexes the list of variables corresponding to a
  // submatrix index.
  void AppendVariablesForSubmatrix(
      int32 submatrix_index,
      std::vector<int32> *variable_indexes) const;

  // note: variables are zero-indexed.
  int32 NumVariables() const { return num_variables_; }

  int32 GetMatrixForVariable(int32 variable) const;

 private:


  // sets up split_points_ and matrix_to_variable_index_.  called from
  // constructor.
  void ComputeSplitPoints(const NnetComputation &computation);
  // sets up variable_ranges_ and full_column_range_.  called from constructor.
  void ComputeVariableRanges(const NnetComputation &computation);
  // sets up variable_to_matrix_.  called from constructor.
  void ComputeVariableToMatrix(const NnetComputation &computation);
  // sets up submatrix_to_matrix_ and submatrix_is_whole_matrix.
  // called from constructor.
  void ComputeSubmatrixInfo(const NnetComputation &computation);

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

  std::vector<int32> submatrix_to_matrix_;
  std::vector<bool> submatrix_is_whole_matrix_;

  // records the matrix index underlying each variable.
  std::vector<int32> variable_to_matrix_;

  int32 num_variables_;

  // maps each submatrix index to the start and end of the corresponding range
  // of variable indexes (note: the actual variable indexes spanned by
  // this submatrix can be expressed as start, start+1 ... end-1, i.e. they
  // don't include the end of the range.
  std::vector<std::pair<int32, int32> > variable_ranges_;

  // indexed by submatrix index, this is true if the submatrix spans the full
  // row range of the underlying matrix.  Affects whether write operations
  // should be classed as write operations or as read-write operations.
  std::vector<bool> full_column_range_;

};


// This struct records an access to a variable (i.e. a row range of a matrix) or
// to a matrix.
struct Access {
  int32 command_index;
  AccessType access_type;
  Access(int32 command_index, AccessType access_type):
      command_index(command_index), access_type(access_type) { }
  bool operator < (const Access &other) const {
    return command_index < other.command_index;
  }
};


/**
   After the command-level attributes have been computed, this function
   organizes them per variable (see class ComputationVariables for how
   a variable is defined; it is part of a matrix).
     @param [in] variables   The definition of variables for this computation
     @param [in] command_attributes  A vector of attributes, one per command, as
                      obtained from ComputeCommandAttributes().
     @param [out] variable_accesses  The output will have a size equal to
                     the number of variables, and each element will be
                     a vector of accesses, sorted by command index; each
                     command will only be listed once in this vector.  */
void ComputeVariableAccesses(
    const ComputationVariables &variables,
    const std::vector<CommandAttributes> &command_attributes,
    std::vector<std::vector<Access> > *variable_accesses);


struct MatrixAccesses {
  /// Index of the command that allocates the matrix, or -1 if the command
  /// doesn't exist (e.g. it is an input).
  int32 allocate_command;
  /// Index of the command that deallocates the matrix, or -1 if never gets
  /// deallocated (e.g. it is an output).
  int32 deallocate_command;
  /// Records the indexes of commands that access the matrix, and the type
  /// (read, read/write, write).  It will be sorted on command index with only
  /// one record per command.  Note: a write to only a part of the matrix
  /// (i.e. a submatrix that isn't the whole thing) will be recorded as an
  /// access of type read/write.
  std::vector<Access> accesses;
  /// true if this matrix is an input to the computation.
  bool is_input;
  /// true if this matrix is an output of the computation.
  bool is_output;
  MatrixAccesses(): allocate_command(-1), deallocate_command(-1),
                    is_input(false), is_output(false) { }
};

/**
   This function organizes information in the CommandAttributes in a way that
   is convenient to access per matrix.  See struct MatrixAccesses for the
   output format; the output "matrix_accesses" is indexed by the matrix index
   (the same index as computation.matrices).
 */
void ComputeMatrixAccesses(
    const Nnet &nnet,
    const NnetComputation &computation,
    const ComputationVariables &variables,
    const std::vector<CommandAttributes> &command_attributes,
    std::vector<MatrixAccesses> *matrix_accesses);

/// This function is to be used in debugging; it produces human-readable output.
void PrintMatrixAccesses(std::ostream &os,
                         const std::vector<MatrixAccesses> &matrix_accesses);

/// This struct exists to set up various pieces of analysis; it helps avoid the
/// repetition of code where we compute all these things in sequence.
struct Analyzer {
  ComputationVariables variables;
  std::vector<CommandAttributes> command_attributes;
  std::vector<std::vector<Access> > variable_accesses;
  std::vector<MatrixAccesses> matrix_accesses;
  void Init(const Nnet &nnet, const NnetComputation &computation);
};


/// This class performs various kinds of specific analysis on top of what class
/// Analyzer gives you immediately.  It mostly contains special-purpose things
/// what were needed by class VariableMergingOptimizer (see nnet-optimize.h, and
/// the extended comment above class VariableMergingOptimizer).
class ComputationAnalysis {
 public:
  /// This class stores the const references provided to its constructor ->
  /// be careful about changing them or deallocating them during the
  /// lifetime of this object.
  ComputationAnalysis(const NnetComputation &computation,
                      const Analyzer &analyzer): computation_(computation),
                                                 analyzer_(analyzer) { }

  /// If the matrix underlying submatrix 's' is an input then this returns -1;
  /// otherwise is returns the first command (read or write) that is not an
  /// allocation command, that accesses any part of 's' [note: deallocation does
  /// not count as a read or write operation].  If there is no such command, it
  /// returns num_commands.
  /// s must be >0 (i.e. not the empty submatrix).
  int32 FirstAccess(int32 s) const;

  /// If the matrix underlying submatrix 's' is an output then this returns
  /// num-commands; otherwise it returns the last non-deallocation command
  /// that accesses any part of submatrix 's'; if there is no such command it
  /// returns -1.
  /// s must be >0 (i.e. not the empty submatrix).
  int32 LastAccess(int32 s) const;

  /// Returns the last command-index that accesses any part of submatrix 's' as
  /// a write operation, or -1 if there is no such operation.  Not: deallocation
  /// does not count as a write operation.
  /// s must be >0 (i.e. not the empty submatrix).
  int32 LastWriteAccess(int32 s) const;

  /// Returns (the first command-index after 'c' that any part of submatrix 's'
  /// is written to); or if there is no such command, then (the
  /// command-index of the command that deallocates the matrix underlying s);
  /// or if there is no such command, then the total number of commands.
  /// s must be >0 (i.e. not the empty submatrix).
  int32 DataInvalidatedCommand(int32 c, int32 s) const;

 private:
  const NnetComputation &computation_;
  const Analyzer &analyzer_;
};


/// This function computes a vector "submat_lists", indexed
/// by matrix index, such that (*submat_lists)[m] is a list of
/// all the submatrix indexes that refer to matrix m.  Note,
/// (*submat_lists)[0] will be the empty vector.
void ComputeSubmatLists(const NnetComputation &computation,
                        std::vector<std::vector<int32> > *submat_lists);


/**
   Returns the total memory, in bytes, used by the computation (just the
   temporary memory, not counting the memory used by the nnet itself).  This is
   defined as the maximum amount of memory used at any one instant.  */
int32 MaxMemoryUsage(const NnetComputation &computation);


// computes a vector of attributes, one for each Command in the computation.
void ComputeCommandAttributes(
    const Nnet &nnet,
    const NnetComputation &computation,
    const ComputationVariables &variables,
    std::vector<CommandAttributes> *attributes);


struct CheckComputationOptions {
  // do the check_rewrite check only for a non-optimized computation, it may
  // legitimately fail after optimization.  see code for details.
  bool check_rewrite;

  CheckComputationOptions(): check_rewrite(false) { }
};


class ComputationChecker {
 public:
  ComputationChecker(const CheckComputationOptions &config,
                     const Nnet &nnet,
                     const ComputationRequest &request,
                     const NnetComputation &computation);
  void Check();  // call this only once.
 private:
  // various dimension consistency checks and checks on properties.
  void CheckComputationIndexes() const;
  // make sure Propagate comes before kNoOpMarker and Backprop comes after it,
  // and that the value of forward_computation_end matches the position of
  // kNoOpMarker.
  void CheckComputationOrder() const;
  // checks for a situation where an undefined variable is read.
  void CheckComputationUndefined() const;
  // checks that all writes are done before reads.  details with implementation.
  void CheckComputationRewrite() const;
  // check matrix accesses make sense.
  void CheckComputationMatrixAccesses() const;


  const CheckComputationOptions &config_;
  const Nnet &nnet_;
  const ComputationRequest &request_;
  const NnetComputation &computation_;
  Analyzer a_;
};










} // namespace nnet3
} // namespace kaldi


#endif

