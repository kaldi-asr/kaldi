// nnet3/nnet-computation.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)
//             2015       Xiaohui Zhang

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

#ifndef KALDI_NNET3_NNET_COMPUTATION_H_
#define KALDI_NNET3_NNET_COMPUTATION_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-nnet.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>


namespace kaldi {
namespace nnet3 {


/**
   \file nnet-computation.h

   The two main classes defined in this header are struct
   ComputationRequest, which basically defines a request for a concrete
   computation that we want the network to do (e.g. "these are the input indexes
   available, and we want these output indexes"), and struct NnetComputation,
   which is a compiled form of the computation that outlines the matrix
   variables and the specific sequence of steps that need to be taken to
   complete the requested computation.
 */


// MiscComputationInfo is a place we enter information about a requested
// computation that doesn't easily fit into the framework as given: things like
// the maximum unrolling we want to do, or how far ahead in time we want a
// particular adaptation method to be able to look.  Elements of this are
// interpreted by individual components, for the most part.
struct MiscComputationInfo {
  // will add members here as needed.

  bool operator== (const MiscComputationInfo &other) const { return true; }
  // This will print this in a human-readable way, for debugging.
  void Print(std::ostream &os) const { };
};


// This defines one type of input that the network gets, or output that it will
// produce.  For inputs, the name should correspond to an input or component
// node name in the nnet (components are allowed so context can be provided in
// recurrent setups); for outputs, the name should be an output node name in the
// Nnet.
// note: this structure is used to represent egs both before and after merging
// into minibatches; if this merging has been done, the indexes will vary in
// the 'n' dimension.
struct IoSpecification {
  std::string name;
  std::vector<Index> indexes;
  bool has_deriv;  // For output nodes, true if a derivative w.r.t. that output
                   // will be supplied.  For input nodes, true if the derivative
                   // w.r.t. that input will be needed.
  IoSpecification(): has_deriv(false) { }

  IoSpecification(const IoSpecification &other):
      name(other.name), indexes(other.indexes), has_deriv(other.has_deriv) { }
  IoSpecification(const std::string &name, const std::vector<Index> &indexes,
                  bool has_deriv = false):
      name(name), indexes(indexes), has_deriv(has_deriv) { }
  // This constructor sets n = 0, x = 0 and t from t_start to t_end-1; and
  // has_deriv to false.
  IoSpecification(const std::string &name, int32 t_start, int32 t_end);

  /// This function is for printing in a human-readable way, for debugging.
  /// Output ends in a newline.
  void Print(std::ostream &os) const;

  void Swap(IoSpecification *other);

  void Read(std::istream &istream, bool binary);

  void Write(std::ostream &ostream, bool binary) const;

  bool operator== (const IoSpecification &other) const;
};

struct IoSpecificationHasher {
  size_t operator () (const IoSpecification &io_spec) const noexcept;
};


// struct ComputationRequest is whatever we need in addition to the
// network itself in order to create the structure of a computation.  The most
// important things it specifies are the available indexes available at
// the input, the indexes requested at various output nodes, and whether or
// not we want to do backprop.
// The same input or output node cannot be listed twice in "inputs" or
// "outputs".
struct ComputationRequest {
  std::vector<IoSpecification> inputs;
  std::vector<IoSpecification> outputs;

  /// if need_model_derivative is true, then we'll be doing either model
  /// training or model-derivative computation, so updatable components need to
  /// be backprop'd.
  bool need_model_derivative;

  /// you should set need_component_stats to true if you need the
  /// average-activation and average-derivative statistics stored by the
  /// StoreStats() functions of components/ such as Tanh, Sigmoid and Softmax.
  bool store_component_stats;

  /// misc_info is for extensibility to things that don't easily fit into the
  /// framework.
  MiscComputationInfo misc_info;

  ComputationRequest(): need_model_derivative(false),
                        store_component_stats(false) { }

  /// returns true if any of inputs[*].has_deriv is true, or
  /// need_model_derivative is true.
  bool NeedDerivatives() const;

  /// Returns the index into "inputs" corresponding to the node with name
  /// "node_name", or -1 if there is no such index.  It is an error if >1 inputs
  /// have the same name.
  int32 IndexForInput(const std::string &node_name) const;

  /// Returns the index into "inputs" corresponding to the node with name
  /// "node_name", or -1 if there is no such index.  It is an error if >1 inputs
  /// have the same name.
  int32 IndexForOutput(const std::string &node_name) const;

  /// This function is for printing info about the computation request
  /// in a human-readable way.
  void Print(std::ostream &os) const;

  void Read(std::istream &istream, bool binary);

  void Write(std::ostream &ostream, bool binary) const;

  bool operator== (const ComputationRequest &other) const;
};

// Hash function for ComputationRequest. It converts
// ComputationRequest to hash code by looking at input
// and output IoSpecifications vectors.
struct ComputationRequestHasher {
  size_t operator()(const ComputationRequest *cr) const noexcept;
};

// Equality function for ComputationRequest pointer
struct ComputationRequestPtrEqual {
 public:
  bool operator() (const ComputationRequest* cr1,
                   const ComputationRequest* cr2) const {
    return (*cr1) == (*cr2);
  }
};


/**
   CommandType is an enum that describes the category of the command used in
   the NnetComputation.  We declare it outside that class because it's so
   frequently used and we got tired of typing NnetComputation:: everywhere.
   We document the commands here.
   Note: for operations that naturally need to operate on entire matrices
   (i.e. allocation commands and input and output commands), we use the
   submatrix indexes of them, which turns out to be more convenient for
   optimization; but these submatrix indexes must refer to the whole of
   a matrix.

   - kAllocMatrix.  Allocate a matrix (its values will be undefined).
                    arg1 = submatrix index, which must refer to a whole matrix.
   - kDeallocMatrix: Deallocate a matrix.  arg1 = submatrix index.
   - kSwapMatrix: initialize matrix with submatrix index arg1 using memory
      from matrix with submatrix index arg2 (using shallow swap).  Both
      submatrices must refer to whole matrices.  The expectation is that
      prior to the swap, arg1 was empty and arg2 was nonempty, but the
      execution code does not enforce this.
   - kSetConst: set all elements of submatrix index 'arg1' to the value 'alpha'.
   - kPropagate: Forward computation of neural net, see Component::Propagate()
     - arg1 is is component-index in neural net
     - arg2 is index into ComponentPrecomputedIndexes (0 if NULL; always 0
       for simple Components)
     - arg3 is sub-matrix index of input
     - arg4 is sub-matrix index of output
     - arg5 is the index of the memo saved from Propagate()'s return value,
        or 0 if it saves no memo.
     - arg6 is 1 if we need to call StoreStats() after the Propagate, or 0
       if we don't.  We used to have a separate command for storing the
       stats, but that has been removed.
   - kBackprop: Do the back-propagation operation, see Component::Backprop()
     - arg1 is index of component in neural net
     - arg2 is index into ComponentPrecomputedIndexes (0 if NULL; always 0
       for simple Components)
     - arg3 is submatrix-index of input value (input to Propagate()); 0 if unused
     - arg4 is submatrix-index of output value (output of Propagate()); 0 if unused
     - arg5 is submatrix-index of output derivative
     - arg6 is submatrix-index of input derivative; 0 if unused.
     - arg7 is the index of the memo which is generated from the corresponding
          Propagate() function if the flag kUsesMemo is set; 0 if unused.
   - kBackpropNoModelUpdate: as kBackprop, but does not set the
     'to_update' argument to the Backprop call, even if the model  is updatable,
     so it skips the model-update phase of backprop.
   - kMatrixCopy: Copy (alpha times contents of sub-matrix arg2)
                  to sub-matrix arg1, currently implemented as copy then scale.
                  Note: to implement scaling a matrix, you can use kMatrixCopy
                  with arg1 == arg2 and it won't do any redundant copying.
   - kMatrixAdd: Add (alpha times contents of sub-matrix arg2)
                 to sub-matrix arg1
   - kCopyRows: call \ref CuMatrix::CopyRows() "CopyRows()" on sub-matrix arg1
                with sub-matrix arg2 and indexes[arg3] as arguments,
                then if alpha != 1.0, scale sub-matrix arg1 by alpha.
   - kAddRows: call \ref CuMatrix::AddRows() "AddRows()" on sub-matrix arg1
               with alpha, sub-matrix arg2 and indexes[arg3] as arguments.
   - kAddRowsMulti, kAddToRowsMulti, kCopyRowsMulti, kCopyToRowsMulti:
       Call the corresponding function in class CuMatrix (Actually the
       names do not have 'Multi' in them, but they are the ones that accept
       a vector of 'Real*'.
        - arg1 is sub-matrix index of *this matrix in operation
        - arg2 is index into "indexes_multi", of which each pair is
      (sub-matrix index, row index) (or (-1,-1) for NULL marker), which
      is turned into a vector of BaseFloat* (pointers to matrix rows)
      before being given as the argument to the function.
      In the 'Add' functions 'alpha' is provided as an argument; for
      the 'Copy' functions, we scale the destination by 'alpha' after
      the copy, if alpha != 1.0.  (However, for implementation reasons,
      kCopyToRowsMulti does not currently support alpha != 1.0 and will
      crash, so we avoid generating this code).
   - kAddRowRanges: call \ref CuMatrix::AddRowRanges() "AddRowRanges()"
     on sub-matrix arg1, with arg2 as source sub-matrix, and indexes given
     indexes_ranges[arg3].  We use the "alpha" as if AddRowRanges()
     accepted that argument, even though it doesn't (we fake it using other
     calls, if alpha != 1.0).
   - kCompressMatrix: Compresses the matrix which should be referred to
     by submatrix-index arg1.  arg2 is a number that determines the
     compression type (it's converted from the enum
     CuCompressedMatrixType; 1=int8, 2=uint8, 3=int16, 4=uint16), and alpha
     determines the 'range' parameter (c.f. NewCuCompressedMatrix()).  arg3
     will be converted to the 'truncate' argument to the class
     CuCompressedMatrix; it should be false (0) if you know that the input is
     limited to the allowed range, and true (1) if the input may exceed that
     range (see docs for CuCompresedMatrix).
   - kDecompressMatrix:  Decompresses the matrix which is referred to
     by submatrix-index arg1 (it should previously have been compressed).
   - kAcceptInput: accepts a matrix of input from the user, which may be either
     features, or derivatives w.r.t. the output.  arg1 is the submatrix index of
     a whole matrix that the input goes to, and arg2 is the index of the network
     node associated with it (e.g. the node of "input" or "ivector"), for
     puroses of double checking.
   - kProvideOutput: outputs a matrix to the user: either a network output, or a
     matrix of derivatives w.r.t. an input.  arg1 is the submatrix index of the
     output (which we expect to be a whole matrix), arg2 is the index of the
     network node associated with it (e.g. the node for "output").
   - kNoOperation: does nothing, and will be removed by optimization code
     (sometimes useful during optimization)
   - kNoOperationPermanent: like kNoOperation, but won't be removed by
     optimization code.  This is used to ensure that for 'trivial'
     computations, which just copy the input to the output, the
     block of commands for the forward or backward propagation is
     nonempty (to avoid confusing the computation code).
   - kNoOperationMarker: does nothing, but used to mark end of a block
     of commands (like forward commands).
   - kNoOperationLabel: does nothing, but is the destination for
     the kGotoLabel command.
   - kGotoLabel: jumps to the kNoOperationLabel command.  arg1 must be set to
     the location of that command.  Since there are no conditionals, the
     kGotoLabel command should be the last command, as remaining commands will
     be unreachable.

*/
enum CommandType {
  kAllocMatrix, kDeallocMatrix, kSwapMatrix, kSetConst,
  kPropagate, kBackprop, kBackpropNoModelUpdate,
  kMatrixCopy, kMatrixAdd, kCopyRows, kAddRows,
  kCopyRowsMulti, kCopyToRowsMulti, kAddRowsMulti, kAddToRowsMulti,
  kAddRowRanges, kCompressMatrix, kDecompressMatrix,
  kAcceptInput, kProvideOutput,
  kNoOperation, kNoOperationPermanent, kNoOperationMarker, kNoOperationLabel,
  kGotoLabel };



// struct NnetComputation defines the specific steps of a neural-net
// computation.  View this as a compiled program; given the Nnet and the
// ComputationRequest, we compile to struct NnetComputation.
struct NnetComputation {
  struct MatrixInfo {
    int32 num_rows;
    int32 num_cols;
    MatrixStrideType stride_type;
    MatrixInfo() { }
    MatrixInfo(int32 num_rows, int32 num_cols,
               MatrixStrideType stride_type):
        num_rows(num_rows), num_cols(num_cols), stride_type(stride_type) {}
    void Read(std::istream &istream, bool binary);
    void Write(std::ostream &ostream, bool binary) const;
  };
  struct MatrixDebugInfo {
    bool is_deriv;  // true if this represents a derivative, not a value.
    std::vector<Cindex> cindexes;
    MatrixDebugInfo(): is_deriv(false) { }
    void Swap(MatrixDebugInfo *other);  // Shallow swap
    void Read(std::istream &istream, bool binary);
    void Write(std::ostream &ostream, bool binary) const;
  };
  struct SubMatrixInfo {
    int32 matrix_index;  // index into "matrices": the underlying matrix.
    int32 row_offset;
    int32 num_rows;
    int32 col_offset;
    int32 num_cols;
    SubMatrixInfo() { }
    SubMatrixInfo(int32 matrix_index, int32 row_offset, int32 num_rows,
                  int32 col_offset, int32 num_cols):
        matrix_index(matrix_index), row_offset(row_offset), num_rows(num_rows),
        col_offset(col_offset), num_cols(num_cols) {}
    void Read(std::istream &istream, bool binary);
    void Write(std::ostream &ostream, bool binary) const;
    bool operator== (const SubMatrixInfo &other) const;
  };
  struct Command {
    CommandType command_type;
    BaseFloat alpha;
    int32 arg1;
    int32 arg2;
    int32 arg3;
    int32 arg4;
    int32 arg5;
    int32 arg6;
    int32 arg7;
    // Constructor where alpha is not specified;
    // This constructor may become deprecated.
    Command(CommandType command_type = kNoOperationMarker,
            int32 arg1 = -1, int32 arg2 = -1, int32 arg3 = -1, int32 arg4 = -1,
            int32 arg5 = -1, int32 arg6 = -1, int32 arg7 = -1):
        command_type(command_type), alpha(1.0),
        arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6),
        arg7(arg7) { }
    // Constructor where you can specify alpha.
    Command(BaseFloat alpha, CommandType command_type = kNoOperationMarker,
            int32 arg1 = -1, int32 arg2 = -1, int32 arg3 = -1, int32 arg4 = -1,
            int32 arg5 = -1, int32 arg6 = -1, int32 arg7 = -1):
        command_type(command_type), alpha(alpha),
        arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6),
        arg7(arg7) { }
    void Read(std::istream &istream, bool binary);
    void Write(std::ostream &ostream, bool binary) const;
  };
  struct PrecomputedIndexesInfo {
    // For each step of the computation for which we might possibly need to store
    // a ComponentPrecomputedIndexes object (and note that this is only applicable
    // for non-simple Components), this struct stores some information.
    // The primary data is in 'data', it's an object of type inheriting from
    // ComponentPrecomputedIndexes.
    // The 'input_indexes' and 'output_indexes' are the vectors that were provided
    // to the function Component::PrecomputeIndexes() when generating these
    // PrecomputedIndexes objects.  They currently only stored in cases where
    // the 'n' values in the computation are numbered only zero and one, because
    // these types of computations are compiled in 'shortcut' compilation, and
    // in that case we'll need these indexes later in order to generate the
    // 'expanded' computation (see the function ExpandComputation()).
    ComponentPrecomputedIndexes *data;
    std::vector<Index> input_indexes;
    std::vector<Index> output_indexes;
    PrecomputedIndexesInfo(): data(NULL) { }
  };


  // "matrices" describes the sizes of the matrices that we use as variables in
  // the computation [note: index zero is reserved for an empty matrix].  Note:
  // we generally don't refer to matrices, even ones known to be whole matrices,
  // using their matrix index directly, but via their submatrix indexes.
  std::vector<MatrixInfo> matrices;

  // debug information for each of the matrices (indexed by matrix-index), only
  // computed if requested in the compiler options.
  std::vector<MatrixDebugInfo> matrix_debug_info;


  // Because some parts of the computation may involve parts of matrix, we
  // declare sub-matrices.  Some of these sub-matrices correspond to entire
  // matrices (this is so that a sub-matrix index can be used to refer to either
  // part of, or all of, a matrix).  The first one (index 0) is an empty
  // sub-matrix, which we use whenever an empty matrix is called for.
  // Note: there is no rule against having identical submatrices.  These
  // will be removed by class ComputationRenumberer in nnet-optimize.cc.
  std::vector<SubMatrixInfo> submatrices;

  // For Components that require precomputed indexes for their Propagate and
  // Backprop operations.  The index into this vector is referred to in
  // kPropagate and kBackprop operations.  Index 0 in the vector is reserved for
  // the NULL pointer, which is used for "simple" components and others that do
  // not require precomputed indexes.
  // These are owned here.
  std::vector<PrecomputedIndexesInfo> component_precomputed_indexes;

  // Used in commands kAddRows, kAddToRows, kCopyRows, which
  // contain indexes into this data-member.
  // Each vector<int32> is a vector of row-indexes (with -1 usually treated as
  // a special case meaning "don't do anything for this row" for add
  // commands, or "use zero" for copy commands.
  std::vector<std::vector<int32> > indexes;

  // Used in commands kAddRowsMulti, kAddToRowsMulti, kCopyRowsMulti and
  // kCopyToRowsMulti.  Contains pairs (sub-matrix index, row index)- or the
  // special pair (-1,-1) meaning "don't do anything for this row" for add
  // commands, or "use zero" for copy commands.
  std::vector<std::vector<std::pair<int32,int32> > > indexes_multi;


  // Indexes used in kAddRowRanges commands, containing pairs (start-index,
  // end-index)
  std::vector<std::vector<std::pair<int32,int32> > > indexes_ranges;

//   // Information about where the values and derivatives of inputs and outputs of
//   // the neural net live.  Indexed by the node_index (the same index as used for
//   // the nodes_ array in the Nnet), each pair is (value_matrix_index,
//   // deriv_matrix_index), with 0 for derivatives that are not present.
//   unordered_map<int32, std::pair<int32, int32> > input_output_info;

  // The sequence of commands.
  std::vector<Command> commands;

  // This is a copy of "need_model_derivative" from the ComputationRequest.
  bool need_model_derivative;

  // computed from "indexes" by ComputeCudaIndexes().
  std::vector<CuArray<int32> > indexes_cuda;

  // computed from "indexes_ranges" by ComputeCudaIndexes().
  std::vector<CuArray<Int32Pair> > indexes_ranges_cuda;


  /// Convenience function used when adding new matrices.  Writes to
  /// 'this->matrices' and 'this->submatrices'; and if 'this->matrix_debug_info'
  /// is nonempty, also increases its size by one.  Returns the *sub-matrix*
  /// index corresponding to the newly added matrix.
  int32 NewMatrix(int32 num_rows, int32 num_cols, MatrixStrideType stride_type);

  /// Convenience function used when adding new sub-matrices.  base_submatrix is
  /// the submatrix of which we want a column and/or row range.  As a
  /// convenience, -1 for the 'num_rows' or the 'num_cols' will be interpreted
  /// as 'as much as possible'.  Returns the new sub-matrix index.  Writes to
  /// 'this->submatrices'.  There is no mechanism to stop duplicates from being
  /// created, but calling RenumberComputation() will remove such duplicates.
  int32 NewSubMatrix(int32 base_submatrix,
                     int32 row_offset, int32 num_rows,
                     int32 col_offset, int32 num_cols);

  // returns true if this submatrix corresponds to the whole of a matrix.
  // submatrix_index must be > 0.
  bool IsWholeMatrix(int32 submatrix_index) const;

  // This must be called after setting up the computation but prior to actually
  // using the Computation object in a computation, to compute CUDA versions of
  // the indexes.
  void ComputeCudaIndexes();

  // This function produces pretty-print ouput intended to allow a human to
  // interpret the computation.
  void Print(std::ostream &os, const Nnet &nnet) const;

  void Read(std::istream &istream, bool binary);
  void Write(std::ostream &ostream, bool binary) const;

  // This function outputs a vector of strings, one for each submatrix,
  // that explains the meaning of each one: something like "m1", "m2";
  // and for parts of matrices, "m1(0:10, 20:40)".
  void GetSubmatrixStrings(const Nnet &nnet,
                           std::vector<std::string> *submat_strings) const;

  // This function outputs a vector, indexed by matrix index, that gives you for
  // each matrix, the index of a submatrix which refers to the whole of that
  // matrix; it makes sure that each matrix has such a submatrix.
  void GetWholeSubmatrices(std::vector<int32> *whole_submatrices) const;


  // This function outputs information similar to Print(), but outputs the
  // preamble as a string and a vector of strings, one per command (with no
  // newlines on these).   This is used in the debugging code in NnetComputer.
  // either pointer argument may be NULL.
  void GetCommandStrings(const Nnet &nnet,
                         std::string *preamble,
                         std::vector<std::string> *command_strings) const;


  // destructor deletes pointers in component_precomputed_indexes.
  ~NnetComputation();
  // removes all information from this struct, makes it as a newly constructed one.
  void Clear() { *this = NnetComputation(); }

  // Copy constructor
  NnetComputation(const NnetComputation &other);
  // Assignment operator.
  NnetComputation &operator = (const NnetComputation &other);
  // Default constructor
  NnetComputation(): need_model_derivative(false) { }
};




// This operator is to print out the NnetComputation in a human-readable way, for
// debugging purposes.
// We don't give Read and Write functions to struct NnetComputation, because we
// don't anticipate needing to write it to disk.
std::ostream &operator << (std::ostream &os,
                           NnetComputation &computation);



} // namespace nnet3
} // namespace kaldi

#endif
