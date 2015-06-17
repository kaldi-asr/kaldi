// nnet3/nnet-computation.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)

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
};


// This defines one type of input that the network gets, or output that it will
// produce.  For inputs, the name should correspond to an input or component
// node name in the nnet (components are allowed so context can be provided in
// recurrent setups); for outputs, the name should be an output node name in the
// Nnet.  In the normal case there will just be one input and one output, and
// the indexes will vary only in the t index, with the others all identical.
struct IoSpecification {
  std::string name;
  std::vector<Index> indexes;
  bool has_deriv;  // For output nodes, true if a derivative w.r.t. that output
                   // will be supplied.  For input nodes, true if the derivative
                   // w.r.t. that input will be needed.
  IoSpecification(): has_deriv(false) { }
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

  // if need_model_derivative is true, then we'll be doing either model training
  // or model-derivative computation.
  bool need_model_derivative;

  // misc_info is for extensibility to things that don't easily fit into the framework
  MiscComputationInfo misc_info;

  ComputationRequest(): need_model_derivative(false) { }

  // returns true if any of inputs[*].has_deriv is true, or model_to_update !=
  // NULL.
  bool NeedDerivatives() const;
};


// struct NnetComputation defines the specific steps of a neural-net
// computation.  View this as a compiled program; given the Nnet and the
// ComputationRequest, we compile to struct NnetComputation.

struct NnetComputation {
  struct MatrixInfo {
    int32 num_rows;
    int32 num_cols;
    MatrixInfo() { }
    MatrixInfo(int32 num_rows, int32 num_cols):
        num_rows(num_rows), num_cols(num_cols) {}
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
  };
  enum CommandType {
    kResizeMatrixZeroed, kResizeMatrixUndefined,
    kResizeMatrixEmpty, kPropagate, kBackprop, kMatrixCopy, kMatrixAdd,
    kCopyRows, kAddRows,
    kCopyRowsMulti, kCopyToRowsMulti, kAddRowsMulti, kAddToRowsMulti,
    kAddRowRanges, kNoOperation, kNoOperationMarker };
  struct Command {
    CommandType command_type;
    // kResizeMatrixZeroed, kResizeMatrixUndefined: arg1 = index of matrix; arg2,arg3 are rows,cols.
    // kResizeMatrixEmpty: arg1 = index of matrix.
    // kPropagate: arg1 = index of component in nnet; arg2 is index of ComponentPrecomputedIndexes
    //   (0 if NULL); arg3, arg4 are sub-matrix indexes of matrix args (input and output)
    // kBackprop: arg1 = index of neural net node (only needed for debug);
    //    arg2 = index of component in nnet; arg3 is index of ComponentPrecomputedIndexes
    //   (0 if NULL); (arg4, arg5, arg6 and arg7) are respectively sub-matrix indexes of
    //   (in-value, output-value, input-deriv, output-deriv).
    // kMatrixCopy,kMatrixAdd: arg1 is source sub-matrix, arg2 is dest sub-matrix.
    // kAddRows, kAddToRows, kCopyRows, kCopyToRows: arg1 (sub-matrix index) is
    //    the *this in operation, arg2 (sub-matrix index) is matrix argument of
    //    operation, changed, arg3 is index into "indexes"
    // kAddRowsMulti, kAddToRowsMulti, kCopyRowsMulti, kCopyToRowsMulti: arg1 is
    //    sub-matrix index of *this matrix in operation; and arg2 is index into
    //    "indexes_multi", of which each pair is (sub-matrix index, row index);
    // kAddRowRanges: arg1 is source matrix, arg2 is dest matrix, arg3 is index
    //   into "indexes_multi".
    // kNoOperation: no operation (sometimes useful during compilation but not
    //  present in final "code").
    // kNoOperationMarker: no operation (sometimes useful during compilation but not
    //  present in final "code").  Used during compilation only.
    int32 arg1;
    int32 arg2;
    int32 arg3;
    int32 arg4;
    int32 arg5;
    int32 arg6;
    int32 arg7;
    Command(CommandType command_type,
            int32 arg1 = -1, int32 arg2 = -1, int32 arg3 = -1, int32 arg4 = -1,
            int32 arg5 = -1, int arg6 = -1, int arg7 = -1):
        command_type(command_type), arg1(arg1), arg2(arg2), arg3(arg3),
        arg4(arg4), arg5(arg5), arg6(arg6) { }
  };

  // "matrices" describes the sizes of the matrices that we use as variables in
  // the computation [note: index zero is reserved for an empty matrix].  Most
  // commands refer to sub_matrices below (note: each matrix will have its own
  // sub-matrix that just refers to the entire matrix).
  std::vector<MatrixInfo> matrices;

  // Because some parts of the computation may involve parts of matrix, we
  // declare sub-matrices.  Some of these sub-matrices correspond to entire
  // matrices (this is so that a sub-matrix index can be used to refer to either
  // part of, or all of, a matrix).  The first one (index 0) is an empty
  // sub-matrix, which we use whenever an empty matrix is called for.
  std::vector<SubMatrixInfo> sub_matrices;

  // For Components that require precomputed indexes for their Propagate and
  // Backprop operations.  The index into this vector is referred to in
  // kPropagate and kBackprop operations.  Index 0 in the vector is reserved for
  // the NULL pointer, which is used for "simple" components and others that do
  // not require precomputed indexes.
  // These are owned here.
  std::vector<ComponentPrecomputedIndexes*> component_precomputed_indexes;

  // used in kAddRows, kAddToRows, kCopyRows, kCopyToRows.  contains row-indexes.
  std::vector<std::vector<int32> > indexes;
  
  // used kAddRowsMulti, kAddToRowsMulti, kCopyRowsMulti, kCopyToRowsMulti.
  // contains pairs (sub-matrix index, row index).
  // also used in kAddRowRanges where it contains pairs (start-index, end-index)
  std::vector<std::vector<std::pair<int32,int32> > > indexes_multi;
  
  // Information about where the values and derivatives of the neural net live,
  // indexed the same index as used for the nodes_ array in the Nnet.
  // each pair is (value_submatrix_index, deriv_submatrix_index), with -1 for
  // derivatives that are not present.
  unordered_map<int32, std::pair<int32, int32> > input_output_info;
  
  // The sequence of commands.
  std::vector<Command> commands;

  // This is a copy of "need_model_derivative" from the ComputationRequest.
  bool need_model_derivative;
  
  // the number of steps in the forward computation, so steps with index >= forward_computation_end
  // are part of the backward computation.
  int32 forward_computation_end;
  
  // computed from "indexes" by ComputeCudaIndexes().
  std::vector<CuArray<int32> > indexes_cuda;

  // computed from "indexes" by ComputeCudaIndexes(), but only
  // those that are used in the kAddRowRanges command are computed.
  std::vector<CuArray<Int32Pair> > indexes_multi_cuda;


  // Convenience function used when adding new matrices.  Returns the corresponding
  // sub-matrix index, which may or not equal the actual matrix index.
  int32 NewMatrix(int32 num_rows, int32 num_cols);

  // Convenience function used when adding new sub-matrices.  Returns the new
  // sub-matrix index.
  int32 NewSubMatrix(int32 base_matrix, int32 dim_offset, int32 dim);

  // This must be called after setting up the computation but prior to actually
  // using the Computation object in a computation, to compute CUDA versions of
  // the indexes.
  void ComputeCudaIndexes();
  
  // destructor deletes pointers in component_precomputed_indexes.
  ~NnetComputation();
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
