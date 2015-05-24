// nnet3/nnet-compile.h

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

#ifndef KALDI_NNET3_NNET_COMPILE_H_
#define KALDI_NNET3_NNET_COMPILE_H_

#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-computation-graph.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {

// This class creates an initial version of the NnetComputation, without any
// optimization or sharing of matrices.  
class Compiler {
 public:
  Compiler(const ComputationRequest &request,
           const Nnet &nnet);
  
  void CreateComputation(NnetComputation *computation);

 private:
  const ComputationRequest &request_;
  const Nnet &nnet_;
  ComputationGraph graph_;

  // Some generic information about each step of the computation... a step is an
  // instance of a NetworkNode, but a NetworkNode may in unrolled computations
  // have multiple steps.  A single step may turn into no commands (for input
  // nodes), or multiple commands.
  struct StepInfo {
    int32 node_index;  // network-node index
    int32 input_step;  // for nodes of type kComponent, the step-index of the
    // step corresponding to the input.
    int32 value;  // matrix index of value that this step outputs.
    int32 deriv;  // matrix index of derivative at the output of this step; zero if
    // not used (note: index zero is reserved for the empty matrix).

    // precomputed_indexes_index is the index into the
    // component_precomputed_indexes array in the NnetComputation.
    int32 precomputed_indexes_index;

    std::vector<Index> output_indexes;      // Indexes that this step outputs.
    std::vector<int32> output_cindex_ids;   // cindex_ids for each of the output
    // indexes.

    // If this component is of type kComponentInput or kOutput (and note that
    // the top-level Descriptor is a concatenation over >= 1 parts), then we set
    // value_parts to a list of submatrix-indexes, each for the corresponding
    // part of the value.  If there is only one part, it will have one element
    // which will be the same as "value".
    std::vector<int32> value_parts;
    // deriv_parts is as "value_parts", but for parts of the derivative (if
    // we're doing backprop).
    std::vector<int32> deriv_parts;

    StepInfo(): node_index(-1), input_step(-1), value(0), deriv(0),
                precomputed_indexes_index(0) { }
  };

  // this sets up cindex_id_to_location_.
  void CreateLocationInfo(const std::vector<std::vector<int32> > &by_step);
  
  // this sets up steps_, destroying "by_step" in the process.
  // It also sets num_matrices_.
  void CreateStepInfo(std::vector<std::vector<int32> > *by_step);



  // Steps of the computation.  Index by step-index.
  std::vector<StepInfo> steps_;

  int32 num_matrices_;
  
  /// This maps each cindex_id to its location.  A location
  /// is a pair (step-index, matrix-row-index).
  std::vector<std::pair<int32, int32> > cindex_id_to_location_;


  // Adds to the computation object the information about the matrix sizes
  void DefineMatrices(NnetComputation *computation) const;

  // sets up the input_output_info of the computation.
  void SetInputOutputInfo(NnetComputation *computation) const;

  // Sets up sub-matrix indexes.  For each matrix index, an equal sub-matrix
  // index is created that corresponds to that entire matrix (including index
  // zero, for the empty sub-matrix corresponding to the empty matrix);
  // and also for those matrices that are multi-part (because they correspond
  // to a Descriptor that has a "parts" vector with size >1), it sets up
  // a sub-matrix for each part and puts the indexes into the "submatrix_indexes"
  // vector of the StepInfo.
  void DefineSubmatrices(NnetComputation *computation);

  // Adds to the computation object the commands to set up the matrices.
  void SetUpMatrices(NnetComputation *computation) const;

  // Sets up the precomputed indexes for each component, and sets the
  // precomputed_indexes_index value for each step.
  void SetUpPrecomputedIndexes(NnetComputation *computation);
  
  // Adds to "computation" the command(s) for the forward computation 
  // for this step.
  void DoForwardComputation(int32 step, NnetComputation *computation) const;

  // Called from DoForwardComputation, handles the case where the step corresponds
  // to a Component.
  void AddPropagateStep(int32 step, NnetComputation *computation) const;

  // Called from DoForwardComputation, handles the case where the step corresponds
  // to types kComponentInput or kOutput.
  void DoForwardComputationDescriptor(
      int32 step, const Descriptor &descriptor,
      NnetComputation *computation) const;

  // Called from DoForwardComputationDescriptor.  
  void DoForwardComputationForwardingDescriptor(
      int32 step,
      int32 value_submatrix_index,
      bool is_first_term_in_sum,
      const ForwardingDescriptor &descriptor,
      NnetComputation *computation) const;

  // Called from DoForwardComputationForwardingDescriptor.
  void DoForwardComputationFromSubmatLocations(
      int32 value_submatrix_index,
      bool is_first_term_in_sum,
      const std::vector<std::pair<int32, int32> > &submat_locations,
      NnetComputation *computation) const;  

  // Called from DoForwardComputationFromSubmatLocations (special
  // case where all input is from the same matrix).
  void DoForwardComputationFromIndexes(
      int32 value_submatrix_index,
      int32 input_submatrix_index,
      bool is_first_term_in_sum,
      const std::vector<int32> &indexes,
      NnetComputation *computation) const;  
  
  
  // Adds to "computation" the command(s) for the backward computation (if any) for
  // this step.
  void DoBackwardComputation(int32 step, NnetComputation *computation) const;

  // Called from DoBackwardComputation, handles the case where the step corresponds
  // to a Component.
  void AddBackpropStep(int32 step, NnetComputation *computation) const;

  // Called from DoBackwardComputation, handles the case where the step
  // corresponds to types kComponentInput or kOutput.
  void DoBackwardComputationDescriptor(
      int32 step, const Descriptor &descriptor,
      NnetComputation *computation) const;

  // Called from DoBackwardComputationDescriptor.  
  void DoBackwardComputationForwardingDescriptor(
      int32 step, int32 deriv_submatrix_index,
      const ForwardingDescriptor &descriptor,
      NnetComputation *computation) const;

  // Called from DoBackwardComputationForwardingDescriptor.
  void DoBackwardComputationFromSubmatLocations(
      int32 deriv_submatrix_index,
      const std::vector<std::pair<int32, int32> > &submat_locations,
      NnetComputation *computation) const;  

  // Called from DoBackwardComputationFromSubmatLocations - special case where
  // input is from just one matrix.
  void DoBackwardComputationFromIndexes(
      int32 deriv_submatrix_index,
      int32 input_deriv_submatrix_index,
      const std::vector<int32> &indexes,
      NnetComputation *computation) const;
  
  
  // [to be called after step_info_ is set up and all the forward and backprop
  // commands have been added].  Adds to the computation the commands that
  // deinitialize all the matrices, except those that may be requested by
  // the user after the computation is done (i.e. outputs of the network,
  // and input derivatives).
  void DestroyMatrices(NnetComputation *computation);

  void AddCommands(NnetComputation *computation);

};



} // namespace nnet3
} // namespace kaldi


#endif

