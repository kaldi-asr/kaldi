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
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-computation-graph.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {

struct CompilerOptions {
  bool output_debug_info;

  CompilerOptions(): output_debug_info(true) { }
};

/// This class creates an initial version of the NnetComputation, without any
/// optimization or sharing of matrices.    Note: for a user-level interface
/// that includes optimization, see class CachingOptimizingCompiler in
/// nnet-optimize.h.
class Compiler {
 public:
  Compiler(const ComputationRequest &request,
           const Nnet &nnet);

  void CreateComputation(const CompilerOptions &opts,
                         NnetComputation *computation);

 private:
  const ComputationRequest &request_;
  const Nnet &nnet_;
  ComputationGraph graph_;

  // Some generic information about each step of the computation... a step is an
  // instance of a NetworkNode, but a NetworkNode may in general have multiple
  // steps.  A single step may turn into no commands (for input nodes), or
  // multiple commands.
  struct StepInfo {
    int32 node_index;  // network-node index
    bool is_input;  // true if step corresponds to an input to the computation.
    int32 value;  // sub-matrix index of value that this step outputs.
    int32 deriv;  // sub-matrix index of derivative at the output of this step; zero
                  // if not used (note: index zero is reserved for the empty
                  // matrix).

    // precomputed_indexes_index is the index into the
    // component_precomputed_indexes array in the NnetComputation, or zero if
    // none needed.
    int32 precomputed_indexes_index;

    std::vector<Index> output_indexes;      // Indexes that this step outputs.
    std::vector<int32> output_cindex_ids;   // cindex_ids corresponding to each
                                            // of the output indexes.

    // If this component is of type kDescriptor (and note that the top-level
    // Descriptor is a concatenation over >= 1 parts), then we set value_parts
    // to a list of submatrix-indexes, each for the corresponding part of the
    // value.  If there is only one part, it will have one element which will be
    // the same as "value".
    std::vector<int32> value_parts;
    // deriv_parts is as "value_parts", but for parts of the derivative (if
    // we're doing backprop).
    std::vector<int32> deriv_parts;

    // for nodes corresponding to descriptors, input_locations_list will contain
    // information about the inputs to this descriptor, telling us for each row
    // of the matrix what other matrix rows it is a summation over.  this is a
    // quantity indexed[part-index][row-index], then a list of pairs (step,
    // row-index), that we store here to avoid computing it twice in forward and
    // backprop.
    std::vector<std::vector<std::vector<std::pair<int32,int32> > > > input_locations_list;

    StepInfo(): node_index(-1), is_input(false), value(0),
                deriv(0), precomputed_indexes_index(0) { }
  };

  // this sets up cindex_id_to_location_.
  void CreateLocationInfo(const std::vector<std::vector<int32> > &by_step);


  // Computes the set of step-indexes of preceding steps that this step depends
  // on.  Assumes CreateLocationInfo() has already been called.  Requires
  // 'step_index' only to handle a special case, that if 'this_step' is a
  // component step, then the only step it depends on is the preceding step
  // (which is the component-input step).
  void ComputeStepDependencies(const std::vector<int32> &this_step,
                               int32 step_index,
                               unordered_set<int32> *dep_steps);

  // This function outputs to each element of "deriv_needed" a bool saying
  // whether, for that step, we need to allocate the matrix of derivatives
  // (interpret this as being at the output of that step).  This variable
  // also tells us whether we need to execute the backprop code for that step.
  void ComputeDerivNeeded(const std::vector<std::vector<int32> > &steps,
                          std::vector<bool> *deriv_needed);

  // this sets up steps_, destroying the input "by_step" in the process.  It
  // also sets various matrix and sub-matrix sizes in "computation".
  void CreateStepInfo(const std::vector<bool> &deriv_needed,
                      std::vector<std::vector<int32> > *by_step,
                      NnetComputation *computation);

  // Gets the stride type, kDefaultStride or kStrideEqualNumCols,
  // at the output of this node: interrogates component flags
  // looking for kInputContiguous or kOutputContiguous.
  MatrixStrideType GetStrideType(int32 node_index) const;


  // Miscellaneous info pertaining to various steps of the computation.  Indexed
  // by step-index.
  std::vector<StepInfo> steps_;

  /// This maps each cindex_id to its location.  However, you should not rely on
  /// its accuracy for cindex_ids that correspond to the Descriptors at
  /// Component inputs, since it's possible in principle for such cindex_ids to
  /// exist at >1 location.  A location is a pair (step-index,
  /// matrix-row-index).
  std::vector<std::pair<int32, int32> > cindex_id_to_location_;


  // Adds to the computation object the information about the matrix sizes
  void DefineMatrices(NnetComputation *computation) const;

  // sets up the input_output_info of the computation (this says where the
  // values and derivatives for the inputs and outputs live).
  void SetInputOutputInfo(NnetComputation *computation) const;

  // Sets up sub-matrix indexes for nodes of type Descriptor (needed mainly
  // because Descriptors in general have many parts corresponding to
  // feature-dimension ranges, and they live in sub-matrices.
  void DefineSubmatrices(NnetComputation *computation);

  // Adds to the computation object the commands to allocate the matrices.
  void AllocateMatrices(NnetComputation *computation) const;

  // Sets up the precomputed indexes for each component, and sets the
  // precomputed_indexes_index value for each step.
  void SetUpPrecomputedIndexes(NnetComputation *computation);

  // Adds to "computation" the command(s) for the forward computation
  // for this step.
  void DoForwardComputation(int32 step, NnetComputation *computation) const;

  // Called from DoForwardComputation, handles the case where the step corresponds
  // to a Component.
  void AddPropagateStep(int32 step, NnetComputation *computation) const;


  // Called from DoForwardComputation, handles the case where the step
  // corresponds to type kDescriptor
  void DoForwardComputationDescriptor(
      int32 step, NnetComputation *computation) const;

  void DoForwardComputationSumDescriptor(
      int32 step, int32 part_index, NnetComputation *computation) const;


  // For the "part_index"'th part of the Descriptor for step "step" (which
  // must correspond to a Descriptor and not an Input or Component), this
  // function computes a vector of lists of submatrix locations of the inputs.
  // It is indexed by the number of rows in the output of this descriptor,
  // and the i'th element of the output is a list of pairs (step-index,
  // row-index-of-matrix).  The output of this row of this row of this part
  // of the computation will be a sum over those pairs.
  void ComputeInputLocationsList(
      int32 step, int32 part_index,
      std::vector<std::vector<std::pair<int32, int32> > > *input_locations)
      const;

  // Changes the format of the location-list produced by ComputeInputLocationsList,
  // to have pairs (sub-matrix, row) instead of (step, row), by replacing each step
  // (i.e. the first of each pair) with steps_[step].value.
  void ComputeValueSubmatLocationsList(
 const std::vector<std::vector<std::pair<int32, int32> > > &input_locations_list,
     std::vector<std::vector<std::pair<int32, int32> > > *submat_locations_list)
      const;


  // Changes the format of the location-list produced by
  // ComputeInputLocationsList, to have pairs (sub-matrix, row) instead of
  // (step, row), but with locations of derivatives not values (for use in
  // backprop).  It does this by replacing each step (i.e. the first of each
  // pair) with steps_[step].deriv, but if this value is zero (i.e. no such
  // derivative exists) it removes the pair.  This could occur in situations
  // where we only need to propagate the derivative selectively to some inputs.
  void ComputeDerivSubmatLocationsList(
 const std::vector<std::vector<std::pair<int32, int32> > > &input_locations_list,
 std::vector<std::vector<std::pair<int32, int32> > > *submat_locations_list)
      const;



  // Called from DoForwardComputationSumDescriptor.
  // Does the forward computation for one sub-matrix of a Descriptor,
  // after the caller has already worked out what the input terms for
  // each row are in terms of a list of sub-matrix locations; each location
  // is pair (submatrix-index, row-index)
  void DoForwardComputationFromSubmatLocationsList(
      int32 value_submatrix_index,
      const std::vector<std::vector<std::pair<int32, int32> > > &submat_locations,
      NnetComputation *computation) const;


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
  // this step.  (non-const only because we clear the cached submat_locations).
  void DoBackwardComputation(int32 step, NnetComputation *computation);

  // Called from DoBackwardComputation, handles the case where the step corresponds
  // to a Component.
  void AddBackpropStep(int32 step, NnetComputation *computation) const;

  // Called from DoBackwardComputation, handles the case where the step
  // corresponds to type kDescriptor.
  void DoBackwardComputationDescriptor(
      int32 step, NnetComputation *computation);

  // Called from DoBackwardComputationSumDescriptor.
  void DoBackwardComputationSumDescriptor(
      int32 step, int32 part_index,
      NnetComputation *computation) const;

  // Called from DoBackwardComputationForwardingDescriptor.
  void DoBackwardComputationFromSubmatLocationsList(
      int32 deriv_submatrix_index,
      const std::vector<std::vector<std::pair<int32, int32> > >&submat_locations,
      NnetComputation *computation) const;


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


  // [to be called after steps_ is set up and all the forward and backprop
  // commands have been added].  Adds to the computation the commands that
  // deinitialize all the matrices, except those that may be requested by
  // the user after the computation is done (i.e. outputs of the network,
  // and input derivatives).
  void DeallocateMatrices(NnetComputation *computation);

  // sets up the debug_info member of "computation".
  void OutputDebugInfo(NnetComputation *computation) const;

  void AddCommands(const std::vector<bool> &deriv_needed,
                   NnetComputation *computation);

};




} // namespace nnet3
} // namespace kaldi


#endif
