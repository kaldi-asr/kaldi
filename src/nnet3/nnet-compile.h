// nnet3/nnet-compile.h

// Copyright 2015-2016    Johns Hopkins University (author: Daniel Povey)

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
  // Constructor that takes one computation request (this is the normal case).
  Compiler(const ComputationRequest &request,
           const Nnet &nnet);

  // Constructor with a sequence of computation requests, for multiple
  // computation segments (used when creating online computations).
  Compiler(const std::vector<const ComputationRequest*> &request,
           const Nnet &nnet);

  void CreateComputation(const CompilerOptions &opts,
                         NnetComputation *computation);

 private:
  // requests_ is the sequence of computation requests, one for each segment; it
  // will contain just one element in the normal case, but more when we're
  // compiling a multi-segment / 'online' computation.
  std::vector<const ComputationRequest*> requests_;
  const Nnet &nnet_;
  ComputationGraph graph_;

  // Some generic information about each step of the computation... a step is an
  // instance of a NetworkNode, but a NetworkNode may in general have multiple
  // steps.  A single step may turn into no commands (for input nodes), or
  // multiple commands.  The StepInfo also contains info about the backprop
  // corresponding to its forward command.
  struct StepInfo {
    int32 node_index;  // network-node index
    int32 value;  // sub-matrix index of value that this step outputs.
    int32 deriv;  // sub-matrix index of derivative at the output of this step; zero
                  // if not used (note: index zero is reserved for the empty
                  // matrix).

    int32 segment;  // normally 0 except for online/multi-segment computations,
                    // identifies the segment of which this step is a part (each
                    // segment in the sequence has a different
                    // ComputationRequest).

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
    // row-index), representing source Cindexes present in a summation, that we
    // store here to avoid computing it twice in forward and backprop.
    std::vector<std::vector<std::vector<std::pair<int32,int32> > > > input_locations_list;

    StepInfo(): node_index(-1), value(0), deriv(0), segment(0),
                precomputed_indexes_index(0) { }
  };

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
  //  'steps' is a vector of steps; each step is a list of cindexes.
  //  'step_to_segment', which should have the same dimension as 'steps',
  //    maps from step index to the segment it occurs in (only interesting
  //    for multi-segment/online computations).
  //  'deriv_needed' will be given the same length as 'steps'.
  void ComputeDerivNeeded(const std::vector<std::vector<int32> > &steps,
                          const std::vector<int32> &step_to_segment,
                          std::vector<bool> *deriv_needed);

  // this sets up steps_, destroying the input "by_step" in the process.  It
  // also sets various matrix and sub-matrix sizes in "computation".  The input
  // 'by_step' is elsewhere referred to as just 'step'; it is a vector of steps,
  // and each step is a vector of cindex_ids that are computed by that step.
  void CreateStepInfo(const std::vector<bool> &deriv_needed,
                      const std::vector<int32> &step_to_segment,
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
  /// exist at >1 location.  (This is not a problem in practice, because we only
  /// need this for the outputs of component-nodes, and for computation inputs).
  /// A location is a pair (step-index, matrix-row-index).
  std::vector<std::pair<int32, int32> > cindex_id_to_location_;


  // Adds to the computation object the information about the matrix sizes
  void DefineMatrices(NnetComputation *computation) const;

  // Sets up sub-matrix indexes for nodes of type Descriptor (needed mainly
  // because Descriptors in general have many parts corresponding to
  // feature-dimension ranges, and they live in sub-matrices.
  void DefineSubmatrices(NnetComputation *computation);

  // Adds to the computation object the commands to allocate the matrices.
  // 'whole_submatrices' is as created by computation->GetWholeSubmatrices(), it
  // gives us the index of a submatrix containing the whole of each matrix.
  void AllocateMatrices(const std::vector<int32> &whole_submatrices,
                        NnetComputation *computation) const;

  // Sets up the precomputed indexes for each component, and sets the
  // precomputed_indexes_index value for each step.
  void SetUpPrecomputedIndexes(const std::vector<int32> &step_to_segment,
                               NnetComputation *computation);

  // Adds to "computation" the command(s) for the forward computation
  // for this step.
  void CompileForward(int32 step, NnetComputation *computation) const;

  // Called from CompileForward, handles the case where the step corresponds
  // to a Component.
  void AddForwardStepComponent(int32 step, NnetComputation *computation) const;

  // Called from CompileForward, handles the case where the step corresponds
  // to an input node.
  void AddForwardStepInput(int32 step, NnetComputation *computation) const;

  // Returns true if step 'step' is an input step.   If step >= steps_.size(),
  // returns false.
  bool IsInputStep(int32 step) const;


  // Called from CompileForward, handles the case where the step
  // corresponds to type kDescriptor
  void CompileForwardDescriptor(
      int32 step, NnetComputation *computation) const;

  void CompileForwardSumDescriptor(
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

  /**
     This function helps to handle scalar factors in Descriptors (expressions
     like `Scale(-1.0, <descriptor)`).  It splits an input_locations_list
     for one SumDescriptor (consisting of one of the appended-together parts
     of a Descriptor) by scale, such that each split-up locations_list
     corresponds to a single scaling factor.
     The scaling factors are all 1.0 by default, but may be different from
     1.0 if the user uses `Scale(...)` expressions in descriptors, e.g.
     `Scale(-1.0, lstm1.z)`.
     To help efficiency, this function treats the case where all the scales
     in the expression are the same (usually 1.0), as a special case.  In this
     case, 'split_locations_lists' will be empty and the shared scale (e.g. 1.0)
     is returned.

       @param [in] descriptor  The SumDescriptor for which we're getting
                      scalar factors.
       @param [in] input_locations_list This is one element of the
                      input_locations_list from the StepInfo of the step we are
                      computing, corresponding to this SumDescriptor (i.e. one
                      part of the Descriptor).  It is indexed by row-index, then
                      it is a list of pairs (step, row-index), representing
                      source Cindexes of a summation.  This function will work
                      out what scaling factors the pairs in these lists have.
       @param [out] split_locations_lists
                      We write to this location.  If all the scaling factors
                      are the same this will be set to the empty list and the
                      common scaling factor returned.  Otherwise +infinity
                      will be returned and the split-up list will be
                      written to the location.  Each element
                      (*split_locations_lists)[i] will be set to a pair
                      (alpha, partial_input_locations_list)
                      where alpha is the scaling factor associated with this
                      split-up piece (e.g. -1.0 if it was part of an expression
                      like `Scale(-1.0, lstm1.z)`), and
                      'partial_input_locations_list' is a vector with the same
                      dimension as 'input_locations_list' (indexed by row-index),
                      where partial_input_locations_list[r] will contain a subset
                      of the pairs present in input_locations_list[r], and
                      if we were to append together all the
                      (*split_locations_lists)[*].second.partial_input_locations_list[r],
                      we'd get a list with the same members as
                      input_locations_list[r], although not necessarily in the same
                      order.
        @return  In the general case (where multiple scales are used), returns
                 +infinity and sets 'split_locations_lists' to the split-up list.
                 In the special, but more common case where only a single scale
                 is used, return that scale (1.0 will be the most common value)
                 and set 'split_locations_lists' to empty; in this special case,
                 which has been made a special case for efficiency reasons,
                 the user should directly use the un-split locations list in
                 'input_locations_list'.
   */
  BaseFloat SplitByScale(const SumDescriptor &descriptor,
   const std::vector<std::vector<std::pair<int32,int32> > > &input_locations_list,
   std::vector<std::pair<BaseFloat,
                         std::vector<std::vector<std::pair<int32,int32> > > > >
                         *split_locations_lists) const;

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



  /** Adds to 'computation' commands for part of the forward computation
      corresponding to a Descriptor.  This is called from
      CompileForwardSumDescriptor.

      @param [in] value_submatrix_index  The submatrix index
               of the quanitity we are computing (part of a Descriptor;
               it's something like Sum(tdnn1, tdnn2) in general).
      @param [in] alpha  The scale (1.0 unless Scale(...) expressions are
               involved in descriptors) with which these terms are present
               in the summation.
      @param [in] submat_locations  Indexed by the row index of
               the submatrix referred to by 'value_submatrix_index', each element is
               a list of sources over which we must sum to obtain
               that row.  Each source is a pair (submatrix-index, row-index).
  */
  void CompileForwardFromSubmatLocationsList(
      int32 value_submatrix_index,
      BaseFloat alpha,
      const std::vector<std::vector<std::pair<int32, int32> > > &submat_locations,
      NnetComputation *computation) const;

  /** Adds to 'computation' commands for part of the forward computation
      corresponding to a Descriptor.  This is called from
      CompileForwardFromSubmatLocationsList.

      @param [in] value_submatrix_index  The submatrix index
               of the quanitity we are computing (part of a Descriptor;
               it's something like Sum(tdnn1, tdnn2) in general).
      @param [in] alpha  The scale (1.0 unless Scale(...) expressions are
               involved in descriptors) with which these terms are present
               in the summation.
      @param [in] submat_locations  Indexed by the row index corresponding
               to the rows of the submatrix referred to by 'value_submatrix_index',
               this reprenents the source vector which we are adding to this row,
               in the format (submatrix-index, row-index), or (-1, -1)
               if in this case there is nothing to add.
       @param [in,out] computation  The computation which we are adding
               commands to.
  */
  void CompileForwardFromSubmatLocations(
      int32 value_submatrix_index,
      BaseFloat alpha,
      const std::vector<std::pair<int32, int32> > &submat_locations,
      NnetComputation *computation) const;


  /** Adds to `computation` a command that adds to the submatrix in
      `value_submatrix_index` a quantity consisting of alpha times
      the submatrix in `input_submatrix_index`, with a row mapping
      given by `indexes`.
  */
  void CompileForwardFromIndexes(
      int32 value_submatrix_index,
      int32 input_submatrix_index,
      BaseFloat alpha,
      const std::vector<int32> &indexes,
      NnetComputation *computation) const;


  // Adds to "computation" the command(s) for the backward computation (if any) for
  // this step.  (non-const only because we clear the cached submat_locations).
  void CompileBackward(int32 step, NnetComputation *computation);

  // Called from CompileBackward, handles the case where the step corresponds
  // to a Component.
  void AddBackwardStepComponent(int32 step, NnetComputation *computation) const;

  // Called from CompileBackward, handles the case where the step
  // corresponds to an input.  If applicable, this generates a command for the
  // network to provide the derivative w.r.t. the input, to the user.
  void AddBackwardStepInput(int32 step, NnetComputation *computation) const;

  // Called from CompileBackward, handles the case where the step
  // corresponds to type kDescriptor.
  void CompileBackwardDescriptor(
      int32 step, NnetComputation *computation);

  // Called from CompileBackwardSumDescriptor.
  void CompileBackwardSumDescriptor(
      int32 step, int32 part_index,
      NnetComputation *computation) const;

  // Called from CompileBackwardForwardingDescriptor.
  void CompileBackwardFromSubmatLocationsList(
      int32 deriv_submatrix_index,
      BaseFloat alpha,
      const std::vector<std::vector<std::pair<int32, int32> > >&submat_locations,
      NnetComputation *computation) const;


  void CompileBackwardFromSubmatLocations(
      int32 deriv_submatrix_index,
      BaseFloat alpha,
      const std::vector<std::pair<int32, int32> > &submat_locations,
      NnetComputation *computation) const;

  // Called from CompileBackwardFromSubmatLocations - special case where
  // input is from just one matrix.
  void CompileBackwardFromIndexes(
      int32 deriv_submatrix_index,
      int32 input_deriv_submatrix_index,
      BaseFloat alpha,
      const std::vector<int32> &indexes,
      NnetComputation *computation) const;


  // [to be called after steps_ is set up and all the forward and backprop
  // commands have been added].  Adds to the computation the commands that
  // deinitialize all the matrices, except those that may be requested by
  // the user after the computation is done (i.e. outputs of the network,
  // and input derivatives).
  // 'whole_submatrices' is as created by computation->GetWholeSubmatrices(), it
  // gives us the index of a submatrix containing the whole of each matrix.
  void DeallocateMatrices(const std::vector<int32> &whole_submatrices,
                          const std::vector<int32> &step_to_segment,
                          NnetComputation *computation);

  // sets up the debug_info member of "computation".
  void OutputDebugInfo(NnetComputation *computation) const;

  void AddCommands(const std::vector<bool> &deriv_needed,
                   const std::vector<int32> &step_to_segment,
                   NnetComputation *computation);

};




} // namespace nnet3
} // namespace kaldi


#endif
