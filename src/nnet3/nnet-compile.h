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
// optimization or sharing of matrices.  It takes the ComputationGraph as input,
// along with some background info.
class ComputationCreator {
 public:
  ComputationCreator(const ComputationRequest &request,
                     const Nnet &nnet,
                     const ComputationGraph &computation_graph);
  
  void CreateComputation(NnetComputation *computation);

 private:
  const ComputationRequest &request_;
  const ComputationGraph &computation_graph_;
  /// shortest_distance_ is shortest distance from inputs (distance 0) to each
  /// cindex_id in the computation graph.
  std::vector<int32> shortest_distance_;
  /// steps_[i] lists the cindex_ids at the output of the i'th component computation.
  std::vector<std::vector<int32> > steps_;

  // This maps each cindex_id to the location
  std::vector<std::pair<int32, int32> > cindex_id_to_location_;

  // Some generic information about each step of the computation... a step
  // is an instance of a NetworkNode, but a NetworkNode may in unrolled computations
  // have multiple steps.
  struct StepInfo {
    int32 node_id;  // network-node id.
    NetworkNode::NodeType node_type;  // enum {kComponent,kInput,kOutput}.
    int32 input_matrix;  // matrix index of input matrix.
    int32 output;  // matrix index of output matrix.
    int32 output_deriv;  // matrix index of output derivative
    int32 input_deriv;  // matrix index of input derivative.
    std::vector<Index> input_indexes;
    // default constructor some elements to -1, but leaves others
    // undefined.
    StepInfo():
        input_matrix(-1), output(-1), output_deriv(-1), input_deriv(-1) { }
  };

  // This
  std::vector<StepInfo> step_info_;

  /// Indexed by the same index as steps_, gives us the input matrix index for
  /// this step.
  std::vector<int32> input_matrices_;
  /// Indexed by the same index as steps_, gives us the list of indexes at the
  /// input location for each step.
  std::vector<std::vector<Index> > input_indexes_;


  // Computes, for each cindex_id, the shortest distance to the input (0 for input
  // components).  It is an error if some cindex_ids are not reachable from the
  // input.
  static void ComputeShortestDistances(
      const ComputationGraph &computation_graph,
      std::vector<int32> *shortest_distance);


  // This works out the steps of the computation and their order.  Once you have
  // the shortest-distances from the input to the Cindexes, this function orders
  // the Cindexes from closest to furthest the input, and then within each
  // category that has the same distance to the input, orders the Cindexes by the
  // component index.  This gives us categories with the same component index and
  // the same distance to the input, and those categories are the steps of the
  // computation.  As special cases, all the accessed input nodes are listed first
  // and all the output nodes are listed last, with each input/output node
  // as a single step; and it makes sure that the order of the Cindexes at
  // the input and output steps is the same as the order specified in the
  // ComputationRequest.
  static void ComputeComputationOrder(
      const ComputationRequest &request,
      const ComputationGraph &computation_graph,
      const std::vector<int32> &shortest_distance,
      std::vector<std::vector<int32> > *steps);

  // This function computes a vector that maps each cindex_id to the
  // corresponding two indices into "steps"; this also gives us
  // the location as (matrix-index, row-index) of each Cindex.
  static void ComputeLocationInfo(
      const std::vector<std::vector<int32> > &steps,
      std::vector<std::pair<int32, int32> > *cindex_id_to_location);


  // Assign some basic information about the steps of the computation, chiefly
  // the locations of the input and output matrices and the corresponding
  // derivatives; and index information about the input of each
  // step (only for real Components[?]).
  // Note, matrix-index zero is at this stage reserved for the empty matrix.
  // In the "computation" object, we store the sizes of these matrices.
  static void GetStepInfo(const ComputationRequest &request,
                          const Nnet &nnet,
                          std::vector<StepInfo> *step_info,
                          NnetComputation *computation);

  // [to be called after step_info_ is set up].  Adds to the computation the
  // commands that initialize all the matrices.  Also defines sub-matrices
  // corresponding to each matrix, with the same index.
  void SetUpForwardMatrices(NnetComputation *computation);

  // Adds to "computation" the commands to set up the input matrix for a step
  // ("step" must correspond to a Component or an output node).  This
  // involves processing the InputDescriptor.
  void SetUpInputForward(int32 step, NnetComputation *computation);

  // Adds to "computation" the command for the forward computation for this step
  // (which must correspond to a real component).
  void DoForwardComputation(int32 step, NnetComputation *computation);

  // Adds to "computation" the commands to do backprop from the input
  // matrix for a step ("step" must correspond to a Component or an output
  // node).  This involves processing the InputDescriptor.
  void BackpropFromInputBackward(int32 step, NnetComputation *computation);
  
  // Adds to "computation" the command for the backward computation for this step
  // (which must correspond to a real component).
  void DoBackwardComputation(int32 step, NnetComputation *computation);
                             
  // [to be called after step_info_ is set up and all the forward and backprop
  // commands have been added].  Adds to the computation the commands that
  // deinitialize all the matrices.
  void DestroyMatrices(NnetComputation *computation);

};



} // namespace nnet3
} // namespace kaldi


#endif

