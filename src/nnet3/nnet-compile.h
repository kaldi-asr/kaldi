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

  // Some generic information about each step of the computation... a step
  // is an instance of a NetworkNode, but a NetworkNode may in unrolled computations
  // have multiple steps.  A single step may turn into multiple commands.
  struct StepInfo {
    int32 node_id;  // network-node id.
    NetworkNode::NodeType node_type;  // enum {kComponent,kComponentInput,kInput,kOutput}.
    int32 value;  // matrix index of value that this step outputs.
    int32 deriv;  // matrix index of derivative at the output of this step.


    std::vector<Index> output_indexes;      // Indexes that this step outputs.
    std::vector<int32> cindex_ids;   // cindex_ids for each of the output
                                     // indexes.
    
  
    // default constructor some elements to -1, but leaves others
    // undefined.
    StepInfo(): value
        input_matrix(-1), output(-1), output_deriv(-1), input_deriv(-1) { }
  };


  // Steps of the computation.  Index by step-index.
  std::vector<StepInfo> steps_;

  /// This maps each cindex_id to its location.  A location
  /// is a pair (step-index, matrix-row-index).
  std::vector<std::pair<int32, int32> > cindex_id_to_location_;

  
  /// This function computes a vector that maps each cindex_id to the
  /// corresponding two indices into "steps"; this also gives us the location as
  /// a pair (matrix-index, row-index) of each Cindex.
  /// The input "steps" is a list, indexed by step, of the cindex_ids
  /// computed by that step.
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

