// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-compile.h"

namespace kaldi {
namespace nnet3 {

ComputationCreator::ComputationCreator(
    const ComputationRequest &request,
    const Nnet &nnet): request_(request), nnet_(nnet) { }


void ComputationCreator::CreateComputation(NnetComputation *computation) {

  ComputeComputationGraph(nnet_, request_, &graph_);
  PruneComputationGraph(nnet_, request_, &graph_);
  std::vector<std::vector<int32> > by_order;
  // see function declaration's comment for meaning of "by_order".
  ComputeComputationOrder(nnet_, request_, graph_, NULL, &by_order);
  std::vector<std::vector<int32> > by_step;
  ComputeComputationSteps(nnet_, request_, graph_, by_order, &by_step);
  by_order.clear();
  CreateLocationInfo(by_step);
  CreateStepInfo(&by_step);

  DefineMatrices(computation);
  DefineSubmatrices(computation);
  int32 arbitrary_factor = 8;
  computation->commands.reserve(num_matrices_ * arbitrary_factor);
  SetUpMatrices(computation);
  int32 num_steps = steps_.size();
  for (int32 step = 0; step < num_steps; step++)
    DoForwardComputation(step, computation);
  if (request_.NeedDerivatives())
    for (int32 step = num_steps; step >= 0; step--)
      DoBackwardComputation(step, computation);
  DestroyMatrices(computation);  
}

void ComputationCreator::CreateStepInfo(
    std::vector<std::vector<int32> > *by_step) {
  KALDI_ASSERT(!by_step->empty());
  KALDI_ASSERT(!cindex_id_to_location_.empty());
  int32 num_steps = by_step->size();
  bool need_derivs = request_.NeedDerivatives();
  steps_.resize(num_steps);
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &this_info = steps_[step];
    this_info.output_cindex_ids.swap((*by_step)[step]);
    int32 num_ids = this_info.output_cindex_ids.size();
    this_info.output_indexes.resize(num_ids);
    for (int32 row_index = 0; row_index < num_ids; row_index++)
      this_info.output_indexes[row_index] =
          graph_.cindexes[this_info.output_cindex_ids[row_index]].second;
    KALDI_ASSERT(num_ids > 0);
    // node id's of all Cindexes are the same, so just use first one.
    this_info.node_index = graph_.cindexes[this_info.output_cindex_ids.front()].first;
    KALDI_ASSERT(this_info.node_index ==
                 graph_.cindexes[this_info.output_cindex_ids.back()].first);

    // set matrix indexes of value, and derivative if needed.  Matrix-index zero
    // is reserved for the empty matrix.
    if (need_derivs) {
      this_info.value = step * 2 + 1;
      this_info.deriv = step * 2 + 2;
    } else {
      this_info.value = step + 1;
    }
    // If this step corresponds to a component, we need to work out the step-index
    // corresponding to its input.
    if (nnet_.GetNode(this_info.node_index).node_type == NetworkNode::kComponent) {
      // choose an arbitrary row index.
      int32 row_index = RandInt(0, this_info.output_cindex_ids.size() - 1) ,
          this_cindex_id = this_info.output_cindex_ids[row_index];
      const std::vector<int32> &dependencies = graph_.dependencies[this_cindex_id];
      KALDI_ASSERT(!dependencies.empty());
      int32 dep_cindex_id = dependencies[0];
      this_info.input_step = cindex_id_to_location_[dep_cindex_id].first;
    }
  }
  num_matrices_ = 1 + num_steps * (need_derivs ? 2 : 1);
}

void ComputationCreator::CreateLocationInfo(const std::vector<std::vector<int32> > &by_step) {
  KALDI_ASSERT(cindex_id_to_location_.empty());
  int32 num_steps = by_step.size();
  for (int32 step = 0; step < num_steps; step++) {
    const std::vector<int32> &output_cindex_ids = by_step[step];
    int32 num_rows = output_cindex_ids.size();
    for (int32 row = 0; row < num_rows; row++) {
      int32 cindex_id = output_cindex_ids[row];
      cindex_id_to_location_[cindex_id] = std::pair<int32,int32>(step, row);
    }
  }
}

// Adds to the computation object the information about the matrix sizes.
void ComputationCreator::DefineMatrices(NnetComputation *computation) const {
  KALDI_ASSERT(computation->matrices.empty());
  computation->matrices.resize(num_matrices_);
  KALDI_ASSERT(num_matrices_ > 0);
  int32 num_steps = steps_.size();
  // matrix-index zero is reserved for the empty matrix.
  computation->matrices[0].num_rows = 0;
  computation->matrices[0].num_cols = 0;
  for (int32 step = 0; step < num_steps; step++) {
    const StepInfo &this_info = steps_[step];
    int32 num_rows = this_info.output_cindex_ids.size(),
        node_index = this_info.node_index,
        num_cols = nnet_.GetNode(node_index).Dim(nnet_);
    if (this_info.value != -1) {
      computation->matrices[this_info.value].num_rows = num_rows;
      computation->matrices[this_info.value].num_cols = num_cols;
    }
    if (this_info.deriv != -1) {
      computation->matrices[this_info.deriv].num_rows = num_rows;
      computation->matrices[this_info.deriv].num_cols = num_cols;
    }
  }
}

void ComputationCreator::DefineSubmatrices(NnetComputation *computation) {
  // First add to the computation all the sub-matrix indexes that correspond to
  // an entire matrix.
  KALDI_ASSERT(computation->sub_matrices.empty());
  int32 num_matrices = computation->matrices.size();
  computation->sub_matrices.resize(num_matrices);
  for (size_t m = 0; m < num_matrices; m++) {
    NnetComputation::MatrixInfo &matrix_info = computation->matrices[m];
    NnetComputation::SubMatrixInfo &submatrix_info = computation->sub_matrices[m];
    submatrix_info.matrix_index = m;
    submatrix_info.row_offset = 0;
    submatrix_info.num_rows = matrix_info.num_rows;
    submatrix_info.col_offset = 0;
    submatrix_info.num_cols = matrix_info.num_cols;
  }
  // check index zero is empty.
  KALDI_ASSERT(computation->sub_matrices[0].num_rows == 0 &&
               computation->sub_matrices[0].num_cols == 0);

  int32 num_steps = steps_.size();
  // Now set up sub-matrices for matrices that are a concatenation over multiple
  // parts.
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &this_info = steps_[step];
    const NetworkNode &node = nnet_.GetNode(this_info.node_index);
    if (node.node_type == NetworkNode::kComponentInput ||
        node.node_type == NetworkNode::kOutput) {
      const Descriptor &desc = node.descriptor;
      KALDI_ASSERT(!desc.parts.empty());
      int32 num_parts = desc.parts.size();
      if (num_parts == 1) {
        this_info.value_parts.push_back(this_info.value);
        this_info.deriv_parts.push_back(this_info.deriv);
      } else { // num_parts > 1.
        int32 cur_dim_offset = 0;
        for (int32 part = 0; part < num_parts; part++) {
          // Have multiple parts, so need to set up sub-matrices.
          this_info.value_parts.resize(num_parts);
          if (this_info.deriv != -1)
            this_info.deriv_parts.resize(num_parts);
          for (int32 part = 0; part < num_parts; part++) {
            int32 this_dim = 1; // <- TODO. HERE
          }
        }
      }
    }
  }
}


void ComputationCreator::DoForwardComputation(int32 step,
                                              NnetComputation *computation) const {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case NetworkNode::kInput: break;  // Nothing to do.
    case NetworkNode::kOutput: case NetworkNode::kComponentInput:
      DoForwardComputationDescriptor(step, node.descriptor, computation);
      break;
    case NetworkNode::kComponent:
      AddPropagateStep(step, computation);
      break;
    default:
      KALDI_ERR << "Invalid node type";
  }      
}

void ComputationCreator::DoBackwardComputation(int32 step,
                                               NnetComputation *computation) const {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case NetworkNode::kInput: break;  // Nothing to do.
    case NetworkNode::kOutput: case NetworkNode::kComponentInput:
      DoBackwardComputationDescriptor(step, node.descriptor, computation);
      break;
    case NetworkNode::kComponent:
      AddBackpropStep(step, computation);
      break;
    default:
      KALDI_ERR << "Invalid node type";
  }      
}


void ComputationCreator::AddPropagateStep(int32 step,
                                          NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 input_step = step_info.input_step;
  KALDI_ASSERT(input_step < step);
  const StepInfo &input_step_info = steps_[input_step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);
  KALDI_ASSERT(node.node_type == NetworkNode::kComponent);  
  
  // in setting the following two variables, we use the fact that the submatrix
  // index of each submatrix that represents an entire matrix, is the same as
  // the matrix index of that matrix.
  int32 input_submatrix_index = input_step_info.value,
      output_submatrix_index = step_info.value;
  NnetComputation::Command c(NnetComputation::kPropagate,
                             node.u.component_index,
                             step_info.precomputed_indexes_index,
                             input_submatrix_index,
                             output_submatrix_index);
  computation->commands.push_back(c);
}


void ComputationCreator::AddBackpropStep(int32 step,
                                         NnetComputation *computation) const {
  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
  const StepInfo &step_info = steps_[step];
  int32 input_step = step_info.input_step;
  KALDI_ASSERT(input_step < step);
  const StepInfo &input_step_info = steps_[input_step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);
  KALDI_ASSERT(node.node_type == NetworkNode::kComponent);
  int32 component_index = node.u.component_index;
  const Component *component = nnet_.GetComponent(component_index);  
  
  // in setting the following two variables, we use the fact that the submatrix
  // index of each submatrix that represents an entire matrix, is the same as
  // the matrix index of that matrix.
  int32 input_submatrix_index = input_step_info.value,
      output_submatrix_index = step_info.value,
      input_deriv_submatrix_index = input_step_info.deriv,
      output_deriv_submatrix_index = step_info.deriv;
  KALDI_ASSERT(input_deriv_submatrix_index > 0 &&
               output_deriv_submatrix_index > 0);
  if (! (component->Properties()&kBackpropNeedsInput))
    input_submatrix_index = 0;
  if (! (component->Properties()&kBackpropNeedsOutput))
    output_submatrix_index = 0;
  
  NnetComputation::Command c(NnetComputation::kPropagate,
                             node.u.component_index,
                             step_info.precomputed_indexes_index,
                             input_submatrix_index,
                             output_submatrix_index,
                             input_deriv_submatrix_index,
                             output_deriv_submatrix_index);
  computation->commands.push_back(c);
}



void ComputationCreator::SetUpMatrices(NnetComputation *computation) const {
  KALDI_ASSERT(computation->commands.empty());
  for (int32 m = 0; m < num_matrices_; m++) {
    // Later in the optimization phase, it turns out that zeroing is not
    // necessary for some matrices, we'll turn these commands into
    // kResizeMatrixUndefined.
    NnetComputation::Command c(NnetComputation::kResizeMatrixZeroed, m);
    computation->commands.push_back(c);
  }
}


void ComputationCreator::SetUpPrecomputedIndexes(
    NnetComputation *computation) {
  int32 num_steps = steps_.size();
  KALDI_ASSERT(computation->component_precomputed_indexes.empty());
  computation->component_precomputed_indexes.push_back(NULL);
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &step_info = steps_[step];
    int32 input_step = step_info.input_step;
    KALDI_ASSERT(static_cast<size_t>(input_step) < steps_.size());
    const StepInfo &input_step_info = steps_[input_step];
    int32 node_index = step_info.node_index;
    const NetworkNode &node = nnet_.GetNode(node_index);
    int32 component_index = node.u.component_index;  
    int32 input_node_index = input_step_info.node_index;
    KALDI_ASSERT(input_node_index == node_index - 1);
    const std::vector<Index> &input_indexes = input_step_info.output_indexes;
    const std::vector<Index> &output_indexes = step_info.output_indexes;
    
    const Component *component = nnet_.GetComponent(component_index);

    bool need_derivs = request_.NeedDerivatives();
    ComponentPrecomputedIndexes *precomputed_indexes =
        component->PrecomputeIndexes(request_.misc_info,
                                     input_indexes, output_indexes,
                                     need_derivs);
    if (precomputed_indexes == NULL) {
      // e.g. simple Components, and some other Components, will return NULL for
      // precomputed_indexes.
      step_info.precomputed_indexes_index = 0;
    } else {
      step_info.precomputed_indexes_index =
          computation->component_precomputed_indexes.size();
      computation->component_precomputed_indexes.push_back(precomputed_indexes);
    }
  }
}



} // namespace nnet3
} // namespace kaldi
