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
#include "nnet3/nnet-computation.h"

namespace kaldi {
namespace nnet3 {

bool ComputationRequest::NeedDerivatives() const {
  bool ans = false;
  if (need_model_derivative)
    ans = true;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].has_deriv) { // derivative requested for this input
      ans = true;
      break;
    }
  }
  if (ans) {
    // check that the output actually provides a derivative, else the
    // request could not be meaningfully satisfied.
    size_t i;
    for (i = 0; i < outputs.size(); i++)
      if (outputs[i].has_deriv)
        break;
    if (i == outputs.size()) {
      KALDI_ERR << "You requested model derivatives or input derivatives, but "
                << "provide no derivatives at the output.";
    }
  }
  return false;
}

int32 ComputationRequest::IndexForInput(
    const std::string &node_name) const {
  int32 ans = -1;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].name == node_name) {
      KALDI_ASSERT(ans == -1 && "Two inputs with the same name");
      ans = i;
    }
  }
  return ans;
}

int32 ComputationRequest::IndexForOutput(
    const std::string &node_name) const {
  int32 ans = -1;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i].name == node_name) {
      KALDI_ASSERT(ans == -1 && "Two inputs with the same name");
      ans = i;
    }
  }
  return ans;
}

NnetComputation::~NnetComputation() {
  for (size_t i = 0; i < component_precomputed_indexes.size(); i++)
    delete component_precomputed_indexes[i];
}

void NnetComputation::ComputeCudaIndexes() {
  indexes_cuda.resize(indexes.size());

  for (size_t i = 0; i < indexes.size(); i++)
    indexes_cuda[i].CopyFromVec(indexes[i]);

  std::vector<bool> need_cuda(indexes_multi.size(), false);
  for (int32 c = 0; c < commands.size(); c++) {
    if (commands[c].command_type == kAddRowRanges) {
      int32 indexes_multi_index = commands[c].arg3;
      KALDI_ASSERT(static_cast<size_t>(indexes_multi_index) < need_cuda.size());
      need_cuda[indexes_multi_index] = true;
    }
  }
  KALDI_ASSERT(sizeof(Int32Pair) == sizeof(std::pair<int32,int32>));
  indexes_multi_cuda.resize(indexes_multi.size());
  for (int32 i = 0; i < indexes_multi.size(); i++) {
    if (need_cuda[i]) {
      const std::vector<std::pair<int32,int32> > *input = &(indexes_multi[i]);
      const std::vector<Int32Pair> *input_cast =
          reinterpret_cast<const std::vector<Int32Pair> *>(input);
      // note: the indexes for CUDA use can't very easily use STL types due to
      // the interface of CUDA being plain C.
      indexes_multi_cuda[i].CopyFromVec(*input_cast);
    }
  }
}

int32 NnetComputation::NewSubMatrix(int32 base_matrix, int32 dim_offset,
                                    int32 dim) {
  KALDI_ASSERT(static_cast<size_t>(base_matrix) < matrices.size());
  int32 num_rows = matrices[base_matrix].num_rows,
      num_cols = matrices[base_matrix].num_cols;
  KALDI_ASSERT(dim_offset >= 0 && dim_offset + dim <= num_cols);
  int32 ans = sub_matrices.size();
  KALDI_ASSERT(ans >= matrices.size());
  sub_matrices.push_back(
      NnetComputation::SubMatrixInfo(base_matrix, 0, num_rows,
                                     dim_offset, dim));
  return ans;
}
  
int32 NnetComputation::NewMatrix(int32 num_rows, int32 num_cols) {
  KALDI_ASSERT(num_rows > 0 && num_cols && 0 &&
               matrices.size() == sub_matrices.size());
  if (matrices.empty()) {  // Set up the zero matrix; index zero is reserved.
    matrices.push_back(MatrixInfo(0, 0));
    sub_matrices.push_back(SubMatrixInfo(0, 0, 0, 0, 0));
  }
  int32 matrix_index = matrices.size(),
      submatrix_index = sub_matrices.size();
  matrices.push_back(MatrixInfo(num_rows, num_cols));
  sub_matrices.push_back(SubMatrixInfo(matrix_index, 0, num_rows, 0, num_cols));
  return submatrix_index;
}

  

} // namespace nnet3
} // namespace kaldi
