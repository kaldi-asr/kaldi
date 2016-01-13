// nnet3/nnet-general-component.cc

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
#include "nnet3/nnet-general-component.h"
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {

//inline
void DistributeComponent::ComputeInputIndexAndBlock(const Index &output_index,
                                                    Index *input_index,
                                                    int32 *block) const {
  int32 num_blocks = input_dim_ / output_dim_;
  *input_index = output_index;
  int32 output_x = output_index.x, input_x;
  if (output_x >= 0) {
    input_x = output_x / num_blocks;
  } else {
    input_x = (output_x - num_blocks + 1) / num_blocks;
  }
  input_index->x = input_x;
  if (block)
    *block = output_x - (input_x * num_blocks);
}

//virtual
void DistributeComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  desired_indexes->resize(1);
  ComputeInputIndexAndBlock(output_index, &((*desired_indexes)[0]), NULL);
}

//virtual
bool DistributeComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  Index input_index;
  ComputeInputIndexAndBlock(output_index, &input_index, NULL);
  if (!input_index_set(input_index))
    return false;
  if (used_inputs) {
    used_inputs->clear();
    used_inputs->push_back(input_index);
  }
  return true;
}

class DistributeComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:

  // each pair is a pair (row, dim_offset), and by
  // computing (input.Data() + row * input.Stride() + dim_offset)
  // we get an address that points to the correct input location.
  std::vector<std::pair<int32, int32> > pairs;

  // this class has a virtual destructor so it can be deleted from a pointer
  // to ComponentPrecomputedIndexes.
  virtual ~DistributeComponentPrecomputedIndexes() { }

  virtual ComponentPrecomputedIndexes* Copy() const {
    return new DistributeComponentPrecomputedIndexes(*this);
  }
};

// virtual
ComponentPrecomputedIndexes* DistributeComponent::PrecomputeIndexes(
    const MiscComputationInfo &, // misc_info
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool) const {  // the bool is 'need_backprop'- unused.
  unordered_map<Index, int32, IndexHasher> index_to_input_dim;
  int32 num_input_indexes = input_indexes.size(),
      num_output_indexes = output_indexes.size();
  for (int32 i = 0; i < num_input_indexes; i++)
    index_to_input_dim[input_indexes[i]] = i;
  DistributeComponentPrecomputedIndexes *ans = new
      DistributeComponentPrecomputedIndexes;
  ans->pairs.resize(output_indexes.size());

  int32 num_blocks = input_dim_ / output_dim_,
      block_size = input_dim_ / num_blocks;

  for (int32 i = 0; i < num_output_indexes; i++) {
    Index input_index;
    int32 block_index;
    ComputeInputIndexAndBlock(output_indexes[i], &input_index, &block_index);
    unordered_map<Index, int32, IndexHasher>::iterator iter =
        index_to_input_dim.find(input_index);
    if (iter == index_to_input_dim.end())
      KALDI_ERR << "Input index not found (code error)";
    int32 input_row = iter->second;
    ans->pairs[i] = std::pair<int32,int32>(input_row, block_index * block_size);
  }
  return ans;
}


void DistributeComponent::ComputeInputPointers(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    int32 num_output_rows,
    std::vector<const BaseFloat*> *input_pointers) const {
  const DistributeComponentPrecomputedIndexes *indexes =
      dynamic_cast<const DistributeComponentPrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL && "Invalid pointer type");
  KALDI_ASSERT(num_output_rows == static_cast<int32>(indexes->pairs.size()));
  input_pointers->resize(num_output_rows);

  const BaseFloat *input_data = in.Data();
  int32 input_stride = in.Stride();
  const BaseFloat **input_pointers_data = &((*input_pointers)[0]);
  const std::pair<int32, int32> *pairs_data = &(indexes->pairs[0]);
  for (int32 i = 0; i < num_output_rows; i++) {
    input_pointers_data[i] = input_data +
        pairs_data[i].first * input_stride +
        pairs_data[i].second;
  }
}

void DistributeComponent::ComputeInputPointers(
    const ComponentPrecomputedIndexes *indexes_in,
    int32 num_output_rows,
    CuMatrixBase<BaseFloat> *in,
    std::vector<BaseFloat*> *input_pointers) const {
  const DistributeComponentPrecomputedIndexes *indexes =
      dynamic_cast<const DistributeComponentPrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL && "Invalid pointer type");
  KALDI_ASSERT(num_output_rows == static_cast<int32>(indexes->pairs.size()));
  input_pointers->resize(num_output_rows);

  BaseFloat *input_data = in->Data();
  int32 input_stride = in->Stride();
  BaseFloat **input_pointers_data = &((*input_pointers)[0]);
  const std::pair<int32, int32> *pairs_data = &(indexes->pairs[0]);
  for (int32 i = 0; i < num_output_rows; i++) {
    input_pointers_data[i] = input_data +
        pairs_data[i].first * input_stride +
        pairs_data[i].second;
  }
}


// virtual
void DistributeComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(indexes != NULL &&
               in.NumCols() == input_dim_ && out->NumCols() == output_dim_);
  int32 num_output_rows = out->NumRows();
  std::vector<const BaseFloat*> input_pointers;
  ComputeInputPointers(indexes, in, num_output_rows, &input_pointers);
  CuArray<const BaseFloat*> input_pointers_cuda(input_pointers);
  out->CopyRows(input_pointers_cuda);
}

// virtual
void DistributeComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &, // in_value,
                                   const CuMatrixBase<BaseFloat> &, // out_value
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *, // to_update,
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv == NULL) return;

  int32 num_blocks = input_dim_ / output_dim_,
      num_output_rows = out_deriv.NumRows();
  if (num_output_rows != in_deriv->NumRows() * num_blocks) {
    // there could be some 'gaps', i.e. some input values that are not ever
    // referred to.  So we need to zero the input.  This would't happen in the
    // setups I plan to use this for.
    in_deriv->SetZero();
  }

  std::vector<BaseFloat*> input_pointers;
  ComputeInputPointers(indexes, num_output_rows, in_deriv, &input_pointers);
  CuArray<BaseFloat*> input_pointers_cuda(input_pointers);
  out_deriv.CopyToRows(input_pointers_cuda);
}


void DistributeComponent::Init(int32 input_dim, int32 output_dim) {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KALDI_ASSERT(input_dim > 0 && output_dim > 0 && input_dim % output_dim == 0);
}

// virtual
void DistributeComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim, output_dim;
  bool ok = cfl->GetValue("input-dim", &input_dim) &&
      cfl->GetValue("output-dim", &output_dim);
  if (!ok || cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  else
    Init(input_dim, output_dim);
}

void DistributeComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DistributeComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</DistributeComponent>");
}

void DistributeComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DistributeComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</DistributeComponent>");
}


} // namespace nnet3
} // namespace kaldi
