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
#include <iomanip>
#include "nnet3/nnet-general-component.h"
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {

// used in I/O
static void CopyPairVector(const CuArray<Int32Pair> &in,
                        std::vector<std::pair<int32, int32> > *out) {
  in.CopyToVec(reinterpret_cast<std::vector<Int32Pair>*>(out));
}
// used in I/O
static void CopyPairVector(const std::vector<std::pair<int32, int32> > &in,
                        CuArray<Int32Pair> *out) {
  const std::vector<Int32Pair> *in_cast =
      reinterpret_cast<const std::vector<Int32Pair>*>(&in);
  out->CopyFromVec(*in_cast);
}



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

void DistributeComponentPrecomputedIndexes::Write(std::ostream &ostream, bool binary) const {
  WriteToken(ostream, binary, "<DistributeComponentPrecomputedIndexes>");
  WriteToken(ostream, binary, "<Pairs>");
  WriteIntegerPairVector(ostream, binary, pairs);
  WriteToken(ostream, binary, "</DistributeComponentPrecomputedIndexes>");
}

void DistributeComponentPrecomputedIndexes::Read(std::istream &istream, bool binary) {
  ExpectOneOrTwoTokens(istream, binary, "<DistributeComponentPrecomputedIndexes>", "<Pairs>");
  ReadIntegerPairVector(istream, binary, &pairs);
  ExpectToken(istream, binary, "</DistributeComponentPrecomputedIndexes>");
}

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
void* DistributeComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(indexes != NULL &&
               in.NumCols() == input_dim_ && out->NumCols() == output_dim_);
  int32 num_output_rows = out->NumRows();
  std::vector<const BaseFloat*> input_pointers;
  ComputeInputPointers(indexes, in, num_output_rows, &input_pointers);
  CuArray<const BaseFloat*> input_pointers_cuda(input_pointers);
  out->CopyRows(input_pointers_cuda);
  return NULL;
}

// virtual
void DistributeComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &, // in_value,
                                   const CuMatrixBase<BaseFloat> &, // out_value
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   void *memo,
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


void StatisticsExtractionComponentPrecomputedIndexes::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<StatisticsExtractionComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > pairs_cpu;
  CopyPairVector(forward_indexes, &pairs_cpu);
  WriteIntegerPairVector(os, binary, pairs_cpu);
  WriteToken(os, binary, "<Counts>");
  counts.Write(os, binary);
  WriteToken(os, binary, "<BackwardIndexes>");
  std::vector<int32> backward_indexes_cpu;
  backward_indexes.CopyToVec(&backward_indexes_cpu);
  WriteIntegerVector(os, binary, backward_indexes_cpu);
  WriteToken(os, binary, "</StatisticsExtractionComponentPrecomputedIndexes>");
}

void StatisticsExtractionComponentPrecomputedIndexes::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<StatisticsExtractionComponentPrecomputedIndexes>",
                       "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > pairs_cpu;
  ReadIntegerPairVector(is, binary, &pairs_cpu);
  CopyPairVector(pairs_cpu, &forward_indexes);
  ExpectToken(is, binary, "<Counts>");
  counts.Read(is, binary);
  ExpectToken(is, binary, "<BackwardIndexes>");
  std::vector<int32> backward_indexes_cpu;
  ReadIntegerVector(is, binary, &backward_indexes_cpu);
  backward_indexes.CopyFromVec(backward_indexes_cpu);
  ExpectToken(is, binary, "</StatisticsExtractionComponentPrecomputedIndexes>");
}

ComponentPrecomputedIndexes*
StatisticsExtractionComponent::PrecomputeIndexes(
    const MiscComputationInfo &misc_info,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool need_backprop) const {
  int32 num_input_indexes = input_indexes.size(),
      num_output_indexes = output_indexes.size();
  StatisticsExtractionComponentPrecomputedIndexes *ans = new
      StatisticsExtractionComponentPrecomputedIndexes();
  // both input and output indexes are assumed sorted first on
  // n and x, then on t.
  Int32Pair invalid_pair;
  invalid_pair.first = -1;
  invalid_pair.second = -1;
  std::vector<Int32Pair> forward_indexes_cpu(output_indexes.size(),
                                             invalid_pair);
  std::vector<int32> backward_indexes_cpu(input_indexes.size(), -1);
  Vector<BaseFloat> counts_cpu(output_indexes.size());

  // this map maps from Index to the position in 'input_indexes'.
  unordered_map<Index, int32, IndexHasher> index_to_input_pos;
  for (int32 i = 0; i < num_input_indexes; i++)
    index_to_input_pos[input_indexes[i]] = i;

  for (int32 i = 0; i < num_output_indexes; i++) {
    Index output_index = output_indexes[i];
    Index input_index(output_index);
    int32 t = output_index.t,
        t_start = output_period_ * (t / output_period_);
    if (t_start > t)                // could happen for negative t_start due to
      t_start -= output_period_;    // the way modulus works in C.
    int32 t_end = t_start + output_period_;
    for (int32 t = t_start; t < t_end; t += input_period_) {
      input_index.t = t;
      unordered_map<Index, int32, IndexHasher>::iterator iter =
          index_to_input_pos.find(input_index);
      if (iter != index_to_input_pos.end()) {
        int32 input_pos = iter->second;
        if (forward_indexes_cpu[i].first == -1) {
          forward_indexes_cpu[i].first = input_pos;
          forward_indexes_cpu[i].second = input_pos + 1;
          counts_cpu(i) = 1.0;
        } else {
          // the following might fail, for instance, if the sorting
          // of the input or output indexes was not as expected.
          KALDI_ASSERT(forward_indexes_cpu[i].second == input_pos);
          forward_indexes_cpu[i].second++;
          counts_cpu(i) += 1.0;
        }
        KALDI_ASSERT(backward_indexes_cpu[input_pos] == -1);
        backward_indexes_cpu[input_pos] = i;
      }
    }
    KALDI_ASSERT(counts_cpu(i) != 0.0);
  }
  for (int32 i = 0; i < num_input_indexes; i++) {
    KALDI_ASSERT(backward_indexes_cpu[i] != -1);
  }
  ans->forward_indexes = forward_indexes_cpu;
  ans->counts = counts_cpu;
  if (need_backprop)
    ans->backward_indexes = backward_indexes_cpu;
  return ans;
}

StatisticsExtractionComponent::StatisticsExtractionComponent():
    input_dim_(-1), input_period_(1), output_period_(1),
    include_variance_(true) { }

StatisticsExtractionComponent::StatisticsExtractionComponent(
    const StatisticsExtractionComponent &other):
    input_dim_(other.input_dim_),
    input_period_(other.input_period_),
    output_period_(other.output_period_),
    include_variance_(other.include_variance_) {
  Check();
}

void StatisticsExtractionComponent::InitFromConfig(ConfigLine *cfl) {
  // input-dim is required.
  bool ok = cfl->GetValue("input-dim", &input_dim_);
  cfl->GetValue("input-period", &input_period_);
  cfl->GetValue("output-period", &output_period_);
  cfl->GetValue("include-variance", &include_variance_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok || input_dim_ <= 0 || input_period_ <= 0 || output_period_ <= 0 ||
      (output_period_ % input_period_ != 0))
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Check();
}

void StatisticsExtractionComponent::Check() const {
  if (!(input_dim_ > 0 && input_period_ > 0 && output_period_ > 0 &&
        (output_period_ % input_period_) == 0))
    KALDI_ERR << "Invalid configuration of StatisticsExtractionComponent";
}

void StatisticsExtractionComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
    std::sort(input_indexes->begin(), input_indexes->end(),
              IndexLessNxt());
    std::sort(output_indexes->begin(), output_indexes->end(),
              IndexLessNxt());
}

bool StatisticsExtractionComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  Index input_index(output_index);
  int32 t = output_index.t,
      t_start = output_period_ * (t / output_period_);
  if (t_start > t)                // could happen for negative t_start due to
    t_start -= output_period_;    // the way modulus works in C.
  int32 t_end = t_start + output_period_;
  if (!used_inputs) {
    for (int32 t = t_start; t < t_end; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index))
        return true;
    }
    return false;
  } else {
    used_inputs->clear();
    bool ans = false;
    for (int32 t = t_start; t < t_end; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index)) {
        ans = true;
        used_inputs->push_back(input_index);
      }
    }
    return ans;
  }
}

void StatisticsExtractionComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  desired_indexes->clear();
  Index input_index(output_index);
  int32 t = output_index.t,
      t_start = output_period_ * (t / output_period_);
  if (t_start > t)                // could happen for negative t due to
    t_start -= output_period_;    // the way modulus works in C
  int32 t_end = t_start + output_period_;
  for (int32 t = t_start; t < t_end; t += input_period_) {
    input_index.t = t;
    desired_indexes->push_back(input_index);
  }
}


void* StatisticsExtractionComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(indexes_in != NULL);
  const StatisticsExtractionComponentPrecomputedIndexes *indexes =
     dynamic_cast<const StatisticsExtractionComponentPrecomputedIndexes*>(
         indexes_in);
  int32 num_rows_out = out->NumRows();
  KALDI_ASSERT(indexes != NULL &&
               indexes->forward_indexes.Dim() == num_rows_out &&
               in.NumCols() == input_dim_ &&
               out->NumCols() == OutputDim());
  out->SetZero();
  // store the counts.
  out->CopyColFromVec(indexes->counts, 0);
  // store the mean stats
  out->ColRange(1, input_dim_).AddRowRanges(in, indexes->forward_indexes);
  if (include_variance_) {
    // store the variance (sum-squared) stats.
    CuMatrix<BaseFloat> in_squared(in);
    in_squared.ApplyPow(2.0);
    out->ColRange(input_dim_ + 1,
                  input_dim_).AddRowRanges(in_squared,
                                           indexes->forward_indexes);
  }
  return NULL;
}

void StatisticsExtractionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *, // to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(indexes_in != NULL);
  const StatisticsExtractionComponentPrecomputedIndexes *indexes =
      dynamic_cast<const StatisticsExtractionComponentPrecomputedIndexes*>(indexes_in);
  in_deriv->SetZero();
  in_deriv->AddRows(1.0, out_deriv.ColRange(1, input_dim_),
                    indexes->backward_indexes);
  if (include_variance_) {
    CuMatrix<BaseFloat> variance_deriv(in_value.NumRows(),
                                       in_value.NumCols(),
                                       kUndefined);
    variance_deriv.CopyRows(out_deriv.ColRange(1 + input_dim_, input_dim_),
                            indexes->backward_indexes);
    in_deriv->AddMatMatElements(2.0, variance_deriv, in_value, 1.0);
  }
}

void StatisticsExtractionComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<StatisticsExtractionComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<InputPeriod>");
  ReadBasicType(is, binary, &input_period_);
  ExpectToken(is, binary, "<OutputPeriod>");
  ReadBasicType(is, binary, &output_period_);
  ExpectToken(is, binary, "<IncludeVarinance>");
  ReadBasicType(is, binary, &include_variance_);
  ExpectToken(is, binary, "</StatisticsExtractionComponent>");
  Check();
}

void StatisticsExtractionComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<StatisticsExtractionComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<InputPeriod>");
  WriteBasicType(os, binary, input_period_);
  WriteToken(os, binary, "<OutputPeriod>");
  WriteBasicType(os, binary, output_period_);
  WriteToken(os, binary, "<IncludeVarinance>");
  WriteBasicType(os, binary, include_variance_);
  WriteToken(os, binary, "</StatisticsExtractionComponent>");
}

void StatisticsPoolingComponentPrecomputedIndexes::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<StatisticsPoolingComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > indexes_cpu;
  CopyPairVector(forward_indexes, &indexes_cpu);
  WriteIntegerPairVector(os, binary, indexes_cpu);
  WriteToken(os, binary, "<BackwardIndexes>");
  CopyPairVector(backward_indexes, &indexes_cpu);
  WriteIntegerPairVector(os, binary, indexes_cpu);
  WriteToken(os, binary, "</StatisticsPoolingComponentPrecomputedIndexes>");
}

void StatisticsPoolingComponentPrecomputedIndexes::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<StatisticsPoolingComponentPrecomputedIndexes>",
                       "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > indexes_cpu;
  ReadIntegerPairVector(is, binary, &indexes_cpu);
  CopyPairVector(indexes_cpu, &forward_indexes);
  ExpectToken(is, binary, "<BackwardIndexes>");
  ReadIntegerPairVector(is, binary, &indexes_cpu);
  CopyPairVector(indexes_cpu, &backward_indexes);
  ExpectToken(is, binary, "</StatisticsPoolingComponentPrecomputedIndexes>");
}

void StatisticsPoolingComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = cfl->GetValue("input-dim", &input_dim_);
  cfl->GetValue("input-period", &input_period_);
  cfl->GetValue("left-context", &left_context_);
  cfl->GetValue("right-context", &right_context_);
  cfl->GetValue("num-log-count-features", &num_log_count_features_);
  cfl->GetValue("output-stddevs", &output_stddevs_);
  cfl->GetValue("variance-floor", &variance_floor_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  // do some basic checks here but Check() will check more completely.
  if (!ok || input_dim_ <= 0 || left_context_ + right_context_ <= 0 ||
      num_log_count_features_ < 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Check();
}

StatisticsPoolingComponent::StatisticsPoolingComponent():
    input_dim_(-1), input_period_(1), left_context_(-1), right_context_(-1),
    num_log_count_features_(0), output_stddevs_(false),
    variance_floor_(1.0e-10) { }


StatisticsPoolingComponent::StatisticsPoolingComponent(
    const StatisticsPoolingComponent &other):
    input_dim_(other.input_dim_), input_period_(other.input_period_),
    left_context_(other.left_context_), right_context_(other.right_context_),
    num_log_count_features_(other.num_log_count_features_),
    output_stddevs_(other.output_stddevs_),
    variance_floor_(1.0e-10) {
  Check();
}

void StatisticsPoolingComponent::Check() const {
  KALDI_ASSERT(input_dim_ > 0);
  KALDI_ASSERT(input_period_ > 0);
  KALDI_ASSERT(left_context_ >= 0 && right_context_ >= 0 &&
               left_context_ + right_context_ > 0);
  KALDI_ASSERT(left_context_ % input_period_ == 0 &&
               right_context_ % input_period_ == 0);
  KALDI_ASSERT(variance_floor_ > 0.0 && variance_floor_ < 1.0);
  KALDI_ASSERT(!output_stddevs_ || (input_dim_ - 1) % 2 == 0);
}

void StatisticsPoolingComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<StatisticsPoolingComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<InputPeriod>");
  ReadBasicType(is, binary, &input_period_);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<NumLogCountFeatures>");
  ReadBasicType(is, binary, &num_log_count_features_);
  ExpectToken(is, binary, "<OutputStddevs>");
  ReadBasicType(is, binary, &output_stddevs_);
  ExpectToken(is, binary, "<VarianceFloor>");
  ReadBasicType(is, binary, &variance_floor_);
  ExpectToken(is, binary, "</StatisticsPoolingComponent>");
  Check();
}

void StatisticsPoolingComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<StatisticsPoolingComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<InputPeriod>");
  WriteBasicType(os, binary, input_period_);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<NumLogCountFeatures>");
  WriteBasicType(os, binary, num_log_count_features_);
  WriteToken(os, binary, "<OutputStddevs>");
  WriteBasicType(os, binary, output_stddevs_);
  WriteToken(os, binary, "<VarianceFloor>");
  WriteBasicType(os, binary, variance_floor_);
  WriteToken(os, binary, "</StatisticsPoolingComponent>");
}

void StatisticsPoolingComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
    std::sort(input_indexes->begin(), input_indexes->end(),
              IndexLessNxt());
    std::sort(output_indexes->begin(), output_indexes->end(),
              IndexLessNxt());
}

void StatisticsPoolingComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  desired_indexes->clear();
  Index input_index(output_index);
  int32 middle_t = output_index.t,
      t_start = middle_t - left_context_,
      t_last = middle_t + right_context_;
  KALDI_ASSERT(middle_t % input_period_ == 0);
  for (int32 t = t_start; t <= t_last; t += input_period_) {
    input_index.t = t;
    desired_indexes->push_back(input_index);
  }
}

bool StatisticsPoolingComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  if (used_inputs)
    used_inputs->clear();
  // you are not supposed to access the output of this component other than at
  // multiples of the input period.  We could make this an error but decided to
  // just have it return false.
  if (output_index.t % input_period_ != 0)
    return false;

  Index input_index(output_index);
  int32 output_t = output_index.t,
      t_start = output_t - left_context_,
      t_last = output_t + right_context_;
  if (!used_inputs) {
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index))
        return true;
    }
    return false;
  } else {
    bool ans = false;
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index)) {
        ans = true;
        used_inputs->push_back(input_index);
      }
    }
    return ans;
  }
}

ComponentPrecomputedIndexes*
StatisticsPoolingComponent::PrecomputeIndexes(
    const MiscComputationInfo &misc_info,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool need_backprop) const {
  int32 num_input_indexes = input_indexes.size(),
      num_output_indexes = output_indexes.size();
  StatisticsPoolingComponentPrecomputedIndexes *ans = new
      StatisticsPoolingComponentPrecomputedIndexes();

  Int32Pair invalid_pair;
  invalid_pair.first = -1;
  invalid_pair.second = -1;
  // forward_indexes_cpu[i] will be the (begin, end) of input indexes
  // included in the sum for the i'th output index.
  std::vector<Int32Pair> forward_indexes_cpu(num_output_indexes,
                                             invalid_pair);
  // backward_indexes_cpu[i] will be the (begin, end) of output indexes
  // for which the i'th input index participates in the sum.
  // because of the way the indexes are sorted (and the fact that only
  // required indexes are present at the input), it naturally has this
  // structure [i.e. no gaps in the sets of indexes].
  std::vector<Int32Pair> backward_indexes_cpu(num_input_indexes,
                                              invalid_pair);

  // this map maps from Index to the position in 'input_indexes'.
  unordered_map<Index, int32, IndexHasher> index_to_input_pos;
  for (int32 i = 0; i < num_input_indexes; i++)
    index_to_input_pos[input_indexes[i]] = i;

  for (int32 i = 0; i < num_output_indexes; i++) {
    Index input_index(output_indexes[i]);
    int32 middle_t = input_index.t,
        t_start = middle_t - left_context_,
        t_last = middle_t + right_context_;
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      unordered_map<Index, int32, IndexHasher>::iterator iter =
          index_to_input_pos.find(input_index);
      if (iter != index_to_input_pos.end()) {
        int32 input_pos = iter->second;
        if (forward_indexes_cpu[i].first == -1) {
          forward_indexes_cpu[i].first = input_pos;
          forward_indexes_cpu[i].second = input_pos + 1;
        } else {
          KALDI_ASSERT(forward_indexes_cpu[i].second == input_pos);
          forward_indexes_cpu[i].second++;
        }
        if (backward_indexes_cpu[input_pos].first == -1) {
          backward_indexes_cpu[input_pos].first = i;
          backward_indexes_cpu[input_pos].second = i + 1;
        } else {
          KALDI_ASSERT(backward_indexes_cpu[input_pos].second == i);
          backward_indexes_cpu[input_pos].second++;
        }
      }
    }
    KALDI_ASSERT(forward_indexes_cpu[i].first != -1);
  }
  for (int32 i = 0; i < num_input_indexes; i++) {
    KALDI_ASSERT(backward_indexes_cpu[i].first != -1);
  }

  ans->forward_indexes = forward_indexes_cpu;
  if (need_backprop)
    ans->backward_indexes = backward_indexes_cpu;
  return ans;
}

void* StatisticsPoolingComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->SetZero();
  KALDI_ASSERT(indexes_in != NULL);
  const StatisticsPoolingComponentPrecomputedIndexes *indexes =
      dynamic_cast<const StatisticsPoolingComponentPrecomputedIndexes*>(indexes_in);
  int32 num_rows_out = out->NumRows();
  KALDI_ASSERT(indexes != NULL &&
               indexes->forward_indexes.Dim() == num_rows_out &&
               in.NumCols() == input_dim_ &&
               out->NumCols() == OutputDim());
  CuVector<BaseFloat> counts(num_rows_out);
  // counts_mat is a fake matrix with one column, containing the counts.
  CuSubMatrix<BaseFloat> counts_mat(counts.Data(), num_rows_out, 1, 1);
  counts_mat.AddRowRanges(in.ColRange(0, 1), indexes->forward_indexes);

  CuSubMatrix<BaseFloat> out_non_count(*out, 0, num_rows_out,
                                       num_log_count_features_, input_dim_ - 1);
  out_non_count.AddRowRanges(in.ColRange(1, input_dim_ - 1),
                             indexes->forward_indexes);
  out_non_count.DivRowsVec(counts);

  if (num_log_count_features_ > 0) {
    counts.ApplyLog();
    CuVector<BaseFloat> ones(num_log_count_features_, kUndefined);
    ones.Set(1.0);
    out->ColRange(0, num_log_count_features_).AddVecVec(1.0, counts, ones);
  }

  if (output_stddevs_) {
    // if this is true, then we assume the input contains x^2 stats as well as x
    // stats, and we want to process them into a standard deviation.
    KALDI_ASSERT((input_dim_ - 1) % 2 == 0);
    int32 feature_dim = (input_dim_ - 1) / 2;
    CuSubMatrix<BaseFloat> mean(*out, 0, num_rows_out,
                                num_log_count_features_, feature_dim),
        variance(*out, 0, num_rows_out,
                 num_log_count_features_ + feature_dim, feature_dim);
    // subtract mean-squared from average of x^2 to get the variance.
    variance.AddMatMatElements(-1.0, mean, mean, 1.0);
    variance.ApplyFloor(variance_floor_);
    // compute the standard deviation via square root.
    variance.ApplyPow(0.5);
  }
  return NULL;
}

void StatisticsPoolingComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv_in,
    void *memo,
    Component *, // to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(indexes_in != NULL);
  const StatisticsPoolingComponentPrecomputedIndexes *indexes =
      dynamic_cast<const StatisticsPoolingComponentPrecomputedIndexes*>(
          indexes_in);
  int32 num_rows_out = out_deriv_in.NumRows();
  CuMatrix<BaseFloat> out_deriv(out_deriv_in);
  if (output_stddevs_) {
    // for now we actually ignore the covariance flooring in the backprop- this
    // is an approximation.  Typically the derivatives computed will be quite
    // tiny for floored variances (they should be zero), so it won't affect the
    // derivatives much.
    int32 feature_dim = (input_dim_ - 1) / 2;
    CuSubMatrix<BaseFloat> mean_deriv(out_deriv, 0, num_rows_out,
                                      num_log_count_features_, feature_dim),
        variance_deriv(out_deriv, 0, num_rows_out,
                       num_log_count_features_ + feature_dim, feature_dim),
        mean_value(out_value, 0, num_rows_out,
                   num_log_count_features_, feature_dim),
        stddev_value(out_value, 0, num_rows_out,
                     num_log_count_features_ + feature_dim, feature_dim);
    // we currently have the deriv w.r.t. the stddev.  step 1 is to get it
    // w.r.t. the centered variance.  If the centered variance is s,
    // and the stddev is sqrt(s), then d/ds sqrt(s) = 0.5 / sqrt(s),
    // so we need to multiply variance_deriv by 0.5 / the stddev.
    variance_deriv.DivElements(stddev_value);
    variance_deriv.Scale(0.5);

    // the deriv w.r.t. the uncentered variance is the same as w.r.t.  the
    // uncentered variance (since they difer by a constant term of -(mean *
    // mean), but we need to add to dF/dmean, the value -2.0 * mean *
    // dF/dvariance.
    mean_deriv.AddMatMatElements(-2.0, mean_value, variance_deriv, 1.0);
  }
  // now we have to account for the effect of division by the count, on
  // the derivative.
  CuVector<BaseFloat> counts(num_rows_out, kUndefined);
  if (num_log_count_features_ > 0) {
    counts.CopyColFromMat(out_value, 0);
    counts.ApplyExp();
  } else {
    counts.SetZero();
    // we need to recompute the counts from the input since they are not in the
    // output.  The submatrix initializer below takes num-rows, num-cols,
    // stride;  num-cols and stride are 1.
    CuSubMatrix<BaseFloat> counts_mat(counts.Data(), num_rows_out, 1, 1);
    counts_mat.AddRowRanges(in_value.ColRange(0, 1), indexes->forward_indexes);
  }
  // Divide the output derivative by the counts.  This is what we want as it
  // concerns the mean and x^2 stats.  As for the counts themselves, the
  // derivative will end up being discarded when we backprop to the
  // StatisticsExtractionComponent (as the count is not differentiable) so it
  // doesn't really matter.
  out_deriv.DivRowsVec(counts);

  // Now propagate the derivative back to the input.  we don't propagate it
  // back for the count's row since it's non-differentiable.
  in_deriv->ColRange(1, input_dim_ - 1).
      AddRowRanges(out_deriv.ColRange(num_log_count_features_, input_dim_ - 1),
                   indexes->backward_indexes);
}

// virtual
void BackpropTruncationComponent::Read(std::istream &is, bool binary) {
  // might not see the "<NaturalGradientAffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, "<BackpropTruncationComponent>",
                       "<Dim>");
  ReadBasicType(is, binary, &dim_);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<Scale>") {
    ReadBasicType(is, binary, &scale_);
    ReadToken(is, binary, &tok);
  } else {
    scale_ = 1.0;
  }
  KALDI_ASSERT(tok == "<ClippingThreshold>");
  ReadBasicType(is, binary, &clipping_threshold_);
  ExpectToken(is, binary, "<ZeroingThreshold>");
  ReadBasicType(is, binary, &zeroing_threshold_);
  ExpectToken(is, binary, "<ZeroingInterval>");
  ReadBasicType(is, binary, &zeroing_interval_);
  ExpectToken(is, binary, "<RecurrenceInterval>");
  ReadBasicType(is, binary, &recurrence_interval_);
  ExpectToken(is, binary, "<NumElementsClipped>");
  ReadBasicType(is, binary, &num_clipped_);
  ExpectToken(is, binary, "<NumElementsZeroed>");
  ReadBasicType(is, binary, &num_zeroed_);
  ExpectToken(is, binary, "<NumElementsProcessed>");
  ReadBasicType(is, binary, &count_);
  ExpectToken(is, binary, "<NumZeroingBoundaries>");
  ReadBasicType(is, binary, &count_zeroing_boundaries_);
  ExpectToken(is, binary, "</BackpropTruncationComponent>");
}

// virtual
void BackpropTruncationComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BackpropTruncationComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<ClippingThreshold>");
  WriteBasicType(os, binary, clipping_threshold_);
  WriteToken(os, binary, "<ZeroingThreshold>");
  WriteBasicType(os, binary, zeroing_threshold_);
  WriteToken(os, binary, "<ZeroingInterval>");
  WriteBasicType(os, binary, zeroing_interval_);
  WriteToken(os, binary, "<RecurrenceInterval>");
  WriteBasicType(os, binary, recurrence_interval_);
  WriteToken(os, binary, "<NumElementsClipped>");
  WriteBasicType(os, binary, num_clipped_);
  WriteToken(os, binary, "<NumElementsZeroed>");
  WriteBasicType(os, binary, num_zeroed_);
  WriteToken(os, binary, "<NumElementsProcessed>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "<NumZeroingBoundaries>");
  WriteBasicType(os, binary, count_zeroing_boundaries_);
  WriteToken(os, binary, "</BackpropTruncationComponent>");
}

void BackpropTruncationComponentPrecomputedIndexes::Write(std::ostream &ostream,
    bool binary) const {
  WriteToken(ostream, binary,
             "<BackpropTruncationComponentPrecomputedIndexes>");
  WriteToken(ostream, binary, "<Zeroing>");
  zeroing.Write(ostream, binary);
  WriteToken(ostream, binary, "<ZeroingSum>");
  WriteBasicType(ostream, binary, zeroing_sum);
  WriteToken(ostream, binary,
             "</BackpropTruncationComponentPrecomputedIndexes>");
}

void BackpropTruncationComponentPrecomputedIndexes::Read(std::istream &istream,
    bool binary) {
  ExpectOneOrTwoTokens(istream, binary,
                       "<BackpropTruncationComponentPrecomputedIndexes>",
                       "<Zeroing>");
  zeroing.Read(istream, binary);
  ExpectToken(istream, binary, "<ZeroingSum>");
  ReadBasicType(istream, binary, &zeroing_sum);
  ExpectToken(istream, binary,
              "</BackpropTruncationComponentPrecomputedIndexes>");
}

std::string BackpropTruncationComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", scale=" << scale_
         << ", count=" << std::setprecision(3) << count_ << std::setprecision(6)
         << ", recurrence-interval=" << recurrence_interval_
         << ", clipping-threshold=" << clipping_threshold_
         << ", clipped-proportion="
         << (count_ > 0.0 ? num_clipped_ / count_ : 0)
         << ", zeroing-threshold=" << zeroing_threshold_
         << ", zeroing-interval=" << zeroing_interval_
         << ", zeroed-proportion="
         << (count_zeroing_boundaries_ > 0.0 ?
             num_zeroed_ / count_zeroing_boundaries_ : 0)
         << ", count-zeroing-boundaries="
         << static_cast<int32>(count_zeroing_boundaries_);
  return stream.str();
}

void BackpropTruncationComponent::Init(
    int32 dim, BaseFloat scale, BaseFloat clipping_threshold,
    BaseFloat zeroing_threshold, int32 zeroing_interval,
    int32 recurrence_interval) {
  KALDI_ASSERT(clipping_threshold >= 0 && zeroing_threshold >= 0 &&
               scale > 0.0 && zeroing_interval > 0 &&
               recurrence_interval > 0 && dim > 0);
  dim_ = dim;
  scale_ = scale;
  clipping_threshold_ = clipping_threshold;
  zeroing_threshold_ = zeroing_threshold;
  zeroing_interval_ = zeroing_interval;
  recurrence_interval_ = recurrence_interval;
  num_clipped_ = 0.0;
  num_zeroed_ = 0.0;
  count_ = 0.0;
  count_zeroing_boundaries_ = 0.0;
}

// virtual
void BackpropTruncationComponent::InitFromConfig(ConfigLine *cfl) {
  int32 dim = 0;
  bool ok = cfl->GetValue("dim", &dim);
  BaseFloat scale = 1.0,
      clipping_threshold = 30.0,
      zeroing_threshold = 15.0;
  int32 zeroing_interval = 20, recurrence_interval = 1;
  cfl->GetValue("scale", &scale);
  cfl->GetValue("clipping-threshold", &clipping_threshold);
  cfl->GetValue("zeroing-threshold", &zeroing_threshold);
  cfl->GetValue("zeroing-interval", &zeroing_interval);
  cfl->GetValue("recurrence-interval", &recurrence_interval);
  if (!ok || cfl->HasUnusedValues() ||
      clipping_threshold < 0 || zeroing_threshold < 0 || zeroing_interval < 1 ||
      recurrence_interval < 1 || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim, scale, clipping_threshold, zeroing_threshold,
      zeroing_interval, recurrence_interval);
}

// virtual
Component* BackpropTruncationComponent::Copy() const {
  BackpropTruncationComponent *ans = new BackpropTruncationComponent();
  ans->dim_ = dim_;
  ans->scale_ = scale_;
  ans->clipping_threshold_ = clipping_threshold_;
  ans->zeroing_threshold_ = zeroing_threshold_;
  ans->zeroing_interval_ = zeroing_interval_;
  ans->recurrence_interval_ = recurrence_interval_;
  ans->num_clipped_ = num_clipped_;
  ans->num_zeroed_ = num_zeroed_;
  ans->count_ = count_;
  ans->count_zeroing_boundaries_ = count_zeroing_boundaries_;
  return ans;
}

// virtual
ComponentPrecomputedIndexes*
BackpropTruncationComponent::PrecomputeIndexes(
    const MiscComputationInfo &misc_info,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool need_backprop) const {
  int32 num_input_indexes = input_indexes.size(),
      num_output_indexes = output_indexes.size();
  KALDI_ASSERT(num_input_indexes == num_output_indexes);
  Vector<BaseFloat> zeroing_cpu(num_output_indexes);

  for (int32 i = 0; i < num_output_indexes; i++) {
    const int32 output_n = output_indexes[i].n;
    const int32 output_t = output_indexes[i].t;
    // checks if output_t crosses a boundary that is a multiple of
    // zeroing_interval_. Note that frame (output_t - recurrence_interval_) is
    // right before frame output_t in RNNs. If the range
    // [output_t - recurrence_interval_, output_t] contains a multiple of
    // zeroing_interval_, then frame output_t crosses the boundary.
    // output_n is used to shift where we put the boundary, so that
    // we don't always zero out gradients on frame 0. It will help avoid
    // learning utterance-boundary effects.
    if (DivideRoundingDown(output_t - output_n, zeroing_interval_) !=
        DivideRoundingDown(output_t - recurrence_interval_ - output_n,
        zeroing_interval_))
      zeroing_cpu(i) = -1.0;
  }

  BackpropTruncationComponentPrecomputedIndexes *ans = new
      BackpropTruncationComponentPrecomputedIndexes();
  ans->zeroing = zeroing_cpu;
  ans->zeroing_sum = -zeroing_cpu.Sum();
  return ans;
}

// virtual
void* BackpropTruncationComponent::Propagate(
                                 const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  if (scale_ != 1.0)
    out->Scale(scale_);
  return NULL;
}

// virtual
void BackpropTruncationComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes_in,
                             const CuMatrixBase<BaseFloat> &, //in_value
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update_in, // may be NULL; may be
                             // identical to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  const BackpropTruncationComponentPrecomputedIndexes *indexes =
      dynamic_cast<const BackpropTruncationComponentPrecomputedIndexes*>(
          indexes_in);
  KALDI_ASSERT(indexes->zeroing.Dim() == out_deriv.NumRows());
  // the following statement will do nothing if in_deriv and out_deriv have same
  // memory.
  in_deriv->CopyFromMat(out_deriv);
  if (scale_ != 1.0)
    in_deriv->Scale(scale_);

  BackpropTruncationComponent *to_update =
      dynamic_cast<BackpropTruncationComponent*>(to_update_in);

  // computes clipping_scales
  BaseFloat clipping_threshold =
      (clipping_threshold_ <= 0.0 ? 1.0e+10 : clipping_threshold_);
  // each row in the derivative matrix, which corresponds to one sample in
  // the mini-batch, is scaled to have a max-norm of clipping_threshold_
  CuVector<BaseFloat> clipping_scales(in_deriv->NumRows());
  clipping_scales.AddDiagMat2(pow(clipping_threshold, -2), *in_deriv,
                              kNoTrans, 0.0);
  // now clipping_scales contains the squared (norm of each row divided by
  //  clipping_threshold)
  int32 num_not_scaled;
  clipping_scales.ApplyFloor(1.0, &num_not_scaled);
  // now clipping_scales contains min(1, squared-(norm/clipping_threshold))
  clipping_scales.ApplyPow(-0.5);
  // now clipping_scales contains max(1, clipping_threshold/vector_norm)
  if (to_update != NULL) {
    to_update->num_clipped_ += (clipping_scales.Dim() - num_not_scaled);
    to_update->count_ += clipping_scales.Dim();
  }

  // computes zeroing_scales
  BaseFloat zeroing_threshold =
      (zeroing_threshold_ <= 0.0 ? 1.0e+10 : zeroing_threshold_);
  // zeroing_scales_vec is actually a 1-row matrix.  (the ApplyHeaviside
  // function isn't defined for vectors).
  CuMatrix<BaseFloat> zeroing_scales(1, in_deriv->NumRows());
  CuSubVector<BaseFloat> zeroing_scales_vec(zeroing_scales, 0);
  zeroing_scales_vec.Set(-pow(zeroing_threshold, 2));
  // now zeroing_scales_vec contains -(squared zeroing_threshold)
  zeroing_scales_vec.AddDiagMat2(1.0, *in_deriv, kNoTrans, 1.0);
  // now zeroing_scales_vec contains squared norm of each row -
  // squared zeroing_threshold
  zeroing_scales.ApplyHeaviside();
  // now the element of zeroing_scales_vec is 1.0 if its corresponding
  // sample's norm exceeds zero_threshold, and 0.0 otherwise
  zeroing_scales_vec.MulElements(indexes->zeroing);
  // now the element of zeroing_scales_vec is -1.0 if we want to zero its
  // corresponding sample's gradient, and 0.0 otherwise
  if (to_update != NULL) {
    to_update->num_zeroed_ -= zeroing_scales_vec.Sum(); // since it is negative
    to_update->count_zeroing_boundaries_ += indexes->zeroing_sum;
  }
  zeroing_scales_vec.Add(1.0);
  // now the element of zeroing_scales_vec is 0.0 if we want to zero its
  // corresponding sample's gradient, and 1.0 otherwise

  // combines clipping_scales and zeroing_scales and applies combined_scales
  // to in_deriv all at once
  CuVector<BaseFloat> combined_scales(clipping_scales);
  combined_scales.MulElements(zeroing_scales_vec);
  in_deriv->MulRowsVec(combined_scales);
}

// virtual
void BackpropTruncationComponent::ZeroStats()  {
  count_ = 0.0;
  count_zeroing_boundaries_ = 0.0;
  num_clipped_ = 0.0;
  num_zeroed_ = 0.0;
}

// virtual
void BackpropTruncationComponent::Scale(BaseFloat scale) {
  count_ *= scale;
  count_zeroing_boundaries_ *= scale;
  num_clipped_ *= scale;
  num_zeroed_ *= scale;
}

// virtual
void BackpropTruncationComponent::Add(BaseFloat alpha,
                                      const Component &other_in) {
  const BackpropTruncationComponent *other =
      dynamic_cast<const BackpropTruncationComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  count_ += alpha * other->count_;
  count_zeroing_boundaries_ += alpha * other->count_zeroing_boundaries_;
  num_clipped_ += alpha * other->num_clipped_;
  num_zeroed_ += alpha * other->num_zeroed_;
}


std::string ConstantComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", " << Type()
         << ", output-dim=" << OutputDim()
         << ", is-updatable=" << std::boolalpha << is_updatable_
         << ", use-natural-gradient=" << std::boolalpha
         << use_natural_gradient_;
  PrintParameterStats(stream, "output", output_, true);
  return stream.str();
}

ConstantComponent::ConstantComponent():
    UpdatableComponent(), is_updatable_(true),
    use_natural_gradient_(true) { }

ConstantComponent::ConstantComponent(
    const ConstantComponent &other):
    UpdatableComponent(other), output_(other.output_),
    is_updatable_(other.is_updatable_),
    use_natural_gradient_(other.use_natural_gradient_),
    preconditioner_(other.preconditioner_) { }

void* ConstantComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyRowsFromVec(output_);
  return NULL;
}

void ConstantComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  // we don't update in_deriv, since we set the flag
  // kBackpropAdds, and the output doesn't depend on the
  // input, so the input-derivative is zero.
  if (to_update_in) {
    ConstantComponent *to_update =
      dynamic_cast<ConstantComponent*>(to_update_in);
    if (to_update->is_updatable_) {
      // only do the update if the is_updatable_ flag is set.
      KALDI_ASSERT(to_update && to_update->is_updatable_);
      if (to_update->use_natural_gradient_ && !to_update->is_gradient_) {
        CuMatrix<BaseFloat> out_deriv_copy(out_deriv);
        BaseFloat scale = 1.0;
        to_update->preconditioner_.PreconditionDirections(&out_deriv_copy,
                                                          &scale);
        to_update->output_.AddRowSumMat(scale * to_update->learning_rate_,
                                        out_deriv_copy);
      } else {
        to_update->output_.AddRowSumMat(to_update->learning_rate_,
                                        out_deriv);
      }
    }
  }
}

void ConstantComponent::Read(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<ConstantComponent>") {
    ReadToken(is, binary, &token);
  }
  if (token == "<LearningRateFactor>") {
    ReadBasicType(is, binary, &learning_rate_factor_);
    ReadToken(is, binary, &token);
  } else {
    learning_rate_factor_ = 1.0;
  }
  if (token == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ReadToken(is, binary, &token);
  } else {
    is_gradient_ = false;
  }
  if (token == "<MaxChange>") {
    ReadBasicType(is, binary, &max_change_);
    ReadToken(is, binary, &token);
  } else {
    max_change_ = 0.0;
  }
  if (token == "<LearningRate>") {
    ReadBasicType(is, binary, &learning_rate_);
    ReadToken(is, binary, &token);
  } else {
    learning_rate_ = 0.001;
  }
  if (token != "<Output>") {
    KALDI_ERR << "Expected token <Output>, got " << token;
  }
  output_.Read(is, binary);
  ExpectToken(is, binary, "<IsUpdatable>");
  ReadBasicType(is, binary, &is_updatable_);
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);
  ExpectToken(is, binary, "</ConstantComponent>");
}

void ConstantComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<Output>");
  output_.Write(os, binary);
  WriteToken(os, binary, "<IsUpdatable>");
  WriteBasicType(os, binary, is_updatable_);
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  WriteToken(os, binary, "</ConstantComponent>");
}

Component* ConstantComponent::Copy() const {
  return new ConstantComponent(*this);
}

void ConstantComponent::Scale(BaseFloat scale) {
  if (is_updatable_) {
    if (scale == 0.0) {
      output_.SetZero();
    } else {
      output_.Scale(scale);
    }
  }
}

void ConstantComponent::Add(BaseFloat alpha, const Component &other_in) {
  if (is_updatable_) {
    const ConstantComponent *other =
        dynamic_cast<const ConstantComponent*>(&other_in);
    KALDI_ASSERT(other != NULL);
    output_.AddVec(alpha, other->output_);
  }
}

void ConstantComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_output(output_.Dim(), kUndefined);
  temp_output.SetRandn();
  output_.AddVec(stddev, temp_output);
}

BaseFloat ConstantComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  KALDI_ASSERT(is_updatable_);
  const ConstantComponent *other =
      dynamic_cast<const ConstantComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return VecVec(output_, other->output_);
}

void ConstantComponent::InitFromConfig(ConfigLine *cfl) {
  int32 output_dim = 0;
  InitLearningRatesFromConfig(cfl);
  bool ok = cfl->GetValue("output-dim", &output_dim);
  cfl->GetValue("is-updatable", &is_updatable_);
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  BaseFloat output_mean = 0.0, output_stddev = 0.0;
  cfl->GetValue("output-mean", &output_mean);
  cfl->GetValue("output-stddev", &output_stddev);
  if (!ok || cfl->HasUnusedValues() || output_dim <= 0) {
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
  }
  Vector<BaseFloat> output(output_dim);
  output.SetRandn();
  output.Scale(output_stddev);
  output.Add(output_mean);
  output_ = output;
}

int32 ConstantComponent::NumParameters() const {
  KALDI_ASSERT(is_updatable_);
  return output_.Dim();
}

void ConstantComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyFromVec(output_);
}

void ConstantComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  output_.CopyFromVec(params);
}

void ConstantComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp(preconditioner_);
  preconditioner_.Swap(&temp);
}

std::string DropoutMaskComponent::Info() const {
  std::ostringstream stream;
  stream << Type()
         << ", output-dim=" << output_dim_
         << ", dropout-proportion=" << dropout_proportion_;
  if (continuous_)
    stream << ", continuous=true";
  return stream.str();
}

DropoutMaskComponent::DropoutMaskComponent():
    output_dim_(-1), dropout_proportion_(0.5), continuous_(false) { }

DropoutMaskComponent::DropoutMaskComponent(
    const DropoutMaskComponent &other):
    output_dim_(other.output_dim_),
    dropout_proportion_(other.dropout_proportion_),
    continuous_(other.continuous_) { }

void* DropoutMaskComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == 0 && out->NumCols() == output_dim_);
  BaseFloat dropout_proportion = dropout_proportion_;
  KALDI_ASSERT(dropout_proportion >= 0.0 && dropout_proportion <= 1.0);

  if (dropout_proportion == 0) {
    out->Set(1.0);
    return NULL;
  }

  if (continuous_) {
    if (test_mode_) {
      out->Set(1.0);
    } else {
      const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);
      out->Scale(dropout_proportion * 4.0);
      // make the expected value 1.0.
      out->Add(1.0 - (2.0 * dropout_proportion));
    }
    return NULL;
  }

  if (test_mode_) {
    out->Set(1.0 - dropout_proportion);
    return NULL;
  }

  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);
  out->Add(-dropout_proportion);
  out->ApplyHeaviside();

  if (out->NumCols() == 2 || out->NumCols() == 3) {
    // This is a kind of special case relevant to LSTms.
    // To generate data where it's never the case that both of the dimensions
    // for a row are zero, we generate uniformly distributed data (call this u_i),
    // and for row i, set (*out)(i, 0) = (0 if u_i < dropout_proportion else 1)
    //                and (*out)(i, 1) = (0 if u_i > 1-dropout_proportion else 1)
    int32 num_rows = out->NumRows();
    // later we may make this a bit more efficient.
    CuVector<BaseFloat> temp(num_rows, kUndefined);
    const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&temp);
    temp.Add(-dropout_proportion);
    out->CopyColFromVec(temp, 0);
    temp.Add(-1.0 + (2.0 * dropout_proportion));
    // Now, 'temp' contains the original uniformly-distributed data plus
    // -(1 - dropout_proportion).
    temp.Scale(-1.0);
    out->CopyColFromVec(temp, 1);
    out->ApplyHeaviside();
  }
  return NULL;
}


void DropoutMaskComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DropoutMaskComponent>", "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<DropoutProportion>");
  ReadBasicType(is, binary, &dropout_proportion_);
  if (PeekToken(is, binary) == 'T') {
    ExpectToken(is, binary, "<TestMode>");
    ReadBasicType(is, binary, &test_mode_);  // read test mode
  } else {
    test_mode_ = false;
  }
  if (PeekToken(is, binary) == 'C') {
    ExpectToken(is, binary, "<Continuous>");
    continuous_ = true;
  } else {
    continuous_ = false;
  }
  ExpectToken(is, binary, "</DropoutMaskComponent>");
}


void DropoutMaskComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DropoutMaskComponent>");
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<DropoutProportion>");
  WriteBasicType(os, binary, dropout_proportion_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  if (continuous_)
    WriteToken(os, binary, "<Continuous>");
  WriteToken(os, binary, "</DropoutMaskComponent>");
}

Component* DropoutMaskComponent::Copy() const {
  return new DropoutMaskComponent(*this);
}

void DropoutMaskComponent::InitFromConfig(ConfigLine *cfl) {
  output_dim_ = 0;
  bool ok = cfl->GetValue("output-dim", &output_dim_);
  KALDI_ASSERT(ok && output_dim_ > 0);
  dropout_proportion_ = 0.5;
  cfl->GetValue("dropout-proportion", &dropout_proportion_);
  continuous_ = false;
  cfl->GetValue("continuous", &continuous_);
  test_mode_ = false;
  cfl->GetValue("test-mode", &test_mode_);
}


std::string GeneralDropoutComponent::Info() const {
  std::ostringstream stream;
  stream << Type()
         << ", dim=" << dim_
         << ", block-dim=" << block_dim_
         << ", dropout-proportion=" << dropout_proportion_;
  if (continuous_)
    stream << ", continuous=true";
  if (time_period_ > 0)
    stream << ", time-period=" << time_period_;
  return stream.str();
}

GeneralDropoutComponent::GeneralDropoutComponent():
    dim_(-1), block_dim_(-1), time_period_(0),
    dropout_proportion_(0.5), continuous_(false) { }

GeneralDropoutComponent::GeneralDropoutComponent(
    const GeneralDropoutComponent &other):
    dim_(other.dim_),
    block_dim_(other.block_dim_),
    time_period_(other.time_period_),
    dropout_proportion_(other.dropout_proportion_),
    continuous_(other.continuous_) { }

void* GeneralDropoutComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {

  KALDI_ASSERT(SameDim(in, *out));

  // The following will do nothing if 'out' and 'in' refer to the same data.
  out->CopyFromMat(in);

  if (test_mode_ || dropout_proportion_ == 0.0)
    return NULL;

  const GeneralDropoutComponentPrecomputedIndexes *indexes =
    dynamic_cast<const GeneralDropoutComponentPrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL);

  CuMatrix<BaseFloat> *mask = GetMemo(indexes->num_mask_rows);

  if (block_dim_ < dim_) {
    KALDI_ASSERT(out->Stride() == out->NumCols());
    int32 num_rows = out->NumRows(),
        dim_multiple = dim_  / block_dim_,
        num_rows_reshaped = num_rows * dim_multiple;
    CuSubMatrix<BaseFloat> out_reshaped(out->Data(), num_rows_reshaped,
                                        block_dim_, block_dim_);
    out_reshaped.MulRows(*mask, indexes->indexes);
  } else {
    out->MulRows(*mask, indexes->indexes);
  }
  return mask;
}

void GeneralDropoutComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(in_deriv != NULL && SameDim(*in_deriv, out_deriv));

  // The following will do no work if in_deriv->Data() == out_deriv.Data().
  in_deriv->CopyFromMat(out_deriv);

  if (test_mode_ || dropout_proportion_ == 0.0) {
    KALDI_ASSERT(memo == NULL);
    return;
  }

  const GeneralDropoutComponentPrecomputedIndexes *indexes =
     dynamic_cast<const GeneralDropoutComponentPrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL && memo != NULL);
  CuMatrix<BaseFloat> *mask = reinterpret_cast<CuMatrix<BaseFloat>*>(memo);

  if (block_dim_ < dim_) {
    KALDI_ASSERT(in_deriv->Stride() == in_deriv->NumCols());
    int32 num_rows = in_deriv->NumRows(),
        dim_multiple = dim_  / block_dim_,
        num_rows_reshaped = num_rows * dim_multiple;
    CuSubMatrix<BaseFloat> in_deriv_reshaped(in_deriv->Data(),
                                             num_rows_reshaped,
                                             block_dim_, block_dim_);
    in_deriv_reshaped.MulRows(*mask, indexes->indexes);
  } else {
    in_deriv->MulRows(*mask, indexes->indexes);
  }
}

void GeneralDropoutComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<GeneralDropoutComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<BlockDim>");
  ReadBasicType(is, binary, &block_dim_);
  ExpectToken(is, binary, "<TimePeriod>");
  ReadBasicType(is, binary, &time_period_);
  ExpectToken(is, binary, "<DropoutProportion>");
  ReadBasicType(is, binary, &dropout_proportion_);
  if (PeekToken(is, binary) == 'T') {
    ExpectToken(is, binary, "<TestMode>");
    test_mode_ = true;
  } else {
    test_mode_ = false;
  }
  if (PeekToken(is, binary) == 'C') {
    ExpectToken(is, binary, "<Continuous>");
    continuous_ = true;
  } else {
    continuous_ = false;
  }
  ExpectToken(is, binary, "</GeneralDropoutComponent>");
}


void GeneralDropoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<GeneralDropoutComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<BlockDim>");
  WriteBasicType(os, binary, block_dim_);
  WriteToken(os, binary, "<TimePeriod>");
  WriteBasicType(os, binary, time_period_);
  WriteToken(os, binary, "<DropoutProportion>");
  WriteBasicType(os, binary, dropout_proportion_);
  if (test_mode_)
    WriteToken(os, binary, "<TestMode>");
  if (continuous_)
    WriteToken(os, binary, "<Continuous>");
  WriteToken(os, binary, "</GeneralDropoutComponent>");
}

Component* GeneralDropoutComponent::Copy() const {
  return new GeneralDropoutComponent(*this);
}

void GeneralDropoutComponent::InitFromConfig(ConfigLine *cfl) {
  dim_ = 0;
  bool ok = cfl->GetValue("dim", &dim_);
  KALDI_ASSERT(ok && dim_ > 0);
  block_dim_ = dim_;
  cfl->GetValue("block-dim", &block_dim_);
  if (!(block_dim_ > 0 && dim_ % block_dim_ == 0))
    KALDI_ERR << "Invalid configuration dim=" << dim_
              << ", block-dim=" << block_dim_;
  time_period_ = 0;
  cfl->GetValue("time-period", &time_period_);
  dropout_proportion_ = 0.5;
  cfl->GetValue("dropout-proportion", &dropout_proportion_);
  continuous_ = false;
  cfl->GetValue("continuous", &continuous_);
  test_mode_ = false;
  cfl->GetValue("test-mode", &test_mode_);
}


CuMatrix<BaseFloat>* GeneralDropoutComponent::GetMemo(
    int32 num_mask_rows) const {
  KALDI_ASSERT(num_mask_rows > 0 && !test_mode_ &&
               dropout_proportion_ > 0.0);
  CuMatrix<BaseFloat> *ans = new CuMatrix<BaseFloat>(num_mask_rows, block_dim_);
  BaseFloat dropout_proportion = dropout_proportion_;

  // This const_cast is only safe assuming you don't attempt
  // to use multi-threaded code with the GPU.
  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(ans);

  if (!continuous_) {
    ans->Add(-dropout_proportion);
    // now, a proportion "dropout_proportion" will be < 0.0. After applying the
    // function (x>0?1:0), a proportion "dropout_proportion" will be zero and (1 -
    // dropout_proportion) will be 1.0.
    ans->ApplyHeaviside();
    ans->Scale(1.0 / (1.0 - dropout_proportion));
  } else {
    ans->Scale(dropout_proportion * 4.0);
    // make the expected value 1.0.
    ans->Add(1.0 - (2.0 * dropout_proportion));
  }
  return ans;
}

ComponentPrecomputedIndexes* GeneralDropoutComponent::PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const {
  KALDI_ASSERT(input_indexes == output_indexes);

  GeneralDropoutComponentPrecomputedIndexes *ans = new
      GeneralDropoutComponentPrecomputedIndexes;
  int32 size = input_indexes.size(), time_period = time_period_,
      cur_row = 0;
  std::vector<int32> indexes(size);
  // the map 'm' will map from a pair from (n, t) value to the row-index of the
  // dropout-mask matrix*.   However, the 't' isn't a real 't' value;
  // if time_period_ == 0, the 't' value will just be zero; otherwise,
  // it will be t divided by time_period_ (rounding towards negative infinity).

  // *before considering effects related to when block_dim_ != dim_.

  std::unordered_map<std::pair<int32,int32>, int32, PairHasher<int32> > m;
  for (int32 i = 0; i < size; i++) {
    int32 n = input_indexes[i].n,
        t = (time_period == 0 ? 0 : DivideRoundingDown(input_indexes[i].t,
                                                       time_period));
    std::pair<int32, int32> p(n, t);

    std::unordered_map<std::pair<int32,int32>, int32,
                       PairHasher<int32> >::const_iterator
        iter = m.find(p);
    if (iter != m.end()) {
      indexes[i] = iter->second;
    } else {
      m[p] = cur_row;
      indexes[i] = cur_row;
      cur_row++;
    }
  }
  int32 multiple = dim_ / block_dim_;
  ans->num_mask_rows = cur_row;
  if (multiple == 1) {
    ans->indexes.CopyFromVec(indexes);
  } else {
    ans->num_mask_rows = cur_row * multiple;
    std::vector<int32> repeated_indexes;
    repeated_indexes.reserve(size * multiple);
    for (int32 i = 0; i < size; i++) {
      int32 row = indexes[i];
      for (int32 j = 0; j < multiple; j++)
        repeated_indexes.push_back(row);
    }
    ans->indexes.CopyFromVec(repeated_indexes);
  }
  return ans;
}

void GeneralDropoutComponentPrecomputedIndexes::Write(std::ostream &os,
    bool binary) const {
  WriteToken(os, binary,
             "<GeneralDropoutComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<NumMaskRows>");
  WriteBasicType(os, binary, num_mask_rows);
  WriteToken(os, binary, "<Indexes>");
  indexes.Write(os, binary);
  WriteToken(os, binary,
             "</GeneralDropoutComponentPrecomputedIndexes>");
}

void GeneralDropoutComponentPrecomputedIndexes::Read(std::istream &is,
    bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<GeneralDropoutComponentPrecomputedIndexes>",
                       "<NumMaskRows>");
  ReadBasicType(is, binary, &num_mask_rows);
  ExpectToken(is, binary, "<Indexes>");
  indexes.Read(is, binary);
  ExpectToken(is, binary,
              "</GeneralDropoutComponentPrecomputedIndexes>");
}


} // namespace nnet3
} // namespace kaldi
