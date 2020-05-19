// nnet3/nnet-attention-component.cc

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)
//                2017  Hossein Hadian

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
#include "nnet3/nnet-attention-component.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-compile-utils.h"

namespace kaldi {
namespace nnet3 {


std::string RestrictedAttentionComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", num-heads=" << num_heads_
         << ", time-stride=" << time_stride_
         << ", key-dim=" << key_dim_
         << ", value-dim=" << value_dim_
         << ", num-left-inputs=" << num_left_inputs_
         << ", num-right-inputs=" << num_right_inputs_
         << ", context-dim=" << context_dim_
         << ", num-left-inputs-required=" << num_left_inputs_required_
         << ", num-right-inputs-required=" << num_right_inputs_required_
         << ", output-context=" << (output_context_ ? "true" : "false")
         << ", key-scale=" << key_scale_;
  if (stats_count_ != 0.0) {
    stream << ", entropy=";
    for (int32 i = 0; i < entropy_stats_.Dim(); i++)
      stream << (entropy_stats_(i) / stats_count_) << ',';
    for (int32 i = 0; i < num_heads_ && i < 5; i++) {
      stream << " posterior-stats[" << i <<"]=";
      for (int32 j = 0; j < posterior_stats_.NumCols(); j++)
        stream << (posterior_stats_(i,j) / stats_count_) << ',';
    }
    stream << " stats-count=" << stats_count_;
  }
  return stream.str();
}

RestrictedAttentionComponent::RestrictedAttentionComponent(
    const RestrictedAttentionComponent &other):
    num_heads_(other.num_heads_),
    key_dim_(other.key_dim_),
    value_dim_(other.value_dim_),
    num_left_inputs_(other.num_left_inputs_),
    num_right_inputs_(other.num_right_inputs_),
    time_stride_(other.time_stride_),
    context_dim_(other.context_dim_),
    num_left_inputs_required_(other.num_left_inputs_required_),
    num_right_inputs_required_(other.num_right_inputs_required_),
    output_context_(other.output_context_),
    key_scale_(other.key_scale_),
    stats_count_(other.stats_count_),
    entropy_stats_(other.entropy_stats_),
    posterior_stats_(other.posterior_stats_) { }



void RestrictedAttentionComponent::InitFromConfig(ConfigLine *cfl) {
  num_heads_ = 1;
  key_dim_ = -1;
  value_dim_ = -1;
  num_left_inputs_ = -1;
  num_right_inputs_ = -1;
  time_stride_ = 1;
  num_left_inputs_required_ = -1;
  num_right_inputs_required_ = -1;
  output_context_ = true;
  key_scale_ = -1.0;


  // mandatory arguments.
  bool ok = cfl->GetValue("key-dim", &key_dim_) &&
      cfl->GetValue("value-dim", &value_dim_) &&
      cfl->GetValue("num-left-inputs", &num_left_inputs_) &&
      cfl->GetValue("num-right-inputs", &num_right_inputs_);

  if (!ok)
    KALDI_ERR << "All of the values key-dim, value-dim, "
        "num-left-inputs and num-right-inputs must be defined.";
  // optional arguments.
  cfl->GetValue("num-heads", &num_heads_);
  cfl->GetValue("time-stride", &time_stride_);
  cfl->GetValue("num-left-inputs-required", &num_left_inputs_required_);
  cfl->GetValue("num-right-inputs-required", &num_right_inputs_required_);
  cfl->GetValue("output-context", &output_context_);
  cfl->GetValue("key-scale", &key_scale_);

  if (key_scale_ < 0.0) key_scale_ = 1.0 / sqrt(key_dim_);
  if (num_left_inputs_required_ < 0)
    num_left_inputs_required_ = num_left_inputs_;
  if (num_right_inputs_required_ < 0)
    num_right_inputs_required_ = num_right_inputs_;

  if (num_heads_ <= 0 || key_dim_ <= 0 || value_dim_ <= 0 ||
      num_left_inputs_ < 0 || num_right_inputs_ < 0 ||
      (num_left_inputs_ + num_right_inputs_) <= 0 ||
      num_left_inputs_required_ > num_left_inputs_ ||
      num_right_inputs_required_ > num_right_inputs_ ||
      time_stride_ <= 0)
    KALDI_ERR << "Config line contains invalid values: "
              << cfl->WholeLine();
  stats_count_ = 0.0;
  context_dim_ = num_left_inputs_ + 1 + num_right_inputs_;
  Check();
}



void*
RestrictedAttentionComponent::Propagate(const ComponentPrecomputedIndexes *indexes_in,
                                        const CuMatrixBase<BaseFloat> &in,
                                        CuMatrixBase<BaseFloat> *out) const {
  const PrecomputedIndexes *indexes = dynamic_cast<const PrecomputedIndexes*>(
      indexes_in);
  KALDI_ASSERT(indexes != NULL &&
               in.NumRows() == indexes->io.num_t_in * indexes->io.num_images &&
               out->NumRows() == indexes->io.num_t_out * indexes->io.num_images);


  Memo *memo = new Memo();
  memo->c.Resize(out->NumRows(), context_dim_ * num_heads_);

  int32 query_dim = key_dim_ + context_dim_;
  int32 input_dim_per_head = key_dim_ + value_dim_ + query_dim,
      output_dim_per_head = value_dim_ + (output_context_ ? context_dim_ : 0);
  for (int32 h = 0; h < num_heads_; h++) {
    CuSubMatrix<BaseFloat> in_part(in, 0, in.NumRows(),
                                   h * input_dim_per_head, input_dim_per_head),
        c_part(memo->c, 0, out->NumRows(),
               h * context_dim_, context_dim_),
        out_part(*out, 0, out->NumRows(),
                 h * output_dim_per_head, output_dim_per_head);
    PropagateOneHead(indexes->io, in_part, &c_part, &out_part);
  }
  return static_cast<void*>(memo);
}

void RestrictedAttentionComponent::PropagateOneHead(
    const time_height_convolution::ConvolutionComputationIo &io,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *c,
    CuMatrixBase<BaseFloat> *out) const {
  int32 query_dim = key_dim_ + context_dim_,
      full_value_dim = value_dim_ + (output_context_ ? context_dim_ : 0);
  KALDI_ASSERT(in.NumRows() == io.num_images * io.num_t_in &&
               out->NumRows() == io.num_images * io.num_t_out &&
               out->NumCols() == full_value_dim &&
               in.NumCols() == (key_dim_ + value_dim_ + query_dim) &&
               io.t_step_in == io.t_step_out &&
               (io.start_t_out - io.start_t_in) % io.t_step_in == 0);

  // 'steps_left_context' is the number of time-steps the input has on the left
  // that don't appear in the output.
  int32 steps_left_context = (io.start_t_out - io.start_t_in) / io.t_step_in,
      rows_left_context = steps_left_context * io.num_images;
  KALDI_ASSERT(rows_left_context >= 0);

  // 'queries' contains the queries.  We don't use all rows of the input
  // queries; only the rows that correspond to the time-indexes at the
  // output, i.e. we exclude the left-context and right-context.
  // 'out'; the remaining rows of 'in' that we didn't select correspond to left
  // and right temporal context.
  CuSubMatrix<BaseFloat> queries(in, rows_left_context, out->NumRows(),
                                 key_dim_ + value_dim_, query_dim);
  // 'keys' contains the keys; note, these are not extended with
  // context information; that happens further in.
  CuSubMatrix<BaseFloat> keys(in, 0, in.NumRows(), 0, key_dim_);

  // 'values' contains the values which we will interpolate.
  // these don't contain the context information; that will be added
  // later if output_context_ == true.
  CuSubMatrix<BaseFloat> values(in, 0, in.NumRows(), key_dim_, value_dim_);

  attention::AttentionForward(key_scale_, keys, queries, values, c, out);
}


void RestrictedAttentionComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    void *memo_in) {
  const Memo *memo = static_cast<const Memo*>(memo_in);
  KALDI_ASSERT(memo != NULL);
  if (entropy_stats_.Dim() != num_heads_) {
    entropy_stats_.Resize(num_heads_);
    posterior_stats_.Resize(num_heads_, context_dim_);
    stats_count_ = 0.0;
  }
  const CuMatrix<BaseFloat> &c = memo->c;
  if (RandInt(0, 2) == 0)
    return;  // only actually store the stats for one in three minibatches, to
             // save time.

  { // first get the posterior stats.
    CuVector<BaseFloat> c_sum(num_heads_ * context_dim_);
    c_sum.AddRowSumMat(1.0, c, 0.0);
    // view the vector as a matrix.
    CuSubMatrix<BaseFloat> c_sum_as_mat(c_sum.Data(), num_heads_,
                                        context_dim_, context_dim_);
    CuMatrix<double> c_sum_as_mat_dbl(c_sum_as_mat);
    posterior_stats_.AddMat(1.0, c_sum_as_mat_dbl);
    KALDI_ASSERT(c.NumCols() == num_heads_ * context_dim_);
  }
  { // now get the entropy stats.
    CuMatrix<BaseFloat> log_c(c);
    log_c.ApplyFloor(1.0e-20);
    log_c.ApplyLog();
    CuVector<BaseFloat> dot_prod(num_heads_ * context_dim_);
    dot_prod.AddDiagMatMat(-1.0, c, kTrans, log_c, kNoTrans, 0.0);
    // dot_prod is the sum over the matrix's rows (which correspond
    // to heads, and context positions), of - c * log(c), which is
    // part of the entropy.  To get the actual contribution to the
    // entropy, we have to sum 'dot_prod' over blocks of
    // size 'context_dim_'; that gives us the entropy contribution
    // per head.  We'd have to divide by c.NumRows() to get the
    // actual entropy, but that's reflected in stats_count_.
    CuSubMatrix<BaseFloat> entropy_mat(dot_prod.Data(), num_heads_,
                                       context_dim_, context_dim_);
    CuVector<BaseFloat> entropy_vec(num_heads_);
    entropy_vec.AddColSumMat(1.0, entropy_mat);
    Vector<double> entropy_vec_dbl(entropy_vec);
    entropy_stats_.AddVec(1.0, entropy_vec_dbl);
  }
  stats_count_ += c.NumRows();
}

void RestrictedAttentionComponent::ZeroStats() {
  entropy_stats_.SetZero();
  posterior_stats_.SetZero();
  stats_count_ = 0.0;
}

void RestrictedAttentionComponent::Scale(BaseFloat scale) {
  entropy_stats_.Scale(scale);
  posterior_stats_.Scale(scale);
  stats_count_ *= scale;
}

void RestrictedAttentionComponent::Add(BaseFloat alpha, const Component &other_in) {
  const RestrictedAttentionComponent *other =
      dynamic_cast<const RestrictedAttentionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  if (entropy_stats_.Dim() == 0 && other->entropy_stats_.Dim() != 0)
    entropy_stats_.Resize(other->entropy_stats_.Dim());
  if (posterior_stats_.NumRows() == 0 && other->posterior_stats_.NumRows() != 0)
    posterior_stats_.Resize(other->posterior_stats_.NumRows(), other->posterior_stats_.NumCols());
  if (other->entropy_stats_.Dim() != 0)
    entropy_stats_.AddVec(alpha, other->entropy_stats_);
  if (other->posterior_stats_.NumRows() != 0)
    posterior_stats_.AddMat(alpha, other->posterior_stats_);
  stats_count_ += alpha * other->stats_count_;
}


void RestrictedAttentionComponent::Check() const {
  KALDI_ASSERT(num_heads_ > 0 && key_dim_ > 0 && value_dim_ > 0 &&
               num_left_inputs_ >= 0 && num_right_inputs_ >= 0 &&
               (num_left_inputs_ + num_right_inputs_) > 0 &&
               time_stride_ > 0 &&
               context_dim_ == (num_left_inputs_ + 1 + num_right_inputs_) &&
               num_left_inputs_required_ >= 0 &&
               num_left_inputs_required_ <= num_left_inputs_ &&
               num_right_inputs_required_ >= 0 &&
               num_right_inputs_required_ <= num_right_inputs_ &&
               key_scale_ > 0.0 && key_scale_ <= 1.0 &&
               stats_count_ >= 0.0);
}


void RestrictedAttentionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo_in,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("RestrictedAttentionComponent::Backprop");
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL);
  Memo *memo = static_cast<Memo*>(memo_in);
  KALDI_ASSERT(memo != NULL);
  const time_height_convolution::ConvolutionComputationIo &io = indexes->io;
  KALDI_ASSERT(indexes != NULL &&
               in_value.NumRows() == io.num_t_in * io.num_images &&
               out_deriv.NumRows() == io.num_t_out * io.num_images &&
               in_deriv != NULL && SameDim(in_value, *in_deriv));

  const CuMatrix<BaseFloat> &c = memo->c;

  int32 query_dim = key_dim_ + context_dim_,
      input_dim_per_head = key_dim_ + value_dim_ + query_dim,
      output_dim_per_head = value_dim_ + (output_context_ ? context_dim_ : 0);

  for (int32 h = 0; h < num_heads_; h++) {
    CuSubMatrix<BaseFloat>
        in_value_part(in_value, 0, in_value.NumRows(),
                      h * input_dim_per_head, input_dim_per_head),
        c_part(c, 0, out_deriv.NumRows(),
               h * context_dim_, context_dim_),
        out_deriv_part(out_deriv, 0, out_deriv.NumRows(),
                       h * output_dim_per_head, output_dim_per_head),
        in_deriv_part(*in_deriv, 0, in_value.NumRows(),
                      h * input_dim_per_head, input_dim_per_head);
    BackpropOneHead(io, in_value_part, c_part, out_deriv_part,
                    &in_deriv_part);
  }
}


void RestrictedAttentionComponent::BackpropOneHead(
    const time_height_convolution::ConvolutionComputationIo &io,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &c,
    const CuMatrixBase<BaseFloat> &out_deriv,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  // the easiest way to understand this is to compare it with PropagateOneHead().
  int32 query_dim = key_dim_ + context_dim_,
      full_value_dim = value_dim_ + (output_context_ ? context_dim_ : 0);
  KALDI_ASSERT(in_value.NumRows() == io.num_images * io.num_t_in &&
               out_deriv.NumRows() == io.num_images * io.num_t_out &&
               out_deriv.NumCols() == full_value_dim &&
               in_value.NumCols() == (key_dim_ + value_dim_ + query_dim) &&
               io.t_step_in == io.t_step_out &&
               (io.start_t_out - io.start_t_in) % io.t_step_in == 0 &&
               SameDim(in_value, *in_deriv) &&
               c.NumRows() == out_deriv.NumRows() &&
               c.NumCols() == context_dim_);

  // 'steps_left_context' is the number of time-steps the input has on the left
  // that don't appear in the output.
  int32 steps_left_context = (io.start_t_out - io.start_t_in) / io.t_step_in,
      rows_left_context = steps_left_context * io.num_images;
  KALDI_ASSERT(rows_left_context >= 0);


  CuSubMatrix<BaseFloat> queries(in_value, rows_left_context, out_deriv.NumRows(),
                                 key_dim_ + value_dim_, query_dim),
      queries_deriv(*in_deriv, rows_left_context, out_deriv.NumRows(),
                    key_dim_ + value_dim_, query_dim),
      keys(in_value, 0, in_value.NumRows(), 0, key_dim_),
      keys_deriv(*in_deriv,  0, in_value.NumRows(), 0, key_dim_),
      values(in_value, 0, in_value.NumRows(), key_dim_, value_dim_),
      values_deriv(*in_deriv, 0, in_value.NumRows(), key_dim_, value_dim_);

  attention::AttentionBackward(key_scale_, keys, queries, values, c, out_deriv,
                               &keys_deriv, &queries_deriv, &values_deriv);
}



void RestrictedAttentionComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
  using namespace time_height_convolution;
  ConvolutionComputationIo io;
  GetComputationStructure(*input_indexes, *output_indexes, &io);
  std::vector<Index> new_input_indexes, new_output_indexes;
  GetIndexes(*input_indexes, *output_indexes, io,
             &new_input_indexes, &new_output_indexes);
  input_indexes->swap(new_input_indexes);
  output_indexes->swap(new_output_indexes);
}

void RestrictedAttentionComponent::GetComputationStructure(
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      time_height_convolution::ConvolutionComputationIo *io) const {
  GetComputationIo(input_indexes, output_indexes, io);
  // if there was only one output and/or input index (unlikely),
  // just let the grid periodicity be t_stride_.
  if (io->t_step_out == 0) io->t_step_out = time_stride_;
  if (io->t_step_in == 0) io->t_step_in = time_stride_;

  // We need the grid size on the input and output to be the same, and to divide
  // t_stride_.  If someone is requesting the output more frequently than
  // t_stride_, then after this change we may end up computing more outputs than
  // we need, but this is not a configuration that I think is very likely.  We
  // let the grid step be the gcd of the input and output steps, and of
  // t_stride_.
  // The next few statements may have the effect of making the grid finer at the
  // input and output, while having the same start and end point.
  int32 t_step = Gcd(Gcd(io->t_step_out, io->t_step_in), time_stride_);
  int32 multiple_out = io->t_step_out / t_step,
      multiple_in = io->t_step_in / t_step;
  io->t_step_in = t_step;
  io->t_step_out = t_step;
  io->num_t_out = 1 + multiple_out * (io->num_t_out - 1);
  io->num_t_in = 1 + multiple_in * (io->num_t_in - 1);

  // Now ensure that the extent of the input has at least
  // the requested left-context and right context; if
  // this increases the amount of input, we'll do zero-padding.
  int32 first_requested_input =
          io->start_t_out - (time_stride_ * num_left_inputs_),
      first_required_input =
         io->start_t_out - (time_stride_ * num_left_inputs_required_),
      last_t_out = io->start_t_out + (io->num_t_out - 1) * t_step,
      last_t_in = io->start_t_in + (io->num_t_in - 1) * t_step,
      last_requested_input = last_t_out + (time_stride_ * num_right_inputs_),
      last_required_input =
           last_t_out + (time_stride_ * num_right_inputs_required_);

  // check that we don't have *more* than the requested context,
  // but that we have at least the required context.
  KALDI_ASSERT(io->start_t_in >= first_requested_input &&
               last_t_in <= last_requested_input &&
               io->start_t_in <= first_required_input &&
               last_t_in >= last_required_input);

  // For the inputs that were requested, but not required,
  // we pad with zeros.  We pad the 'io' object, adding these
  // extra inputs structurally; in runtime they'll be set to zero.
  io->start_t_in = first_requested_input;
  io->num_t_in = 1 + (last_requested_input - first_requested_input) / t_step;
}

void RestrictedAttentionComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<RestrictedAttentionComponent>");
  WriteToken(os, binary, "<NumHeads>");
  WriteBasicType(os, binary, num_heads_);
  WriteToken(os, binary, "<KeyDim>");
  WriteBasicType(os, binary, key_dim_);
  WriteToken(os, binary, "<ValueDim>");
  WriteBasicType(os, binary, value_dim_);
  WriteToken(os, binary, "<NumLeftInputs>");
  WriteBasicType(os, binary, num_left_inputs_);
  WriteToken(os, binary, "<NumRightInputs>");
  WriteBasicType(os, binary, num_right_inputs_);
  WriteToken(os, binary, "<TimeStride>");
  WriteBasicType(os, binary, time_stride_);
  WriteToken(os, binary, "<NumLeftInputsRequired>");
  WriteBasicType(os, binary, num_left_inputs_required_);
  WriteToken(os, binary, "<NumRightInputsRequired>");
  WriteBasicType(os, binary, num_right_inputs_required_);
  WriteToken(os, binary, "<OutputContext>");
  WriteBasicType(os, binary, output_context_);
  WriteToken(os, binary, "<KeyScale>");
  WriteBasicType(os, binary, key_scale_);
  WriteToken(os, binary, "<StatsCount>");
  WriteBasicType(os, binary, stats_count_);
  WriteToken(os, binary, "<EntropyStats>");
  entropy_stats_.Write(os, binary);
  WriteToken(os, binary, "<PosteriorStats>");
  posterior_stats_.Write(os, binary);
  WriteToken(os, binary, "</RestrictedAttentionComponent>");
}

void RestrictedAttentionComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<RestrictedAttentionComponent>",
                       "<NumHeads>");
  ReadBasicType(is, binary, &num_heads_);
  ExpectToken(is, binary, "<KeyDim>");
  ReadBasicType(is, binary, &key_dim_);
  ExpectToken(is, binary, "<ValueDim>");
  ReadBasicType(is, binary, &value_dim_);
  ExpectToken(is, binary, "<NumLeftInputs>");
  ReadBasicType(is, binary, &num_left_inputs_);
  ExpectToken(is, binary, "<NumRightInputs>");
  ReadBasicType(is, binary, &num_right_inputs_);
  ExpectToken(is, binary, "<TimeStride>");
  ReadBasicType(is, binary, &time_stride_);
  ExpectToken(is, binary, "<NumLeftInputsRequired>");
  ReadBasicType(is, binary, &num_left_inputs_required_);
  ExpectToken(is, binary, "<NumRightInputsRequired>");
  ReadBasicType(is, binary, &num_right_inputs_required_);
  ExpectToken(is, binary, "<OutputContext>");
  ReadBasicType(is, binary, &output_context_);
  ExpectToken(is, binary, "<KeyScale>");
  ReadBasicType(is, binary, &key_scale_);
  ExpectToken(is, binary, "<StatsCount>");
  ReadBasicType(is, binary, &stats_count_);
  ExpectToken(is, binary, "<EntropyStats>");
  entropy_stats_.Read(is, binary);
  ExpectToken(is, binary, "<PosteriorStats>");
  posterior_stats_.Read(is, binary);
  ExpectToken(is, binary, "</RestrictedAttentionComponent>");

  context_dim_ = num_left_inputs_ + 1 + num_right_inputs_;
}


void RestrictedAttentionComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  int32 first_time = output_index.t - (time_stride_ * num_left_inputs_),
      last_time = output_index.t + (time_stride_ * num_right_inputs_);
  desired_indexes->clear();
  desired_indexes->resize(context_dim_);
  int32 n = output_index.n, x = output_index.x,
      i = 0;
  for (int32 t = first_time; t <= last_time; t += time_stride_, i++) {
    (*desired_indexes)[i].n = n;
    (*desired_indexes)[i].t = t;
    (*desired_indexes)[i].x = x;
  }
  KALDI_ASSERT(i == context_dim_);
}


bool RestrictedAttentionComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  Index index(output_index);

  if (used_inputs != NULL) {
    int32 first_time = output_index.t - (time_stride_ * num_left_inputs_),
        last_time = output_index.t + (time_stride_ * num_right_inputs_);
    used_inputs->clear();
    used_inputs->reserve(context_dim_);

    for (int32 t = first_time; t <= last_time; t += time_stride_) {
      index.t = t;
      if (input_index_set(index)) {
        // This input index is available.
        used_inputs->push_back(index);
      } else {
        // This input index is not available.
        int32 offset = (t - output_index.t) / time_stride_;
        if (offset >= -num_left_inputs_required_ &&
            offset <= num_right_inputs_required_) {
          used_inputs->clear();
          return false;
        }
      }
    }
    // All required time-offsets of the output were computable. -> return true.
    return true;
  } else {
    int32 t = output_index.t,
        first_time_required = t - (time_stride_ * num_left_inputs_required_),
        last_time_required = t + (time_stride_ * num_right_inputs_required_);
    for (int32 t = first_time_required;
         t <= last_time_required;
         t += time_stride_) {
      index.t = t;
      if (!input_index_set(index))
        return false;
    }
    return true;
  }
}


// static
void RestrictedAttentionComponent::CreateIndexesVector(
    const std::vector<std::pair<int32, int32> > &n_x_pairs,
    int32 t_start, int32 t_step, int32 num_t_values,
    const std::unordered_set<Index, IndexHasher> &index_set,
    std::vector<Index> *output_indexes) {
  output_indexes->resize(static_cast<size_t>(num_t_values) * n_x_pairs.size());
  std::vector<Index>::iterator out_iter = output_indexes->begin();
  for (int32 t = t_start; t < t_start + (t_step * num_t_values); t += t_step) {
    std::vector<std::pair<int32, int32> >::const_iterator
        iter = n_x_pairs.begin(), end = n_x_pairs.end();
    for (; iter != end; ++iter) {
      out_iter->n = iter->first;
      out_iter->t = t;
      out_iter->x = iter->second;
      if (index_set.count(*out_iter) == 0)
        out_iter->t = kNoTime;
      ++out_iter;
    }
  }
  KALDI_ASSERT(out_iter == output_indexes->end());
}

void RestrictedAttentionComponent::GetIndexes(
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      time_height_convolution::ConvolutionComputationIo &io,
      std::vector<Index> *new_input_indexes,
      std::vector<Index> *new_output_indexes) const {

  std::unordered_set<Index, IndexHasher> input_set, output_set;
  for (std::vector<Index>::const_iterator iter = input_indexes.begin();
       iter != input_indexes.end(); ++iter)
    input_set.insert(*iter);
  for (std::vector<Index>::const_iterator iter = output_indexes.begin();
       iter != output_indexes.end(); ++iter)
    output_set.insert(*iter);

  std::vector<std::pair<int32, int32> > n_x_pairs;
  GetNxList(input_indexes, &n_x_pairs);  // the n,x pairs at the output will be
                                         // identical.
  KALDI_ASSERT(n_x_pairs.size() == io.num_images);
  CreateIndexesVector(n_x_pairs, io.start_t_in, io.t_step_in, io.num_t_in,
                      input_set, new_input_indexes);
  CreateIndexesVector(n_x_pairs, io.start_t_out, io.t_step_out, io.num_t_out,
                      output_set, new_output_indexes);
}

ComponentPrecomputedIndexes* RestrictedAttentionComponent::PrecomputeIndexes(
    const MiscComputationInfo &,  // misc_info
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool) // need_backprop
    const {
  PrecomputedIndexes *ans = new PrecomputedIndexes();
  GetComputationStructure(input_indexes, output_indexes, &(ans->io));
  if (GetVerboseLevel() >= 2) {
    // what goes next is just a check.
    std::vector<Index> new_input_indexes, new_output_indexes;
    GetIndexes(input_indexes, output_indexes, ans->io,
               &new_input_indexes, &new_output_indexes);
    // input_indexes and output_indexes should be the ones that were
    // output by ReorderIndexes(), so they should already
    // have gone through the GetComputationStructure()->GetIndexes()
    // procedure.  Applying the same procedure twice is supposed to
    // give an unchanged results.
    KALDI_ASSERT(input_indexes == new_input_indexes &&
                 output_indexes == new_output_indexes);
  }
  return ans;
}



RestrictedAttentionComponent::PrecomputedIndexes*
RestrictedAttentionComponent::PrecomputedIndexes::Copy() const {
  return new PrecomputedIndexes(*this);
}

void RestrictedAttentionComponent::PrecomputedIndexes::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<RestrictedAttentionComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<Io>");
  io.Write(os, binary);
  WriteToken(os, binary, "</RestrictedAttentionComponentPrecomputedIndexes>");
}

void RestrictedAttentionComponent::PrecomputedIndexes::Read(
    std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<RestrictedAttentionComponentPrecomputedIndexes>",
                       "<Io>");
  io.Read(is, binary);
  ExpectToken(is, binary, "</RestrictedAttentionComponentPrecomputedIndexes>");
}


} // namespace nnet3
} // namespace kaldi
