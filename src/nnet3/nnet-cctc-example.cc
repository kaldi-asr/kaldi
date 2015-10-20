// nnet3/nnet-ctcexample.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

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

#include <cmath>
#include "nnet3/nnet-cctc-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

void NnetCctcSupervision::Write(std::ostream &os, bool binary) const {
  CheckDim();
  WriteToken(os, binary, "<NnetCctcSup>");
  WriteToken(os, binary, name);
  WriteIndexVector(os, binary, indexes);
  WriteToken(os, binary, "<NumSups>");
  int32 size = supervision.size();
  KALDI_ASSERT(size > 0 && "Attempting to write empty NnetCctcSupervision.");
  WriteBasicType(os, binary, size);
  if (!binary) os << "\n";
  for (int32 i = 0; i < size; i++) {
    supervision[i].Write(os, binary);
    if (!binary) os << "\n";
  }
  WriteToken(os, binary, "</NnetCctcSup>");
}

bool NnetCctcSupervision::operator == (const NnetCctcSupervision &other) const {
  return name == other.name && indexes == other.indexes &&
      supervision == other.supervision;
}
void NnetCctcSupervision::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetCctcSup>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  { // temp. to delete soon.
    std::string s;
    ReadToken(is, binary, &s);
  }
  // will be replaced with: ExpectToken(is, binary, "<NumSups>");
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size > 0 && size < 1000000);
  supervision.resize(size);
  for (int32 i = 0; i < size; i++)
    supervision[i].Read(is, binary);
  ExpectToken(is, binary, "</NnetCctcSup>");
  CheckDim();
}



// returns the number of separate sequences represented in the range of indexes
// starting from 'indexes_offset' and of length 'num_frames'... This is the same
// as the number of distinct 'n' values in the indexes.  This funtion will crash
// if all the sequences don't have the same number of time steps.  this will
// usually be the same as the --minibatch-size used in merging egs.
static int32 NumSequences(const NnetCctcSupervision &supervision,
                          int32 indexes_offset, int32 num_frames) {
  KALDI_ASSERT(indexes_offset >= 0 &&
               indexes_offset + num_frames <=
               static_cast<int32>(supervision.indexes.size()));
  std::vector<Index>::const_iterator
      begin = supervision.indexes.begin() + indexes_offset,
      end = begin + num_frames,
      iter = begin;
  std::vector<int32> change_points;
  int32 cur_n = iter->n;
  for (++iter; iter != end; ++iter) {
    if (iter->n != cur_n) {
      change_points.push_back(iter - begin);
      cur_n = iter->n;
    }
  }
  change_points.push_back(num_frames);
  int32 num_sequences = change_points.size();
  int32 first_change = change_points[0];
  for (int32 i = 1; i < change_points.size(); i++) {
    if (change_points[i] != first_change * (i+1))
      KALDI_ERR << "Cctc supervision object doesn't have the "
                << "regular structure we expect.";
  }
  return num_sequences;
}


void NnetCctcSupervision::ComputeObjfAndDerivs(
    const ctc::CctcTrainingOptions &opts,
    const ctc::CctcTransitionModel &cctc_trans_model,
    const CuMatrix<BaseFloat> &cu_weights,
    const CuMatrixBase<BaseFloat> &nnet_output,
    BaseFloat *tot_weight_out,
    BaseFloat *tot_num_objf_out,
    BaseFloat *tot_den_objf_out,
    CuMatrixBase<BaseFloat> *nnet_out_deriv) const {
  static int32 num_warnings = 50;
  int32 num_output_indexes = cctc_trans_model.NumOutputIndexes();
  KALDI_ASSERT(nnet_output.NumCols() == num_output_indexes);
  int32 cur_offset = 0;
  const BaseFloat error_logprob_per_frame = -10.0;
  BaseFloat tot_weight = 0.0, tot_num_objf = 0.0, tot_den_objf = 0.0;
  std::vector<ctc::CctcSupervision>::const_iterator
      iter = supervision.begin(), end = supervision.end();
  if (nnet_out_deriv)
    nnet_out_deriv->SetZero();
  // Normally this loop will only loop once.
  for (; iter != end; cur_offset += iter->num_frames,++iter) {
    const ctc::CctcSupervision &supervision = *iter;
    const CuSubMatrix<BaseFloat> nnet_output_part(nnet_output, cur_offset,
                                                  supervision.num_frames,
                                                  0, num_output_indexes);
    int32 num_sequences = NumSequences(*this, cur_offset,
                                       supervision.num_frames);
    ctc::CctcCommonComputation computation(opts, cctc_trans_model,
                                           cu_weights, supervision,
                                           num_sequences, nnet_output_part);
    tot_weight += supervision.num_frames * supervision.weight;
    BaseFloat this_num_part, this_den_part, this_weight;
    computation.Forward(&this_num_part, &this_den_part, &this_weight);
    if (this_num_part - this_num_part == 0.0 &&
        this_den_part - this_den_part == 0.0) {
      // no NaN's or inf's.
      tot_num_objf += this_num_part;
      tot_den_objf += this_den_part;
      tot_weight += this_weight;
    } else {
      tot_num_objf += supervision.num_frames *
          supervision.weight * error_logprob_per_frame;
      tot_weight += this_weight;
      if (num_warnings > 0) {
        num_warnings--;
        KALDI_WARN << "Bad prob = " << this_num_part << " + "
                   << this_den_part << " encountered in CTC computation";
      }
      continue;  // Don't do the backprop.
    }
    if (nnet_out_deriv == NULL)
      continue;
    // Now do the backward phase, if requested.
    CuSubMatrix<BaseFloat> out_deriv_part(*nnet_out_deriv, cur_offset,
                                          supervision.num_frames,
                                          0, num_output_indexes);
    computation.Backward(&out_deriv_part);
  }
  KALDI_ASSERT(cur_offset == nnet_output.NumRows());
  *tot_weight_out = tot_weight;
  *tot_num_objf_out = tot_num_objf;
  *tot_den_objf_out = tot_den_objf;
}

void NnetCctcSupervision::CheckDim() const {
  int32 num_indexes = indexes.size(), num_indexes_check = 0,
      label_dim = -1;
  std::vector<ctc::CctcSupervision>::const_iterator
      iter = supervision.begin(), end = supervision.end();
  for (; iter != end; ++iter) {
    num_indexes_check += iter->num_frames;
    if (label_dim == -1) {
      label_dim = iter->label_dim;
    } else {
      KALDI_ASSERT(label_dim == iter->label_dim);
    }
  }
  KALDI_ASSERT(num_indexes == num_indexes_check);
}

NnetCctcSupervision::NnetCctcSupervision(const NnetCctcSupervision &other):
    name(other.name),
    indexes(other.indexes),
    supervision(other.supervision) { CheckDim(); }

void NnetCctcSupervision::Swap(NnetCctcSupervision *other) {
  name.swap(other->name);
  indexes.swap(other->indexes);
  supervision.swap(other->supervision);
  CheckDim();
}

NnetCctcSupervision::NnetCctcSupervision(
    const ctc::CctcSupervision &ctc_supervision,
    const std::string &name,
    int32 first_frame,
    int32 frame_skip):
    name(name) {
  supervision.resize(1);
  supervision[0] = ctc_supervision;
  int32 num_frames = ctc_supervision.num_frames;
  KALDI_ASSERT(num_frames > 0 && frame_skip > 0);
  indexes.resize(num_frames);
  // leave n and x in the indexes at zero at this point.
  for (int32 i = 0; i < num_frames; i++)
    indexes[i].t = first_frame + i * frame_skip;
  CheckDim();
}


void NnetCctcExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<Nnet3CctcEg>");
  WriteToken(os, binary, "<NumInputs>");
  int32 size = inputs.size();
  WriteBasicType(os, binary, size);
  KALDI_ASSERT(size > 0 && "Attempting to write NnetCctcExample with no inputs");
  if (!binary) os << '\n';
  for (int32 i = 0; i < size; i++) {
    inputs[i].Write(os, binary);
    if (!binary) os << '\n';
  }
  WriteToken(os, binary, "<NumOutputs>");
  size = outputs.size();
  WriteBasicType(os, binary, size);
  KALDI_ASSERT(size > 0 && "Attempting to write NnetCctcExample with no outputs");
  if (!binary) os << '\n';
  for (int32 i = 0; i < size; i++) {
    outputs[i].Write(os, binary);
    if (!binary) os << '\n';
  }
  WriteToken(os, binary, "</Nnet3CctcEg>");
}

void NnetCctcExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3CctcEg>");
  ExpectToken(is, binary, "<NumInputs>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size < 1 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  inputs.resize(size);
  for (int32 i = 0; i < size; i++)
    inputs[i].Read(is, binary);
  ExpectToken(is, binary, "<NumOutputs>");
  ReadBasicType(is, binary, &size);
  if (size < 1 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  outputs.resize(size);
  for (int32 i = 0; i < size; i++)
    outputs[i].Read(is, binary);
  ExpectToken(is, binary, "</Nnet3CctcEg>");
}

void NnetCctcExample::Swap(NnetCctcExample *other) {
  inputs.swap(other->inputs);
  outputs.swap(other->outputs);
}

void NnetCctcExample::Compress() {
  std::vector<NnetIo>::iterator iter = inputs.begin(), end = inputs.end();
  // calling features.Compress() will do nothing if they are sparse or already
  // compressed.
  for (; iter != end; ++iter) iter->features.Compress();
}

NnetCctcExample::NnetCctcExample(const NnetCctcExample &other):
    inputs(other.inputs), outputs(other.outputs) { }



// called from MergeCctcExamplesInternal, this function merges the CctcSupervision
// objects into one.  Requires (and checks) that they all have the same name.
static void MergeCctcSupervision(
    const std::vector<const NnetCctcSupervision*> &inputs,
    bool compactify,
    NnetCctcSupervision *output) {
  int32 num_inputs = inputs.size(),
      num_indexes = 0;
  for (int32 n = 0; n < num_inputs; n++) {
    KALDI_ASSERT(inputs[n]->name == inputs[0]->name);
    num_indexes += inputs[n]->indexes.size();
  }
  output->name = inputs[0]->name;
  std::vector<const ctc::CctcSupervision*> input_supervision;
  input_supervision.reserve(inputs.size());
  for (int32 n = 0; n < num_inputs; n++) {
    std::vector<ctc::CctcSupervision>::const_iterator
        iter = inputs[n]->supervision.begin(),
        end = inputs[n]->supervision.end();
    for (; iter != end; ++iter)
      input_supervision.push_back(&(*iter));
  }
  AppendCctcSupervision(input_supervision,
                       compactify,
                       &(output->supervision));
  output->indexes.clear();
  output->indexes.reserve(num_indexes);
  for (int32 n = 0; n < num_inputs; n++) {
    const std::vector<Index> &src_indexes = inputs[n]->indexes;
    int32 cur_size = output->indexes.size();
    output->indexes.insert(output->indexes.end(),
                           src_indexes.begin(), src_indexes.end());
    std::vector<Index>::iterator iter = output->indexes.begin() + cur_size,
        end = output->indexes.end();
    // change the 'n' index to correspond to the index into 'input'.
    // Each example gets a different 'n' value, starting from 0.
    for (; iter != end; ++iter) {
      KALDI_ASSERT(iter->n == 0 && "Merging already-merged CTC egs");
      iter->n = n;
    }
  }
  KALDI_ASSERT(output->indexes.size() == num_indexes);
}


void MergeCctcExamples(bool compress,
                       bool compactify,
                       std::vector<NnetCctcExample> *input,
                       NnetCctcExample *output) {
  int32 num_examples = input->size();
  KALDI_ASSERT(num_examples > 0);
  // we temporarily make the input-features in 'input' look like regular NnetExamples,
  // so that we can recycle the MergeExamples() function.
  std::vector<NnetExample> eg_inputs(num_examples);
  for (int32 i = 0; i < num_examples; i++)
    eg_inputs[i].io.swap((*input)[i].inputs);
  NnetExample eg_output;
  MergeExamples(eg_inputs, compress, &eg_output);
  // swap the inputs back so that they are not really changed.
  for (int32 i = 0; i < num_examples; i++)
    eg_inputs[i].io.swap((*input)[i].inputs);
  // write to 'output->inputs'
  eg_output.io.swap(output->inputs);

  // Now deal with the CTC-supervision 'outputs'.  There will
  // normally be just one of these, with name "output", but we
  // handle the more general case.
  int32 num_output_names = (*input)[0].outputs.size();
  output->outputs.resize(num_output_names);
  for (int32 i = 0; i < num_output_names; i++) {
    std::vector<const NnetCctcSupervision*> to_merge(num_examples);
    for (int32 j = 0; j < num_examples; j++) {
      KALDI_ASSERT((*input)[j].outputs.size() == num_output_names);
      to_merge[j] = &((*input)[j].outputs[i]);
    }
    MergeCctcSupervision(to_merge,
                        compactify,
                        &(output->outputs[i]));
  }
}

void GetCctcComputationRequest(const Nnet &nnet,
                               const NnetCctcExample &eg,
                               bool need_model_derivative,
                               bool store_component_stats,
                               ComputationRequest *request) {
  request->inputs.clear();
  request->inputs.reserve(eg.inputs.size());
  request->outputs.clear();
  request->outputs.reserve(eg.outputs.size());
  request->need_model_derivative = need_model_derivative;
  request->store_component_stats = store_component_stats;
  for (size_t i = 0; i < eg.inputs.size(); i++) {
    const NnetIo &io = eg.inputs[i];
    const std::string &name = io.name;
    int32 node_index = nnet.GetNodeIndex(name);
    if (node_index == -1 &&
        !nnet.IsInputNode(node_index))
      KALDI_ERR << "Nnet example has input named '" << name
                << "', but no such input node is in the network.";

    request->inputs.resize(request->inputs.size() + 1);
    IoSpecification &io_spec = request->inputs.back();
    io_spec.name = name;
    io_spec.indexes = io.indexes;
    io_spec.has_deriv = false;
  }
  for (size_t i = 0; i < eg.outputs.size(); i++) {
    // there will normally be exactly one output , named "output"
    const NnetCctcSupervision &sup = eg.outputs[i];
    const std::string &name = sup.name;
    int32 node_index = nnet.GetNodeIndex(name);
    if (node_index == -1 &&
        !nnet.IsOutputNode(node_index))
      KALDI_ERR << "Nnet example has output named '" << name
                << "', but no such output node is in the network.";
    request->outputs.resize(request->outputs.size() + 1);
    IoSpecification &io_spec = request->outputs.back();
    io_spec.name = name;
    io_spec.indexes = sup.indexes;
    io_spec.has_deriv = need_model_derivative;
  }
  // check to see if something went wrong.
  if (request->inputs.empty())
    KALDI_ERR << "No inputs in computation request.";
  if (request->outputs.empty())
    KALDI_ERR << "No outputs in computation request.";
}

void ShiftCctcExampleTimes(int32 frame_shift,
                           const std::vector<std::string> &exclude_names,
                           NnetCctcExample *eg) {
  std::vector<NnetIo>::iterator input_iter = eg->inputs.begin(),
      input_end = eg->inputs.end();
  for (; input_iter != input_end; ++input_iter) {
    bool must_exclude = false;
    std::vector<string>::const_iterator exclude_iter = exclude_names.begin(),
        exclude_end = exclude_names.end();
    for (; exclude_iter != exclude_end; ++exclude_iter)
      if (input_iter->name == *exclude_iter)
        must_exclude = true;
    if (!must_exclude) {
      std::vector<Index>::iterator indexes_iter = input_iter->indexes.begin(),
          indexes_end = input_iter->indexes.end();
      for (; indexes_iter != indexes_end; ++indexes_iter)
        indexes_iter->t += frame_shift;
    }
  }
  // note: we'll normally choose a small enough shift that the output-data
  // shift will be zero.
  std::vector<NnetCctcSupervision>::iterator
      sup_iter = eg->outputs.begin(),
      sup_end = eg->outputs.end();
  for (; sup_iter != sup_end; ++sup_iter) {
    std::vector<Index> &indexes = sup_iter->indexes;
    KALDI_ASSERT(indexes.size() >= 2 && indexes[0].n == indexes[1].n &&
                 indexes[0].x == indexes[1].x);
    int32 frame_subsampling_factor = indexes[1].t - indexes[0].t;
    KALDI_ASSERT(frame_subsampling_factor > 0);

    // We need to shift by a multiple of frame_subsampling_factor.
    // Round to the closest multiple.
    int32 supervision_frame_shift =
        frame_subsampling_factor *
        std::floor(0.5 + (frame_shift * 1.0 / frame_subsampling_factor));
    if (supervision_frame_shift == 0)
      continue;
    std::vector<Index>::iterator indexes_iter = indexes.begin(),
        indexes_end = indexes.end();
    for (; indexes_iter != indexes_end; ++indexes_iter)
      indexes_iter->t += supervision_frame_shift;
  }
}

} // namespace nnet3
} // namespace kaldi
