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

#include "nnet3/nnet-cctc-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

void NnetCctcSupervision::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetCctcSup>");
  WriteToken(os, binary, name);
  WriteToken(os, binary, "<NumOutputs>");
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


void NnetCctcSupervision::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetCctcSup>");
  ReadToken(is, binary, &name);
  ExpectToken(is, binary, "<NumOutputs>");
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size > 0 && size < 1000000);
  supervision.resize(size);
  for (int32 i = 0; i < size; i++)
    supervision[i].Read(is, binary);
  ExpectToken(is, binary, "</NnetCctcSup>");
}


void NnetCctcSupervision::ComputeObjfAndDerivs(
    const ctc::CctcTrainingOptions &opts,
    const ctc::CctcTransitionModel &cctc_trans_model,
    const CuMatrix<BaseFloat> &cu_weights,
    const CuMatrixBase<BaseFloat> &nnet_output,
    BaseFloat *tot_weight_out,
    BaseFloat *tot_objf_out,
    CuMatrixBase<BaseFloat> *nnet_out_deriv) const {
  static int32 num_warnings = 50;
  int32 num_output_indexes = cctc_trans_model.NumOutputIndexes();
  KALDI_ASSERT(nnet_output.NumCols() == num_output_indexes);
  int32 cur_offset = 0;
  const BaseFloat error_logprob_per_frame = -10.0;
  BaseFloat tot_weight = 0.0, tot_objf = 0.0;
  std::vector<ctc::CctcSupervision>::const_iterator
      iter = supervision.begin(), end = supervision.end();
  if (nnet_out_deriv)
    nnet_out_deriv->SetZero();
  for (; iter != end; cur_offset += iter->num_frames,++iter) {
    const ctc::CctcSupervision &supervision = *iter;
    const CuSubMatrix<BaseFloat> nnet_output_part(nnet_output, cur_offset,
                                                  supervision.num_frames,
                                                  0, num_output_indexes);
    ctc::CctcComputation computation(opts, cctc_trans_model, cu_weights,
                                 supervision, nnet_output_part);
    tot_weight += supervision.num_frames * supervision.weight;
    BaseFloat tot_log_prob = computation.Forward();
    if (tot_log_prob == tot_log_prob && tot_log_prob - tot_log_prob == 0.0) {
      tot_objf += supervision.weight * tot_log_prob;
    } else {  // NaN or inf
      tot_objf += supervision.num_frames *
          supervision.weight * error_logprob_per_frame;
      if (num_warnings > 0) {
        num_warnings--;
        KALDI_WARN << "Bad forward prob " << tot_log_prob
                   << " encountered in CTC computation";
      }
      continue;  // Don't do the backprop.
    }
    if (nnet_out_deriv == NULL)
      continue;
    // Now do the backward phase, if requested.
    CuSubMatrix<BaseFloat> out_deriv_part(*nnet_out_deriv, cur_offset,
                                          supervision.num_frames,
                                          0, num_output_indexes);
    if (!computation.Backward(&out_deriv_part)) {
      nnet_out_deriv->Range(cur_offset, supervision.num_frames,
                            0, num_output_indexes).SetZero();
      if (num_warnings > 0) {
        num_warnings--;
        KALDI_WARN << "NaN's or inf's encountered in CTC backprop";
      }
    }                                              
  }
  KALDI_ASSERT(cur_offset == nnet_output.NumRows());
  *tot_weight_out = tot_weight;
  *tot_objf_out = tot_objf;
}

NnetCctcSupervision::NnetCctcSupervision(const NnetCctcSupervision &other):
    name(other.name),
    indexes(other.indexes),
    supervision(other.supervision) { }

void NnetCctcSupervision::Swap(NnetCctcSupervision *other) {
  name.swap(other->name);
  indexes.swap(other->indexes);
  supervision.swap(other->supervision);
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

} // namespace nnet3
} // namespace kaldi
