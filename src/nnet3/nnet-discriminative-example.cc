// nnet3/nnet-discriminative-example.cc

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
#include "nnet3/nnet-discriminative-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {


void NnetDiscriminativeSupervision::Write(std::ostream &os, bool binary) const {
  CheckDim();
  WriteToken(os, binary, "<NnetDiscriminativeSup>");
  WriteToken(os, binary, name);
  WriteIndexVector(os, binary, indexes);
  supervision.Write(os, binary);
  WriteToken(os, binary, "<DW>");  // for DerivWeights.  Want to save space.
  WriteVectorAsChar(os, binary, deriv_weights);
  WriteToken(os, binary, "</NnetDiscriminativeSup>");
}

bool NnetDiscriminativeSupervision::operator == (const NnetDiscriminativeSupervision &other) const {
  return name == other.name && indexes == other.indexes &&
      supervision == other.supervision &&
      deriv_weights.ApproxEqual(other.deriv_weights);
}

void NnetDiscriminativeSupervision::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetDiscriminativeSup>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  supervision.Read(is, binary);
  ExpectToken(is, binary, "<DW>");
  ReadVectorAsChar(is, binary, &deriv_weights);
  ExpectToken(is, binary, "</NnetDiscriminativeSup>");
  CheckDim();
}


void NnetDiscriminativeSupervision::CheckDim() const {
  if (supervision.frames_per_sequence == -1) {
    // this object has not been set up.
    KALDI_ASSERT(indexes.empty());
    return;
  }
  KALDI_ASSERT(indexes.size() == supervision.num_sequences *
               supervision.frames_per_sequence && !indexes.empty() &&
               supervision.frames_per_sequence > 1);
  int32 first_frame = indexes[0].t,
      frame_skip = indexes[supervision.num_sequences].t - first_frame,
      num_sequences = supervision.num_sequences,
      frames_per_sequence = supervision.frames_per_sequence;
  int32 k = 0;
  for (int32 i = 0; i < frames_per_sequence; i++) {
    for (int32 j = 0; j < num_sequences; j++,k++) {
      int32 n = j, t = i * frame_skip + first_frame, x = 0;
      Index index(n, t, x);
      KALDI_ASSERT(indexes[k] == index);
    }
  }
  if (deriv_weights.Dim() != 0) {
    KALDI_ASSERT(deriv_weights.Dim() == indexes.size());
    KALDI_ASSERT(deriv_weights.Min() >= 0.0 &&
                 deriv_weights.Max() <= 1.0);
  }
}

NnetDiscriminativeSupervision::NnetDiscriminativeSupervision(const NnetDiscriminativeSupervision &other):
    name(other.name),
    indexes(other.indexes),
    supervision(other.supervision),
    deriv_weights(other.deriv_weights) { CheckDim(); }

NnetDiscriminativeSupervision::NnetDiscriminativeSupervision(
    const std::string &name,
    const discriminative::DiscriminativeSupervision &supervision,
    const VectorBase<BaseFloat> &deriv_weights,
    int32 first_frame,
    int32 frame_skip):
    name(name),
    supervision(supervision),
    deriv_weights(deriv_weights) {
  // note: this will set the 'x' index to zero.
  indexes.resize(supervision.num_sequences *
                 supervision.frames_per_sequence);
  int32 k = 0, num_sequences = supervision.num_sequences,
      frames_per_sequence = supervision.frames_per_sequence;
  for (int32 i = 0; i < frames_per_sequence; i++) {
    for (int32 j = 0; j < num_sequences; j++,k++) {
      indexes[k].n = j;
      indexes[k].t = i * frame_skip + first_frame;
    }
  }
  KALDI_ASSERT(k == indexes.size());
  CheckDim();
}

void NnetDiscriminativeSupervision::Swap(NnetDiscriminativeSupervision *other) {
  name.swap(other->name);
  indexes.swap(other->indexes);
  supervision.Swap(&(other->supervision));
  deriv_weights.Swap(&(other->deriv_weights));
  if (RandInt(0, 5) == 0)
    CheckDim();
}


void NnetDiscriminativeExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<Nnet3DiscriminativeEg>");
  WriteToken(os, binary, "<NumInputs>");
  int32 size = inputs.size();
  WriteBasicType(os, binary, size);
  KALDI_ASSERT(size > 0 && "Attempting to write NnetDiscriminativeExample with no inputs");
  if (!binary) os << '\n';
  for (int32 i = 0; i < size; i++) {
    inputs[i].Write(os, binary);
    if (!binary) os << '\n';
  }
  WriteToken(os, binary, "<NumOutputs>");
  size = outputs.size();
  WriteBasicType(os, binary, size);
  KALDI_ASSERT(size > 0 && "Attempting to write NnetDiscriminativeExample with no outputs");
  if (!binary) os << '\n';
  for (int32 i = 0; i < size; i++) {
    outputs[i].Write(os, binary);
    if (!binary) os << '\n';
  }
  WriteToken(os, binary, "</Nnet3DiscriminativeEg>");
}

void NnetDiscriminativeExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3DiscriminativeEg>");
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
  ExpectToken(is, binary, "</Nnet3DiscriminativeEg>");
}

void NnetDiscriminativeExample::Swap(NnetDiscriminativeExample *other) {
  inputs.swap(other->inputs);
  outputs.swap(other->outputs);
}

void NnetDiscriminativeExample::Compress() {
  std::vector<NnetIo>::iterator iter = inputs.begin(), end = inputs.end();
  // calling features.Compress() will do nothing if they are sparse or already
  // compressed.
  for (; iter != end; ++iter) iter->features.Compress();
}

NnetDiscriminativeExample::NnetDiscriminativeExample(const NnetDiscriminativeExample &other):
    inputs(other.inputs), outputs(other.outputs) { }

void MergeSupervision(
    const std::vector<const NnetDiscriminativeSupervision*> &inputs,
    NnetDiscriminativeSupervision *output) {
  int32 num_inputs = inputs.size(),
      num_indexes = 0;
  for (int32 n = 0; n < num_inputs; n++) {
    KALDI_ASSERT(inputs[n]->name == inputs[0]->name);
    num_indexes += inputs[n]->indexes.size();
  }
  output->name = inputs[0]->name;
  std::vector<const discriminative::DiscriminativeSupervision*> input_supervision;
  input_supervision.reserve(inputs.size());
  for (int32 n = 0; n < num_inputs; n++)
    input_supervision.push_back(&(inputs[n]->supervision));
  discriminative::DiscriminativeSupervision output_supervision;
  discriminative::AppendSupervision(input_supervision,
                         &output_supervision);
  output->supervision.Swap(&(output_supervision));

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
      KALDI_ASSERT(iter->n == 0 && "Merging already-merged discriminative egs");
      iter->n = n;
    }
  }
  KALDI_ASSERT(output->indexes.size() == num_indexes);
  // OK, at this point the 'indexes' will be in the wrong order,
  // because they should be first sorted by 't' and next by 'n'.
  // 'sort' will fix this, due to the operator < on type Index.
  // TODO: Is this required?
  std::sort(output->indexes.begin(), output->indexes.end());

  // merge the deriv_weights.
  if (inputs[0]->deriv_weights.Dim() != 0) {
    int32 frames_per_sequence = inputs[0]->deriv_weights.Dim();
    output->deriv_weights.Resize(output->indexes.size(), kUndefined);
    KALDI_ASSERT(output->deriv_weights.Dim() ==
                 frames_per_sequence * num_inputs);
    for (int32 n = 0; n < num_inputs; n++) {
      const Vector<BaseFloat> &src_deriv_weights = inputs[n]->deriv_weights;
      KALDI_ASSERT(src_deriv_weights.Dim() == frames_per_sequence);
      // the ordering of the deriv_weights corresponds to the ordering of the
      // Indexes, where the time dimension has the greater stride.
      for (int32 t = 0; t < frames_per_sequence; t++) {
        output->deriv_weights(t * num_inputs + n) = src_deriv_weights(t);
      }
    }
  }
  output->CheckDim();
}


void MergeDiscriminativeExamples(
    bool compress,
    std::vector<NnetDiscriminativeExample> *input,
    NnetDiscriminativeExample *output) {
  int32 num_examples = input->size();
  KALDI_ASSERT(num_examples > 0);
  // we temporarily make the input-features in 'input' look like regular
  // NnetExamples, so that we can recycle the
  // MergeExamples() function.
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

  // Now deal with the discriminative-supervision 'outputs'.  There will
  // normally be just one of these, with name "output", but we
  // handle the more general case.
  int32 num_output_names = (*input)[0].outputs.size();
  output->outputs.resize(num_output_names);
  for (int32 i = 0; i < num_output_names; i++) {
    std::vector<const NnetDiscriminativeSupervision*> to_merge(num_examples);
    for (int32 j = 0; j < num_examples; j++) {
      KALDI_ASSERT((*input)[j].outputs.size() == num_output_names);
      to_merge[j] = &((*input)[j].outputs[i]);
    }
    MergeSupervision(to_merge,
                     &(output->outputs[i]));
  }
}


void GetDiscriminativeComputationRequest(const Nnet &nnet,
                                         const NnetDiscriminativeExample &eg,
                                         bool need_model_derivative,
                                         bool store_component_stats,
                                         bool use_xent_regularization,
                                         bool use_xent_derivative,
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
    const NnetDiscriminativeSupervision &sup = eg.outputs[i];
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

    if (use_xent_regularization) {
      size_t cur_size = request->outputs.size();
      request->outputs.resize(cur_size + 1);
      IoSpecification &io_spec = request->outputs[cur_size - 1],
          &io_spec_xent = request->outputs[cur_size];
      // the IoSpecification for the -xent output is the same
      // as for the regular output, except for its name which has
      // the -xent suffix (and the has_deriv member may differ).
      io_spec_xent = io_spec;
      io_spec_xent.name = name + "-xent";
      io_spec_xent.has_deriv = use_xent_derivative;
    }
  }
  // check to see if something went wrong.
  if (request->inputs.empty())
    KALDI_ERR << "No inputs in computation request.";
  if (request->outputs.empty())
    KALDI_ERR << "No outputs in computation request.";
}

void ShiftDiscriminativeExampleTimes(int32 frame_shift,
                            const std::vector<std::string> &exclude_names,
                            NnetDiscriminativeExample *eg) {
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
  // shift will be zero after dividing by frame_subsampling_factor
  // (e.g. frame_subsampling_factor == 3 and shift = 0 or 1.
  std::vector<NnetDiscriminativeSupervision>::iterator
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

size_t NnetDiscriminativeExampleStructureHasher::operator () (
    const NnetDiscriminativeExample &eg) const noexcept {
  // these numbers were chosen at random from a list of primes.
  NnetIoStructureHasher io_hasher;
  size_t size = eg.inputs.size(), ans = size * 35099;
  for (size_t i = 0; i < size; i++)
    ans = ans * 19157 + io_hasher(eg.inputs[i]);
  for (size_t i = 0; i < eg.outputs.size(); i++) {
    const NnetDiscriminativeSupervision &sup = eg.outputs[i];
    StringHasher string_hasher;
    IndexVectorHasher indexes_hasher;
    ans = ans * 17957 +
        string_hasher(sup.name) + indexes_hasher(sup.indexes);
  }
  return ans;
}

bool NnetDiscriminativeExampleStructureCompare::operator () (
    const NnetDiscriminativeExample &a,
    const NnetDiscriminativeExample &b) const {
  NnetIoStructureCompare io_compare;
  if (a.inputs.size() != b.inputs.size() ||
      a.outputs.size() != b.outputs.size())
    return false;
  size_t size = a.inputs.size();
  for (size_t i = 0; i < size; i++)
    if (!io_compare(a.inputs[i], b.inputs[i]))
      return false;
  size = a.outputs.size();
  for (size_t i = 0; i < size; i++)
    if (a.outputs[i].name != b.outputs[i].name ||
        a.outputs[i].indexes != b.outputs[i].indexes)
      return false;
  return true;
}


int32 GetNnetDiscriminativeExampleSize(const NnetDiscriminativeExample &a) {
  int32 ans = 0;
  for (size_t i = 0; i < a.inputs.size(); i++) {
    int32 s = a.inputs[i].indexes.size();
    if (s > ans)
      ans = s;
  }
  for (size_t i = 0; i < a.outputs.size(); i++) {
    int32 s = a.outputs[i].indexes.size();
    if (s > ans)
      ans = s;
  }
  return ans;
}


DiscriminativeExampleMerger::DiscriminativeExampleMerger(const ExampleMergingConfig &config,
                             NnetDiscriminativeExampleWriter *writer):
    finished_(false), num_egs_written_(0),
    config_(config), writer_(writer) { }


void DiscriminativeExampleMerger::AcceptExample(NnetDiscriminativeExample *eg) {
  KALDI_ASSERT(!finished_);
  // If an eg with the same structure as 'eg' is already a key in the
  // map, it won't be replaced, but if it's new it will be made
  // the key.  Also we remove the key before making the vector empty.
  // This way we ensure that the eg in the key is always the first
  // element of the vector.
  std::vector<NnetDiscriminativeExample*> &vec = eg_to_egs_[eg];
  vec.push_back(eg);
  int32 eg_size = GetNnetDiscriminativeExampleSize(*eg),
      num_available = vec.size();
  bool input_ended = false;
  int32 minibatch_size = config_.MinibatchSize(eg_size, num_available,
                                               input_ended);
  if (minibatch_size != 0) {  // we need to write out a merged eg.
    KALDI_ASSERT(minibatch_size == num_available);

    std::vector<NnetDiscriminativeExample*> vec_copy(vec);
    eg_to_egs_.erase(eg);

    // MergeDiscriminativeExamples() expects a vector of NnetDiscriminativeExample, not of pointers,
    // so use swap to create that without doing any real work.
    std::vector<NnetDiscriminativeExample> egs_to_merge(minibatch_size);
    for (int32 i = 0; i < minibatch_size; i++) {
      egs_to_merge[i].Swap(vec_copy[i]);
      delete vec_copy[i];  // we owned those pointers.
    }
    WriteMinibatch(&egs_to_merge);
  }
}

void DiscriminativeExampleMerger::WriteMinibatch(
    std::vector<NnetDiscriminativeExample> *egs) {
  KALDI_ASSERT(!egs->empty());
  int32 eg_size = GetNnetDiscriminativeExampleSize((*egs)[0]);
  NnetDiscriminativeExampleStructureHasher eg_hasher;
  size_t structure_hash = eg_hasher((*egs)[0]);
  int32 minibatch_size = egs->size();
  stats_.WroteExample(eg_size, structure_hash, minibatch_size);
  NnetDiscriminativeExample merged_eg;
  MergeDiscriminativeExamples(config_.compress, egs, &merged_eg);
  std::ostringstream key;
  key << "merged-" << (num_egs_written_++) << "-" << minibatch_size;
  writer_->Write(key.str(), merged_eg);
}

void DiscriminativeExampleMerger::Finish() {
  if (finished_) return;  // already finished.
  finished_ = true;

  // we'll convert the map eg_to_egs_ to a vector of vectors to avoid
  // iterator invalidation problems.
  std::vector<std::vector<NnetDiscriminativeExample*> > all_egs;
  all_egs.reserve(eg_to_egs_.size());

  MapType::iterator iter = eg_to_egs_.begin(), end = eg_to_egs_.end();
  for (; iter != end; ++iter)
    all_egs.push_back(iter->second);
  eg_to_egs_.clear();

  for (size_t i = 0; i < all_egs.size(); i++) {
    int32 minibatch_size;
    std::vector<NnetDiscriminativeExample*> &vec = all_egs[i];
    KALDI_ASSERT(!vec.empty());
    int32 eg_size = GetNnetDiscriminativeExampleSize(*(vec[0]));
    bool input_ended = true;
    while (!vec.empty() &&
           (minibatch_size = config_.MinibatchSize(eg_size, vec.size(),
                                                   input_ended)) != 0) {
      // MergeDiscriminativeExamples() expects a vector of
      // NnetDiscriminativeExample, not of pointers, so use swap to create that
      // without doing any real work.
      std::vector<NnetDiscriminativeExample> egs_to_merge(minibatch_size);
      for (int32 i = 0; i < minibatch_size; i++) {
        egs_to_merge[i].Swap(vec[i]);
        delete vec[i];  // we owned those pointers.
      }
      vec.erase(vec.begin(), vec.begin() + minibatch_size);
      WriteMinibatch(&egs_to_merge);
    }
    if (!vec.empty()) {
      int32 eg_size = GetNnetDiscriminativeExampleSize(*(vec[0]));
      NnetDiscriminativeExampleStructureHasher eg_hasher;
      size_t structure_hash = eg_hasher(*(vec[0]));
      int32 num_discarded = vec.size();
      stats_.DiscardedExamples(eg_size, structure_hash, num_discarded);
      for (int32 i = 0; i < num_discarded; i++)
        delete vec[i];
      vec.clear();
    }
  }
  stats_.PrintStats();
}



} // namespace nnet3
} // namespace kaldi
