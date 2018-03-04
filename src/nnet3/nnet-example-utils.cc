// nnet3/nnet-example-utils.cc

// Copyright 2012-2015    Johns Hopkins University (author: Daniel Povey)
//                2014    Vimal Manohar

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

#include "nnet3/nnet-example-utils.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"
#include "util/text-utils.h"
#include <numeric>
#include <iomanip>

namespace kaldi {
namespace nnet3 {


// get a sorted list of all NnetIo names in all examples in the list (will
// normally be just the strings "input" and "output", but maybe also "ivector").
static void GetIoNames(const std::vector<NnetExample> &src,
                            std::vector<std::string> *names_vec) {
  std::set<std::string> names;
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {
    std::vector<NnetIo>::const_iterator iter2 = iter->io.begin(),
                                         end2 = iter->io.end();
    for (; iter2 != end2; ++iter2)
      names.insert(iter2->name);
  }
  CopySetToVector(names, names_vec);
}

// Get feature "sizes" for each NnetIo name, which are the total number of
// Indexes for that NnetIo (needed to correctly size the output matrix).  Also
// make sure the dimensions are consistent for each name.
static void GetIoSizes(const std::vector<NnetExample> &src,
                       const std::vector<std::string> &names,
                       std::vector<int32> *sizes) {
  std::vector<int32> dims(names.size(), -1);  // just for consistency checking.
  sizes->clear();
  sizes->resize(names.size(), 0);
  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {
    std::vector<NnetIo>::const_iterator iter2 = iter->io.begin(),
                                         end2 = iter->io.end();
    for (; iter2 != end2; ++iter2) {
      const NnetIo &io = *iter2;
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, io.name);
      KALDI_ASSERT(*names_iter == io.name);
      int32 i = names_iter - names_begin;
      int32 this_dim = io.features.NumCols();
      if (dims[i] == -1) {
        dims[i] = this_dim;
      } else if (dims[i] != this_dim) {
        KALDI_ERR << "Merging examples with inconsistent feature dims: "
                  << dims[i] << " vs. " << this_dim << " for '"
                  << io.name << "'.";
      }
      KALDI_ASSERT(io.features.NumRows() == io.indexes.size());
      int32 this_size = io.indexes.size();
      (*sizes)[i] += this_size;
    }
  }
}




// Do the final merging of NnetIo, once we have obtained the names, dims and
// sizes for each feature/supervision type.
static void MergeIo(const std::vector<NnetExample> &src,
                    const std::vector<std::string> &names,
                    const std::vector<int32> &sizes,
                    bool compress,
                    NnetExample *merged_eg) {
  // The total number of Indexes we have across all examples.
  int32 num_feats = names.size();

  std::vector<int32> cur_size(num_feats, 0);

  // The features in the different NnetIo in the Indexes across all examples
  std::vector<std::vector<GeneralMatrix const*> > output_lists(num_feats);

  // Initialize the merged_eg
  merged_eg->io.clear();
  merged_eg->io.resize(num_feats);
  for (int32 f = 0; f < num_feats; f++) {
    NnetIo &io = merged_eg->io[f];
    int32 size = sizes[f];
    KALDI_ASSERT(size > 0);
    io.name = names[f];
    io.indexes.resize(size);
  }

  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();
  std::vector<NnetExample>::const_iterator eg_iter = src.begin(),
    eg_end = src.end();
  for (int32 n = 0; eg_iter != eg_end; ++eg_iter, ++n) {
    std::vector<NnetIo>::const_iterator io_iter = eg_iter->io.begin(),
      io_end = eg_iter->io.end();
    for (; io_iter != io_end; ++io_iter) {
      const NnetIo &io = *io_iter;
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, io.name);
      KALDI_ASSERT(*names_iter == io.name);

      int32 f = names_iter - names_begin;
      int32 this_size = io.indexes.size();
      int32 &this_offset = cur_size[f];
      KALDI_ASSERT(this_size + this_offset <= sizes[f]);

      // Add f'th Io's features
      output_lists[f].push_back(&(io.features));

      // Work on the Indexes for the f^th Io in merged_eg
      NnetIo &output_io = merged_eg->io[f];
      std::copy(io.indexes.begin(), io.indexes.end(),
                output_io.indexes.begin() + this_offset);
      std::vector<Index>::iterator output_iter = output_io.indexes.begin();
      // Set the n index to be different for each of the original examples.
      for (int32 i = this_offset; i < this_offset + this_size; i++) {
        // we could easily support merging already-merged egs, but I don't see a
        // need for it right now.
        KALDI_ASSERT(output_iter[i].n == 0 &&
                     "Merging already-merged egs?  Not currentlysupported.");
        output_iter[i].n = n;
      }
      this_offset += this_size;  // note: this_offset is a reference.
    }
  }
  KALDI_ASSERT(cur_size == sizes);
  for (int32 f = 0; f < num_feats; f++) {
    AppendGeneralMatrixRows(output_lists[f],
                            &(merged_eg->io[f].features));
    if (compress) {
      // the following won't do anything if the features were sparse.
      merged_eg->io[f].features.Compress();
    }
  }
}



void MergeExamples(const std::vector<NnetExample> &src,
                   bool compress,
                   NnetExample *merged_eg) {
  KALDI_ASSERT(!src.empty());
  std::vector<std::string> io_names;
  GetIoNames(src, &io_names);
  // the sizes are the total number of Indexes we have across all examples.
  std::vector<int32> io_sizes;
  GetIoSizes(src, io_names, &io_sizes);
  MergeIo(src, io_names, io_sizes, compress, merged_eg);
}

void ShiftExampleTimes(int32 t_offset,
                       const std::vector<std::string> &exclude_names,
                       NnetExample *eg) {
  if (t_offset == 0)
    return;
  std::vector<NnetIo>::iterator iter = eg->io.begin(),
      end = eg->io.end();
  for (; iter != end; iter++) {
    bool name_is_excluded = false;
    std::vector<std::string>::const_iterator
        exclude_iter = exclude_names.begin(),
        exclude_end = exclude_names.end();
    for (; exclude_iter != exclude_end; ++exclude_iter) {
      if (iter->name == *exclude_iter) {
        name_is_excluded = true;
        break;
      }
    }
    if (!name_is_excluded) {
      // name is not something like "ivector" that we exclude from shifting.
      std::vector<Index>::iterator index_iter = iter->indexes.begin(),
          index_end = iter->indexes.end();
      for (; index_iter != index_end; ++index_iter)
        index_iter->t += t_offset;
    }
  }
}

void GetComputationRequest(const Nnet &nnet,
                           const NnetExample &eg,
                           bool need_model_derivative,
                           bool store_component_stats,
                           ComputationRequest *request) {
  request->inputs.clear();
  request->inputs.reserve(eg.io.size());
  request->outputs.clear();
  request->outputs.reserve(eg.io.size());
  request->need_model_derivative = need_model_derivative;
  request->store_component_stats = store_component_stats;
  for (size_t i = 0; i < eg.io.size(); i++) {
    const NnetIo &io = eg.io[i];
    const std::string &name = io.name;
    int32 node_index = nnet.GetNodeIndex(name);
    if (node_index == -1 &&
        !nnet.IsInputNode(node_index) && !nnet.IsOutputNode(node_index))
      KALDI_ERR << "Nnet example has input or output named '" << name
                << "', but no such input or output node is in the network.";

    std::vector<IoSpecification> &dest =
        nnet.IsInputNode(node_index) ? request->inputs : request->outputs;
    dest.resize(dest.size() + 1);
    IoSpecification &io_spec = dest.back();
    io_spec.name = name;
    io_spec.indexes = io.indexes;
    io_spec.has_deriv = nnet.IsOutputNode(node_index) && need_model_derivative;
  }
  // check to see if something went wrong.
  if (request->inputs.empty())
    KALDI_ERR << "No inputs in computation request.";
  if (request->outputs.empty())
    KALDI_ERR << "No outputs in computation request.";
}

void WriteVectorAsChar(std::ostream &os,
                       bool binary,
                       const VectorBase<BaseFloat> &vec) {
  if (binary) {
    int32 dim = vec.Dim();
    std::vector<unsigned char> char_vec(dim);
    const BaseFloat *data = vec.Data();
    for (int32 i = 0; i < dim; i++) {
      BaseFloat value = data[i];
      KALDI_ASSERT(value >= 0.0 && value <= 1.0);
      // below, the adding 0.5 is done so that we round to the closest integer
      // rather than rounding down (since static_cast will round down).
      char_vec[i] = static_cast<unsigned char>(255.0 * value + 0.5);
    }
    WriteIntegerVector(os, binary, char_vec);
  } else {
    // the regular floating-point format will be more readable for text mode.
    vec.Write(os, binary);
  }
}

void ReadVectorAsChar(std::istream &is,
                      bool binary,
                      Vector<BaseFloat> *vec) {
  if (binary) {
    BaseFloat scale = 1.0 / 255.0;
    std::vector<unsigned char> char_vec;
    ReadIntegerVector(is, binary, &char_vec);
    int32 dim = char_vec.size();
    vec->Resize(dim, kUndefined);
    BaseFloat *data = vec->Data();
    for (int32 i = 0; i < dim; i++)
      data[i] = scale * char_vec[i];
  } else {
    vec->Read(is, binary);
  }
}

void RoundUpNumFrames(int32 frame_subsampling_factor,
                      int32 *num_frames,
                      int32 *num_frames_overlap) {
  if (*num_frames % frame_subsampling_factor != 0) {
    int32 new_num_frames = frame_subsampling_factor *
        (*num_frames / frame_subsampling_factor + 1);
    KALDI_LOG << "Rounding up --num-frames=" << (*num_frames)
              << " to a multiple of --frame-subsampling-factor="
              << frame_subsampling_factor
              << ", now --num-frames=" << new_num_frames;
    *num_frames = new_num_frames;
  }
  if (*num_frames_overlap % frame_subsampling_factor != 0) {
    int32 new_num_frames_overlap = frame_subsampling_factor *
        (*num_frames_overlap / frame_subsampling_factor + 1);
    KALDI_LOG << "Rounding up --num-frames-overlap=" << (*num_frames_overlap)
              << " to a multiple of --frame-subsampling-factor="
              << frame_subsampling_factor
              << ", now --num-frames-overlap=" << new_num_frames_overlap;
    *num_frames_overlap = new_num_frames_overlap;
  }
  if (*num_frames_overlap < 0 || *num_frames_overlap >= *num_frames) {
    KALDI_ERR << "--num-frames-overlap=" << (*num_frames_overlap) << " < "
              << "--num-frames=" << (*num_frames);
  }
}

void ExampleGenerationConfig::ComputeDerived() {
  if (!SplitStringToIntegers(num_frames_str, ",", false, &num_frames) ||
      num_frames.empty()) {
    KALDI_ERR << "Invalid option (expected comma-separated list of integers): "
              << "--num-frames=" << num_frames_str;
  }

  int32 m = frame_subsampling_factor;
  if (m < 1) {
    KALDI_ERR << "Invalid value --frame-subsampling-factor=" << m;
  }
  bool changed = false;
  for (size_t i = 0; i < num_frames.size(); i++) {
    int32 value = num_frames[i];
    if (value <= 0) {
      KALDI_ERR << "Invalid option --num-frames=" << num_frames_str;
    }
    if (value % m != 0) {
      value = m * ((value / m) + 1);
      changed = true;
    }
    num_frames[i] = value;
  }
  if (changed) {
    std::ostringstream rounded_num_frames_str;
    for (size_t i = 0; i < num_frames.size(); i++) {
      if (i > 0)
        rounded_num_frames_str << ',';
      rounded_num_frames_str << num_frames[i];
    }
    KALDI_LOG << "Rounding up --num-frames=" << num_frames_str
              << " to multiples of --frame-subsampling-factor=" << m
              << ", to: " << rounded_num_frames_str.str();
  }
}


UtteranceSplitter::UtteranceSplitter(const ExampleGenerationConfig &config):
    config_(config),
    total_num_utterances_(0), total_input_frames_(0),
    total_frames_overlap_(0), total_num_chunks_(0),
    total_frames_in_chunks_(0) {
  if (config.num_frames.empty()) {
    KALDI_ERR << "You need to call ComputeDerived() on the "
                 "ExampleGenerationConfig().";
  }
  InitSplitForLength();
}

UtteranceSplitter::~UtteranceSplitter() {
  KALDI_LOG << "Split " << total_num_utterances_ << " utts, with "
            << "total length " << total_input_frames_ << " frames ("
            << (total_input_frames_ / 360000.0) << " hours assuming "
            << "100 frames per second)";
  float average_chunk_length = total_frames_in_chunks_ * 1.0 / total_num_chunks_,
      overlap_percent = total_frames_overlap_ * 100.0 / total_input_frames_,
      output_percent = total_frames_in_chunks_ * 100.0 / total_input_frames_,
      output_percent_no_overlap = output_percent - overlap_percent;

  KALDI_LOG << "Average chunk length was " << average_chunk_length
            << " frames; overlap between adjacent chunks was "
            << overlap_percent << "% of input length; length of output was "
            << output_percent << "% of input length (minus overlap = "
            << output_percent_no_overlap << "%).";
  if (chunk_size_to_count_.size() > 1) {
    std::ostringstream os;
    os << std::setprecision(4);
    for (std::map<int32, int32>::iterator iter = chunk_size_to_count_.begin();
         iter != chunk_size_to_count_.end(); ++iter) {
      int32 chunk_size = iter->first,
          num_frames = chunk_size * iter->second;
      float percent_of_total = num_frames * 100.0 / total_frames_in_chunks_;
      if (iter != chunk_size_to_count_.begin()) os << ", ";
      os << chunk_size << " = " << percent_of_total << "%";
    }
    KALDI_LOG << "Output frames are distributed among chunk-sizes as follows: "
              << os.str();
  }
}

float UtteranceSplitter::DefaultDurationOfSplit(
    const std::vector<int32> &split) const {
  if (split.empty())  // not a valid split, but useful to handle this case.
    return 0.0;
  float principal_num_frames = config_.num_frames[0],
      num_frames_overlap = config_.num_frames_overlap;
  KALDI_ASSERT(num_frames_overlap < principal_num_frames &&
               "--num-frames-overlap value is too high");
  float overlap_proportion = num_frames_overlap / principal_num_frames;
  float ans = std::accumulate(split.begin(), split.end(), int32(0));
  for (size_t i = 0; i + 1 < split.size(); i++) {
    float min_adjacent_chunk_length = std::min(split[i], split[i + 1]),
        overlap = overlap_proportion * min_adjacent_chunk_length;
    ans -= overlap;
  }
  KALDI_ASSERT(ans > 0.0);
  return ans;
}

/*
  This comment describes the idea behind what InitChunkSize() is supposed to do,
  and how it relates to the purpose of class UtteranceSplitter.

  Class UtteranceSplitter is supposed to tell us, for a given utterance length,
  what chunk sizes to use.  The chunk sizes it may choose are:
    - zero or more chunks of the 'principal' size (the first-listed value in
      --num-frames option)
    - at most two chunks of 'alternative' num-frames (meaning, any but the
      first-listed choice in the --num-frames option).

  (note: an empty list of chunks is not allowed as a split).  A split is
  a list of chunk-sizes in increasing order (we when we actually split the
  utterance into chunks, we may, at random, reverse the order.

  The choice of split to use for a given utterance-length is determined as
  follows.  Firstly, for each split we compute a 'default duration' (see
  DefaultDurationOfSplit()... if --num-frames-overlap is zero, this is just the
  sum of the chunk sizes).  We then use by a cost-function that depends on
  default-duration and the length of the utterance: the idea is that these two
  should be as close as possible, but penalizing the default-duration being
  larger than the utterance-length (which in the normal case of
  --num-frames-overlap=0 would lead to gaps between the segments), twice as much
  as the other sign of difference.

  Specifically:
    cost(default_duration, utt_length) = (default_duration > utt_length ?
                                         default_duration - utt_length :
                                         2.0 * (utt_length - default_duration))
  [but as a special case, set c to infinity if the largest chunk size in the
   split is longer than the utterance length; we couldn't, in that case, use
   this split for this utterance].

  We want to make sure a good variety of combinations of chunk sizes are chosen
  in case there are ties from the cost function.  For each utterance length
  we store the set of splits, whose costs are within 2
  of the best cost available for that utterance length.  When asked to find
  chunks for a particular utterance of that length, we will choose randomly
  from that pool of splits.
 */
void UtteranceSplitter::InitSplitForLength() {
  int32 max_utterance_length = MaxUtteranceLength();

  // The 'splits' vector is a list of possible splits (a split being
  // a sorted vector of chunk-sizes).
  // The vector 'splits' is itself sorted.
  std::vector<std::vector<int32> > splits;
  InitSplits(&splits);


  // Define a split-index 0 <= s < splits.size() as index into the 'splits'
  // vector, and let a cost c >= 0 represent the mismatch between an
  // utterance length and the total length of the chunk sizes in a split:

  //  c(default_duration, utt_length) = (default_duration > utt_length ?
  //                                    default_duration - utt_length :
  //                                    2.0 * (utt_length - default_duration))
  // [but as a special case, set c to infinity if the largest chunk size in the
  //  split is longer than the utterance length; we couldn't, in that case, use
  //  this split for this utterance].

  // 'costs_for_length[u][s]', indexed by utterance-length u and then split,
  // contains the cost for utterance-length u and split s.

  std::vector<std::vector<float> > costs_for_length(
      max_utterance_length + 1);
  int32 num_splits = splits.size();

  for (int32 u = 0; u <= max_utterance_length; u++)
    costs_for_length[u].reserve(num_splits);

  for (int32 s = 0; s < num_splits; s++) {
    const std::vector<int32> &split = splits[s];
    float default_duration = DefaultDurationOfSplit(split);
    int32 max_chunk_size = *std::max_element(split.begin(), split.end());
    for (int32 u = 0; u <= max_utterance_length; u++) {
      // c is the cost for this utterance length and this split.  We penalize
      // gaps twice as strongly as overlaps, based on the intuition that
      // completely throwing out frames of data is worse than counting them
      // twice.
      float c = (default_duration > float(u) ? default_duration - float(u) :
                 2.0 * (u - default_duration));
      if (u < max_chunk_size)  // can't fit the largest of the chunks in this
                               // utterance
        c = std::numeric_limits<float>::max();
      KALDI_ASSERT(c >= 0);
      costs_for_length[u].push_back(c);
    }
  }


  splits_for_length_.resize(max_utterance_length + 1);

  for (int32 u = 0; u <= max_utterance_length; u++) {
    const std::vector<float> &costs = costs_for_length[u];
    float min_cost = *std::min_element(costs.begin(), costs.end());
    if (min_cost == std::numeric_limits<float>::max()) {
      // All costs were infinity, becaues this utterance-length u is shorter
      // than the smallest chunk-size.  Leave splits_for_length_[u] as empty
      // for this utterance-length, meaning we will not be able to choose any
      // split, and such utterances will be discarded.
      continue;
    }
    float cost_threshold = 1.9999; // We will choose pseudo-randomly from splits
                                   // that are within this distance from the
                                   // best cost.  Make the threshold just
                                   // slightly less than 2...  this will
                                   // hopefully make the behavior more
                                   // deterministic for ties.
    std::vector<int32> possible_splits;
    std::vector<float>::const_iterator iter = costs.begin(), end = costs.end();
    int32 s = 0;
    for (; iter != end; ++iter,++s)
      if (*iter < min_cost + cost_threshold)
        splits_for_length_[u].push_back(splits[s]);
  }

  if (GetVerboseLevel() >= 3) {
    std::ostringstream os;
    for (int32 u = 0; u <= max_utterance_length; u++) {
      if (!splits_for_length_[u].empty()) {
        os << u << "=(";
        std::vector<std::vector<int32 > >::const_iterator
            iter1 = splits_for_length_[u].begin(),
            end1 = splits_for_length_[u].end();

        while (iter1 != end1) {
          std::vector<int32>::const_iterator iter2 = iter1->begin(),
              end2 = iter1->end();
          while (iter2 != end2) {
            os << *iter2;
            ++iter2;
            if (iter2 != end2) os << ",";
          }
          ++iter1;
          if (iter1 != end1) os << "/";
        }
        os << ")";
        if (u < max_utterance_length) os << ", ";
      }
    }
    KALDI_VLOG(3) << "Utterance-length-to-splits map is: " << os.str();
  }
}


bool UtteranceSplitter::LengthsMatch(const std::string &utt,
                                     int32 utterance_length,
                                     int32 supervision_length,
                                     int32 length_tolerance) const {
  int32 sf = config_.frame_subsampling_factor,
      expected_supervision_length = (utterance_length + sf - 1) / sf;
  if (std::abs(supervision_length - expected_supervision_length) 
      <= length_tolerance) {
    return true;
  } else {
    if (sf == 1) {
      KALDI_WARN << "Supervision does not have expected length for utterance "
                 << utt << ": expected length = " << utterance_length
                 << ", got " << supervision_length;
    } else {
      KALDI_WARN << "Supervision does not have expected length for utterance "
                 << utt << ": expected length = (" << utterance_length
                 << " + " << sf << " - 1) / " << sf << " = "
                 << expected_supervision_length
                 << ", got: " << supervision_length
                 << " (note: --frame-subsampling-factor=" << sf << ")";
    }
    return false;
  }
}


void UtteranceSplitter::GetChunkSizesForUtterance(
    int32 utterance_length, std::vector<int32> *chunk_sizes) const {
  KALDI_ASSERT(!splits_for_length_.empty());
  // 'primary_length' is the first-specified num-frames.
  // It's the only chunk that may be repeated an arbitrary number
  // of times.
  int32 primary_length = config_.num_frames[0],
      num_frames_overlap = config_.num_frames_overlap,
      max_tabulated_length = splits_for_length_.size() - 1,
      num_primary_length_repeats = 0;
  KALDI_ASSERT(primary_length - num_frames_overlap > 0);
  KALDI_ASSERT(utterance_length >= 0);
  while (utterance_length > max_tabulated_length) {
    utterance_length -= (primary_length - num_frames_overlap);
    num_primary_length_repeats++;
  }
  KALDI_ASSERT(utterance_length >= 0);
  const std::vector<std::vector<int32> > &possible_splits =
      splits_for_length_[utterance_length];
  if (possible_splits.empty()) {
    chunk_sizes->clear();
    return;
  }
  int32 num_possible_splits = possible_splits.size(),
      randomly_chosen_split = RandInt(0, num_possible_splits - 1);
  *chunk_sizes = possible_splits[randomly_chosen_split];
  for (int32 i = 0; i < num_primary_length_repeats; i++)
    chunk_sizes->push_back(primary_length);

  std::sort(chunk_sizes->begin(), chunk_sizes->end());
  if (RandInt(0, 1) == 0) {
    std::reverse(chunk_sizes->begin(), chunk_sizes->end());
  }
}


int32 UtteranceSplitter::MaxUtteranceLength() const {
  int32 num_lengths = config_.num_frames.size();
  KALDI_ASSERT(num_lengths > 0);
  // 'primary_length' is the first-specified num-frames.
  // It's the only chunk that may be repeated an arbitrary number
  // of times.
  int32 primary_length = config_.num_frames[0],
      max_length = primary_length;
  for (int32 i = 0; i < num_lengths; i++) {
    KALDI_ASSERT(config_.num_frames[i] > 0);
    max_length = std::max(config_.num_frames[i], max_length);
  }
  return 2 * max_length + primary_length;
}

void UtteranceSplitter::InitSplits(std::vector<std::vector<int32> > *splits) const {
  // we consider splits whose default duration (as returned by
  // DefaultDurationOfSplit()) is up to MaxUtteranceLength() + primary_length.
  // We can be confident without doing a lot of math, that splits above this
  // length will never be chosen for any utterance-length up to
  // MaxUtteranceLength() (which is the maximum we use).
  int32 primary_length = config_.num_frames[0],
      default_duration_ceiling = MaxUtteranceLength() + primary_length;

  typedef unordered_set<std::vector<int32>, VectorHasher<int32> > SetType;

  SetType splits_set;

  int32 num_lengths = config_.num_frames.size();

  // The splits we are allow are: zero to two 'alternate' lengths, plus
  // an arbitrary number of repeats of the 'primary' length.  The repeats
  // of the 'primary' length are handled by the inner loop over n.
  // The zero to two 'alternate' lengths are handled by the loops over
  // i and j.  i == 0 and j == 0 are special cases; they mean, no
  // alternate is chosen.
  for (int32 i = 0; i < num_lengths; i++) {
    for (int32 j = 0; j < num_lengths; j++) {
      std::vector<int32> vec;
      if (i > 0)
        vec.push_back(config_.num_frames[i]);
      if (j > 0)
        vec.push_back(config_.num_frames[j]);
      int32 n = 0;
      while (DefaultDurationOfSplit(vec) <= default_duration_ceiling) {
        if (!vec.empty()) // Don't allow the empty vector as a split.
          splits_set.insert(vec);
        n++;
        vec.push_back(primary_length);
        std::sort(vec.begin(), vec.end());
      }
    }
  }
  for (SetType::const_iterator iter = splits_set.begin();
       iter != splits_set.end(); ++iter)
    splits->push_back(*iter);
  std::sort(splits->begin(), splits->end());  // make the order deterministic,
                                              // for consistency of output
                                              // between runs and C libraries.
}


// static
void UtteranceSplitter::DistributeRandomlyUniform(int32 n, std::vector<int32> *vec) {
  KALDI_ASSERT(!vec->empty());
  int32 size = vec->size();
  if (n < 0) {
    DistributeRandomlyUniform(-n, vec);
    for (int32 i = 0; i < size; i++)
      (*vec)[i] *= -1;
    return;
  }
  // from this point we know n >= 0.
  int32 common_part = n / size,
      remainder = n % size, i;
  for (i = 0; i < remainder; i++) {
    (*vec)[i] = common_part + 1;
  }
  for (; i < size; i++) {
    (*vec)[i] = common_part;
  }
  std::random_shuffle(vec->begin(), vec->end());
  KALDI_ASSERT(std::accumulate(vec->begin(), vec->end(), int32(0)) == n);
}


// static
void UtteranceSplitter::DistributeRandomly(int32 n,
                                           const std::vector<int32> &magnitudes,
                                           std::vector<int32> *vec) {
  KALDI_ASSERT(!vec->empty() && vec->size() == magnitudes.size());
  int32 size = vec->size();
  if (n < 0) {
    DistributeRandomly(-n, magnitudes, vec);
    for (int32 i = 0; i < size; i++)
      (*vec)[i] *= -1;
    return;
  }
  float total_magnitude = std::accumulate(magnitudes.begin(), magnitudes.end(),
                                          int32(0));
  KALDI_ASSERT(total_magnitude > 0);
  // note: 'partial_counts' contains the negative of the partial counts, so
  // when we sort the larger partial counts come first.
  std::vector<std::pair<float, int32> > partial_counts;
  int32 total_count = 0;
  for (int32 i = 0; i < size; i++) {
    float this_count = n * float(magnitudes[i]) / total_magnitude;
    // note: cast of float to int32 rounds towards zero (down, in this
    // case, since this_count >= 0).
    int32 this_whole_count = static_cast<int32>(this_count),
        this_partial_count = this_count - this_whole_count;
    (*vec)[i] = this_whole_count;
    total_count += this_whole_count;
    partial_counts.push_back(std::pair<float, int32>(-this_partial_count, i));
  }
  KALDI_ASSERT(total_count <= n && total_count + size >= n);
  std::sort(partial_counts.begin(), partial_counts.end());
  int32 i = 0;
  // Increment by one the elements of the vector that has the largest partial
  // count, then the next largest partial count, and so on... until we reach the
  // desired total-count 'n'.
  for(; total_count < n; i++,total_count++) {
    (*vec)[partial_counts[i].second]++;
  }
  KALDI_ASSERT(std::accumulate(vec->begin(), vec->end(), int32(0)) == n);
}


void UtteranceSplitter::GetGapSizes(int32 utterance_length,
                                    bool enforce_subsampling_factor,
                                    const std::vector<int32> &chunk_sizes,
                                    std::vector<int32> *gap_sizes) const {
  if (chunk_sizes.empty()) {
    gap_sizes->clear();
    return;
  }
  if (enforce_subsampling_factor && config_.frame_subsampling_factor > 1) {
    int32 sf = config_.frame_subsampling_factor, size = chunk_sizes.size();
    int32 utterance_length_reduced = (utterance_length + (sf - 1)) / sf;
    std::vector<int32> chunk_sizes_reduced(chunk_sizes);
    for (int32 i = 0; i < size; i++) {
      KALDI_ASSERT(chunk_sizes[i] % config_.frame_subsampling_factor == 0);
      chunk_sizes_reduced[i] /= config_.frame_subsampling_factor;
    }
    GetGapSizes(utterance_length_reduced, false,
                chunk_sizes_reduced, gap_sizes);
    KALDI_ASSERT(gap_sizes->size() == static_cast<size_t>(size));
    for (int32 i = 0; i < size; i++)
      (*gap_sizes)[i] *= config_.frame_subsampling_factor;
    return;
  }
  int32 num_chunks = chunk_sizes.size(),
      total_of_chunk_sizes = std::accumulate(chunk_sizes.begin(),
                                             chunk_sizes.end(),
                                             int32(0)),
      total_gap = utterance_length - total_of_chunk_sizes;
  gap_sizes->resize(num_chunks);

  if (total_gap < 0) {
    // there is an overlap.  Overlaps can only go between chunks, not at the
    // beginning or end of the utterance.  Also, we try to make the length of
    // overlap proportional to the size of the smaller of the two chunks
    // that the overlap is between.
    if (num_chunks == 1) {
      // there needs to be an overlap, but there is only one chunk... this means
      // the chunk-size exceeds the utterance length, which is not allowed.
      KALDI_ERR << "Chunk size is " << chunk_sizes[0]
                << " but utterance length is only "
                << utterance_length;
    }

    // note the elements of 'overlaps' will be <= 0.
    std::vector<int32> magnitudes(num_chunks - 1),
        overlaps(num_chunks - 1);
    // the 'magnitudes' vector will contain the minimum of the lengths of the
    // two adjacent chunks between which are are going to consider having an
    // overlap.  These will be used to assign the overlap proportional to that
    // size.
    for (int32 i = 0; i + 1 < num_chunks; i++) {
      magnitudes[i] = std::min<int32>(chunk_sizes[i], chunk_sizes[i + 1]);
    }
    DistributeRandomly(total_gap, magnitudes, &overlaps);
    for (int32 i = 0; i + 1 < num_chunks; i++) {
      // If the following condition does not hold, it's possible we
      // could get chunk start-times less than zero.  I don't believe
      // it's possible for this condition to fail, but we're checking
      // for it at this level to make debugging easier, just in case.
      KALDI_ASSERT(overlaps[i] <= magnitudes[i]);
    }

    (*gap_sizes)[0] = 0;  // no gap before 1st chunk.
    for (int32 i = 1; i < num_chunks; i++)
      (*gap_sizes)[i] = overlaps[i-1];
  } else {
    // There may be a gap.  Gaps can go at the start or end of the utterance, or
    // between segments.  We try to distribute the gaps evenly.
    std::vector<int32> gaps(num_chunks + 1);
    DistributeRandomlyUniform(total_gap, &gaps);
    // the last element of 'gaps', the one at the end of the utterance, is
    // implicit and doesn't have to be written to the output.
    for (int32 i = 0; i < num_chunks; i++)
      (*gap_sizes)[i] = gaps[i];
  }
}


void UtteranceSplitter::GetChunksForUtterance(
    int32 utterance_length,
    std::vector<ChunkTimeInfo> *chunk_info) {
  std::vector<int32> chunk_sizes;
  GetChunkSizesForUtterance(utterance_length, &chunk_sizes);
  std::vector<int32> gaps(chunk_sizes.size());
  GetGapSizes(utterance_length, true, chunk_sizes, &gaps);
  int32 num_chunks = chunk_sizes.size();
  chunk_info->resize(num_chunks);
  int32 t = 0;
  for (int32 i = 0; i < num_chunks; i++) {
    t += gaps[i];
    ChunkTimeInfo &info = (*chunk_info)[i];
    info.first_frame = t;
    info.num_frames = chunk_sizes[i];
    info.left_context = (i == 0 && config_.left_context_initial >= 0 ?
                         config_.left_context_initial : config_.left_context);
    info.right_context = (i == num_chunks - 1 && config_.right_context_final >= 0 ?
                          config_.right_context_final : config_.right_context);
    t += chunk_sizes[i];
  }
  SetOutputWeights(utterance_length, chunk_info);
  AccStatsForUtterance(utterance_length, *chunk_info);
  // check that the end of the last chunk doesn't go more than
  // 'config_.frame_subsampling_factor - 1' frames past the end
  // of the utterance.  That amount, we treat as rounding error.
  KALDI_ASSERT(t - utterance_length < config_.frame_subsampling_factor);
}

void UtteranceSplitter::AccStatsForUtterance(
    int32 utterance_length,
    const std::vector<ChunkTimeInfo> &chunk_info) {
  total_num_utterances_ += 1;
  total_input_frames_ += utterance_length;

  for (size_t c = 0; c < chunk_info.size(); c++) {
    int32 chunk_size = chunk_info[c].num_frames;
    if (c > 0) {
      int32 last_chunk_end = chunk_info[c-1].first_frame +
          chunk_info[c-1].num_frames;
      if (last_chunk_end > chunk_info[c].first_frame)
        total_frames_overlap_ += last_chunk_end - chunk_info[c].first_frame;
    }
    std::map<int32, int32>::iterator iter = chunk_size_to_count_.find(
        chunk_size);
    if (iter == chunk_size_to_count_.end())
      chunk_size_to_count_[chunk_size] = 1;
    else
      iter->second++;
    total_num_chunks_ += 1;
    total_frames_in_chunks_ += chunk_size;
  }
}


void UtteranceSplitter::SetOutputWeights(
    int32 utterance_length,
    std::vector<ChunkTimeInfo> *chunk_info) const {
  int32 sf = config_.frame_subsampling_factor;
  int32 num_output_frames = (utterance_length + sf - 1) / sf;
  // num_output_frames is the number of frames of supervision.  'count[t]' will
  // be the number of chunks that this output-frame t appears in.  Note: the
  // 'first_frame' and 'num_frames' members of ChunkTimeInfo will always be
  // multiples of frame_subsampling_factor.
  std::vector<int32> count(num_output_frames, 0);
  int32 num_chunks = chunk_info->size();
  for (int32 i = 0; i < num_chunks; i++) {
    ChunkTimeInfo &chunk = (*chunk_info)[i];
    for (int32 t = chunk.first_frame / sf;
         t < (chunk.first_frame + chunk.num_frames) / sf;
         t++)
      count[t]++;
  }
  for (int32 i = 0; i < num_chunks; i++) {
    ChunkTimeInfo &chunk = (*chunk_info)[i];
    chunk.output_weights.resize(chunk.num_frames / sf);
    int32 t_start = chunk.first_frame / sf;
    for (int32 t = t_start;
         t < (chunk.first_frame + chunk.num_frames) / sf;
         t++)
      chunk.output_weights[t - t_start] = 1.0 / count[t];
  }
}

int32 ExampleMergingConfig::IntSet::LargestValueInRange(int32 max_value) const {
  KALDI_ASSERT(!ranges.empty());
  int32 ans = 0, num_ranges = ranges.size();
  for (int32 i = 0; i < num_ranges; i++) {
    int32 possible_ans = 0;
    if (max_value >= ranges[i].first) {
      if (max_value >= ranges[i].second)
        possible_ans = ranges[i].second;
      else
        possible_ans = max_value;
    }
    if (possible_ans > ans)
      ans = possible_ans;
  }
  return ans;
}

// static
bool ExampleMergingConfig::ParseIntSet(const std::string &str,
                                       ExampleMergingConfig::IntSet *int_set) {
  std::vector<std::string> split_str;
  SplitStringToVector(str, ",", false, &split_str);
  if (split_str.empty())
    return false;
  int_set->largest_size = 0;
  int_set->ranges.resize(split_str.size());
  for (size_t i = 0; i < split_str.size(); i++) {
    std::vector<int32> split_range;
    SplitStringToIntegers(split_str[i], ":", false, &split_range);
    if (split_range.size() < 1 || split_range.size() > 2 ||
        split_range[0] > split_range.back() || split_range[0] <= 0)
      return false;
    int_set->ranges[i].first = split_range[0];
    int_set->ranges[i].second = split_range.back();
    int_set->largest_size = std::max<int32>(int_set->largest_size,
                                            split_range.back());
  }
  return true;
}

void ExampleMergingConfig::ComputeDerived() {
  if (measure_output_frames != "deprecated") {
    KALDI_WARN << "The --measure-output-frames option is deprecated "
        "and will be ignored.";
  }
  if (discard_partial_minibatches != "deprecated") {
    KALDI_WARN << "The --discard-partial-minibatches option is deprecated "
        "and will be ignored.";
  }
  std::vector<std::string> minibatch_size_split;
  SplitStringToVector(minibatch_size, "/", false, &minibatch_size_split);
  if (minibatch_size_split.empty()) {
    KALDI_ERR << "Invalid option --minibatch-size=" << minibatch_size;
  }

  rules.resize(minibatch_size_split.size());
  for (size_t i = 0; i < minibatch_size_split.size(); i++) {
    int32 &eg_size = rules[i].first;
    IntSet &int_set = rules[i].second;
    // 'this_rule' will be either something like "256" or like "64-128,256"
    // (but these two only if  minibatch_size_split.size() == 1, or something with
    // an example-size specified, like "256=64-128,256"
    std::string &this_rule = minibatch_size_split[i];
    if (this_rule.find('=') != std::string::npos) {
      std::vector<std::string> rule_split;  // split on '='
      SplitStringToVector(this_rule, "=", false, &rule_split);
      if (rule_split.size() != 2) {
        KALDI_ERR << "Could not parse option --minibatch-size="
                  << minibatch_size;
      }
      if (!ConvertStringToInteger(rule_split[0], &eg_size) ||
          !ParseIntSet(rule_split[1], &int_set))
        KALDI_ERR << "Could not parse option --minibatch-size="
                  << minibatch_size;

    } else {
      if (minibatch_size_split.size() != 1) {
        KALDI_ERR << "Could not parse option --minibatch-size="
                  << minibatch_size << " (all rules must have "
                  << "eg-size specified if >1 rule)";
      }
      if (!ParseIntSet(this_rule, &int_set))
        KALDI_ERR << "Could not parse option --minibatch-size="
                  << minibatch_size;
    }
  }
  {
    // check that no size is repeated.
    std::vector<int32> all_sizes(minibatch_size_split.size());
    for (size_t i = 0; i < minibatch_size_split.size(); i++)
      all_sizes[i] = rules[i].first;
    std::sort(all_sizes.begin(), all_sizes.end());
    if (!IsSortedAndUniq(all_sizes)) {
      KALDI_ERR << "Invalid --minibatch-size=" << minibatch_size
                << " (repeated example-sizes)";
    }
  }
}

int32 ExampleMergingConfig::MinibatchSize(int32 size_of_eg,
                                          int32 num_available_egs,
                                          bool input_ended) const {
  KALDI_ASSERT(num_available_egs > 0 && size_of_eg > 0);
  int32 num_rules = rules.size();
  if (num_rules == 0)
    KALDI_ERR << "You need to call ComputeDerived() before calling "
        "MinibatchSize().";
  int32 min_distance = std::numeric_limits<int32>::max(),
      closest_rule_index = 0;
  for (int32 i = 0; i < num_rules; i++) {
    int32 distance = std::abs(size_of_eg - rules[i].first);
    if (distance < min_distance) {
      min_distance = distance;
      closest_rule_index = i;
    }
  }
  if (!input_ended) {
    // until the input ends, we can only use the largest available
    // minibatch-size (otherwise, we could expect more later).
    int32 largest_size = rules[closest_rule_index].second.largest_size;
    if (largest_size <= num_available_egs)
      return largest_size;
    else
      return 0;
  } else {
    int32 s = rules[closest_rule_index].second.LargestValueInRange(
        num_available_egs);
    KALDI_ASSERT(s <= num_available_egs);
    return s;
  }
}


void ExampleMergingStats::WroteExample(int32 example_size,
                                    size_t structure_hash,
                                    int32 minibatch_size) {
  std::pair<int32, size_t> p(example_size, structure_hash);


  unordered_map<int32, int32> &h = stats_[p].minibatch_to_num_written;
  unordered_map<int32, int32>::iterator iter = h.find(minibatch_size);
  if (iter == h.end())
    h[minibatch_size] = 1;
  else
    iter->second += 1;
}

void ExampleMergingStats::DiscardedExamples(int32 example_size,
                                         size_t structure_hash,
                                         int32 num_discarded) {
  std::pair<int32, size_t> p(example_size, structure_hash);
  stats_[p].num_discarded += num_discarded;
}


void ExampleMergingStats::PrintStats() const {
  PrintSpecificStats();
  PrintAggregateStats();
}

void ExampleMergingStats::PrintAggregateStats() const {
  // First print some aggregate stats.
  int64 num_distinct_egs_types = 0,  // number of distinct types of input egs
                                     // (differing in size or structure).
      total_discarded_egs = 0, // total number of discarded egs.
      total_discarded_egs_size = 0, // total number of discarded egs each multiplied by size
                                    // of that eg
      total_non_discarded_egs = 0,  // total over all minibatches written, of
                                    // minibatch-size, equals number of input egs
                                    // that were not discarded.
      total_non_discarded_egs_size = 0,  // total over all minibatches of size-of-eg
                                     // * minibatch-size.
      num_minibatches = 0,  // total number of minibatches
      num_distinct_minibatch_types = 0;  // total number of combination of
                                         // (type-of-eg, number of distinct
                                         // minibatch-sizes for that eg-type)-
                                         // reflects the number of time we have
                                         // to compile.

  StatsType::const_iterator eg_iter = stats_.begin(), eg_end = stats_.end();

  for (; eg_iter != eg_end; ++eg_iter) {
    int32 eg_size = eg_iter->first.first;
    const StatsForExampleSize &stats = eg_iter->second;
    num_distinct_egs_types++;
    total_discarded_egs += stats.num_discarded;
    total_discarded_egs_size += stats.num_discarded * eg_size;

    unordered_map<int32, int32>::const_iterator
        mb_iter = stats.minibatch_to_num_written.begin(),
        mb_end = stats.minibatch_to_num_written.end();
    for (; mb_iter != mb_end; ++mb_iter) {
      int32 mb_size = mb_iter->first,
          num_written = mb_iter->second;
      num_distinct_minibatch_types++;
      num_minibatches += num_written;
      total_non_discarded_egs += num_written * mb_size;
      total_non_discarded_egs_size += num_written * mb_size * eg_size;
    }
  }
  // the averages are written as integers- we don't really need more precision
  // than that.
  int64 total_input_egs = total_discarded_egs + total_non_discarded_egs,
      total_input_egs_size =
      total_discarded_egs_size + total_non_discarded_egs_size;

  float avg_input_egs_size = total_input_egs_size * 1.0 / total_input_egs;
  float percent_discarded = total_discarded_egs * 100.0 / total_input_egs;
  // note: by minibatch size we mean the number of egs per minibatch, it
  // does not take note of the size of the input egs.
  float avg_minibatch_size = total_non_discarded_egs * 1.0 / num_minibatches;

  std::ostringstream os;
  os << std::setprecision(4);
  os << "Processed " << total_input_egs
     << " egs of avg. size " << avg_input_egs_size
     << " into " << num_minibatches << " minibatches, discarding "
     << percent_discarded <<  "% of egs.  Avg minibatch size was "
     << avg_minibatch_size << ", #distinct types of egs/minibatches "
     << "was " << num_distinct_egs_types << "/"
     << num_distinct_minibatch_types;
  KALDI_LOG << os.str();
}

void ExampleMergingStats::PrintSpecificStats() const {
  KALDI_LOG << "Merged specific eg types as follows [format: <eg-size1>="
      "{<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>}"
      ",<egs-size2>={...},... (note,egs-size == number of input "
      "frames including context).";
  std::ostringstream os;

  // copy from unordered map to map to get sorting, for consistent output.
  typedef std::map<std::pair<int32, size_t>, StatsForExampleSize> SortedMapType;

  SortedMapType stats;
  stats.insert(stats_.begin(), stats_.end());
  SortedMapType::const_iterator eg_iter = stats.begin(), eg_end = stats.end();
  for (; eg_iter != eg_end; ++eg_iter) {
    int32 eg_size = eg_iter->first.first;
    if (eg_iter != stats.begin())
      os << ",";
    os << eg_size << "={";
    const StatsForExampleSize &stats = eg_iter->second;
    unordered_map<int32, int32>::const_iterator
        mb_iter = stats.minibatch_to_num_written.begin(),
        mb_end =  stats.minibatch_to_num_written.end();
    for (; mb_iter != mb_end; ++mb_iter) {
      int32 mb_size = mb_iter->first,
          num_written = mb_iter->second;
      if (mb_iter != stats.minibatch_to_num_written.begin())
        os << ",";
      os << mb_size << "->" << num_written;
    }
    os << ",d=" << stats.num_discarded << "}";
  }
  KALDI_LOG << os.str();
}



int32 GetNnetExampleSize(const NnetExample &a) {
  int32 ans = 0;
  for (size_t i = 0; i < a.io.size(); i++) {
    int32 s = a.io[i].indexes.size();
    if (s > ans)
      ans = s;
  }
  return ans;
}

ExampleMerger::ExampleMerger(const ExampleMergingConfig &config,
                             NnetExampleWriter *writer):
    finished_(false), num_egs_written_(0),
    config_(config), writer_(writer) { }


void ExampleMerger::AcceptExample(NnetExample *eg) {
  KALDI_ASSERT(!finished_);
  // If an eg with the same structure as 'eg' is already a key in the
  // map, it won't be replaced, but if it's new it will be made
  // the key.  Also we remove the key before making the vector empty.
  // This way we ensure that the eg in the key is always the first
  // element of the vector.
  std::vector<NnetExample*> &vec = eg_to_egs_[eg];
  vec.push_back(eg);
  int32 eg_size = GetNnetExampleSize(*eg),
      num_available = vec.size();
  bool input_ended = false;
  int32 minibatch_size = config_.MinibatchSize(eg_size, num_available,
                                               input_ended);
  if (minibatch_size != 0) {  // we need to write out a merged eg.
    KALDI_ASSERT(minibatch_size == num_available);

    std::vector<NnetExample*> vec_copy(vec);
    eg_to_egs_.erase(eg);

    // MergeExamples() expects a vector of NnetExample, not of pointers,
    // so use swap to create that without doing any real work.
    std::vector<NnetExample> egs_to_merge(minibatch_size);
    for (int32 i = 0; i < minibatch_size; i++) {
      egs_to_merge[i].Swap(vec_copy[i]);
      delete vec_copy[i];  // we owned those pointers.
    }
    WriteMinibatch(egs_to_merge);
  }
}

void ExampleMerger::WriteMinibatch(const std::vector<NnetExample> &egs) {
  KALDI_ASSERT(!egs.empty());
  int32 eg_size = GetNnetExampleSize(egs[0]);
  NnetExampleStructureHasher eg_hasher;
  size_t structure_hash = eg_hasher(egs[0]);
  int32 minibatch_size = egs.size();
  stats_.WroteExample(eg_size, structure_hash, minibatch_size);
  NnetExample merged_eg;
  MergeExamples(egs, config_.compress, &merged_eg);
  std::ostringstream key;
  key << "merged-" << (num_egs_written_++) << "-" << minibatch_size;
  writer_->Write(key.str(), merged_eg);
}

void ExampleMerger::Finish() {
  if (finished_) return;  // already finished.
  finished_ = true;

  // we'll convert the map eg_to_egs_ to a vector of vectors to avoid
  // iterator invalidation problems.
  std::vector<std::vector<NnetExample*> > all_egs;
  all_egs.reserve(eg_to_egs_.size());

  MapType::iterator iter = eg_to_egs_.begin(), end = eg_to_egs_.end();
  for (; iter != end; ++iter)
    all_egs.push_back(iter->second);
  eg_to_egs_.clear();

  for (size_t i = 0; i < all_egs.size(); i++) {
    int32 minibatch_size;
    std::vector<NnetExample*> &vec = all_egs[i];
    KALDI_ASSERT(!vec.empty());
    int32 eg_size = GetNnetExampleSize(*(vec[0]));
    bool input_ended = true;
    while (!vec.empty() &&
           (minibatch_size = config_.MinibatchSize(eg_size, vec.size(),
                                                   input_ended)) != 0) {
      // MergeExamples() expects a vector of NnetExample, not of pointers,
      // so use swap to create that without doing any real work.
      std::vector<NnetExample> egs_to_merge(minibatch_size);
      for (int32 i = 0; i < minibatch_size; i++) {
        egs_to_merge[i].Swap(vec[i]);
        delete vec[i];  // we owned those pointers.
      }
      vec.erase(vec.begin(), vec.begin() + minibatch_size);
      WriteMinibatch(egs_to_merge);
    }
    if (!vec.empty()) {
      int32 eg_size = GetNnetExampleSize(*(vec[0]));
      NnetExampleStructureHasher eg_hasher;
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
