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
      if (dims[i] == -1)
        dims[i] = this_dim;
      else if(dims[i] != this_dim) {
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
  int32 num_feats = names.size();
  std::vector<int32> cur_size(num_feats, 0);
  std::vector<std::vector<GeneralMatrix const*> > output_lists(num_feats);
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
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (int32 n = 0; iter != end; ++iter,++n) {
    std::vector<NnetIo>::const_iterator iter2 = iter->io.begin(),
                                         end2 = iter->io.end();
    for (; iter2 != end2; ++iter2) {
      const NnetIo &io = *iter2;
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, io.name);
      KALDI_ASSERT(*names_iter == io.name);
      int32 f = names_iter - names_begin;
      int32 this_size = io.indexes.size(),
          &this_offset = cur_size[f];
      KALDI_ASSERT(this_size + this_offset <= sizes[f]);
      output_lists[f].push_back(&(io.features));
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

bool ContainsSingleExample(const NnetExample &eg,
                           int32 *min_input_t,
                           int32 *max_input_t,
                           int32 *min_output_t,
                           int32 *max_output_t) {
  bool done_input = false, done_output = false;
  int32 num_indexes = eg.io.size();
  for (int32 i = 0; i < num_indexes; i++) {
    const NnetIo &io = eg.io[i];
    std::vector<Index>::const_iterator iter = io.indexes.begin(),
                                        end = io.indexes.end();
    // Should not have an empty input/output type.
    KALDI_ASSERT(!io.indexes.empty());
    if (io.name == "input" || io.name == "output") {
      int32 min_t = iter->t, max_t = iter->t;
      for (; iter != end; ++iter) {
        int32 this_t = iter->t;
        min_t = std::min(min_t, this_t);
        max_t = std::max(max_t, this_t);
        if (iter->n != 0) {
          KALDI_WARN << "Example does not contain just a single example; "
                     << "too late to do frame selection or reduce context.";
          return false;
        }
      }
      if (io.name == "input") {
        done_input = true;
        *min_input_t = min_t;
        *max_input_t = max_t;
      } else {
        KALDI_ASSERT(io.name == "output");
        done_output = true;
        *min_output_t = min_t;
        *max_output_t = max_t;
      }
    } else {
      for (; iter != end; ++iter) {
        if (iter->n != 0) {
          KALDI_WARN << "Example does not contain just a single example; "
                     << "too late to do frame selection or reduce context.";
          return false;
        }
      }
    }
  }
  if (!done_input) {
    KALDI_WARN << "Example does not have any input named 'input'";
    return false;
  }
  if (!done_output) {
    KALDI_WARN << "Example does not have any output named 'output'";
    return false;
  }
  return true;
}

void FilterExample(const NnetExample &eg,
                   int32 min_input_t,
                   int32 max_input_t,
                   int32 min_output_t,
                   int32 max_output_t,
                   NnetExample *eg_out) {
  eg_out->io.clear();
  eg_out->io.resize(eg.io.size());
  for (size_t i = 0; i < eg.io.size(); i++) {
    bool is_input_or_output;
    int32 min_t, max_t;
    const NnetIo &io_in = eg.io[i];
    NnetIo &io_out = eg_out->io[i];
    const std::string &name = io_in.name;
    io_out.name = name;
    if (name == "input") {
      min_t = min_input_t;
      max_t = max_input_t;
      is_input_or_output = true;
    } else if (name == "output") {
      min_t = min_output_t;
      max_t = max_output_t;
      is_input_or_output = true;
    } else {
      is_input_or_output = false;
    }
    if (!is_input_or_output) {  // Just copy everything.
      io_out.indexes = io_in.indexes;
      io_out.features = io_in.features;
    } else {
      const std::vector<Index> &indexes_in = io_in.indexes;
      std::vector<Index> &indexes_out = io_out.indexes;
      indexes_out.reserve(indexes_in.size());
      int32 num_indexes = indexes_in.size(), num_kept = 0;
      KALDI_ASSERT(io_in.features.NumRows() == num_indexes);
      std::vector<bool> keep(num_indexes, false);
      std::vector<Index>::const_iterator iter_in = indexes_in.begin(),
                                          end_in = indexes_in.end();
      std::vector<bool>::iterator iter_out = keep.begin();
      for (; iter_in != end_in; ++iter_in,++iter_out) {
        int32 t = iter_in->t;
        bool is_within_range = (t >= min_t && t <= max_t);
        *iter_out = is_within_range;
        if (is_within_range) {
          indexes_out.push_back(*iter_in);
          num_kept++;
        }
      }
      KALDI_ASSERT(iter_out == keep.end());
      if (num_kept == 0)
        KALDI_ERR << "FilterExample removed all indexes for '" << name << "'";

      FilterGeneralMatrixRows(io_in.features, keep,
                              &io_out.features);
      KALDI_ASSERT(io_out.features.NumRows() == num_kept &&
                   indexes_out.size() == static_cast<size_t>(num_kept));
    }
  }
}

bool SelectFromExample(const NnetExample &eg,
                       std::string frame_str,
                       int32 left_context,
                       int32 right_context,
                       int32 frame_shift,
                       NnetExample *eg_out) {
  int32 min_input_t, max_input_t,
      min_output_t, max_output_t;
  if (!ContainsSingleExample(eg, &min_input_t, &max_input_t,
                             &min_output_t, &max_output_t))
    KALDI_ERR << "Too late to perform frame selection/context reduction on "
              << "these examples (already merged?)";
  if (frame_str != "") {
    // select one frame.
    if (frame_str == "random") {
      min_output_t = max_output_t = RandInt(min_output_t,
                                                          max_output_t);
    } else {
      int32 frame;
      if (!ConvertStringToInteger(frame_str, &frame))
        KALDI_ERR << "Invalid option --frame='" << frame_str << "'";
      if (frame < min_output_t || frame > max_output_t) {
        // Frame is out of range.  Should happen only rarely.  Calling code
        // makes sure of this.
        return false;
      }
      min_output_t = max_output_t = frame;
    }
  }
  // There may come a time when we want to remove or make it possible to disable
  // the error messages below.  The std::max and std::min expressions may seem
  // unnecessary but are intended to make life easier if and when we do that.
  if (left_context != -1) {
    if (min_input_t > min_output_t - left_context)
      KALDI_ERR << "You requested --left-context=" << left_context
                << ", but example only has left-context of "
                <<  (min_output_t - min_input_t);
    min_input_t = std::max(min_input_t, min_output_t - left_context);
  }
  if (right_context != -1) {
    if (max_input_t < max_output_t + right_context)
      KALDI_ERR << "You requested --right-context=" << right_context
                << ", but example only has right-context of "
                <<  (max_input_t - max_output_t);
    max_input_t = std::min(max_input_t, max_output_t + right_context);
  }
  FilterExample(eg,
                min_input_t, max_input_t,
                min_output_t, max_output_t,
                eg_out);
  if (frame_shift != 0) {
    std::vector<std::string> exclude_names;  // we can later make this
    exclude_names.push_back(std::string("ivector")); // configurable.
    ShiftExampleTimes(frame_shift, exclude_names, eg_out);
  }
  return true;
}

} // namespace nnet3
} // namespace kaldi
