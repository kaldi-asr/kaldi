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


} // namespace nnet3
} // namespace kaldi
