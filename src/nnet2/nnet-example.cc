// nnet2/nnet-example.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)
//                2014  Vimal Manohar

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

#include "nnet2/nnet-example.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet2 {

// This function returns true if the example has labels which, for each frame,
// have a single element with probability one; and if so, it outputs them to the
// vector in the associated pointer.  This enables us to write the egs more
// compactly to disk in this common case.
bool HasSimpleLabels(
    const NnetExample &eg,
    std::vector<int32> *simple_labels) {
  size_t num_frames = eg.labels.size();
  for (int32 t = 0; t < num_frames; t++)
    if (eg.labels[t].size() != 1 || eg.labels[t][0].second != 1.0)
      return false;
  simple_labels->resize(num_frames);
  for (int32 t = 0; t < num_frames; t++)
    (*simple_labels)[t] = eg.labels[t][0].first;
  return true;
}


void NnetExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<NnetExample>");

  // At this point, we write <Lab1> if we have "simple" labels, or
  // <Lab2> in general.  Previous code (when we had only one frame of
  // labels) just wrote <Labels>.
  std::vector<int32> simple_labels;
  if (HasSimpleLabels(*this, &simple_labels)) {
    WriteToken(os, binary, "<Lab1>");
    WriteIntegerVector(os, binary, simple_labels);
  } else {
    WriteToken(os, binary, "<Lab2>");
    int32 num_frames = labels.size();
    WriteBasicType(os, binary, num_frames);
    for (int32 t = 0; t < num_frames; t++) {
      int32 size = labels[t].size();
      WriteBasicType(os, binary, size);
      for (int32 i = 0; i < size; i++) {
        WriteBasicType(os, binary, labels[t][i].first);
        WriteBasicType(os, binary, labels[t][i].second);
      }
    }
  }
  WriteToken(os, binary, "<InputFrames>");
  input_frames.Write(os, binary);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context);
  WriteToken(os, binary, "<SpkInfo>");
  spk_info.Write(os, binary);
  WriteToken(os, binary, "</NnetExample>");
}

void NnetExample::Read(std::istream &is, bool binary) {
  // Note: weight, label, input_frames, left_context and spk_info are members.
  // This is a struct.
  ExpectToken(is, binary, "<NnetExample>");

  std::string token;
  ReadToken(is, binary, &token);
  if (!strcmp(token.c_str(), "<Lab1>")) {  // simple label format
    std::vector<int32> simple_labels;
    ReadIntegerVector(is, binary, &simple_labels);
    labels.resize(simple_labels.size());
    for (size_t i = 0; i < simple_labels.size(); i++) {
      labels[i].resize(1);
      labels[i][0].first = simple_labels[i];
      labels[i][0].second = 1.0;
    }
  } else if (!strcmp(token.c_str(), "<Lab2>")) {  // generic label format
    int32 num_frames;
    ReadBasicType(is, binary, &num_frames);
    KALDI_ASSERT(num_frames > 0);
    labels.resize(num_frames);
    for (int32 t = 0; t < num_frames; t++) {
      int32 size;
      ReadBasicType(is, binary, &size);
      KALDI_ASSERT(size >= 0);
      labels[t].resize(size);
      for (int32 i = 0; i < size; i++) {
        ReadBasicType(is, binary, &(labels[t][i].first));
        ReadBasicType(is, binary, &(labels[t][i].second));
      }
    }
  } else if (!strcmp(token.c_str(), "<Labels>")) {  // back-compatibility
    labels.resize(1);  // old format had 1 frame of labels.
    int32 size;
    ReadBasicType(is, binary, &size);
    labels[0].resize(size);
    for (int32 i = 0; i < size; i++) {
      ReadBasicType(is, binary, &(labels[0][i].first));
      ReadBasicType(is, binary, &(labels[0][i].second));
    }
  } else {
    KALDI_ERR << "Expected token <Lab1>, <Lab2> or <Labels>, got " << token;
  }
  ExpectToken(is, binary, "<InputFrames>");
  input_frames.Read(is, binary);
  ExpectToken(is, binary, "<LeftContext>"); // Note: this member is
  // recently added, but I don't think we'll get too much back-compatibility
  // problems from not handling the old format.
  ReadBasicType(is, binary, &left_context);
  ExpectToken(is, binary, "<SpkInfo>");
  spk_info.Read(is, binary);
  ExpectToken(is, binary, "</NnetExample>");
}

void NnetExample::SetLabelSingle(int32 frame, int32 pdf_id, BaseFloat weight) {
  KALDI_ASSERT(static_cast<size_t>(frame) < labels.size());
  labels[frame].clear();
  labels[frame].push_back(std::make_pair(pdf_id, weight));
}

int32 NnetExample::GetLabelSingle(int32 frame, BaseFloat *weight) {
  BaseFloat max = -1.0;
  int32 pdf_id = -1;
  KALDI_ASSERT(static_cast<size_t>(frame) < labels.size());
  for (int32 i = 0; i < labels[frame].size(); i++) {
    if (labels[frame][i].second > max) {
      pdf_id = labels[frame][i].first;
      max = labels[frame][i].second;
    }
  }
  if (weight != NULL) *weight = max;
  return pdf_id;
}



static bool nnet_example_warned_left = false, nnet_example_warned_right = false;

// Self-constructor that can reduce the number of frames and/or context.
NnetExample::NnetExample(const NnetExample &input,
                         int32 start_frame,
                         int32 new_num_frames,
                         int32 new_left_context,
                         int32 new_right_context): spk_info(input.spk_info) {
  int32 num_label_frames = input.labels.size();
  if (start_frame < 0) start_frame = 0;  // start_frame is offset in the labeled
                                         // frames.
  KALDI_ASSERT(start_frame < num_label_frames);
  if (start_frame + new_num_frames > num_label_frames || new_num_frames == -1)
    new_num_frames = num_label_frames - start_frame;
  // compute right-context of input.
  int32 input_right_context =
      input.input_frames.NumRows() - input.left_context - num_label_frames;
  if (new_left_context == -1) new_left_context = input.left_context;
  if (new_right_context == -1) new_right_context = input_right_context;
  if (new_left_context > input.left_context) {
    if (!nnet_example_warned_left) {
      nnet_example_warned_left = true;
      KALDI_WARN << "Requested left-context " << new_left_context
                 << " exceeds input left-context " << input.left_context
                 << ", will not warn again.";
    }
    new_left_context = input.left_context;
  }
  if (new_right_context > input_right_context) {
    if (!nnet_example_warned_right) {
      nnet_example_warned_right = true;
      KALDI_WARN << "Requested right-context " << new_right_context
                 << " exceeds input right-context " << input_right_context
                 << ", will not warn again.";
    }
    new_right_context = input_right_context;
  }

  int32 new_tot_frames = new_left_context + new_num_frames + new_right_context,
      left_frames_lost = (input.left_context - new_left_context) + start_frame;
  
  CompressedMatrix new_input_frames(input.input_frames,
                                    left_frames_lost,
                                    new_tot_frames,
                                    0, input.input_frames.NumCols());
  new_input_frames.Swap(&input_frames);  // swap with class-member.
  left_context = new_left_context;  // set class-member.
  labels.clear();
  labels.insert(labels.end(),
                input.labels.begin() + start_frame,
                input.labels.begin() + start_frame + new_num_frames);
}

void ExamplesRepository::AcceptExamples(
    std::vector<NnetExample> *examples) {
  KALDI_ASSERT(!examples->empty());
  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  examples_.swap(*examples);
  full_semaphore_.Signal();
}

void ExamplesRepository::ExamplesDone() {
  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  done_ = true;
  full_semaphore_.Signal();
}

bool ExamplesRepository::ProvideExamples(
    std::vector<NnetExample> *examples) {
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal(); // Increment the semaphore so
    // the call by the next thread will not block.
    return false; // no examples to return-- all finished.
  } else {
    KALDI_ASSERT(!examples_.empty() && examples->empty());
    examples->swap(examples_);
    empty_semaphore_.Signal();
    return true;
  }
}


void DiscriminativeNnetExample::Write(std::ostream &os,
                                              bool binary) const {
  // Note: weight, num_ali, den_lat, input_frames, left_context and spk_info are
  // members.  This is a struct.
  WriteToken(os, binary, "<DiscriminativeNnetExample>");
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight);
  WriteToken(os, binary, "<NumAli>");
  WriteIntegerVector(os, binary, num_ali);
  if (!WriteCompactLattice(os, binary, den_lat)) {
    // We can't return error status from this function so we
    // throw an exception. 
    KALDI_ERR << "Error writing CompactLattice to stream";
  }
  WriteToken(os, binary, "<InputFrames>");
  {
    CompressedMatrix cm(input_frames); // Note: this can be read as a regular
                                       // matrix.
    cm.Write(os, binary);
  }
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context);
  WriteToken(os, binary, "<SpkInfo>");
  spk_info.Write(os, binary);
  WriteToken(os, binary, "</DiscriminativeNnetExample>");
}

void DiscriminativeNnetExample::Read(std::istream &is,
                                             bool binary) {
  // Note: weight, num_ali, den_lat, input_frames, left_context and spk_info are
  // members.  This is a struct.
  ExpectToken(is, binary, "<DiscriminativeNnetExample>");
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight);
  ExpectToken(is, binary, "<NumAli>");
  ReadIntegerVector(is, binary, &num_ali);
  CompactLattice *den_lat_tmp = NULL;
  if (!ReadCompactLattice(is, binary, &den_lat_tmp) || den_lat_tmp == NULL) {
    // We can't return error status from this function so we
    // throw an exception. 
    KALDI_ERR << "Error reading CompactLattice from stream";
  }
  den_lat = *den_lat_tmp;
  delete den_lat_tmp;
  ExpectToken(is, binary, "<InputFrames>");
  input_frames.Read(is, binary);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context);
  ExpectToken(is, binary, "<SpkInfo>");
  spk_info.Read(is, binary);
  ExpectToken(is, binary, "</DiscriminativeNnetExample>");
}

void DiscriminativeNnetExample::Check() const {
  KALDI_ASSERT(weight > 0.0);
  KALDI_ASSERT(!num_ali.empty());
  int32 num_frames = static_cast<int32>(num_ali.size());


  std::vector<int32> times;
  int32 num_frames_den = CompactLatticeStateTimes(den_lat, &times);
  KALDI_ASSERT(num_frames == num_frames_den);
  KALDI_ASSERT(input_frames.NumRows() >= left_context + num_frames);
}


} // namespace nnet2
} // namespace kaldi
