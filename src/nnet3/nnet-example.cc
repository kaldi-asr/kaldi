// nnet3/nnet-example.cc

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

#include "nnet3/nnet-example.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet3 {

void InputFeature::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<feat>");
  WriteToken(os, binary, name);
  WriteIndexVector(os, binary, indexes);
  features.Write(os, binary);
}

void InputFeature::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<feat>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  features.Read(is, binary);
}


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
  WriteToken(os, binary, "<Nnet3Eg>");

  WriteBasicType(os, binary, t0);

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
  int32 num_inputs = features.size();
  KALDI_ASSERT(num_inputs > 0);
  WriteBasicType(os, binary, num_inputs);
  for (int32 i = 0; i < num_inputs; i++)
    features[i].Write(os, binary);
  WriteToken(os, binary, "</Nnet3Eg>");
}

void NnetExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3Eg>");

  ReadBasicType(is, binary, &t0);

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
  } else {
    KALDI_ERR << "Expected token <Lab1> or <Lab2>, got " << token;
  }
  int32 num_inputs;
  ReadBasicType(is, binary, &num_inputs);
  if (num_inputs < 0 || num_inputs > 100000)
    KALDI_ERR << "num_inputs out of range: " << num_inputs;
  features.resize(num_inputs);
  for (int32 i = 0; i < num_inputs; i++)
    features[i].Read(is, binary);
  ExpectToken(is, binary, "</Nnet3Eg>");
}


} // namespace nnet3
} // namespace kaldi
