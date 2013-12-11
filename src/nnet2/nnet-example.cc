// nnet/nnet-example.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)

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

void NnetExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<NnetExample>");
  WriteToken(os, binary, "<Labels>");
  int32 size = labels.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++) {
    WriteBasicType(os, binary, labels[i].first);
    WriteBasicType(os, binary, labels[i].second);
  }
  WriteToken(os, binary, "<InputFrames>");
  input_frames.Write(os, binary); // can be read as regular Matrix.
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
  ExpectToken(is, binary, "<Labels>");
  int32 size;
  ReadBasicType(is, binary, &size);
  labels.resize(size);
  for (int32 i = 0; i < size; i++) {
    ReadBasicType(is, binary, &(labels[i].first));
    ReadBasicType(is, binary, &(labels[i].second));
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
