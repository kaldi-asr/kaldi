// nnet/nnet-example.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

namespace kaldi {


void NnetTrainingExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<NnetTrainingExample>");
  WriteToken(os, binary, "<Labels>");
  int32 size = labels.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++) {
    WriteBasicType(os, binary, labels[i].first);
    WriteBasicType(os, binary, labels[i].second);
  }
  WriteToken(os, binary, "<InputFrames>");
  input_frames.Write(os, binary);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context);
  WriteToken(os, binary, "<SpkInfo>");
  spk_info.Write(os, binary);
  WriteToken(os, binary, "</NnetTrainingExample>");
}
void NnetTrainingExample::Read(std::istream &is, bool binary) {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  ExpectToken(is, binary, "<NnetTrainingExample>");  
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
  ExpectToken(is, binary, "</NnetTrainingExample>");
}


  
  
} // namespace
