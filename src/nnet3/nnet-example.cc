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

void Feature::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<feat>");
  WriteToken(os, binary, name);
  WriteIndexVector(os, binary, indexes);
  features.Write(os, binary);
}

void Feature::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<feat>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  features.Read(is, binary);
}



void NnetExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<Nnet3Eg>");

  WriteToken(os, binary, "<Input>");
  int32 size = input.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    input[i].Write(os, binary);
  WriteToken(os, binary, "<Supervision>");  
  size = supervision.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    supervision[i].Write(os, binary);
  WriteToken(os, binary, "</Nnet3Eg>");
}

void NnetExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3Eg>");
  ExpectToken(is, binary, "<Input>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size < 0 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  input.resize(size);
  for (int32 i = 0; i < size; i++)
    input[i].Read(is, binary);
  ExpectToken(is, binary, "<Supervision>");  
  ReadBasicType(is, binary, &size);
  if (size < 0 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  supervision.resize(size);
  for (int32 i = 0; i < size; i++)
    supervision[i].Read(is, binary);
  ExpectToken(is, binary, "</Nnet3Eg>");
}

} // namespace nnet3
} // namespace kaldi
