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

#include "nnet3/nnet-ctc-example.h"

namespace kaldi {
namespace nnet3 {

void NnetCtcOutput::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetCtcOutput>");
  WriteToken(os, binary, name);
  WriteToken(os, binary, "<NumOutputs>");
  int32 size = supervision.size();
  KALDI_ASSERT(size > 0 && "Attempting to write empty NnetCtcOutput.");
  WriteBasicType(os, binary, size);
  if (!binary) os << "\n";  
  for (int32 i = 0; i < size; i++) {
    supervision[i].Write(os, binary);
    if (!binary) os << "\n";
  }
  WriteToken(os, binary, "</NnetCtcOutput>");
}

void NnetCtcOutput::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetCtcOutput>");
  ReadToken(is, binary, &name);
  ExpectToken(is, binary, "<NumOutputs>");
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size > 0 && size < 1000000);
  supervision.resize(size);
  for (int32 i = 0; i < size; i++)
    supervision[i].Read(is, binary);
  ExpectToken(is, binary, "</NnetCtcOutput>");
}


void NnetCtcOutput::GetIndexes(std::vector<Index> *indexes) const {
  KALDI_ASSERT(!supervision.empty());
  int32 total_size = 0;
  std::vector<ctc::CtcSupervision>::const_iterator
      iter = supervision.begin(), end = supervision.end();
  for (; iter != end; ++iter)
    total_size += iter->num_frames;
  indexes->resize(total_size);
  std::vector<Index>::iterator out_iter = indexes->begin();
  int32 n = 0;
  for (iter = supervision.begin(); iter != end; ++iter,++n) {
    int32 this_first_frame = iter->first_frame,
        this_frame_skip = iter->frame_skip,
        this_num_frames = iter->num_frames;
    for (int32 i = 0; i < this_num_frames; i++, ++out_iter) {
      int32 t = this_first_frame + i * this_frame_skip, x = 0;
      *out_iter = Index(n, t, x);
    }
  }
  KALDI_ASSERT(out_iter == indexes->end());
}

void NnetCtcExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<Nnet3CtcEg>");
  WriteToken(os, binary, "<NumInputs>");
  int32 size = inputs.size();
  WriteBasicType(os, binary, size);
  KALDI_ASSERT(size > 0 && "Attempting to write NnetCtcExample with no inputs");
  if (!binary) os << '\n';
  for (int32 i = 0; i < size; i++) {
    inputs[i].Write(os, binary);
    if (!binary) os << '\n';
  }
  size = outputs.size();
  WriteBasicType(os, binary, size);
  KALDI_ASSERT(size > 0 && "Attempting to write NnetCtcExample with no outputs");
  if (!binary) os << '\n';
  for (int32 i = 0; i < size; i++) {
    outputs[i].Write(os, binary);
    if (!binary) os << '\n';
  }
  WriteToken(os, binary, "</Nnet3CtcEg>");
}

void NnetCtcExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3CtcEg>");
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
  ExpectToken(is, binary, "</Nnet3CtcEg>");
}

void NnetCtcExample::Swap(NnetCtcExample *other) {
  inputs.swap(other->inputs);
  outputs.swap(other->outputs);
}

void NnetCtcExample::Compress() {
  std::vector<NnetIo>::iterator iter = inputs.begin(),
      end = inputs.end();
  // calling features.Compress() will do nothing if they are sparse or already
  // compressed.
  for (; iter != end; ++iter)
    iter->features.Compress();
}

} // namespace nnet3
} // namespace kaldi
