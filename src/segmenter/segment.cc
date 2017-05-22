// segmenter/segment.cc

// Copyright 2016    Vimal Manohar

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

#include "segmenter/segment.h"

namespace kaldi {
namespace segmenter {

void Segment::Write(std::ostream &os, bool binary) const {
  if (binary) {
    os.write(reinterpret_cast<const char *>(&start_frame), sizeof(start_frame));
    os.write(reinterpret_cast<const char *>(&end_frame), sizeof(start_frame));
    os.write(reinterpret_cast<const char *>(&class_id), sizeof(class_id));
  } else {
    WriteBasicType(os, binary, start_frame);
    WriteBasicType(os, binary, end_frame);
    WriteBasicType(os, binary, Label());
  }
}

void Segment::Read(std::istream &is, bool binary) {
  if (binary) {
    is.read(reinterpret_cast<char *>(&start_frame), sizeof(start_frame));
    is.read(reinterpret_cast<char *>(&end_frame), sizeof(end_frame));
    is.read(reinterpret_cast<char *>(&class_id), sizeof(class_id));
  } else {
    ReadBasicType(is, binary, &start_frame);
    ReadBasicType(is, binary, &end_frame);
    int32 label;
    ReadBasicType(is, binary, &label);
    SetLabel(label);
  }
    
  KALDI_ASSERT(end_frame >= start_frame && start_frame >= 0);
}

std::ostream& operator<<(std::ostream& os, const Segment &seg) {
  os << "[ ";
  seg.Write(os, false);
  os << "]";
  return os;
}  

} // end namespace segmenter 
} // end namespace kaldi
