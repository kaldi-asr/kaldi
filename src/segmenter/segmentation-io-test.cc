// segmenter/segmentation-io-test.cc

// Copyright 2015  Vimal Manohar (Johns Hopkins University)

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

#include "segmenter/segmentation.h"

namespace kaldi {
namespace segmenter {

void UnitTestSegmentationIo() {
  Segmentation seg;
  int32 max_length = RandInt(0, 1000),
        max_segment_length = max_length / 10,
        num_classes = RandInt(0, 3);

  if (max_segment_length == 0) 
    max_segment_length = 1;

  seg.GenRandomSegmentation(max_length, max_segment_length, num_classes);

  bool binary = ( RandInt(0,1) == 0 );
  std::ostringstream os;

  seg.Write(os, binary);
  
  Segmentation seg2;
  std::istringstream is(os.str());
  seg2.Read(is, binary);

  std::ostringstream os2;
  seg2.Write(os2, binary);

  KALDI_ASSERT(os2.str() == os.str());
}

} // namespace segmenter
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::segmenter;

  for (int32 i = 0; i < 100; i++)
    UnitTestSegmentationIo();
  return 0;
}
  

