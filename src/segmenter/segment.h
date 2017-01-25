// segmenter/segment.h"

// Copyright 2016   Vimal Manohar (Johns Hopkins University)

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

#ifndef KALDI_SEGMENTER_SEGMENT_H_
#define KALDI_SEGMENTER_SEGMENT_H_

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {
namespace segmenter {

/**
 * This structure defines a single segment. It consists of the following basic
 * properties:
 * 1) start_frame : This is the frame index of the first frame in the
 *                  segment.
 * 2) end_frame   : This is the frame index of the last frame in the segment.
 *                  Note that the end_frame is included in the segment.
 * 3) class_id    : This is the class corresponding to the segments. For e.g.,
 *                  could be 0, 1 or 2 depending on whether the segment is 
 *                  silence, speech or noise. In general, it can be any
 *                  integer class label.
**/

struct Segment {
  int32 start_frame;
  int32 end_frame;
  int32 class_id;

  // Accessors for labels or class id. This is useful in the future when 
  // we might change the type of label.
  inline int32 Label() const { return class_id; }
  inline void SetLabel(int32 label) { class_id = label; }
  inline int32 Length() const { return end_frame - start_frame + 1; }
  
  // This is the default constructor that sets everything to undefined values.
  Segment() : start_frame(-1), end_frame(-1), class_id(-1) { }

  // This constructor initializes the segmented with the provided start and end
  // frames and the segment label. This is the main constructor.
  Segment(int32 start, int32 end, int32 label) : 
    start_frame(start), end_frame(end), class_id(label) { }

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // This is a function that returns the size of the elements in the structure.
  // It is used during I/O in binary mode, which checks for the total size
  // required to store the segment.
  static size_t SizeInBytes() {
    return (sizeof(int32) + sizeof(int32) + sizeof(int32));
  }

  void Reset() {
      start_frame = -1;
      end_frame = -1;
      class_id = -1;
  }
};

/** 
 * Comparator to order segments based on start frame
**/

class SegmentComparator {
  public:
    bool operator() (const Segment &lhs, const Segment &rhs) const {
      return lhs.start_frame < rhs.start_frame;
    }
};

/** 
 * Comparator to order segments based on length
**/

class SegmentLengthComparator {
  public:
    bool operator() (const Segment &lhs, const Segment &rhs) const {
      return lhs.Length() < rhs.Length();
    }
};
  
std::ostream& operator<<(std::ostream& os, const Segment &seg);

} // end namespace segmenter 
} // end namespace kaldi

#endif // KALDI_SEGMENTER_SEGMENT_H_
