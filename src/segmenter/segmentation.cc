// segmenter/segmentation.cc

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

#include "segmenter/segmentation.h"
#include <algorithm>

namespace kaldi {
namespace segmenter {

void Segmentation::PushBack(const Segment &seg) {
  dim_++;
  segments_.push_back(seg);
}

SegmentList::iterator Segmentation::Insert(SegmentList::iterator it,
                                           const Segment &seg) {
  dim_++;
  return segments_.insert(it, seg);
}

void Segmentation::EmplaceBack(int32 start_frame, int32 end_frame,
                               int32 class_id) {
  dim_++;
  Segment seg(start_frame, end_frame, class_id);
  segments_.push_back(seg);
}

SegmentList::iterator Segmentation::Emplace(SegmentList::iterator it,
                                            int32 start_frame, int32 end_frame,
                                            int32 class_id) {
  dim_++;
  Segment seg(start_frame, end_frame, class_id);
  return segments_.insert(it, seg);
}

SegmentList::iterator Segmentation::Erase(SegmentList::iterator it) {
  dim_--;
  return segments_.erase(it);
}

void Segmentation::Clear() {
  segments_.clear();
  dim_ = 0;
}

void Segmentation::Read(std::istream &is, bool binary) {
  Clear();

  if (binary) {
    int32 sz = is.peek();
    if (sz == Segment::SizeInBytes()) {
      is.get();
    } else {
      KALDI_ERR << "Segmentation::Read: expected to see Segment of size "
                << Segment::SizeInBytes() << ", saw instead " << sz
                << ", at file position " << is.tellg();
    }

    int32 segmentssz;
    is.read(reinterpret_cast<char *>(&segmentssz), sizeof(segmentssz));
    if (is.fail() || segmentssz < 0)
      KALDI_ERR << "Segmentation::Read: read failure at file position "
        << is.tellg();

    for (int32 i = 0; i < segmentssz; i++) {
      Segment seg;
      seg.Read(is, binary);
      segments_.push_back(seg);
    }
    dim_ = segmentssz;
  } else {
    Segment seg;
    while (1) {
      int i = is.peek();
      if (i == -1) {
        KALDI_ERR << "Unexpected EOF";
      } else if (static_cast<char>(i) == '\n') {
        if (seg.start_frame != -1) {
          KALDI_ERR << "No semicolon before newline (wrong format)";
        } else {
          is.get();
          break;
        }
      } else if (std::isspace(i)) {
        is.get();
      } else if (static_cast<char>(i) == ';') {
        if (seg.start_frame != -1) {
          segments_.push_back(seg);
          dim_++;
          seg.Reset();
        } else {
          is.get();
          KALDI_ASSERT(static_cast<char>(is.peek()) == '\n');
          is.get();
          break;
        }
        is.get();
      } else {
        seg.Read(is, false);
      }
    }
  }
#ifdef KALDI_PARANOID
  Check();
#endif
}

void Segmentation::Write(std::ostream &os, bool binary) const {
#ifdef KALDI_PARANOID
  Check();
#endif

  SegmentList::const_iterator it = Begin();
  if (binary) {
    char sz = Segment::SizeInBytes();
    os.write(&sz, 1);

    int32 segmentssz = static_cast<int32>(Dim());
    KALDI_ASSERT((size_t)segmentssz == Dim());

    os.write(reinterpret_cast<const char *>(&segmentssz), sizeof(segmentssz));

    for (; it != End(); ++it) {
      it->Write(os, binary);
    }
  } else {
    if (Dim() == 0) {
      os << ";";
    }
    for (; it != End(); ++it) {
      it->Write(os, binary);
      os << "; ";
    }
    os << std::endl;
  }
}

void Segmentation::Check() const {
  int32 dim = 0;
  for (SegmentList::const_iterator it = Begin(); it != End(); ++it, dim++) {
    KALDI_ASSERT(it->start_frame >= 0);
    KALDI_ASSERT(it->end_frame >= 0);
    KALDI_ASSERT(it->Label() >= 0);
  }
  KALDI_ASSERT(dim == dim_);
}

void Segmentation::Sort() {
  segments_.sort(SegmentComparator());
}

void Segmentation::SortByLength() {
  segments_.sort(SegmentLengthComparator());
}

SegmentList::iterator Segmentation::MinElement() {
  return std::min_element(segments_.begin(), segments_.end(),
                          SegmentLengthComparator());
}

SegmentList::iterator Segmentation::MaxElement() {
  return std::max_element(segments_.begin(), segments_.end(),
                          SegmentLengthComparator());
}

Segmentation::Segmentation() {
  Clear();
}


void Segmentation::GenRandomSegmentation(int32 max_length,
                                         int32 max_segment_length,
                                         int32 num_classes) {
  Clear();
  int32 st = 0;
  int32 end = 0;

  while (st < max_length) {
    int32 segment_length = RandInt(1, max_segment_length);

    end = st + segment_length - 1;

    // Choose random class id
    int32 k = RandInt(-1, num_classes - 1);

    if (k >= 0) {
      Segment seg(st, end, k);
      segments_.push_back(seg);
      dim_++;
    }

    // Choose random shift i.e. the distance between two adjacent segments
    int32 shift = RandInt(0, max_segment_length);
    st = end + shift;
  }

  Check();
}

}  // namespace segmenter
}  // namespace kaldi
