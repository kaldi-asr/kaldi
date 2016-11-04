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

// The following functions construct new segment in-place in the
// segmentation and increments the dim_ of the segmentation. There's one
// emplace for each constructor in Segment.
void Segmentation::EmplaceBack(int32 start_frame, int32 end_frame, int32 class_id) {
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
    if (int c = is.peek() != static_cast<int>('[')) {
      KALDI_ERR << "Segmentation::Read: expected to see [, saw "
                << static_cast<char>(c) << ", at file position " << is.tellg();
    }
    is.get();   // consume the '['
    while (is.peek() != static_cast<int>(']')) {
      KALDI_ASSERT(!is.eof());
      Segment seg;
      seg.Read(is, binary);
      segments_.push_back(seg);
      dim_++;
      is >> std::ws;
    }
    is.get();
    KALDI_ASSERT(!is.eof());
  }
  Check();
}

void Segmentation::Write(std::ostream &os, bool binary) const {
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
    os << "[ ";
    for (; it != End(); ++it) {
      it->Write(os, binary);
      os << std::endl;
    } 
    os << "]" << std::endl;
  }
}

void Segmentation::Check() const {
  int32 dim = 0;
  for (SegmentList::const_iterator it = Begin(); it != End(); ++it, dim++) {
    KALDI_ASSERT(it->start_frame >= 0);
    KALDI_ASSERT(it->end_frame >= 0);
    KALDI_ASSERT(it->Label() >= 0);
  };
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

}
}
