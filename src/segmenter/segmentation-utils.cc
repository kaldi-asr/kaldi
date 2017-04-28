// segmenter/segmentation-utils.cc 
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

#include "segmenter/segmentation-utils.h"

namespace kaldi {
namespace segmenter {

void MergeLabels(const std::vector<int32> &merge_labels,
                 int32 dest_label,
                 Segmentation *segmentation) {
  KALDI_ASSERT(segmentation);

  // Check if sorted and unique
  KALDI_ASSERT(std::adjacent_find(merge_labels.begin(),
                                  merge_labels.end(), std::greater<int32>())
               == merge_labels.end());

  for (SegmentList::iterator it = segmentation->Begin();
       it != segmentation->End(); ++it) {
    if (std::binary_search(merge_labels.begin(), merge_labels.end(),
                           it->Label())) {
      it->SetLabel(dest_label);
    }
  }
#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

void RelabelSegmentsUsingMap(const unordered_map<int32, int32> &label_map,
                             Segmentation *segmentation) {
  int32 default_label = -1;
  unordered_map<int32, int32>::const_iterator it = label_map.find(-1);
  if (it != label_map.end()) {
    default_label = it->second; // default_label is defined.
    KALDI_ASSERT(default_label != -1);
  } // else no default_label

  for (SegmentList::iterator it = segmentation->Begin();
       it != segmentation->End(); ) {
    unordered_map<int32, int32>::const_iterator map_it = label_map.find(
        it->Label());
    int32 dest_label = -100;
    if (map_it == label_map.end()) {
      if (default_label == -1)  // no default_label
        KALDI_ERR << "Could not find label " << it->Label()
                  << " in label map.";
      else
        dest_label = default_label;
    } else {
      dest_label = map_it->second;
    }

    if (dest_label == -1) {
      // Remove segments that will be mapped to label -1.
      it = segmentation->Erase(it);  
      continue;
    } 
    it->SetLabel(dest_label);
    ++it;
  }
}

void RelabelAllSegments(int32 label, Segmentation *segmentation) {
  for (SegmentList::iterator it = segmentation->Begin();
       it != segmentation->End(); ++it)
    it->SetLabel(label);
}

void ScaleFrameShift(BaseFloat factor, Segmentation *segmentation) {
  for (SegmentList::iterator it = segmentation->Begin();
       it != segmentation->End(); ++it) {
    it->start_frame *= factor;
    it->end_frame *= factor;
  }
}

void RemoveSegments(int32 label, Segmentation *segmentation) {
  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ) {
    if (it->Label() == label) {
      it = segmentation->Erase(it);
    } else {
      ++it;
    }
  }
#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

void RemoveSegments(const std::vector<int32> &labels,
                    int32 max_remove_length, 
                    Segmentation *segmentation) {
  // Check if sorted and unique
  KALDI_ASSERT(std::adjacent_find(labels.begin(),
               labels.end(), std::greater<int32>()) == labels.end());

  KALDI_ASSERT (max_remove_length >= -1);
  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ) {
    if (max_remove_length == -1) {  // remove all segments
      if (std::binary_search(labels.begin(), labels.end(), 
                             it->Label()))
        it = segmentation->Erase(it);
      else
        ++it;
    } else if (it->Length() < max_remove_length) {
      if (std::binary_search(labels.begin(), labels.end(), 
                             it->Label()) ||
          (labels.size() == 1 && labels[0] == -1))
        it = segmentation->Erase(it);
      else 
        ++it;
    } else {
      ++it;
    }
  }
#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

void KeepSegments(int32 label, Segmentation *segmentation) {
  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ) {
    if (it->Label() != label) {
      it = segmentation->Erase(it);
    } else {
      ++it;
    }
  }
#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

// TODO(Vimal): Write test function for this.
void SplitInputSegmentation(const Segmentation &in_segmentation,
                            int32 segment_length,
                            Segmentation *out_segmentation) {
  out_segmentation->Clear();
  for (SegmentList::const_iterator it = in_segmentation.Begin();
        it != in_segmentation.End(); ++it) {
    int32 length = it->Length();

    // Since ceil is used, this results in all pieces to be smaller than
    // segment_length rather than being larger.
    int32 num_chunks = std::ceil(static_cast<BaseFloat>(length)
                                 / segment_length);
    int32 actual_segment_length = static_cast<BaseFloat>(length) / num_chunks;

    int32 start_frame = it->start_frame;
    for (int32 j = 0; j < num_chunks; j++) {
      int32 end_frame = std::min(start_frame + actual_segment_length - 1,
                                 it->end_frame);
      out_segmentation->EmplaceBack(start_frame, end_frame, it->Label());
      start_frame = end_frame + 1;
    }
  }
#ifdef KALDI_PARANOID
  out_segmentation->Check();
#endif
}

// TODO(Vimal): Write test function for this.
void SplitSegments(int32 segment_length, int32 min_remainder,
                   int32 overlap_length, int32 segment_label,
                   Segmentation *segmentation) {
  KALDI_ASSERT(segmentation);
  KALDI_ASSERT(segment_length > 0 && min_remainder > 0);
  KALDI_ASSERT(overlap_length >= 0);

  KALDI_ASSERT(overlap_length < segment_length);
  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ++it) {
    if (segment_label != -1 && it->Label() != segment_label) continue;

    int32 start_frame = it->start_frame;
    int32 length = it->Length();

    if (length > segment_length + min_remainder) {
      // Split segment
      // To show what this is doing, consider the following example, where it is
      // currently pointing to B.
      // A <--> B <--> C

      // Modify the start_frame of the current frame. This prepares the current
      // segment to be used as the "next segment" when we move the iterator in
      // the next statement.
      // In the example, the start_frame for B has just been modified.
      it->start_frame = start_frame + segment_length - overlap_length;

      // Create a new segment and add it to the where the current iterator is.
      // The statement below results in this:
      // A <--> B1 <--> B <--> C
      // with the iterator it pointing at B1. So when the iterator is
      // incremented in the for loop, it will point to B again, but whose
      // start_frame had been modified.
      it = segmentation->Emplace(it, start_frame,
                                 start_frame + segment_length - 1,
                                 it->Label());
    }
  }
#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

// TODO(Vimal): Write test code for this
void SplitSegmentsUsingAlignment(int32 segment_length,
                                 int32 segment_label,
                                 const std::vector<int32> &ali,
                                 int32 ali_label,
                                 int32 min_silence_length,
                                 Segmentation *segmentation) {
  KALDI_ASSERT(segmentation);
  KALDI_ASSERT(segment_length > 0);

  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End();) {
    // Safety check. In practice, should never fail.
    KALDI_ASSERT(segmentation->Dim() <= ali.size());

    if (segment_label != -1 && it->Label() != segment_label) {
      ++it;
      continue;
    }

    int32 start_frame = it->start_frame;
    int32 length = it->Length();
    int32 label = it->Label();

    if (length <= segment_length) {
      ++it;
      continue;
    }

    // Split segment
    // To show what this is doing, consider the following example, where it is
    // currently pointing to B.
    // A <--> B <--> C

    Segmentation ali_segmentation;
    InsertFromAlignment(ali, start_frame,
                        start_frame + length,
                        0, &ali_segmentation, NULL);
    KeepSegments(ali_label, &ali_segmentation);
    MergeAdjacentSegments(0, &ali_segmentation);

    // Get largest chunk of alignment where label == ali_label
    SegmentList::iterator s_it = ali_segmentation.MaxElement();

    if (s_it == ali_segmentation.End() || s_it->Length() < min_silence_length) {
      // The largest chunk is smaller than min_silence_length, so
      // skip splitting this segment.
      ++it;
      continue;
    }

    KALDI_ASSERT(s_it->start_frame >= start_frame);
    KALDI_ASSERT(s_it->end_frame <= start_frame + length);

    // Modify the start_frame of the current frame. This prepares the current
    // segment to be used as the "next segment" when we move the iterator in
    // the next statement.
    // In the example, the start_frame for B has just been modified.
    int32 end_frame;
    if (s_it->Length() > 1) {
      end_frame = s_it->start_frame + s_it->Length() / 2 - 2;
      it->start_frame = end_frame + 2;
    } else {
      end_frame = s_it->start_frame - 1;
      it->start_frame = s_it->end_frame + 1;
    }

    // end_frame is within this current segment
    KALDI_ASSERT(end_frame < start_frame + length);
    // The first new segment length is smaller than the old segment length
    KALDI_ASSERT(end_frame - start_frame + 1 < length);

    // The second new segment length is smaller than the old segment length
    KALDI_ASSERT(it->end_frame - end_frame - 1 < length);

    if (it->Length() < 0) {
      // This is possible when the beginning of the segment is silence
      it = segmentation->Erase(it);
    }

    // Create a new segment and add it to the where the current iterator is.
    // The statement below results in this:
    // A <--> B1 <--> B <--> C
    // with the iterator it pointing at B1.
    if (end_frame >= start_frame) {
      it = segmentation->Emplace(it, start_frame, end_frame, label);
    }
  }
#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

// TODO(Vimal): Write test code for this
void MergeAdjacentSegments(int32 max_intersegment_length,
                           Segmentation *segmentation,
                           bool sort) {
  if (sort) Sort(segmentation);
#ifdef KALDI_PARANOID
  else KALDI_ASSERT(IsSorted(*segmentation));
#endif

  SegmentList::iterator it = segmentation->Begin(),
                   prev_it = segmentation->Begin();

  while (it != segmentation->End()) {
    KALDI_ASSERT(it->start_frame >= prev_it->start_frame);

    if (it != segmentation->Begin() &&
        it->Label() == prev_it->Label() &&
        prev_it->end_frame + max_intersegment_length + 1 >= it->start_frame) {
      // merge segments
      if (prev_it->end_frame < it->end_frame) {
        // If the previous segment end before the current segment, then
        // extend the previous segment to the end_frame of the current
        // segment and remove the current segment.
        prev_it->end_frame = it->end_frame;
      }   // else current segment is entirely within the previous segment 
          // and can simple by removed
      it = segmentation->Erase(it);  
      // After erase, it points to the next segment.
    } else {
      // no merging of segments
      prev_it = it;
      ++it;
    }
  }

#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

void PadSegments(int32 label, int32 length, Segmentation *segmentation) {
  KALDI_ASSERT(segmentation);
  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ++it) {
    if (it->Label() != label) continue;

    it->start_frame -= length;
    it->end_frame += length;

    if (it->start_frame < 0) it->start_frame = 0;
  }
}

void ShrinkSegments(int32 label, int32 length, Segmentation *segmentation) {
  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ) {
    if (it->Label() == label) {
      if (it->Length() <= 2 * length) {
        it = segmentation->Erase(it);
      } else {
        it->start_frame += length;
        it->end_frame -= length;
        ++it;
      }
    } else {
      ++it;
    }
  }

#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

void BlendShortSegmentsWithNeighbors(int32 label, int32 max_length,
                                     int32 max_intersegment_length,
                                     Segmentation *segmentation, 
                                     bool sort) {
  if (sort) Sort(segmentation);
#ifdef KALDI_PARANOID
  else KALDI_ASSERT(IsSorted(*segmentation));
#endif

  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ) {
    if (it == segmentation->Begin()) {
      // Can't blend the first segment
      ++it;
      continue;
    }

    SegmentList::iterator next_it = it;
    ++next_it;

    if (next_it == segmentation->End())   // End of segmentation
      break;

    SegmentList::iterator prev_it = it;
    --prev_it;

    // If the previous and current segments have different labels,
    // then ensure that they are not overlapping
    KALDI_ASSERT(it->start_frame >= prev_it->start_frame &&
                 (prev_it->Label() == it->Label() ||
                  prev_it->end_frame < it->start_frame));

    KALDI_ASSERT(next_it->start_frame >= it->start_frame &&
                 (it->Label() == next_it->Label() ||
                  it->end_frame < next_it->start_frame));

    if (next_it->Label() != prev_it->Label() || it->Label() != label ||
        it->Length() >= max_length ||
        next_it->start_frame - it->end_frame - 1 > max_intersegment_length ||
        it->start_frame - prev_it->end_frame - 1 > max_intersegment_length) {
      ++it;
      continue;
    }

    prev_it->end_frame = next_it->end_frame;
    segmentation->Erase(it);
    it = segmentation->Erase(next_it);
  }
#ifdef KALDI_PARANOID
  segmentation->Check();
#endif
}

bool ConvertToAlignment(const Segmentation &segmentation,
                        int32 default_label, int32 length,
                        int32 tolerance,
                        std::vector<int32> *alignment) {
  KALDI_ASSERT(alignment);
  alignment->clear();

  if (length != -1) {
    KALDI_ASSERT(length >= 0);
    alignment->resize(length, default_label);
  }

  SegmentList::const_iterator it = segmentation.Begin();
  for (; it != segmentation.End(); ++it) {
    if (length != -1 && it->end_frame >= length + tolerance) {
      KALDI_WARN << "End frame (" << it->end_frame << ") "
                 << ">= length (" << length
                 << ") + tolerance (" << tolerance << ")."
                 << "Conversion failed.";
      return false;
    }

    int32 end_frame = it->end_frame;
    if (length == -1) {
      alignment->resize(it->end_frame + 1, default_label);
    } else {
      if (it->end_frame >= length)
        end_frame = length - 1;
    }

    KALDI_ASSERT(end_frame < alignment->size());
    for (int32 i = it->start_frame; i <= end_frame; i++) {
      (*alignment)[i] = it->Label();
    }
  }
  return true;
}

int32 InsertFromAlignment(const std::vector<int32> &alignment,
                          int32 start, int32 end,
                          int32 start_time_offset,
                          Segmentation *segmentation,
                          std::map<int32, int64> *frame_counts_per_class) {
  KALDI_ASSERT(segmentation);

  if (end <= start) return 0;   // nothing to insert

  // Correct boundaries
  if (end > alignment.size()) end = alignment.size();
  if (start < 0) start = 0;

  KALDI_ASSERT(end > start);    // This is possible if end was originally
                                // greater than alignment.size().
                                // The user must resize alignment appropriately
                                // before passing to this function.

  int32 num_segments = 0;
  int32 state = -100, start_frame = -1;
  for (int32 i = start; i < end; i++) {
    KALDI_ASSERT(alignment[i] >= -1);
    if (alignment[i] != state) {
      // Change of state i.e. a different class id.
      // So the previous segment has ended.
      if (start_frame != -1) {
        // start_frame == -1 in the beginning of the alignment. That is just
        // initialization step and hence no creation of segment.
        segmentation->EmplaceBack(start_frame + start_time_offset,
                                  i-1 + start_time_offset, state);
        num_segments++;

        if (frame_counts_per_class)
          (*frame_counts_per_class)[state] += i - start_frame;
      }
      start_frame = i;
      state = alignment[i];
    }
  }

  KALDI_ASSERT(state >= -1 && start_frame >= 0 && start_frame < end);
  segmentation->EmplaceBack(start_frame + start_time_offset,
                            end-1 + start_time_offset, state);
  num_segments++;
  if (frame_counts_per_class)
    (*frame_counts_per_class)[state] += end - start_frame;

#ifdef KALDI_PARANOID
  segmentation->Check();
#endif

  return num_segments;
}

int32 InsertFromSegmentation(
    const Segmentation &in_segmentation, int32 start_time_offset,
    bool sort,
    Segmentation *out_segmentation,
    std::vector<int64> *frame_counts_per_class) {
  KALDI_ASSERT(out_segmentation);

  if (in_segmentation.Dim() == 0) return 0;   // nothing to insert

  int32 num_segments = 0;

  for (SegmentList::const_iterator it = in_segmentation.Begin();
        it != in_segmentation.End(); ++it) {
    out_segmentation->EmplaceBack(it->start_frame + start_time_offset,
                                  it->end_frame + start_time_offset,
                                  it->Label());
    num_segments++;
    if (frame_counts_per_class) {
      if (frame_counts_per_class->size() <= it->Label()) {
        frame_counts_per_class->resize(it->Label() + 1, 0);
      }
      (*frame_counts_per_class)[it->Label()] += it->Length();
    }
  }

  if (sort) out_segmentation->Sort();

#ifdef KALDI_PARANOID
  out_segmentation->Check();
#endif

  return num_segments;
}

void ExtendSegmentation(const Segmentation &in_segmentation,
                        bool sort,
                        Segmentation *segmentation) {
  InsertFromSegmentation(in_segmentation, 0, sort, segmentation, NULL);
}

bool GetClassCountsPerFrame(
    const Segmentation &segmentation,
    int32 length, int32 tolerance,
    std::vector<std::map<int32, int32> > *class_counts_per_frame) {
  KALDI_ASSERT(class_counts_per_frame);

  if (length != -1) {
    KALDI_ASSERT(length >= 0);
    class_counts_per_frame->resize(length, std::map<int32, int32>());
  }

  SegmentList::const_iterator it = segmentation.Begin();
  for (; it != segmentation.End(); ++it) {
    if (length != -1 && it->end_frame >= length + tolerance) {
      KALDI_WARN << "End frame (" << it->end_frame << ") "
                 << ">= length + tolerance (" << length + tolerance << ")."
                 << "Conversion failed.";
      return false;
    }

    int32 end_frame = it->end_frame;
    if (length == -1) {
      class_counts_per_frame->resize(it->end_frame + 1,
                                     std::map<int32, int32>());
    } else {
      if (it->end_frame >= length)
        end_frame = length - 1;
    }

    KALDI_ASSERT(end_frame < class_counts_per_frame->size());
    for (int32 i = it->start_frame; i <= end_frame; i++) {
      std::map<int32, int32> &this_class_counts = (*class_counts_per_frame)[i];
      std::map<int32, int32>::iterator c_it = this_class_counts.lower_bound(
          it->Label());
      if (c_it == this_class_counts.end() || it->Label() < c_it->first) {
        this_class_counts.insert(c_it, std::make_pair(it->Label(), 1));
      } else {
        c_it->second++;
      }
    }
  }

  return true;
}

bool IsNonOverlapping(const Segmentation &segmentation) {
  std::vector<bool> vec;
  for (SegmentList::const_iterator it = segmentation.Begin();
        it != segmentation.End(); ++it) {
    vec.resize(it->end_frame + 1, false);
    for (int32 i = it->start_frame; i <= it->end_frame; i++) {
      if (vec[i]) return false;
      vec[i] = true;
    }
  }
  return true;
}

bool IsSorted(const Segmentation &segmentation) {
  SegmentList::const_iterator it = segmentation.Begin(),
    prev_it = segmentation.Begin();

  if (segmentation.Dim() <= 1) return true;
  ++it;

  while (it != segmentation.End()) {
    if (prev_it->start_frame > it->start_frame) 
      return false;
    ++it; ++prev_it;
  }

  return true;
}

void Sort(Segmentation *segmentation) {
  segmentation->Sort();
}

void TruncateToLength(int32 length, Segmentation *segmentation) {
  for (SegmentList::iterator it = segmentation->Begin();
        it != segmentation->End(); ) {
    if (it->start_frame >= length) {
      it = segmentation->Erase(it);
      continue;
    }

    if (it->end_frame >= length)
      it->end_frame = length - 1;
    ++it;
  }
}

} // end namespace segmenter
} // end namespace kaldi
