#include "segmenter/segmenter.h"
#include <algorithm>

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
}

void HistogramEncoder::Initialize(int32 num_bins, BaseFloat bin_w, BaseFloat min_s) {
  bin_sizes.clear();
  bin_sizes.resize(num_bins, 0);
  bin_width = bin_w;
  min_score = min_s;
}

void HistogramEncoder::Encode(BaseFloat x, int32 n) {
  int32 i = (x - min_score ) / bin_width;
  if (i < 0) i = 0;
  if (i >= NumBins()) i = NumBins() - 1;
  bin_sizes[i] += n;
}

void Segmentation::GenRandomSegmentation(int32 max_length, int32 num_classes) {
  Clear();
  int32 s = max_length;
  int32 e = max_length;

  while (s >= 0) {
    int32 chunk_size = rand() % (max_length / 10);
    s = e - chunk_size + 1;
    int32 k = rand() % num_classes;

    if (k != 0) {
      Segment seg(s,e,k);
      segments_.push_front(seg);
      dim_++;
    }
    e = s - 1;
  }
  Check();
}

/**
 * This function splits an input segmentation in_segmentation into pieces of
 * approximately segment_length. Each piece is given the same class id as the
 * original segment.
 * The way this function is written is that it first figures out the number of
 * pieces that the segment must be broken into. Then it creates that many pieces
 * of equal size (actual_segment_length).
**/
void Segmentation::SplitSegments(
    const Segmentation &in_segmentation,
    int32 segment_length) {
  Clear();
  for (SegmentList::const_iterator it = in_segmentation.Begin(); 
        it != in_segmentation.End(); ++it) {
    int32 length = it->Length();

    // Adding 0.5 here makes num_chunks like the ceil of the number that it
    // actually is. This results in all pieces to be smaller than
    // segment_length rather than being larger.
    int32 num_chunks = (static_cast<BaseFloat>(length)) / segment_length + 0.5;
    int32 actual_segment_length = static_cast<BaseFloat>(length) / num_chunks 
                                  + 0.5;
    
    int32 start_frame = it->start_frame;
    for (int32 j = 0; j < num_chunks; j++) {
      int32 end_frame = std::min(start_frame + actual_segment_length - 1, 
                                 it->end_frame);
      Emplace(start_frame, end_frame, it->class_id);
      start_frame = end_frame + 1;
    }
  }
  Check();
}

/**
 * This function splits the current segmentation into pieces of segment_length.
 * But if the last remaining piece is smaller than min_remainder, then the last
 * piece is merged to the piece before it, resulting in a piece that is of
 * length < segment_length + min_remainder.
 * The way this function works it is it looks at the current segment length and
 * checks if it is larger than segment_length + min_remainder. If it is larger,
 * then it must be split. To do this, it first modifies the start_frame of
 * the current frame to start_frame + segment_length - overlap.
 * It then creates a new segment of length segment_length from the original 
 * start_frame to start_frame + segment_length - 1 and adds it just before the
 * current segment. So in the next iteration, we would actually be back to the
 * same segment, but whose start_frame had just been modified.
**/
void Segmentation::SplitSegments(int32 segment_length,
                                 int32 min_remainder, int32 overlap,
                                 int32 label) {
  KALDI_ASSERT(overlap < segment_length);
  for (SegmentList::iterator it = segments_.begin(); 
      it != segments_.end(); ++it) {
    if (label != -1 && it->Label() != label) continue;
    
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
      it->start_frame = start_frame + segment_length - overlap;

      // Create a new segment and add it to the where the current iterator is.
      // The statement below results in this:
      // A <--> B1 <--> B <--> C
      // with the iterator it pointing at B1. So when the iterator is
      // incremented in the for loop, it will point to B again, but whose
      // start_frame had been modified.
      it = segments_.emplace(it, start_frame, 
                             start_frame + segment_length - 1, it->Label());

      // Forward list code
      // it->end_frame = start_frame + segment_length - 1;
      // it = segments_.emplace(it+1, it->end_frame + 1, end_frame, it->Label());
      dim_++;
    }
  }
  Check();
}

/**
 * This function is very straight forward. It just merges the labels in
 * merge_labels to the class_id dest_label. This means any segment that originally
 * had the class_id as any of the labels in merge_labels would end up having the
 * class_id dest_label.
**/
void Segmentation::MergeLabels(const std::vector<int32> &merge_labels,
                            int32 dest_label) {
  std::is_sorted(merge_labels.begin(), merge_labels.end());
  int32 size = 0;
  for (SegmentList::iterator it = segments_.begin(); 
       it != segments_.end(); ++it, size++) {
    if (std::binary_search(merge_labels.begin(), merge_labels.end(), it->Label())) {
      it->SetLabel(dest_label);
    }
  }
  KALDI_ASSERT(size == Dim());
  Check();
}

/**
 * This function is used to merge segments next to each other in the SegmentList
 * and within a distance of max_intersegment_length frames from each other,
 * provided the segments are of the same class_id.
**/
void Segmentation::MergeAdjacentSegments(int32 max_intersegment_length) {
  for (SegmentList::iterator it = segments_.begin(), prev_it = segments_.begin(); 
      it != segments_.end();) {

    if (it != segments_.begin() &&
        it->Label() == prev_it->Label() && 
        prev_it->end_frame + max_intersegment_length >= it->start_frame) {
      // merge segments
      if (prev_it->end_frame < it->end_frame) {
        // This is to avoid cases with overlapping segments where the previous 
        // segment ends after the current segment ends. In that case, the 
        // current segment must be simply removed.
        // Otherwise the previous segment must be modified to cover the current
        // segment.
        prev_it->end_frame = it->end_frame;
      }
      it = segments_.erase(it);
      dim_--;
    } else {
      // no merging of segments
      prev_it = it;
      ++it;
    }
  }
  Check();
}

/** 
 * Create a HistogramEncoder object based on this segmentation
**/
void Segmentation::CreateHistogram(
    int32 label, const Vector<BaseFloat> &scores, 
    const HistogramOptions &opts, HistogramEncoder *hist_encoder) {
  if (Dim() == 0)
    KALDI_ERR << "Segmentation must not be empty";

  int32 num_bins = opts.num_bins;
  BaseFloat min_score = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat max_score = -std::numeric_limits<BaseFloat>::infinity();

  mean_scores_.clear();
  mean_scores_.resize(Dim(), std::numeric_limits<BaseFloat>::quiet_NaN());
  
  std::vector<int32> num_frames(Dim(), 0);

  int32 i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); ++it, i++) {
    if (it->Label() != label) continue;
    SubVector<BaseFloat> this_segment_scores(scores, it->start_frame, it->end_frame - it->start_frame + 1);
    BaseFloat mean_score = this_segment_scores.Sum() / this_segment_scores.Dim();
    
    mean_scores_[i] = mean_score;
    num_frames[i] = this_segment_scores.Dim();

    if (mean_score > max_score) max_score = mean_score;
    if (mean_score < min_score) min_score = mean_score;
  }
  KALDI_ASSERT(i == mean_scores_.size());

  if (opts.select_above_mean) {
    min_score = scores.Sum() / scores.Dim();
  }

  BaseFloat bin_width = (max_score - min_score) / num_bins;
  hist_encoder->Initialize(num_bins, bin_width, min_score);

  hist_encoder->select_from_full_histogram = opts.select_from_full_histogram;

  i = 0;
  for (SegmentList::const_iterator it = segments_.begin(); it != segments_.end(); ++it, i++) {
    if (it->Label() != label) continue;
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));

    if (opts.select_above_mean && mean_scores_[i] < min_score) continue;
    KALDI_ASSERT(mean_scores_[i] >= min_score);

    hist_encoder->Encode(mean_scores_[i], num_frames[i]);
  }
  KALDI_ASSERT(i == mean_scores_.size());
  Check();
}

int32 Segmentation::SelectTopBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 dst_label, int32 reject_label,
    int32 num_frames_select, bool remove_rejected_frames) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  KALDI_ASSERT(dst_label >=0 && reject_label >= 0);
  KALDI_ASSERT(num_frames_select >= 0);

  BaseFloat min_score_for_selection = std::numeric_limits<BaseFloat>::infinity();
  int32 num_top_frames = 0, i = hist_encoder.NumBins() - 1;
  while (i >= (hist_encoder.select_from_full_histogram ? 0 : (hist_encoder.NumBins() / 2))) {
    num_top_frames += hist_encoder.BinSize(i);
    if (num_top_frames >= num_frames_select) {
      num_top_frames -= hist_encoder.BinSize(i);
      if (num_top_frames == 0) {
        num_top_frames += hist_encoder.BinSize(i);
        i--;
      }
      break;
    }
    i--;
  }
  min_score_for_selection = hist_encoder.min_score + (i+1) * hist_encoder.bin_width;

  i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); i++) {
    if (it->Label() != src_label) {
      ++it;
      continue;
    }
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));
    if (mean_scores_[i] >= min_score_for_selection) {
      it->SetLabel(dst_label);
      ++it;
    } else {
      if (remove_rejected_frames) {
        it = segments_.erase(it);
        dim_--;
      } else {
        it->SetLabel(reject_label);
        ++it;
      }
    }
  }
  KALDI_ASSERT(i == mean_scores_.size());

  if (remove_rejected_frames) mean_scores_.clear();

  Check();
  return num_top_frames;
}

int32 Segmentation::SelectBottomBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 dst_label, int32 reject_label, 
    int32 num_frames_select, bool remove_rejected_frames) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  KALDI_ASSERT(dst_label >=0 && reject_label >= 0);
  KALDI_ASSERT(num_frames_select >= 0);

  BaseFloat max_score_for_selection = -std::numeric_limits<BaseFloat>::infinity();
  int32 num_bottom_frames = 0, i = 0;
  while (i < (hist_encoder.select_from_full_histogram ? hist_encoder.NumBins() : (hist_encoder.NumBins() / 2))) {
    num_bottom_frames += hist_encoder.BinSize(i);
    if (num_bottom_frames >= num_frames_select) {
      num_bottom_frames -= hist_encoder.BinSize(i);
      if (num_bottom_frames == 0) {
        num_bottom_frames += hist_encoder.BinSize(i);
        i++;
      }
      break;
    }
    i++;
  }
  max_score_for_selection = hist_encoder.min_score + i * hist_encoder.bin_width;

  i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); i++) {
    if (it->Label() != src_label) {
      ++it; 
      continue;
    }
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));
    if (mean_scores_[i] < max_score_for_selection) {
      it->SetLabel(dst_label);
      ++it;
    } else {
      if (remove_rejected_frames) {
        it = segments_.erase(it);
        dim_--;
      } else {
        it->SetLabel(reject_label);
        ++it;
      }
    }
  }
  KALDI_ASSERT(i == mean_scores_.size());

  if (remove_rejected_frames) mean_scores_.clear();

  Check();
  return num_bottom_frames;
}

std::pair<int32,int32> Segmentation::SelectTopAndBottomBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 top_label, int32 num_frames_top,
    int32 bottom_label, int32 num_frames_bottom,
    int32 reject_label, bool remove_rejected_frames) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  KALDI_ASSERT(top_label >= 0 && bottom_label >= 0 && reject_label >= 0);
  KALDI_ASSERT(num_frames_top >= 0 && num_frames_bottom >= 0);
  
  BaseFloat min_score_for_selection = std::numeric_limits<BaseFloat>::infinity();
  int32 num_selected_top = 0, i = hist_encoder.NumBins() - 1;
  while (i >= hist_encoder.NumBins() / 2) {
    int32 this_selected = hist_encoder.BinSize(i);
    num_selected_top += this_selected;
    if (num_selected_top >= num_frames_top) {
      num_selected_top -= this_selected;
      if (num_selected_top == 0) {
        num_selected_top += this_selected;
        i--;
      }
      break;
    }
    i--;
  }
  min_score_for_selection = hist_encoder.min_score + (i+1) * hist_encoder.bin_width;
  
  BaseFloat max_score_for_selection = -std::numeric_limits<BaseFloat>::infinity();
  int32 num_selected_bottom= 0;
  i = 0;
  while (i < hist_encoder.NumBins() / 2) {
    int32 this_selected = hist_encoder.BinSize(i);
    num_selected_bottom += this_selected;
    if (num_selected_bottom >= num_frames_bottom) {
      num_selected_bottom -= this_selected;
      if (num_selected_bottom == 0) {
        num_selected_bottom += this_selected;
        i++;
      }
      break;
    }
    i++;
  }
  max_score_for_selection = hist_encoder.min_score + i * hist_encoder.bin_width;

  i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); i++) {
    if (it->Label() != src_label) {
      ++it; 
      continue;
    }
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));
    if (mean_scores_[i] >= min_score_for_selection) {
      it->SetLabel(top_label);
      ++it;
    } else if (mean_scores_[i] < max_score_for_selection) {
      it->SetLabel(bottom_label);
      ++it;
    } else {
      if (remove_rejected_frames) {
        it = segments_.erase(it);
        dim_--;
      } else {
        it->SetLabel(reject_label);
        ++it;
      }
    }
  }
  KALDI_ASSERT(i == mean_scores_.size());
  
  if (remove_rejected_frames) mean_scores_.clear();

  Check();
  return std::make_pair(num_selected_top, num_selected_bottom);
}

/**
 * This function intersects the segmentation with the filter segmentation
 * and includes only sub-segments where the filter segmentation has the label
 * filter_label. 
 * If filter_label is -1, the filter_label would dynamically change to be the
 * label of the primary segmentation.
 * For e.g. if the segmentation is 
 * start_frame end_frame label
 * 5 10 1
 * 8 12 2
 * and filter_segmentation is 
 * 0 7 1
 * 7 10 2
 * 10 13 1.
 * And filter_label is 1. Then after intersection, this 
 * object would hold 
 * 5 7 1
 * 8 10 2
 * 10 12 2
**/ 
void Segmentation::IntersectSegments(
    const Segmentation &secondary_segmentation, 
    Segmentation *out_seg, int32 mismatch_label) const {
  KALDI_ASSERT(secondary_segmentation.Dim() > 0);
  KALDI_ASSERT(out_seg);
  out_seg->Clear();
  SegmentList::const_iterator s_it = secondary_segmentation.Begin();
  for (SegmentList::const_iterator p_it = Begin(); p_it != End(); ++p_it) {
    if (s_it == secondary_segmentation.End()) 
      --s_it;
    // This statement was necessary so that it would not crash at the next
    // statement.
    KALDI_ASSERT(s_it != secondary_segmentation.End());
    
    // If the secondary segment start is beyond the start of the current
    // segment, then move the secondary segment iterator back.
    while (s_it != secondary_segmentation.Begin() &&
           s_it->start_frame > p_it->start_frame) 
      --s_it;
    KALDI_ASSERT(s_it != secondary_segmentation.End());
    
    // Now, we can move the secondary segment iterator until the end of the
    // secondary segment is just at the current segment.
    // There are two possibilities here:
    // (i)  The secondary segment ends are on either side of the primary's
    //      start_frame. 
    //      i.e. s_it->start_frame < p_it->start_frame <= s_it->end_frame 
    // (ii) The secondary segment ends are after the primary's start_frame.
    //      i.e. p_it->start_frame <= s_it->start_frame <= s_it->end_frame
    while (s_it != secondary_segmentation.End() &&
            s_it->end_frame < p_it->start_frame) ++s_it;

    // Actual intersection is done here.
    int32 start_frame = p_it->start_frame;
    for (; s_it != secondary_segmentation.End() &&
            s_it->start_frame <= p_it->end_frame; ++s_it) {
      int32 new_label = p_it->Label();
      if (s_it->Label() != p_it->Label()) {
        new_label = mismatch_label;
      }

      if (start_frame < s_it->start_frame) {
        // This is the first part of handling case (ii)
        if (mismatch_label != -1)
          out_seg->Emplace(start_frame, s_it->start_frame - 1, 
                           mismatch_label);
        start_frame = s_it->start_frame;
      } 
      
      KALDI_ASSERT(start_frame == std::max(p_it->start_frame, s_it->start_frame));
      // Once this is done, it reduces to case (i)

      if (s_it->end_frame < p_it->end_frame) {
        // This is the case in which there is a part of primary segment that is
        // not intersected by the secondary segment. We split the primary
        // segment into two parts, the first of which is created using the
        // emplace statement below and the second is created by modifying the
        // current segment to the contain only the part after secondary
        // segment's end_frame.
        if (start_frame <= s_it->end_frame) {
          if (new_label != -1)
            out_seg->Emplace(start_frame, s_it->end_frame, new_label);
          start_frame = s_it->end_frame + 1;
        }
        KALDI_ASSERT(start_frame <= p_it->end_frame);
      } else { // if (s_it->end_frame > p_it->end_frame)
        if (new_label != -1)
          out_seg->Emplace(start_frame, p_it->end_frame, new_label);
        start_frame = p_it->end_frame + 1;
      }
    }
    
    if (s_it == secondary_segmentation.End()) {
      --s_it;
      if (start_frame < p_it->end_frame) {
        if (mismatch_label != -1)
          out_seg->Emplace(start_frame, p_it->end_frame, mismatch_label);
        start_frame = p_it->end_frame + 1;
      }
      ++s_it;
    }
  }
}

/*
void Segmentation::IntersectSegments(
    const Segmentation &filter_segmentation,
    int32 filter_label) {
  SegmentList::iterator it = segments_.begin();
  SegmentList::const_iterator filter_it = filter_segmentation.Begin();
  
  int32 orig_filter_label = filter_label;

  while (it != segments_.end()) {
    if (orig_filter_label == -1) filter_label = it->Label();
    
    // Move the filter iterator up to the first segment where the end point of
    // the filter segment is just after the start of the current segment.
    while (filter_it != filter_segmentation.End() && 
           (filter_it->end_frame < it->start_frame || 
           filter_it->Label() != filter_label)) {
      ++filter_it;
    }
    
    // If the filter has reached the end, then we are done
    if (filter_it == filter_segmentation.End()) {
      while (it != segments_.end()) {
        // There is no segment in the filter_segmentation beyond this. So the
        // intersection is empty. Hence erase the remaining segments.
        it = segments_.erase(it);
        dim_--;
      }
      break;
    }

    // If the segment in the filter is beyond the end of the current segment,
    // then there is no intersection between this segment and the
    // filter_segmentation. Hence remove this segment.
    if (filter_it->start_frame > it->end_frame) {
      it = segments_.erase(it);
      dim_--;
      continue;
    }

    // Filter start_frame is after the start_frame of this segment. 
    // So throw away the initial part of this segment as it is not in the
    // filter. i.e. Set the start of this segment to be the start of the filter
    // segment.
    if (filter_it->start_frame > it->start_frame) 
      it->start_frame = filter_it->start_frame;
      
    if (filter_it->end_frame < it->end_frame) {
      // filter segment ends before the end of the current segment. Then end 
      // the current segment right at the end of the filter and leave the 
      // iterator at the remaining part.
      int32 start_frame = it->start_frame;
      it->start_frame = filter_it->end_frame + 1;
      segments_.emplace(it, start_frame, filter_it->end_frame, it->Label());
      dim_++;
    } else {
      // filter segment ends after the end of this current segment. So 
      // we don't need to create any new segment. Just advance the iterator 
      // to the next segment.
      ++it;
    }
  }
  Check();
}
*/

/**
 * A very straight forward function to extend this segmentation by adding
 * segments from another segmentation seg. If sort is called, the segments would
 * be sorted after extension. This can be skipped if its known that the segments
 * would be sorted.
**/
void Segmentation::Extend(const Segmentation &seg, bool sort) {
  for (SegmentList::const_iterator it = seg.Begin(); it != seg.End(); ++it) {
    segments_.push_back(*it);
    dim_++;
  }
  if (sort) Sort();
}

/**
 * This function is a little complicated in what it does. But this is required
 * for one of the applications.  This function creates a new segmentation by
 * sub-segmenting an overlapping primary segmentation and assign new class_id to
 * the regions where the primary segmentation intersects the non-overlapping
 * secondary segmentation segments with class_id secondary_label.  
 * This is similar to the function "IntersectSegments", but instead of keeping
 * only the filtered subsegments, all the subsegments are kept, while only
 * changing the class_id of the filtered sub-segments. 
 * For the sub-segments, where the secondary segment class_id is
 * secondary_label, the created sub-segment is labeled "subsegment_label",
 * provided it is non-negative. Otherwise, the created sub-segment is labeled
 * the class_id of the secondary segment.
 * For the other sub-segments, where the secondary segment
 * class_id is not secondary_label, the created sub-segment retains the class_id
 * of the parent segment. 
 * Additionally this program adds the secondary segmentation's vector_value
 * along with this segmentation's string_value if they exist.
**/

void Segmentation::SubSegmentUsingNonOverlappingSegments(
    const Segmentation &secondary_segmentation, int32 secondary_label,
    int32 subsegment_label, Segmentation *out_seg) const {
  KALDI_ASSERT(secondary_segmentation.Dim() > 0);
  KALDI_ASSERT(secondary_segmentation.IsNonOverlapping());
  SegmentList::const_iterator s_it = secondary_segmentation.Begin();
  for (SegmentList::const_iterator p_it = Begin(); p_it != End(); ++p_it) {
    if (s_it == secondary_segmentation.End()) 
      --s_it;
    // This statement was necessary so that it would not crash at the next
    // statement.
    KALDI_ASSERT(s_it != secondary_segmentation.End());
    
    // If the secondary segment start is beyond the start of the current
    // segment, then move the secondary segment iterator back.
    while (s_it != secondary_segmentation.Begin() &&
           s_it->start_frame > p_it->start_frame) 
      --s_it;
    KALDI_ASSERT(s_it != secondary_segmentation.End());
    
    // Now, we can move the secondary segment iterator until the end of the
    // secondary segment is just at the current segment.
    // There are two possibilities here:
    // (i)  The secondary segment ends are on either side of the primary's
    //      start_frame. 
    //      i.e. s_it->start_frame < p_it->start_frame <= s_it->end_frame 
    // (ii) The secondary segment ends are after the primary's start_frame.
    //      i.e. p_it->start_frame <= s_it->start_frame <= s_it->end_frame
    while (s_it != secondary_segmentation.End() &&
            s_it->end_frame < p_it->start_frame) ++s_it;

    // Actual intersection is done here.
    int32 start_frame = p_it->start_frame;
    for (; s_it != secondary_segmentation.End() &&
            s_it->start_frame <= p_it->end_frame; ++s_it) {
      int32 new_label = p_it->Label();

      if (s_it->Label() == secondary_label) {
        new_label = (subsegment_label >= 0 ? subsegment_label : s_it->Label());
      }

      if (start_frame < s_it->start_frame) {
        // This is the first part of handling case (ii)
        out_seg->Emplace(start_frame, s_it->start_frame - 1, 
                         p_it->Label());
        start_frame = s_it->start_frame;
      } // Once this is done, it reduces to case (i)
      KALDI_ASSERT(start_frame == std::max(p_it->start_frame, s_it->start_frame));

      if (s_it->end_frame < p_it->end_frame) {
        // This is the case in which there is a part of primary segment that is
        // not intersected by the secondary segment. We split the primary
        // segment into two parts, the first of which is created using the
        // emplace statement below and the second is created by modifying the
        // current segment to the contain only the part after secondary
        // segment's end_frame.
        out_seg->Emplace(start_frame, s_it->end_frame, new_label);
        start_frame = s_it->end_frame + 1;
        KALDI_ASSERT(start_frame <= p_it->end_frame);
      } else { // if (s_it->end_frame > p_it->end_frame)
        out_seg->Emplace(start_frame, p_it->end_frame, new_label);
      }
    }
  }
}

void Segmentation::SubSegmentUsingSmallOverlapSegments(
    const Segmentation &secondary_segmentation, int32 secondary_label,
    int32 subsegment_label, Segmentation *out_seg) const {
  // TODO: When the secondary segmentation has overlap, it just considers the 
  // label of the latest segment.
  KALDI_ASSERT(secondary_segmentation.Dim() > 0);
  KALDI_ASSERT(secondary_segmentation.HasSmallOverlap());
  SegmentList::const_iterator s_it = secondary_segmentation.Begin();
  for (SegmentList::const_iterator p_it = Begin(); p_it != End(); ++p_it) {
    if (s_it == secondary_segmentation.End()) 
      --s_it;
    // This statement was necessary so that it would not crash at the next
    // statement.
    KALDI_ASSERT(s_it != secondary_segmentation.End());
    
    // If the secondary segment start is beyond the start of the current
    // segment, then move the secondary segment iterator back.
    while (s_it != secondary_segmentation.Begin() &&
           s_it->start_frame > p_it->start_frame) 
      --s_it;
    KALDI_ASSERT(s_it != secondary_segmentation.End());
    
    // Now, we can move the secondary segment iterator until the end of the
    // secondary segment is just at the current segment.
    // There are two possibilities here:
    // (i)  The secondary segment ends are on either side of the primary's
    //      start_frame. 
    //      i.e. s_it->start_frame < p_it->start_frame <= s_it->end_frame 
    // (ii) The secondary segment ends are after the primary's start_frame.
    //      i.e. p_it->start_frame <= s_it->start_frame <= s_it->end_frame
    while (s_it != secondary_segmentation.End() &&
            s_it->end_frame < p_it->start_frame) ++s_it;

    // Actual intersection is done here.
    int32 start_frame = p_it->start_frame;
    for (; s_it != secondary_segmentation.End() &&
            s_it->start_frame <= p_it->end_frame; ++s_it) {
      SegmentList::const_iterator s_it_next(s_it);
      ++s_it_next;

      int32 end_frame = s_it->end_frame;
      if (s_it_next != secondary_segmentation.End() && 
          s_it_next->start_frame <= s_it->end_frame) {
        end_frame = s_it_next->start_frame - 1;
      }

      int32 new_label = p_it->Label();

      if (s_it->Label() == secondary_label) {
        new_label = (subsegment_label >= 0 ? subsegment_label : s_it->Label());
      }

      if (start_frame < s_it->start_frame) {
        // This is the first part of handling case (ii)
        out_seg->Emplace(start_frame, s_it->start_frame - 1, 
                         p_it->Label());
        start_frame = s_it->start_frame;
      } // Once this is done, it reduces to case (i)
      KALDI_ASSERT(start_frame == std::max(p_it->start_frame, s_it->start_frame));

      if (end_frame < p_it->end_frame) {
        // This is the case in which there is a part of primary segment that is
        // not intersected by the secondary segment. We split the primary
        // segment into two parts, the first of which is created using the
        // emplace statement below and the second is created by modifying the
        // current segment to the contain only the part after secondary
        // segment's end_frame.
        out_seg->Emplace(start_frame, end_frame, new_label);
        start_frame = end_frame + 1;
        KALDI_ASSERT(start_frame <= p_it->end_frame);
      } else { // if (end_frame > p_it->end_frame)
        out_seg->Emplace(start_frame, p_it->end_frame, new_label);
      }
    }
  }
}


//    const Segmentation &nonoverlapping_segmentation,
//    int32 secondary_label, int32 subsegment_label,
//    Segmentation *out_segmentation) const {
//  out_segmentation->Clear();
//  SegmentList::const_iterator s_it = nonoverlapping_segmentation.Begin();
//  for (SegmentList::const_iterator p_it = Begin(); p_it != End(); ++p_it) {
//    if (s_it == nonoverlapping_segmentation.End()) --s_it;
//    // This statement was necessary so that it would not crash at the next
//    // statement.
//
//
//    // The following two statements may be a little inefficient and there might
//    // be better way to do this. This is a TODO.
//    
//    // If the secondary segment start is beyond the start of the current
//    // segment, then move the secondary segment iterator back.
//    while (s_it->start_frame > p_it->start_frame) --s_it;
//    // Now, we can move the secondary segment iterator until the end of the
//    // secondary segment is just before the current segment.
//    while (s_it != nonoverlapping_segmentation.End() &&
//            s_it->end_frame < p_it->start_frame) ++s_it;
//    // This is so that state is equalized and we can be sure that always, the
//    // secondary segment is just one segment before the current segment.
//
//    // Actual intersection is done here.
//    for (; s_it->start_frame <= p_it->end_frame && 
//          s_it != nonoverlapping_segmentation.End(); ++s_it) {
//      int32 new_label = p_it->Label();
//      
//      if (s_it->Label() == secondary_label) {
//        new_label = (subsegment_label >= 0 ? 
//                         subsegment_label : s_it->Label());
//      }
//
//      out_segmentation->Emplace(
//          std::max(s_it->start_frame, p_it->start_frame),
//          std::min(s_it->end_frame, p_it->end_frame), new_label,
//          s_it->VectorValue(), p_it->StringValue());
//    }
//  }
//}

/**
void Segmentation::CreateSubSegmentsOld(
    const Segmentation &filter_segmentation,
    int32 filter_label,
    int32 subsegment_label) {
  SegmentList::iterator it = segments_.begin();
  SegmentList::const_iterator filter_it = filter_segmentation.Begin();

  while (it != segments_.end()) {
    
    // If the start of the segment in the filter is before the current
    // segment then move the filter iterator up to the first segment where the
    // end point of the filter segment is just after the start of the current
    // segment
    while (filter_it != filter_segmentation.End() && 
           (filter_it->end_frame < it->start_frame || 
           filter_it->Label() != filter_label)) {
      ++filter_it;
    }
    
    // If the filter has reached the end, then we are done
    if (filter_it == filter_segmentation.End()) {
      break;
    }

    // If the segment in the filter is beyond the end of the current segment,
    // then increment the iterator until the current segment end 
    // point is just after the start of the filter segment
    if (filter_it->start_frame > it->end_frame) {
      ++it; 
      continue;
    }

    // filter start_frame is after the start_frame of this segment. 
    // So split the segment into two parts at filter_start. 
    // Create a new segment for the
    // first part which retains the same label as before.
    // For now, retain the same label for the second part. The 
    // label would change while processing the end of the subsegment.
    if (filter_it->start_frame > it->start_frame) {
      segments_.emplace(it, it->start_frame, filter_it->start_frame - 1, it->Label());
      dim_++;
      it->start_frame = filter_it->start_frame;
    }
      
    if (filter_it->end_frame < it->end_frame) {
      // filter segment ends before the end of the current segment. Then end 
      // the current segment right at the end of the filter and leave the 
      // remaining part for the next segment
      int32 start_frame = it->start_frame;
      it->start_frame = filter_it->end_frame + 1;
      segments_.emplace(it, start_frame, filter_it->end_frame, subsegment_label);
      dim_++;
    } else {
      // filter segment ends after the end of this current segment. 
      // So change the label of this frame to
      // subsegment_label
      it->SetLabel(subsegment_label);
      ++it;
    }
  }
  Check();
}
**/

/**
 * This function is used to widen segments of class_id "label" by "length" frames
 * on either side of the segment. This is useful to widen segments of speech.
 * While widening, it also reduces the length of the segment adjacent to it.
 * This may not be required in some applications, but it is ok for speech /
 * silence. We are calling frames within a "length" number of frames near the
 * speech segment as speech and hence we reduce the width of the silence segment
 * before it.
**/
void Segmentation::WidenSegments(int32 label, int32 length) {
  for (SegmentList::iterator it = segments_.begin();
        it != segments_.end(); ++it) {
    if (it->Label() == label) {
      if (it != segments_.begin()) {
        // it is not the beginning of the segmentation, so we can widen it on
        // the start_frame side
        SegmentList::iterator prev_it = it;
        --prev_it;
        it->start_frame -= length;
        if (prev_it->Label() == label && it->start_frame < prev_it->end_frame) {
          // After widening this segment, it overlaps the previous segment that
          // also has the same class_id. Then turn this segment into a composite
          // one 
          it->start_frame = prev_it->start_frame;
          // and remove the previous segment from the list.
          Erase(prev_it);
        } else if (prev_it->Label() != label && 
            it->start_frame < prev_it->end_frame) {
          // Previous segment is not the same class_id, so we cannot turn this into
          // a composite segment.
          if (it->start_frame <= prev_it->start_frame) {
            // The extended segment absorbs the previous segment into it
            // So remove the previous segment
            Erase(prev_it);
          } else {
            // The extended segment reduces the length of the previous
            // segment. But does not completely overlap it.
            prev_it->end_frame -= length;
          }
        }
      }
      SegmentList::iterator next_it = it;
      ++next_it;

      if (next_it != segments_.end())
        it->end_frame += length;          // Line (1)
    } else { // if (it->Label() != label)
      if (it != segments_.begin()) {
        SegmentList::iterator prev_it = it;
        --prev_it;
        if (prev_it->end_frame >= it->end_frame) {
          // The extended previous segment in Line (1) completely
          // overlaps the current segment. So remove the current segment. 
          it = Erase(it);
          --it;   // So that we can increment in the for loop
        } else if (prev_it->end_frame >= it->start_frame) {
          // The extended previous segment in Line (1) reduces the length of
          // this segment.
          it->start_frame = prev_it->end_frame + 1;
        }
      } 
    }
  }
}

void Segmentation::ShrinkSegments(int32 label, int32 length) {
  for (SegmentList::iterator it = segments_.begin();
        it != segments_.end();) {
    if (it->Label() == label) {
      if (it->Length() <= 2 * length) {
        it = segments_.erase(it);
        dim_--;
      } else {
        it->start_frame += length;
        it->end_frame -= length;
        ++it;
      }
    } else 
      ++it;
  }
}

/**
 * This function relabels segments of class_id "label" that are shorter than
 * max_length frames, provided the segments before and after it are of the same
 * class_id "other_label". Now all three segments have the same class_id
 * "other_label" and hence can be merged into a composite segment.  
 * An example where this is useful is when there is a short segment of silence
 * with speech segments on either sides. Then the short segment of silence is
 * removed and called speech instead. The three continguous segments of speech
 * are merged into a single composite segment.
**/
void Segmentation::RelabelShortSegments(int32 label, int32 max_length) {
  for (SegmentList::iterator it = segments_.begin();
        it != segments_.end();) {
    if (it == segments_.begin()) {
      ++it;
      continue;
    }
    
    SegmentList::iterator next_it = it;
    ++next_it;
    if (next_it == segments_.end()) break;
    
    SegmentList::iterator prev_it = it;
    --prev_it;

    if (next_it->Label() == prev_it->Label() && it->Label() == label 
        && it->Length() < max_length) {
      prev_it->end_frame = next_it->end_frame;
      segments_.erase(it);
      it = segments_.erase(next_it);
      dim_ -= 2;
    } else 
      ++it;
  }
}

/**
 * This is very straight forward. It removes all segments of class_id "label"
**/
void Segmentation::RemoveSegments(int32 label) {
  for (SegmentList::iterator it = segments_.begin();
        it != segments_.end();) {
    if (it->Label() == label) {
      it = segments_.erase(it);
      dim_--;
    } else {
      ++it;
    }
  }
  Check();
}

/**
 * This is very straight forward. It removes any segment whose class_id is
 * contained in the vector "labels"
**/
void Segmentation::RemoveSegments(const std::vector<int32> &labels) {
  KALDI_ASSERT(std::is_sorted(labels.begin(), labels.end()));
  for (SegmentList::iterator it = segments_.begin();
        it != segments_.end();) {
    if (std::binary_search(labels.begin(), labels.end(), it->Label())) {
      it = segments_.erase(it);
      dim_--;
    } else {
      ++it;
    }
  }
  Check();
}

void Segmentation::Clear() {
  segments_.clear();
  dim_ = 0;
  mean_scores_.clear();
}

void Segmentation::Read(std::istream &is, bool binary) {
  Clear();
  
  if (binary) {
    int32 sz = is.peek();
    if (sz == Segment::SizeOf()) {
      is.get();
    } else {
      KALDI_ERR << "Segmentation::Read: expected to see Segment of size "
                << Segment::SizeOf() << ", saw instead " << sz
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
  SegmentList::const_iterator it = segments_.begin();
  if (binary) {
    char sz = Segment::SizeOf();
    os.write(&sz, 1);

    int32 segmentssz = static_cast<int32>(Dim());
    KALDI_ASSERT((size_t)segmentssz == Dim());
  
    os.write(reinterpret_cast<const char *>(&segmentssz), sizeof(segmentssz));

    for (; it != segments_.end(); ++it) {
      it->Write(os, binary);
    } 
  } else {
    os << "[ ";
    for (; it != segments_.end(); ++it) {
      it->Write(os, binary);
      os << std::endl;
    } 
    os << "]" << std::endl;
  }
}

/**
 * This function is used to write the segmentation in RTTM format. Each class is
 * treated as a "SPEAKER". If map_to_speech_and_sil is true, then the class_id 0
 * is treated as SILENCE and every other class_id as SPEECH. The argument
 * start_time is used to set what the time corresponding to the 0 frame in the
 * segment.  Each segment is converted into the following line,
 * SPEAKER <file-id> 1 <start-time> <duration> <NA> <NA> <speaker> <NA>
 * ,where
 * <file-id> is the file_id supplied as an argument
 * <start-time> is the start time of the segment in seconds
 * <duration> is the length of the segment in seconds
 * <speaker> is the class_id stored in the segment. If map_to_speech_and_sil is
 * set true then <speaker> is either SPEECH or SILENCE.
 * The function retunns the largest class_id that it encounters.
**/
int32 Segmentation::WriteRttm(std::ostream &os, const std::string &file_id, const std::string &channel, 
                              BaseFloat frame_shift, BaseFloat start_time, 
                              bool map_to_speech_and_sil) const {
  SegmentList::const_iterator it = segments_.begin();
  int32 largest_class = 0;
  for (; it != segments_.end(); ++it) {
    os << "SPEAKER " << file_id << " " << channel << " "
       << it->start_frame * frame_shift + start_time << " " 
       << (it->Length()) * frame_shift << " <NA> <NA> ";
    if (map_to_speech_and_sil) {
      switch (it->Label()) {
        case 1:
          os << "SPEECH ";
          break;
        default:
          os << "SILENCE ";
          break;
      }
      largest_class = 1;
    } else {
      if (it->Label() >= 0) {
        os << it->Label() << " ";
        if (it->Label() > largest_class)
          largest_class = it->Label();
      }
    }
    os << "<NA>" << std::endl;
  } 
  return largest_class;
}

/**
 * This function is used to convert the segmentation into frame-level alignment
 * with the label for each frame begin the class_id of segment the frame belongs
 * to.
 * The arguments are used to provided extended functionality that are required
 * for most cases.
 * default_label : the label that is used as filler in regions where the frame
 *                 is not in any of the segments. In most applications, certain
 *                 segments are removed, such as the ones that are silence. Then
 *                 the segments would not span the entire duration of the file.
 *                 e.g. 
 *                 10 35 1
 *                 41 190 2
 *                 ...
 *                 Here there is no segment from 36-40. These frames are
 *                 filled with default_label.
 * length        : the number of frames required in the alignment. In most
 *                 applications, the length of the alignment required is known.
 *                 Usually it must match the length of the features (obtained
 *                 using feat-to-len). Then the alignment is resized to this
 *                 length and filled with default_label. The segments are then
 *                 read and the frames corresponding to the segments are
 *                 relabeled with the class_id of the respective segments.
 * tolerance     : the tolerance in number of frames that we allow for the
 *                 frame index corresponding to the end_frame of the last
 *                 segment. Since, we use 25 ms widows with 10 ms frame shift, 
 *                 it is possible that the features length is 2 frames less than
 *                 the end of the last segment. So the user can set the
 *                 tolerance to 2 in order to avoid returning with error in this
 *                 function.
**/
bool Segmentation::ConvertToAlignment(std::vector<int32> *alignment,
                                      int32 default_label, int32 length, 
                                      int32 tolerance) const {
  KALDI_ASSERT(alignment != NULL);
  alignment->clear();

  if (length != -1) {
    KALDI_ASSERT(length >= 0);
    alignment->resize(length, default_label);
  }
  
  SegmentList::const_iterator it = segments_.begin();
  for (; it != segments_.end(); ++it) {
    if (length != -1 && it->end_frame >= length + tolerance) {
      KALDI_WARN << "End frame (" << it->end_frame << ") "
                 << ">= length + tolerance (" << length + tolerance << ")."
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
    for (size_t i = it->start_frame; i <= end_frame; i++) {
      (*alignment)[i] = it->Label();
    }
  } 
  return true;
}

int32 Segmentation::InsertFromAlignment(
    const std::vector<int32> &alignment,
    int32 start_time_offset,
    std::vector<int64> *frame_counts_per_class) {
  if (alignment.size() == 0) return 0;

  int32 num_segments = 0;
  int32 state = -1, start_frame = -1; 
  for (int32 i = 0; i < alignment.size(); i++) {
    if (alignment[i] != state) {  
      // Change of state i.e. a different class id. 
      // So the previous segment has ended.
      if (state != -1) {
        // state == -1 in the beginning of the alignment. That is just
        // initialization step and hence no creation of segment.
        Emplace(start_frame + start_time_offset, 
                i-1 + start_time_offset, state);
        num_segments++;

        if (frame_counts_per_class) {
          if (frame_counts_per_class->size() <= state) {
            frame_counts_per_class->resize(state + 1, 0);
          }
          (*frame_counts_per_class)[state] += i - start_frame;
        }
      }
      start_frame = i;
      state = alignment[i];
    }
  }

  KALDI_ASSERT(state >= 0 && start_frame < alignment.size());
  Emplace(start_frame + start_time_offset, 
          alignment.size()-1 + start_time_offset, state);
  num_segments++;
  if (frame_counts_per_class) {
    if (frame_counts_per_class->size() <= state) {
      frame_counts_per_class->resize(state + 1, 0);
    }
    (*frame_counts_per_class)[state] += alignment.size() - start_frame;
  }

  return num_segments;
}

int32 Segmentation::InsertFromSegmentation(
    const Segmentation &seg,
    int32 start_time_offset,
    std::vector<int64> *frame_counts_per_class) {
  if (seg.Dim() == 0) return 0;

  int32 num_segments = 0;

  for (SegmentList::const_iterator it = seg.Begin(); it != seg.End(); ++it) {
    Emplace(it->start_frame + start_time_offset,
            it->end_frame + start_time_offset, it->Label());
    num_segments++;
    if (frame_counts_per_class) {
      if (frame_counts_per_class->size() <= it->Label()) {
        frame_counts_per_class->resize(it->Label() + 1, 0);
      }
      (*frame_counts_per_class)[it->Label()] += it->Length();
    }
  }

  return num_segments;
}

void Segmentation::Check() const {
  int32 dim = 0;
  for (SegmentList::const_iterator it = segments_.begin();
        it != segments_.end(); ++it, dim++) {
    KALDI_ASSERT(it->Label() >= 0);
  };
  KALDI_ASSERT(dim == dim_);
  KALDI_ASSERT(mean_scores_.size() == 0 || mean_scores_.size() == dim_);
}

bool Segmentation::IsNonOverlapping() const {
  int32 end_frame = Begin()->end_frame;
  int32 start_frame = Begin()->start_frame;
  for (SegmentList::const_iterator it = Begin(); it != End(); ++it) {
    if (it == Begin()) continue;
    if (it->start_frame <= end_frame || it->start_frame < start_frame) 
      return false;
    end_frame = it->end_frame;
    start_frame = it->start_frame;
  }
  return true;
}

bool Segmentation::HasSmallOverlap() const {
  int32 end_frame = Begin()->end_frame;
  int32 start_frame = Begin()->start_frame;
  for (SegmentList::const_iterator it = Begin(); it != End(); ++it) {
    if (it == Begin()) continue;
    if (it->end_frame < end_frame || it->start_frame < start_frame) 
      return false;
    SegmentList::const_iterator next_it = it;
    ++next_it;
    if (next_it != End() && next_it->start_frame <= end_frame)
      return false;
    end_frame = it->end_frame;
    start_frame = it->start_frame;
  }
  return true;
}

SegmentationPostProcessor::SegmentationPostProcessor(
    const SegmentationPostProcessingOptions &opts) : opts_(opts) {
  if (!opts_.filter_in_fn.empty()) {
    if (ClassifyRspecifier(opts_.filter_in_fn, NULL, NULL) ==
        kNoRspecifier) {
      bool binary_read;
      Input ki(opts_.filter_in_fn, &binary_read);
      filter_segmentation_.Read(ki.Stream(), binary_read);
    } else {
      filter_reader_.Open(opts_.filter_in_fn);
    }
  }
  
  if (!opts_.remove_labels_csl.empty()) {
    if (!SplitStringToIntegers(opts_.remove_labels_csl, ":",
          false, &remove_labels_)) {
      KALDI_ERR << "Bad value for --remove-labels option: "
                << opts_.remove_labels_csl;
    }
    std::sort(remove_labels_.begin(), remove_labels_.end());
  }
    
  if (!opts_.merge_labels_csl.empty()) {
    if (!SplitStringToIntegers(opts_.merge_labels_csl, ":", 
          false, &merge_labels_)) {
      KALDI_ERR << "Bad value for --merge-labels option: "
                << opts_.merge_labels_csl;
    }
    std::sort(merge_labels_.begin(), merge_labels_.end());
  }

  Check();
}

void SegmentationPostProcessor::Check() const {
  if ( IsFilteringToBeDone() && opts_.post_process_label < 0) {
    KALDI_ERR << "Invalid value " << opts_.post_process_label << " for option "
              << "--post-process-label. It must be non-negative.";
  }

  if (IsWideningSegmentsToBeDone() && opts_.widen_label < 0) {
    KALDI_ERR << "Invalid value " << opts_.widen_label << " for option "
              << "--widen-label. It must be non-negative.";
  }

  if (IsWideningSegmentsToBeDone() && opts_.widen_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.widen_length << " for option "
              << "--widen-length. It must be positive.";
  }

  if (IsShrinkingSegmentsToBeDone() && opts_.shrink_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.shrink_length << " for option "
              << "--shrink-length. It must be positive.";
  }

  if (IsRelabelingShortSegmentsToBeDone() && 
      opts_.relabel_short_segments_class < 0) {
    KALDI_ERR << "Invalid value " << opts_.relabel_short_segments_class 
              << " for option " << "--relabel-short-segments-class. "
              << "It must be non-negative.";
  }
  
  if (IsRelabelingShortSegmentsToBeDone() && opts_.max_relabel_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.max_relabel_length << " for option "
              << "--max-relabel-length. It must be positive.";
  }

  if (IsRemovingSegmentsToBeDone() && remove_labels_[0] < 0) {
    KALDI_ERR << "Invalid value " << opts_.remove_labels_csl
              << " for option " << "--remove-labels. "
              << "The labels must be non-negative.";
  }

  if (IsMergingAdjacentSegmentsToBeDone() && 
      opts_.max_intersegment_length < 0) {
    KALDI_ERR << "Invalid value " << opts_.max_intersegment_length 
              << " for option "
              << "--max-intersegment-length. It must be non-negative.";
  }
  
  if (IsSplittingSegmentsToBeDone() && opts_.max_segment_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.max_segment_length 
              << " for option "
              << "--max-segment-length. It must be positive.";
  }

  if (opts_.post_process_label != -1 && opts_.post_process_label < 0) {
    KALDI_ERR << "Invalid value " << opts_.post_process_label << " for option "
              << "--post-process-label. It must be non-negative.";
  }
}

bool SegmentationPostProcessor::FilterAndPostProcess(Segmentation *seg, const
                                                     std::string *key) {
  if (!key) {
    Filter(seg);
  } else {
    if (!Filter(*key, seg)) return false;
  }

  return PostProcess(seg);
}

bool SegmentationPostProcessor::PostProcess(Segmentation *seg) const { 
  MergeLabels(seg);
  WidenSegments(seg);
  ShrinkSegments(seg);
  RelabelShortSegments(seg);
  RemoveSegments(seg);
  MergeAdjacentSegments(seg);
  SplitSegments(seg);

  return true;
}

void SegmentationPostProcessor::Filter(Segmentation *seg) const {
  if (!IsFilteringToBeDone()) return;
  KALDI_ASSERT(ClassifyRspecifier(opts_.filter_in_fn, NULL, NULL) ==
      kNoRspecifier);
  Segmentation tmp_seg(*seg);
  tmp_seg.IntersectSegments(filter_segmentation_, seg);
}

bool SegmentationPostProcessor::Filter(const std::string &key, 
                                     Segmentation *seg) {
  if (!IsFilteringToBeDone()) return true;
  KALDI_ASSERT(ClassifyRspecifier(opts_.filter_in_fn, NULL, NULL) !=
               kNoRspecifier);
  if (!filter_reader_.HasKey(key)) {
    KALDI_WARN << "Could not find filter for utterance " << key;
    if (!opts_.ignore_missing_filter_keys) return false;
    return true;
  }
  
  Segmentation tmp_seg(*seg);
  tmp_seg.IntersectSegments(filter_reader_.Value(key), seg);
  return true;
}

void SegmentationPostProcessor::MergeLabels(Segmentation *seg) const {
  if (!IsMergingLabelsToBeDone()) return;
  seg->MergeLabels(merge_labels_, opts_.merge_dst_label);
}

void SegmentationPostProcessor::WidenSegments(Segmentation *seg) const {
  if (!IsWideningSegmentsToBeDone()) return;
  seg->WidenSegments(opts_.widen_label, opts_.widen_length);
}

void SegmentationPostProcessor::ShrinkSegments(Segmentation *seg) const {
  if (!IsShrinkingSegmentsToBeDone()) return;
  seg->ShrinkSegments(opts_.widen_label, opts_.shrink_length);
}

void SegmentationPostProcessor::RelabelShortSegments(Segmentation *seg) const {
  if (!IsRelabelingShortSegmentsToBeDone()) return;
  seg->RelabelShortSegments(opts_.relabel_short_segments_class, 
                            opts_.max_relabel_length);
}

void SegmentationPostProcessor::RemoveSegments(Segmentation *seg) const {
  if (!IsRemovingSegmentsToBeDone()) return;
  seg->RemoveSegments(remove_labels_);
}

void SegmentationPostProcessor::MergeAdjacentSegments(Segmentation *seg) const {
  if (!IsMergingAdjacentSegmentsToBeDone()) return;
  seg->MergeAdjacentSegments(opts_.max_intersegment_length);
}

void SegmentationPostProcessor::SplitSegments(Segmentation *seg) const {
  if (!IsSplittingSegmentsToBeDone()) return;
  seg->SplitSegments(opts_.max_segment_length, opts_.max_segment_length / 2,
                     opts_.overlap_length, 
                     opts_.post_process_label);
}

}
}
