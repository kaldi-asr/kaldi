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

void Segmentation::SplitSegments(
    const Segmentation &in_segmentation,
    int32 segment_length) {
  Clear();
  for (SegmentList::const_iterator it = in_segmentation.Begin(); 
        it != in_segmentation.End(); ++it) {
    int32 length = it->end_frame - it->start_frame;
    int32 num_chunks = (static_cast<BaseFloat>(length)) / segment_length + 0.5;
    int32 segment_length = static_cast<BaseFloat>(length) / num_chunks + 0.5;
    
    int32 start_frame = it->start_frame;
    for (int32 j = 0; j < num_chunks; j++) {
      int32 end_frame = std::min(start_frame + segment_length, it->end_frame);
      Emplace(start_frame, end_frame, it->class_id);
      start_frame = end_frame + 1;
    }
  }
  Check();
}

void Segmentation::SplitSegments(int32 segment_length,
                                 int32 min_remainder) {
  for (SegmentList::iterator it = segments_.begin(); 
      it != segments_.end(); ++it) {
    int32 start_frame = it->start_frame;
    int32 end_frame = it->end_frame;
    int32 length = end_frame - start_frame;

    if (length > segment_length + min_remainder) {
      // Split segment
      it->start_frame = start_frame + segment_length;
      it = segments_.emplace(it, start_frame, start_frame + segment_length - 1, it->Label());

      // Forward list code
      // it->end_frame = start_frame + segment_length - 1;
      // it = segments_.emplace(it+1, it->end_frame + 1, end_frame, it->Label());
      
      dim_++;
    }
  }
  Check();
}

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

void Segmentation::IntersectSegments(
    const Segmentation &filter_segmentation,
    int32 filter_label) {
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
      while (it != segments_.end()) {
        it = segments_.erase(it);
        dim_--;
      }
      break;
    }

    // If the segment in the filter is beyond the end of the current segment,
    // then remove the segments until the current segment end 
    // point is just after the start of the filter segment
    if (filter_it->start_frame > it->end_frame) {
      it = segments_.erase(it);
      dim_--;
      continue;
    }

    // filter start_frame is after the start_frame of this segment. 
    // So throw away the initial part of this segment as it is not in the
    // filter. i.e. Set the start of this segment to be the start of the filter
    // segment.
    if (filter_it->start_frame > it->start_frame) 
      it->start_frame = filter_it->start_frame;
      
    if (filter_it->end_frame < it->end_frame) {
      // filter segment ends before the end of the current segment. Then end 
      // the current segment right at the end of the filter and leave the 
      // remaining part for the next segment
      
      int32 start_frame = it->start_frame;
      it->start_frame = filter_it->end_frame + 1;
      segments_.emplace(it, start_frame, filter_it->end_frame, it->Label());

      //Forward list
      //int32 end_frame = it->end_frame;
      //it->end_frame = filter_it->end_frame;
      //it = segments_.emplace(it+1, filter_it->end_frame + 1, 
      //    end_frame, it->Label());
      
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
    
void Segmentation::WidenSegments(int32 label, int32 length) {
  for (SegmentList::iterator it = segments_.begin();
        it != segments_.end(); ++it) {
    if (it->Label() == label) {
      if (it != segments_.begin()) {
        SegmentList::iterator prev_it = it;
        --prev_it;
        it->start_frame -= length;
        if (prev_it->Label() == label && it->start_frame < prev_it->end_frame) {
          it->start_frame = prev_it->start_frame;
          Erase(prev_it);
        } else if (prev_it->Label() != label && 
            it->start_frame < prev_it->end_frame) {
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
        it->end_frame += length;
    } else { // if (it->Label() != label)
      if (it != segments_.begin()) {
        SegmentList::iterator prev_it = it;
        --prev_it;
        if (prev_it->end_frame >= it->end_frame) {
          // The extended previous SPEECH segment completely overlaps the current
          // SILENCE segment. So remove the SILENCE segment.
          it = Erase(it);
          --it;   // So that we can increment in the for loop
        } else if (prev_it->end_frame >= it->start_frame) {
          // The extended previous SPEECH segment reduces the length of this 
          // SILENCE segment.
          it->start_frame = prev_it->end_frame + 1;
        }
      } 
    }
  }
}

void Segmentation::RemoveShortSegments(int32 label, int32 max_length) {
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
        && it->end_frame - it->start_frame + 1 < max_length) {
      prev_it->end_frame = next_it->end_frame;
      segments_.erase(it);
      it = segments_.erase(next_it);
      dim_ -= 2;
    } else 
      ++it;
  }
}

void Segmentation::Clear() {
  segments_.clear();
  dim_ = 0;
  mean_scores_.clear();
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

void Segmentation::WriteRttm(std::ostream &os, std::string key, BaseFloat frame_shift, BaseFloat start_time) const {
  SegmentList::const_iterator it = segments_.begin();
  for (; it != segments_.end(); ++it) {
    os << "SPEAKER " << key << " 1 "
       << it->start_frame * frame_shift + start_time << " " 
       << (it->end_frame - it->start_frame + 1) * frame_shift << " <NA> <NA> ";
    switch (it->Label()) {
      case 1:
        os << "SPEECH ";
        break;
      default:
        os << "SILENCE ";
        break;
    }
    os << "<NA>" << std::endl;
  } 
}

bool Segmentation::ConvertToAlignment(std::vector<int32> *alignment,
    int32 default_label, int32 length, int32 tolerance) const {
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
                << ">= length + tolerance (" << length + tolerance << ").";
      return false;
    }

    int32 end_frame = it->end_frame;
    if (length == -1) {
      alignment->resize(it->end_frame + 1, default_label);
    } else {
      if (it->end_frame >= length) 
        end_frame = length - 1;
    }

    for (size_t i = it->start_frame; i <= end_frame; i++) {
      (*alignment).at(i) = it->Label();
    }
  } 
  return true;
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

void Segmentation::Emplace(int32 start_frame, int32 end_frame, ClassId class_id) {
  dim_++;
  segments_.emplace_back(start_frame, end_frame, class_id);
}

SegmentList::iterator Segmentation::Erase(SegmentList::iterator it) {
  dim_--;
  return segments_.erase(it);
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

void Segmentation::Check() const {
  int32 dim = 0;
  for (SegmentList::const_iterator it = segments_.begin();
        it != segments_.end(); ++it, dim++);
  KALDI_ASSERT(dim == dim_);
  KALDI_ASSERT(mean_scores_.size() == 0 || mean_scores_.size() == dim_);
}

}
}
