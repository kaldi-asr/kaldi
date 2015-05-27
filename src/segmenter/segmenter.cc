#include "segmenter/segmenter.h"
#include <algorithm>

namespace kaldi {
namespace segmenter {

void Segment::Write(std::ostream &os, bool binary) const {
  if (binary) {
    os.write(reinterpret_cast<const char *>(start_frame), sizeof(start_frame));
    os.write(reinterpret_cast<const char *>(end_frame), sizeof(start_frame));
    os.write(reinterpret_cast<const char *>(class_id), sizeof(class_id));
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
  int32 i = (x - min_score ) / NumBins();
  if (i < 0) i = 0;
  if (i >= NumBins()) i = NumBins() - 1;
  bin_sizes[i] += n;
}

void Segmentation::SplitSegments(
    const Segmentation &in_segmentation,
    int32 segment_length) {
  Clear();
  for (std::forward_list<Segment>::const_iterator it = in_segmentation.Begin(); 
        it != in_segmentation.End(); ++it) {
    int32 length = it->end_frame - it->start_frame;
    int32 num_chunks = (static_cast<BaseFloat>(length)) / segment_length + 0.5;
    int32 segment_length = static_cast<BaseFloat>(length) / num_chunks + 0.5;
    
    int32 start_frame = it->start_frame;
    for (int32 j = 0; j < num_chunks; j++) {
      int32 end_frame = std::min(start_frame + segment_length, it->end_frame);
      Emplace(start_frame, end_frame, it->class_id);
      dim_++;
      start_frame = end_frame + 1;
    }
  }
}

void Segmentation::SplitSegments(int32 segment_length,
                                 int32 min_remainder) {
  for (std::forward_list<Segment>::iterator it = segments_.begin(); 
      it != segments_.end(); ++it) {
    int32 start_frame = it->start_frame;
    int32 end_frame = it->end_frame;
    int32 length = end_frame - start_frame;

    if (length > segment_length + min_remainder) {
      // Split segment
      it->end_frame = start_frame + segment_length - 1;
      segments_.emplace_after(it, it->end_frame + 1, end_frame, it->Label());
      dim_++;
    }
  }
}

void Segmentation::MergeLabels(const std::vector<int32> &merge_labels,
                            int32 dest_label) {
  std::is_sorted(merge_labels.begin(), merge_labels.end());
  int32 size = 0;
  for (std::forward_list<Segment>::iterator it = segments_.begin(); 
       it != segments_.end(); ++it, size++) {
    if (std::binary_search(merge_labels.begin(), merge_labels.end(), it->Label())) {
      it->SetLabel(dest_label);
    }
  }
  KALDI_ASSERT(size == Dim());
}

void Segmentation::CreateHistogram(
    int32 label, const Vector<BaseFloat> &scores, 
    int32 num_bins, HistogramEncoder *hist_encoder) {
  if (Dim() == 0)
    KALDI_ERR << "Segmentation must not be empty";

  BaseFloat min_score = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat max_score = -std::numeric_limits<BaseFloat>::infinity();

  mean_scores_.clear();
  mean_scores_.resize(Dim(), std::numeric_limits<BaseFloat>::quiet_NaN());
  
  std::vector<int32> num_frames;
  int32 i = 0;
  for (std::forward_list<Segment>::iterator it = segments_.begin(); 
        it != segments_.end(); ++it, i++) {
    if (it->Label() != label) continue;
    SubVector<BaseFloat> this_segment_scores(scores, it->start_frame, it->end_frame - it->start_frame + 1);
    BaseFloat mean_score = this_segment_scores.Sum() / this_segment_scores.Dim();
    
    mean_scores_[i] = mean_score;
    num_frames.push_back(this_segment_scores.Dim());

    if (mean_score > max_score) max_score = mean_score;
    if (mean_score < min_score) min_score = mean_score;
  }

  BaseFloat bin_width = (max_score - min_score) / num_bins;
  hist_encoder->Initialize(num_bins, bin_width, min_score);

  i = 0;
  for (std::forward_list<Segment>::const_iterator it = segments_.begin(); it != segments_.end(); ++it) {
    if (it->Label() != label) continue;
    hist_encoder->Encode(mean_scores_[i], num_frames[i]);
    i++;
  }
}

int32 Segmentation::SelectTopBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 dst_label, int32 reject_label,
    int32 num_frames_select) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  if (num_frames_select == -1) return 0;

  BaseFloat min_score_for_selection = std::numeric_limits<BaseFloat>::infinity();
  int32 num_top_frames = 0, i = hist_encoder.NumBins() - 1;
  while (i >= hist_encoder.NumBins() / 2) {
    num_top_frames += hist_encoder.BinSize(i);
    if (num_top_frames >= num_frames_select) break;
    i--;
  }
  min_score_for_selection = hist_encoder.min_score + i * hist_encoder.bin_width;

  i = 0;
  for (std::forward_list<Segment>::iterator it = segments_.begin(); 
        it != segments_.end(); ++it, i++) {
    if (it->Label() != src_label) continue;
    if (mean_scores_[i] >= min_score_for_selection) {
      it->SetLabel(dst_label);
    } else {
      it->SetLabel(reject_label);
    }
  }

  return num_top_frames;
}

int32 Segmentation::SelectBottomBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 dst_label, int32 reject_label, 
    int32 num_frames_select) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  if (num_frames_select == -1) return 0;

  BaseFloat max_score_for_selection = -std::numeric_limits<BaseFloat>::infinity();
  int32 num_bottom_frames = 0, i = 0;
  while (i < hist_encoder.NumBins() / 2) {
    num_bottom_frames += hist_encoder.BinSize(i);
    if (num_bottom_frames >= num_frames_select) break;
    i++;
  }
  max_score_for_selection = hist_encoder.min_score + (i+1) * hist_encoder.bin_width;

  i = 0;
  for (std::forward_list<Segment>::iterator it = segments_.begin(); 
        it != segments_.end(); ++it) {
    if (it->Label() != src_label) continue;
    if (mean_scores_[i] < max_score_for_selection) {
      it->SetLabel(dst_label);
    } else {
      it->SetLabel(reject_label);
    }
  }

  return num_bottom_frames;
}

void Segmentation::IntersectSegments(
    const Segmentation &in_segmentation,
    const Segmentation &filter_segmentation,
    int32 filter_label) {
  Clear();

  std::forward_list<Segment>::const_iterator it = in_segmentation.Begin(),
                  filter_it = filter_segmentation.Begin();

  int32 start_frame = it->start_frame;
  while (it != in_segmentation.End()) {
    while (filter_it != filter_segmentation.End() && 
           filter_it->end_frame < start_frame && 
           filter_it->Label() != filter_label) {
      ++filter_it;
    }
    
    if (filter_it == filter_segmentation.End()) 
      break;

    if (filter_it->start_frame > it->end_frame) {
      ++it;
      continue;
    }

    if (filter_it->start_frame > start_frame) 
      start_frame = filter_it->start_frame;
      
    if (filter_it->end_frame < it->end_frame) {
      Emplace(start_frame,
              filter_it->end_frame, it->Label());
      dim_++;
      start_frame = filter_it->end_frame + 1;
    } else {
      Emplace(start_frame,
              it->end_frame, it->Label());
      dim_++;
      ++it;
      if (it != in_segmentation.End())
        start_frame = it->start_frame;
    }
  }
}

void Segmentation::IntersectSegments(
    const Segmentation &filter_segmentation,
    int32 filter_label) {
  std::forward_list<Segment>::iterator it = segments_.begin();
  std::forward_list<Segment>::const_iterator filter_it = filter_segmentation.Begin();

  int32 start_frame = it->start_frame;
  while (it != segments_.end()) {
    while (filter_it != filter_segmentation.End() && 
           filter_it->end_frame < start_frame && 
           filter_it->Label() != filter_label) {
      ++filter_it;
    }
    
    if (filter_it == filter_segmentation.End()) 
      break;

    if (filter_it->start_frame > it->end_frame) {
      ++it;
      continue;
    }

    if (filter_it->start_frame > start_frame) 
      start_frame = filter_it->start_frame;
      
    if (filter_it->end_frame < it->end_frame) {
      segments_.emplace_after(it, start_frame,
              filter_it->end_frame, it->Label());
      dim_++;
      start_frame = filter_it->end_frame + 1;
    } else {
      segments_.emplace_after(it, start_frame,
              it->end_frame, it->Label());
      dim_++;
      ++it;
      if (it != segments_.end())
        start_frame = it->start_frame;
    }
  }
}

void Segmentation::Clear() {
  segments_.clear();
  dim_ = 0;
  mean_scores_.clear();
  current_ = segments_.before_begin();
}

void Segmentation::Write(std::ostream &os, bool binary) const {
  std::forward_list<Segment>::const_iterator it = segments_.begin();
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

void Segmentation::Read(std::istream &is, bool binary) {
  Clear();
  
  std::forward_list<Segment>::iterator it = segments_.before_begin();

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

    for (int32 i = 0; i < segmentssz; i++, ++it) {
      Segment seg;
      seg.Read(is, binary);
      segments_.insert_after(it, seg);
    }
    dim_ = segmentssz;
  } else {
    is >> std::ws;
    if (is.peek() != static_cast<int>('[')) {
      KALDI_ERR << "Segmentation::Read: expected to see [, saw "
                << is.peek() << ", at file position " << is.tellg();
    }
    is.get();   // consume the '['
    while (is.peek() != static_cast<int>(']')) {
      Segment seg;
      seg.Read(is, binary);
      segments_.insert_after(it++, seg);
      dim_++;
    }
  }
}

void Segmentation::Emplace(int32 start_frame, int32 end_frame, ClassId class_id) {
  segments_.emplace_after(current_, start_frame, end_frame, class_id);
  ++current_;
}

}
}
