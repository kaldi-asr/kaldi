namespace kaldi {

namespace Segmenter {

void Histogram::Initialize(int32 num_bins, BaseFloat bin_w, BaseFloat min_s) {
  bin_sizes.clear();
  bin_sizes.resize(num_bins, 0);
  bin_width = bin_w;
  min_score = min_s;
}

void Histogram::Encode(BaseFloat x, int32 n) {
  int32 i = (x - min_score ) / NumBins();
  if (i < 0) i = 0;
  if (i >= NumBins()) i = NumBins() - 1;
  bin_sizes[i] += n;
}

void SplitSegments(const std::forward_list<Segment> &in_segments,
                              int32 segment_length,
                              std::forward_list<Segment> *out_segments) {
  for (const_iterator it = in_segments.begin(); it != in_segments.end(); ++it) {
    int32 length = it->end_frame - it->start_frame;
    int32 num_chunks = (static<BaseFloat>(length)) / segment_length + 0.5;
    int32 segment_length = static_cast<BaseFloat>(length) / num_chunks + 0.5;
    
    int32 start_frame = it->start_frame;
    for (int32 j = 0; j < num_chunks; j++) {
      int32 end_frame = std::min(start_frame + segment_length, it->end_frame);
      Segment segment(start, frame, it->class_id);
      out_segments->push_back(segment);
      start_frame = end_frame + 1;
    }
  }
}

void SplitSegments(int32 segment_length,
                              int32 min_remainder,
                              std::forward_list<Segment> *segments) {
  for (iterator it = segments->begin(); it != segments->end(); ++it) {
    int32 start_frame = it->start_frame;
    int32 end_frame = it->end_frame;
    int32 length = end_frame - start_frame;

    if (length > segment_length + min_remainder) {
      // Split segment
      it->end_frame = start_frame + segment_length - 1;
      segments->emplace_after(it, it->end_frame + 1, it->end_frame, it->GetLabel());
    }
  }
}

void MergeLabels(const std::vector<int32> &merge_labels,
                            int32 dest_label,
                            std::forward_list<Segment> *segments) {
  is_sorted(merge_labels.begin(), merge_labels.end());
  for (iterator it = segments->begin(), it != segments->end(); ++it) {
    if (std::binary_search(merge_labels.begin(), merge_labels.end(), it->GetLabel())) {
      segments->SetLabel(dest_label);
    }
  }
}

void CreateHistogram(int32 label, const Vector<BaseFloat> &scores, int32 num_bins,
                     std::forward_list<Segment> *segments,  
                     HistogramEncoder *hist_encoder) {
  if (segments == NULL) 
    KALDI_ERR << "Segments must not be NULL to create a histogram encoder";

  BaseFloat min_score = std::numeric_limits<BaseFloat>::max();
  BaseFloat max_score = -std::numeric_limist<BaseFloat>::max();
  std::vector<BaseFloat> mean_scores;
  std::vector<int32> num_frames;
  for (std::forward_list<Segment>::iterator it = segments.begin(); 
        it != segments.end(); ++it) {
    if (it->GetLabel() != label) continue;
    SubVector this_segment_scores(scores, it->start_frame, it->end_frame - it->start_frame + 1);
    BaseFloat mean_score = this_segment_scores.Sum() / this_segment_scores.Dim();
    it->SetScore(mean_score);
    
    mean_scores.push_back(mean_score);
    num_frames.push_back(this_segment_scores.Dim());

    if (mean_score > max_score) max_score = mean_score;
    if (mean_score < min_score) min_score = mean_score;
  }

  BaseFloat bin_width = (max_score - min_score) / num_bins;
  histogram_encoder->Initialize(num_bins, bin_width, min_score);

  int32 i = 0;
  for (std::forward_list<Segment>::const_iterator it = segments.begin(); it != segments.end(); ++it) {
    if (it->GetLabel() != label) continue;
    histogram->Encode(mean_scores[i], num_frames[i]);
    i++;
  }
}

int32 SelectTopBins(const Histogram &histogram, int32 src_label,
                              int32 dst_label, int32 num_frames_select,
                              std::forward_list<Segment> *segments) {
  BaseFloat min_score_for_selection = std::numeric_limits<BaseFloat>::max();
  int32 num_top_frames = 0, i = histogram.NumBins() - 1;
  while (i >= histogram.NumBins() / 2) {
    num_top_frames += histogram.BinSize(i);
    if (num_top_frames >= num_frames_select) break;
    i--;
  }
  min_score_for_selection = histogram.min_score + i * histogram.bin_width;

  for (std::forward_list<Segment>::iterator it = segments->begin(); 
        it != segments->end(); ++it) {
    if (it->GetLabel() != src_label) continue;
    if (it->GetScore() >= min_score_for_selection) {
      it->SetLabel(dst_label);
    }
  }

  return num_top_frames;
}

int32 SelectBottomBins(const Histogram &histogram, int32 src_label,
                                  int32 dst_label, int32 num_frames_select,
                                  std::forward_list<Segment> *segments) {
  BaseFloat max_score_for_selection = -std::numeric_limits<BaseFloat>::max();
  int32 num_bottom_frames = 0, i = 0;
  while (i < histogram.NumBins() / 2) {
    num_bottom_frames += histogram.BinSize(i);
    if (num_bottom_frames >= num_frames_select) break;
    i++;
  }
  max_score_for_selection = histogram.min_score + (i+1) * histogram.bin_width;

  for (std::forward_list<Segment>::iterator it = segments->begin(); 
        it != segments->end(); ++it) {
    if (it->GetLabel() != src_label) continue;
    if (it->GetScore() < max_score_for_selection) {
      it->SetLabel(dst_label);
    }
  }

  return num_bottom_frames;
}

void IntersectSegments(const std::forward_list<Segment> &in_segments,
                       const std::forward_list<Segment> &filter_segments,
                       int32 filter_label,
                       std::forward_list<Segment> *segments) {

  std::forward_list<Segment>::iterator out_it = segments->before_begin();
  std::forward_list<Segment>::const_iterator it = in_segments.begin(),
                  filter_it = filter_segments.begin();

  int32 start_frame = it->start_frame;
  while (it != in_segments.end()) {
    while (filter_it != filter_segments.end() && 
           filter_it->end_frame < start_frame && 
           filter_it->Label() != filter_label) {
      ++filter_it;
    }
    
    if (filter_it == filter_segments.end()) 
      break;

    if (filter_it->start_frame > it->end_frame) {
      ++it;
      continue;
    }

    if (filter_it->start_frame > start_frame) 
      start_frame = filter_it->start_frame;
      
    if (filter_it->end_frame < it->end_frame) {
      segments->emplace_after(out_it, start_frame,
              filter_it->end_frame, it->Label());
      start_frame = filter_it->end_frame + 1;
    } else {
      segments->emplace_after(out_it, start_frame,
                it->end_frame, it->Label());
      ++it;
      if (it != in_segments.end())
        start_frame = it->start_frame;
    }
    
    ++out_it;
  }
}

}
}
