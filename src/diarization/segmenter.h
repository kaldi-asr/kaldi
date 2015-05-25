#ifndef SEGMENTER_H
#define SEGMENTER_H

#include "kaldi-common.h"

namespace kaldi {

namespace Segmenter {

struct Segment {
  int32 start_frame;
  int32 end_frame;
  ClassId class_id;
};

struct HistogramEncoder {
  BaseFloat bin_width;
  BaseFloat min_score; 
  vector<int32> bin_sizes;
  
  int32 NumBins() const { return bin_sizes.size(); } 
  void Initialize(int32 num_bins, BaseFloat bin_w, BaseFloat min_s);
}

void SplitSegments(const std::forward_list<Segment> &in_segments, 
                   int32 segment_length,
                   std::forward_list<Segment> *out_segments);
void SplitSegments(int32 segment_length,
                   int32 min_remainder,
                   std::forward_list<Segment> *segments);
void MergeLabels(const std::forward_list<Segment> &in_segments,
                 const std::vector<int32> &merge_labels,
                 std::forward_list<Segment> *out_segments);
void CreateHistogram(int32 label, const Vector<BaseFloat> &score, int32 num_bins,
                     std::forward_list<Segment> *segments,  
                     Histogram *histogram);
int32 SelectTopBins(const Histogram &histogram, int32 src_label, 
                   int32 dst_label, int32 num_frames_select,
                   std::forward_list<Segment> *segments);
int32 SelectBottomBins(const Histogram &histogram, int32 src_label,
                      int32 dst_label, int32 num_frames_select,
                      std::forward_list<Segment> *segments);

void IntersectSegments(const std::forward_list<Segment> &in_segments,
                       const std::forward_list<Segment> &filter_segments,
                       int32 filter_label,
                       std::forward_list<Segment> *segments);
}
}

#endif // SEGMENTER_H
