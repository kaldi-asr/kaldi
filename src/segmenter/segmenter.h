#ifndef SEGMENTER_H
#define SEGMENTER_H

#include <list>
#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "segmenter/segmenter.h"
#include "util/kaldi-table.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace segmenter {

typedef int32 ClassId;
struct Segment {
  int32 start_frame;
  int32 end_frame;
  ClassId class_id;

  inline int32 Label() const { return class_id; }
  inline void SetLabel(int32 label) { class_id = label; }

  Segment(int32 start, int32 end, int32 label) : 
    start_frame(start), end_frame(end), class_id(label) { }
  Segment() : start_frame(-1), end_frame(-1), class_id(-1) { }

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  static size_t SizeOf() {
    return (sizeof(int32) + sizeof(int32) + sizeof(ClassId));
  }
};

struct HistogramEncoder {
  BaseFloat bin_width;
  BaseFloat min_score; 
  std::vector<int32> bin_sizes;
  bool select_from_full_histogram;

  HistogramEncoder(): bin_width(-1), 
                      min_score(std::numeric_limits<BaseFloat>::infinity()),
                      select_from_full_histogram(false) {}

  inline int32 NumBins() const { return bin_sizes.size(); } 
  inline int32 BinSize(int32 i) const { return bin_sizes.at(i); }

  void Initialize(int32 num_bins, BaseFloat bin_w, BaseFloat min_s);
  void Encode(BaseFloat x, int32 n);
};

struct SegmentationOptions {
  std::string merge_labels_csl;
  int32 merge_dst_label, filter_label, mask_label;
  std::string filter_rspecifier, mask_rspecifier;

  SegmentationOptions() : merge_dst_label(-1), filter_label(-1), mask_label(-1) { }
  
  void Register(OptionsItf *po) {
    po->Register("merge-labels", &merge_labels_csl, "Merge labels into a single "
                "label defined by merge-dst-label. "
                "The labels to be merged are to "
                "be specified as a colon-separated list");
    po->Register("merge-dst-label", &merge_dst_label, "Merge labels into this "
                "label");
    po->Register("filter-rspecifier", &filter_rspecifier, "Filter and select "
                 "only those regions that have label filter-label in this "
                 "filter segmentation");
    po->Register("filter-label", &filter_label, "The label on which the "
                 "filtering is done");
    po->Register("mask-rspecifier", &mask_rspecifier, "Unselect "
                 "those regions that have label mask-label in this "
                 "mask-segmentation");
    po->Register("mask-label", &mask_label, "The label on which the "
                 "masking is done");
  }
};

struct HistogramOptions {
  int32 num_bins;
  bool select_above_mean;
  bool select_from_full_histogram;

  HistogramOptions() : num_bins(100), select_above_mean(false), select_from_full_histogram(false) {}
  
  void Register(OptionsItf *po) {
    po->Register("num-bins", &num_bins, "Number of bins in the histogram "
                 "created using the scores. Use larger number of bins to "
                 "make a finer selection");
    po->Register("select-above-mean", &select_above_mean, "If true, "
                 "use mean as the reference instead of min");
    po->Register("select-from-full-histogram", &select_from_full_histogram,
                 "Do not restrict selection to one half");

  }

};

class SegmentComparator {
  public:
    bool operator() (const Segment &lhs, const Segment &rhs) const {
      return lhs.start_frame < rhs.start_frame;
    }
};

typedef std::list<Segment> SegmentList;

class Segmentation {
  public:

    Segmentation() {
      Clear();
    }
    
    void GenRandomSegmentation(int32 max_length, int32 num_classes);

    // Split the input segmentation into pieces of 
    // approximately segment_length and store it in
    // this segmentation
    void SplitSegments(const Segmentation &in_segments,
                       int32 segment_length);
    
    // Split this segmentation into pieces of size 
    // segment_length such that the last remaining piece
    // is not longer than min_remainder.
    // Typically used to create 1s windows from 10 minute long chunks
    void SplitSegments(int32 segment_length,
                       int32 min_remainder);

    // Modify this segmentation to merge labels in merge_labels vector into a
    // single label dest_label.
    // e.g Merge noise and silence into a single silence label
    void MergeLabels(const std::vector<int32> &merge_labels,
                     int32 dest_label);

    // Merge adjacent segments of the same label
    void MergeAdjacentSegments();

    // Create a Histogram Encoder that can map a segment to 
    // a bin based on the average score
    void CreateHistogram(int32 label, const Vector<BaseFloat> &score, 
                         const HistogramOptions &opts, HistogramEncoder *hist);

    // Modify this segmentation to select the top bins in the 
    // histogram. Assumes that this segmentation also has the 
    // average scores.
    int32 SelectTopBins(const HistogramEncoder &hist, 
                        int32 src_label, int32 dst_label, int32 reject_label,
                        int32 num_frames_select, bool remove_rejected_frames);

    // Modify this segmentation to select the bottom bins in the histogram.
    // Assumes that this segmentation also has the average scores.
    int32 SelectBottomBins(const HistogramEncoder &hist, 
                           int32 src_label, int32 dst_label, int32 reject_label,
                           int32 num_frames_select, bool remove_rejected_frames);

    // Modify this segmentation to select the top and bottom bins in the 
    // histogram. Assumes that this segmentation also has the average scores.
    std::pair<int32,int32> SelectTopAndBottomBins(
        const HistogramEncoder &hist_encoder, 
        int32 src_label, int32 top_label, int32 num_frames_top,
        int32 bottom_label, int32 num_frames_bottom,
        int32 reject_label, bool remove_rejected_frames);

    // Initialize this segmentation from in_segmentation.
    // But select subsegments of this segmentation by including
    // only regions for which the "filter_segmentation" has 
    // the label "filter_label". 
    void IntersectSegments(const Segmentation &in_segmentation,
                           const Segmentation &filter_segmentation,
                           int32 filter_label);

    // Select subsegments of this segmentation by including
    // only regions for which the "filter_segmentation" has 
    // the label "filter_label". 
    // For e.g. if the segmentation is 
    // start_frame end_frame label
    // 5 10 1
    // 8 12 2
    // and filter_segmentation is 
    // 0 7 1
    // 7 10 2
    // 10 13 1.
    // And filter_label is 1. Then after intersection, this 
    // object would hold 
    // 5 7 1
    // 8 10 2
    // 10 12 2
    void IntersectSegments(const Segmentation &filter_segmentation,
                           int32 filter_label);

    // Create subsegments of this segmentation and assign new labels to 
    // the filtered regions. This is similar to "IntersectSegments" but 
    // instead of keeping only the filtered subsegments, all the 
    // subsegments are kept, while only changing the labels of the 
    // filtered subsegment to "subsegment_label"
    void CreateSubSegments(const Segmentation &filter_segmentation, 
                           int32 filter_label,
                           int32 subsegment_label);

    // Widen segments of label "label" by "length" frames 
    // on either side. But don't increase the length beyond the
    // neighboring segment. Also if the neighboring segment is
    // of a different type than "label", that segment is 
    // shortened to fix the boundary betten the segment and the 
    // neighbor
    void WidenSegments(int32 label, int32 length);

    // Remove segments of label "label" if they have a length
    // less than "max_length", label "label" and the previous
    // and next segments have the same label (not necessarily "label")
    void RemoveShortSegments(int32 label, int32 max_length);

    // Remove segments of label "label"
    void RemoveSegments(int32 label);

    // Reset segmentation i.e. clear all values
    void Clear();
    
    // Read segmentation object from input stream
    void Read(std::istream &is, bool binary);

    // Write segmentation object to output stream
    void Write(std::ostream &os, bool binary) const;

    // Write the segmentation in the form of an RTTM
    int32 WriteRttm(std::ostream &os, std::string key, BaseFloat frame_shift, BaseFloat start_time, bool map_to_speech_and_sil) const;
    
    // Convert current segmentation to alignment
    bool ConvertToAlignment(std::vector<int32> *alignment, 
                            int32 default_label = 0, int32 length = -1,
                            int32 tolerance = 2) const;

    SegmentList::iterator Erase(SegmentList::iterator it);
    void Emplace(int32 start_frame, int32 end_frame, ClassId class_id);
    void Check() const;
    void Sort();
  
    inline int32 Dim() const { return segments_.size(); }
    SegmentList::iterator Begin() { return segments_.begin(); }
    SegmentList::const_iterator Begin() const { return segments_.begin(); }
    SegmentList::iterator End() { return segments_.end(); }
    SegmentList::const_iterator End() const { return segments_.end(); }

  private: 
    int32 dim_;
    SegmentList segments_;
    std::vector<BaseFloat> mean_scores_;
    SegmentList::iterator current_;
};

typedef TableWriter<KaldiObjectHolder<Segmentation> > SegmentationWriter;
typedef SequentialTableReader<KaldiObjectHolder<Segmentation> > SequentialSegmentationReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Segmentation> > RandomAccessSegmentationReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<Segmentation> >  RandomAccessBaseFloatMatrixReaderMapped;

}
}

#endif // SEGMENTER_H
