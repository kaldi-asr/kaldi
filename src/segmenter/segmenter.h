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

// ClassId is just an integer for now. We could change it
// later if needed.
typedef int32 ClassId;

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
 * Some other properties that a segment might hold temporarily are 
 * vector_value   : This is some real valued vector such as average energy or 
 *                  ivector for the segment.
 * string_value   : Some string value such as segment_id that is characteristic
 *                  of the segment.
**/

struct Segment {
  int32 start_frame;
  int32 end_frame;
  ClassId class_id;
  Vector<BaseFloat> vector_value;
  std::string string_value;

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


  // This constructor is an extension to the above main constructor and also 
  // initializes the vector_value of the segment.
  Segment(int32 start, int32 end, int32 label, const Vector<BaseFloat>& vec) : 
    Segment(start, end, label) { 
    vector_value.Resize(vec.Dim());
    vector_value.CopyFromVec(vec);
  }
  
  // This constructor is an extension to the above constructor and 
  // initializes the string_value along with the vector_value
  Segment(int32 start, int32 end, int32 label, 
          const Vector<BaseFloat>& vec, const std::string &str) : 
    Segment(start, end, label, vec) { 
    string_value = str;
  }
 
  // This constructor is an extension to the main constructor and
  // additionally initializes the string_value of the segment.
  Segment(int32 start, int32 end, int32 label, const std::string &str) : 
    Segment(start, end, label) {
    string_value = str;
  } 

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // This is a function that returns the size of the elements in the structure.
  // It is used during I/O in binary mode, which checks for the total size
  // required to store the segment.
  static size_t SizeOf() {
    return (sizeof(int32) + sizeof(int32) + sizeof(ClassId));
  }

  // Accessors to get vector and string values corresponding to the segment.
  const Vector<BaseFloat>& VectorValue() const { return vector_value; }
  const std::string& StringValue() const { return string_value; }

  // Accessor to set the vector value not during initialization.
  void SetVectorValue(const VectorBase<BaseFloat>& vec) {
    vector_value.Resize(vec.Dim());
    vector_value.CopyFromVec(vec);
  }
};

/** This structure is used to encode some vector of real values into bins. This
 *  is mainly used in the classification of segments into speech, silence and
 *  noise depending on the vector of frame-level energy and/or zero-crossing
 *  of the frames in the segment.
**/

struct HistogramEncoder {
  // Width of the bins in the histogram of real values
  BaseFloat bin_width;

  // Minimum score corresponding to the lowest bin of the histogram
  BaseFloat min_score; 

  // This is a vector that stores the number of real values contained in the
  // different bins.
  std::vector<int32> bin_sizes;

  // A flag that is relevant only in a particular function. See the comments 
  // in Encode function for details.
  bool select_from_full_histogram;

  // default constructor
  HistogramEncoder(): bin_width(-1), 
                      min_score(std::numeric_limits<BaseFloat>::infinity()),
                      select_from_full_histogram(false) {}

  // Accessors for different quantities
  inline int32 NumBins() const { return bin_sizes.size(); } 
  inline int32 BinSize(int32 i) const { return bin_sizes[i]; }

  // Initialize the container to a specific number of bins and also size 
  // and the value each bin represents.
  void Initialize(int32 num_bins, BaseFloat bin_w, BaseFloat min_s);

  // Insert the real value 'x' with a count of 'n' times into the appropriate
  // bin in the histogram.
  void Encode(BaseFloat x, int32 n);
};

/**
 * Structure for some common options related to segmentation that would be used
 * in multiple segmentation programs. Some of the operations include merging,
 * filtering etc.
**/

struct SegmentationPostProcessingOptions {
  std::string filter_in_fn; 
  int32 filter_label;
  bool ignore_missing_filter_keys;
  std::string merge_labels_csl;
  int32 merge_dst_label;
  int32 widen_label;
  int32 widen_length;
  int32 shrink_label;
  int32 shrink_length;
  int32 relabel_short_segments_class; 
  int32 max_relabel_length;
  std::string remove_labels_csl;
  bool merge_adjacent_segments;
  int32 max_intersegment_length;
  int32 max_segment_length;
  int32 overlap_length;
  int32 post_process_label;

  SegmentationPostProcessingOptions() : 
    filter_label(-1), ignore_missing_filter_keys(false), merge_dst_label(-1), 
    widen_label(-1), widen_length(-1),
    shrink_label(-1), shrink_length(-1),
    relabel_short_segments_class(-1), max_relabel_length(-1), 
    merge_adjacent_segments(false), max_intersegment_length(0),
    max_segment_length(-1), overlap_length(0), post_process_label(-1) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("filter-in-fn", &filter_in_fn,
                   "The segmentation that is used as a filter for the "
                   "Intersection or Filtering post-processing operation. "
                   "Refer to the IntersectSegments() code for details. "
                   "Used in conjunction with the option --filter-label.");
    //opts->Register("filter-label", &filter_label, "The label on which the "
    //               "Intersection or Filtering operation is done. "
    //               "Refer to the IntersectSegments() code for details. "
    //               "Used in conjunction with the options --filter-in-fn.");
    opts->Register("ignore-missing-filter-keys", &ignore_missing_filter_keys, 
                   "If this is true and a key could not be found in the "
                   "filter, the post-processing skips the Filtering operation. "
                   "Otherwise, it counts it as an error. Applicable only when "
                   "--filter-in-fn is an archive. "
                   "Used in conjunction with the option --filter-in-fn.");
    opts->Register("merge-labels", &merge_labels_csl, "Merge labels into a "
                   "single label defined by merge-dst-label."
                   "The labels are specified as a colon-separated list. "
                   "Refer to the MergeLabels() code for details. "
                   "Used in conjunction with the option --merge-dst-label");
    opts->Register("merge-dst-label", &merge_dst_label, 
                   "Merge labels into this label. "
                   "Refer to the MergeLabels() code for details. "
                   "Used in conjunction with the option --merge-labels.");
    opts->Register("widen-label", &widen_label, 
                   "Widen segments of this class_id "
                   "by shrinking the adjacent segments of other class_ids or "
                   "merging with adjacent segments of the same class_id. "
                   "Refer to the WidenSegments() code for details. "
                   "Used in conjunction with the option --widen-length.");
    opts->Register("widen-length", &widen_length, "Widen segments by this many "
                   "frames on either side. "
                   "See option --widen-label for details. "
                   "Refer to the WidenSegments() code for details. "
                   "Used in conjunction with the option --widen-label.");
    opts->Register("shrink-label", &shrink_label, 
                   "Shrink segments of this class_id "
                   "by shrinking the adjacent segments of other class_ids or "
                   "merging with adjacent segments of the same class_id. "
                   "Refer to the ShrinkSegments() code for details. "
                   "Used in conjunction with the option --widen-length.");
    opts->Register("shrink-length", &shrink_length, "Shrink segments by this many "
                   "frames on either side. "
                   "See option --shrink-label for details. "
                   "Refer to the ShrinkSegments() code for details. "
                   "Used in conjunction with the option --shrink-label.");
    opts->Register("relabel-short-segments-class", &relabel_short_segments_class, 
                   "The class_id for which the short segments are to be "
                   "relabeled as the class_id of the neighboring segments. "
                   "Refer to RelabelShortSegments() code for details. "
                   "Used in conjunction with the option --max-relabel-length.");
    opts->Register("max-relabel-length", &max_relabel_length, 
                   "The maximum length of segment in number of frames that "
                   "will be relabeled to the class-id of the adjacent "
                   "segments, provided the adjacent segments both have the "
                   "same class-id. "
                   "Refer to RelabelShortSegments() code for details. "
                   "Used in conjunction with the option "
                   "--relabel-short-segments-class");
    opts->Register("remove-labels", &remove_labels_csl, 
                   "Remove any segment whose class_id is contained in "
                   "remove_labels_csl. "
                   "Refer to the RemoveLabels() code for details.");
    opts->Register("merge-adjacent-segments", &merge_adjacent_segments, 
                   "Merge adjacent segments of the same label if they are "
                   "within max-intersegment-length distance. "
                   "Refer to the MergeAdjacentSegments() code for details. "
                   "Used in conjunction with the option "
                   "--max-intersegment-length\n");
    opts->Register("max-intersegment-length", &max_intersegment_length,  
                   "The maximum intersegment length that is allowed for "
                   "two adjacent segments to be merged. "
                   "Refer to the MergeAdjacentSegments() code for details. "
                   "Used in conjunction with the option "
                   "--merge-adjacent-segments\n");
    opts->Register("max-segment-length", &max_segment_length, 
                   "If segment is longer than this length, split it into "
                   "pieces with less than these many frames. "
                   "Refer to the SplitSegments() code for details. "
                   "Used in conjunction with the option --overlap-length.");
    opts->Register("overlap-length", &overlap_length,
                   "When splitting segments longer than max-segment-length, "
                   "have the pieces overlap by these many frames. "
                   "Refer to the SplitSegments() code for details. "
                   "Used in conjunction with the option --max-segment-length.");
    opts->Register("post-process-label", &post_process_label, 
                   "Do post processing only on this label. This option is "
                   "applicable to only a few operations including "
                   "SplitSegments"); 
    //opts->Register("mask-rspecifier", &mask_rspecifier, "Unselect "
    //             "those regions that have label mask-label in this "
    //             "mask-segmentation");
    //opts->Register("mask-label", &mask_label, "The label on which the "
    //             "masking is done");
  }
};

/** 
 * Structure for options for histogram encoding 
**/

struct HistogramOptions {
  int32 num_bins;
  bool select_above_mean;
  bool select_from_full_histogram;

  HistogramOptions() : num_bins(100), select_above_mean(false), select_from_full_histogram(false) {}
  
  void Register(OptionsItf *opts) {
    opts->Register("num-bins", &num_bins, "Number of bins in the histogram "
                   "created using the scores. Use larger number of bins to "
                   "make a finer selection");
    opts->Register("select-above-mean", &select_above_mean, "If true, "
                   "use mean as the reference instead of min");
    opts->Register("select-from-full-histogram", &select_from_full_histogram,
                   "Do not restrict selection to one half");

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

// Segments are stored as a doubly-linked-list. This could be changed later 
// if needed. Hence defining a typedef SegmentList.
typedef std::list<Segment> SegmentList;

/**
 * The main class to store segmentation and do operations on it. The segments
 * are stored in the structure SegmentList, which is currently a doubly-linked
 * list.
 * See the .cc file for details of implementation of the different functions.
 * This file gives only a small description of the functions.
**/

class Segmentation {
  public:
    // Default constructor
    Segmentation() {
      Clear();
    }
    
    // Create random segmentation. Useful for debugging purposes.
    void GenRandomSegmentation(int32 max_length, int32 num_classes);

    // Split the input segmentation into pieces of approximately
    // segment_length and store it in this segmentation.
    // Most probably, you want to use the split segments version that is below
    // this one.
    void SplitSegments(const Segmentation &in_segments,
                       int32 segment_length);
    
    // Split this segmentation into pieces of size 
    // segment_length such that the last remaining piece
    // is not longer than min_remainder.
    // Optionally create overlapping pieces with the number
    // of overlapping frames specified by overlap.
    // Typically used to create 1s windows from 10 minute long chunks
    void SplitSegments(int32 segment_length,
                       int32 min_remainder, int32 overlap = 0,
                       int32 label = -1);

    // Modify this segmentation to merge labels in merge_labels vector into a
    // single label dest_label.
    // e.g Merge noise and silence into a single silence label
    void MergeLabels(const std::vector<int32> &merge_labels,
                     int32 dest_label);

    // Merge adjacent segments of the same label. "Adjacent" is defined as being
    // within max_intersegment_length of each other. i.e. start_frame of next
    // segment must not be greater than max_intersegment_length away from
    // end_frame of the current segment.
    void MergeAdjacentSegments(int32 max_intersegment_length = 1);

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
    //void IntersectSegments(const Segmentation &in_segmentation,
    //                       const Segmentation &filter_segmentation,
    //                       int32 filter_label);

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
    void IntersectSegments(const Segmentation &secondary_segmentation, 
                           Segmentation *out_seg, 
                           int32 mismatch_label = -1) const;

    // Extend a segmentation by adding another one. By default, the
    // resultant segmentation would be sorted. If its known that the other
    // segmentation must all be after this segmentation, then sort may be given
    // false.
    void Extend(const Segmentation &other_seg, bool sort = true);

    // Create new segmentation by sub-segmenting this segmentation and 
    // assign new labels to the filtered regions from secondary segmentation.
    // This is similar to "IntersectSegments", but instead of keeping only 
    // the filtered subsegments, all the subsegments are kept, while only 
    // changing the labels of the filtered subsegment to "subsegment_label".
    // Additionally this program adds the secondary segmentation's 
    // vector_value along with this segmentation's string_value
    void CreateSubSegments(const Segmentation &secondary_segmentation,
                           int32 secondary_label, int32 subsegment_label,
                           Segmentation *out_seg) const {
      SubSegmentUsingNonOverlappingSegments(secondary_segmentation,
                           secondary_label, subsegment_label,
                           out_seg);
    }

    void SubSegmentUsingNonOverlappingSegments(const Segmentation &secondary_segmentation,
                           int32 secondary_label, int32 subsegment_label,
                           Segmentation *out_seg) const;
    
    void SubSegmentUsingSmallOverlapSegments(const Segmentation &secondary_segmentation,
                           int32 secondary_label, int32 subsegment_label,
                           Segmentation *out_seg) const;

    void CreateSubSegmentsOld(const Segmentation &filter_segmentation, 
                           int32 filter_label,
                           int32 subsegment_label);

    // Widen segments of label "label" by "length" frames 
    // on either side. But don't increase the length beyond the
    // neighboring segment. Also if the neighboring segment is
    // of a different type than "label", that segment is 
    // shortened to fix the boundary betten the segment and the 
    // neighbor
    void WidenSegments(int32 label, int32 length);
    void ShrinkSegments(int32 label, int32 length);

    // Relabel segments of label "label" if they have a length
    // less than "max_length", label "label" and the previous
    // and next segments have the same label (not necessarily "label")
    // The three contiguous segments have the same label and hence are merged
    // together.
    void RelabelShortSegments(int32 label, int32 max_length);

    // Remove segments of label "label"
    void RemoveSegments(int32 label);

    // Remove segments of labels "labels"
    void RemoveSegments(const std::vector<int32> &labels);

    // Reset segmentation i.e. clear all values
    void Clear();
    
    // Read segmentation object from input stream
    void Read(std::istream &is, bool binary);

    // Write segmentation object to output stream
    void Write(std::ostream &os, bool binary) const;

    // Write the segmentation in the form of an RTTM
    int32 WriteRttm(std::ostream &os, const std::string &file_id, const std::string &channel, 
                    BaseFloat frame_shift, BaseFloat start_time, 
                    bool map_to_speech_and_sil) const;
    
    // Convert current segmentation to alignment
    bool ConvertToAlignment(std::vector<int32> *alignment, 
                            int32 default_label = 0, int32 length = -1,
                            int32 tolerance = 2) const;

    // Insert segments created from alignment whose 0th frame corresponds to 
    // start_time_offset
    int32 InsertFromAlignment(const std::vector<int32> &alignment,
                              int32 start_time_offset = 0, 
                              std::vector<int64> *frame_counts_per_class = NULL);

    int32 InsertFromSegmentation(const Segmentation &seg,
                                 int32 start_time_offset = 0,
                                 std::vector<int64> *frame_counts_per_class = NULL);

    // The following functions construct new segment in-place in the
    // segmentation and increments the dim_ of the segmentation. There's one
    // emplace for each constructor in Segment.
    inline void Emplace(int32 start_frame, int32 end_frame, ClassId class_id) {
      dim_++;
      segments_.emplace_back(start_frame, end_frame, class_id);
    }

    inline void Emplace(int32 start_frame, int32 end_frame, ClassId class_id, 
                        const Vector<BaseFloat> &vec) {
      dim_++;
      segments_.emplace_back(start_frame, end_frame, class_id, vec);
    }

    inline void Emplace(int32 start_frame, int32 end_frame, ClassId class_id, 
                        const std::string &str) {
      dim_++;
      segments_.emplace_back(start_frame, end_frame, class_id, str);
    }

    inline void Emplace(int32 start_frame, int32 end_frame, ClassId class_id, 
                        const Vector<BaseFloat> &vec, const std::string &str) {
      dim_++;
      segments_.emplace_back(start_frame, end_frame, class_id, vec, str);
    }

    // Call erase operation on the SegmentList and returns the iterator pointing
    // to the next segment in the SegmentList and also decrements dim_.
    inline SegmentList::iterator Erase(SegmentList::iterator it) {
      dim_--;
      return segments_.erase(it);
    }

    // Check if all segments have class_id >=0 and if dim_ matches the number of
    // segments.
    void Check() const;

    // Check if segmentation is non-overlapping.
    bool IsNonOverlapping() const;
    
    // Check if segmentation does not have large overlaps.
    bool HasSmallOverlap() const;

    // Sort the segments on the start_frame
    inline void Sort() { segments_.sort(SegmentComparator()); };
  
    // Public accessors
    inline int32 Dim() const { return dim_; }
    SegmentList::iterator Begin() { return segments_.begin(); }
    SegmentList::const_iterator Begin() const { return segments_.cbegin(); }
    SegmentList::iterator End() { return segments_.end(); }
    SegmentList::const_iterator End() const { return segments_.cend(); }

    const SegmentList* Data() const { return &segments_; }

  private: 
    // number of segments in the segmentation
    int32 dim_;

    // list of segments in the segmentation
    SegmentList segments_;
    
    // the score for each segment in the segmentation. If it has a non-zero
    // size, then the size must equal dim_.
    std::vector<BaseFloat> mean_scores_;

    friend class SegmentationPostProcessor;
};

typedef TableWriter<KaldiObjectHolder<Segmentation> > SegmentationWriter;
typedef SequentialTableReader<KaldiObjectHolder<Segmentation> > SequentialSegmentationReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Segmentation> > RandomAccessSegmentationReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<Segmentation> >  RandomAccessBaseFloatMatrixReaderMapped;

class SegmentationPostProcessor {
 public:
  explicit SegmentationPostProcessor(
      const SegmentationPostProcessingOptions &opts);
 
  bool FilterAndPostProcess(Segmentation *seg, const std::string *key = NULL);
  bool PostProcess(Segmentation *seg) const;
  
  bool Filter(const std::string &key, Segmentation *seg);
  void Filter(Segmentation *seg) const;
  void MergeLabels(Segmentation *seg) const;
  void WidenSegments(Segmentation *seg) const;
  void ShrinkSegments(Segmentation *seg) const;
  void RelabelShortSegments(Segmentation *seg) const;
  void RemoveSegments(Segmentation *seg) const;
  void MergeAdjacentSegments(Segmentation *seg) const;
  void SplitSegments(Segmentation *seg) const;

 private:
  const SegmentationPostProcessingOptions &opts_;
  std::vector<int32> merge_labels_;
  std::vector<int32> remove_labels_;
  Segmentation filter_segmentation_;
  RandomAccessSegmentationReader filter_reader_;

  inline bool IsFilteringToBeDone() const {
    return (!opts_.filter_in_fn.empty());
  }

  inline bool IsMergingLabelsToBeDone() const { 
    return (!opts_.merge_labels_csl.empty() || opts_.merge_dst_label != -1);
  }

  inline bool IsWideningSegmentsToBeDone() const {
    return (opts_.widen_label != -1 || opts_.widen_length != -1);
  }
  
  inline bool IsShrinkingSegmentsToBeDone() const {
    return (opts_.shrink_label != -1 || opts_.shrink_length != -1);
  }
  
  inline bool IsRelabelingShortSegmentsToBeDone() const {
    return (opts_.relabel_short_segments_class != -1 || opts_.max_relabel_length != -1);
  }

  inline bool IsRemovingSegmentsToBeDone() const { 
    return (!opts_.remove_labels_csl.empty()); 
  }

  inline bool IsMergingAdjacentSegmentsToBeDone() const {
    return (opts_.merge_adjacent_segments);
  }

  inline bool IsSplittingSegmentsToBeDone() const {
    return (opts_.max_segment_length != -1);
  }

  void Check() const;
};

}
}

#endif // SEGMENTER_H
