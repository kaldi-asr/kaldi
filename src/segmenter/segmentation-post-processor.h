// segmenter/segmentation-post-processor.h

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

#ifndef KALDI_SEGMENTER_SEGMENTATION_POST_PROCESSOR_H_
#define KALDI_SEGMENTER_SEGMENTATION_POST_PROCESSOR_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "segmenter/segmentation.h"

namespace kaldi {
namespace segmenter {

/**
 * Structure for some common options related to segmentation that would be used
 * in multiple segmentation programs. Some of the operations include merging,
 * filtering etc.
**/

struct SegmentationPostProcessingOptions {
  std::string merge_labels_csl;
  int32 merge_dst_label;

  int32 pad_label;
  int32 pad_length;

  int32 shrink_label;
  int32 shrink_length;

  int32 blend_short_segments_class;
  int32 max_blend_length;

  std::string remove_labels_csl;
  int32 max_remove_length;

  bool merge_adjacent_segments;
  int32 max_intersegment_length;

  int32 max_segment_length;
  int32 overlap_length;

  int32 post_process_label;

  SegmentationPostProcessingOptions() :
    merge_dst_label(-1),
    pad_label(-1), pad_length(-1),
    shrink_label(-1), shrink_length(-1),
    blend_short_segments_class(-1), max_blend_length(-1),
    max_remove_length(-1), 
    merge_adjacent_segments(false), 
    max_intersegment_length(0),
    max_segment_length(-1), overlap_length(0),
    post_process_label(-1) { }

  void Register(OptionsItf *opts) {
    opts->Register("merge-labels", &merge_labels_csl, "Merge labels into a "
                   "single label defined by merge-dst-label. "
                   "The labels are specified as a colon-separated list. "
                   "Refer to the MergeLabels() code for details. "
                   "Used in conjunction with the option --merge-dst-label");
    opts->Register("merge-dst-label", &merge_dst_label,
                   "Merge labels specified by merge-labels into this label. "
                   "Refer to the MergeLabels() code for details. "
                   "Used in conjunction with the option --merge-labels.");
    opts->Register("pad-label", &pad_label,
                   "Pad segments of this label by pad_length frames."
                   "Refer to the PadSegments() code for details. "
                   "Used in conjunction with the option --pad-length.");
    opts->Register("pad-length", &pad_length, "Pad segments by this many "
                   "frames on either side. "
                   "Refer to the PadSegments() code for details. "
                   "Used in conjunction with the option --pad-label.");
    opts->Register("shrink-label", &shrink_label,
                   "Shrink segments of this label by shrink_length frames. "
                   "Refer to the ShrinkSegments() code for details. "
                   "Used in conjunction with the option --shrink-length.");
    opts->Register("shrink-length", &shrink_length, "Shrink segments by this "
                   "many frames on either side. "
                   "Refer to the ShrinkSegments() code for details. "
                   "Used in conjunction with the option --shrink-label.");
    opts->Register("blend-short-segments-class", &blend_short_segments_class,
                   "The label for which the short segments are to be "
                   "blended with the neighboring segments that are less than "
                   "max_intersegment_length frames away. "
                   "Refer to BlendShortSegments() code for details. "
                   "Used in conjunction with the option --max-blend-length "
                   "and --max-intersegment-length.");
    opts->Register("max-blend-length", &max_blend_length,
                   "The maximum length of segment in number of frames that "
                   "will be blended with the neighboring segments provided "
                   "they both have the same label. "
                   "Refer to BlendShortSegments() code for details. "
                   "Used in conjunction with the option "
                   "--blend-short-segments-class");
    opts->Register("remove-labels", &remove_labels_csl,
                   "Remove any segment whose label is contained in "
                   "remove_labels_csl. "
                   "Refer to the RemoveLabels() code for details.");
    opts->Register("max-remove-length", &max_remove_length,
                   "If provided, specifies the maximum length of segments "
                   "that will be removed by --remove-labels option");
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
                   "--merge-adjacent-segments or "
                   "--blend-short-segments-class\n");
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
  }
};

class SegmentationPostProcessor {
 public:
  explicit SegmentationPostProcessor(
      const SegmentationPostProcessingOptions &opts);

  bool PostProcess(Segmentation *seg) const;

  void DoMergingLabels(Segmentation *seg) const;
  void DoPaddingSegments(Segmentation *seg) const;
  void DoShrinkingSegments(Segmentation *seg) const;
  void DoBlendingShortSegments(Segmentation *seg) const;
  void DoRemovingSegments(Segmentation *seg) const;
  void DoMergingAdjacentSegments(Segmentation *seg) const;
  void DoSplittingSegments(Segmentation *seg) const;

 private:
  const SegmentationPostProcessingOptions &opts_;
  std::vector<int32> merge_labels_;
  std::vector<int32> remove_labels_;

  void Check() const;
};

} // end namespace segmenter
} // end namespace kaldi

#endif // KALDI_SEGMENTER_SEGMENTATION_POST_PROCESSOR_H_
