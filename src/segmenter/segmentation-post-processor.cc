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

#include "segmenter/segmentation-utils.h"
#include "segmenter/segmentation-post-processor.h"

namespace kaldi {
namespace segmenter {

static inline bool IsMergingLabelsToBeDone(
    const SegmentationPostProcessingOptions &opts) {
  return (!opts.merge_labels_csl.empty() || opts.merge_dst_label != -1);
}

static inline bool IsPaddingSegmentsToBeDone(
    const SegmentationPostProcessingOptions &opts) {
  return (opts.pad_label != -1 || opts.pad_length != -1);
}

static inline bool IsShrinkingSegmentsToBeDone(
    const SegmentationPostProcessingOptions &opts) {
  return (opts.shrink_label != -1 || opts.shrink_length != -1);
}

static inline bool IsBlendingShortSegmentsToBeDone(
    const SegmentationPostProcessingOptions &opts) {
  return (opts.blend_short_segments_class != -1 || opts.max_blend_length != -1);
}

static inline bool IsRemovingSegmentsToBeDone(
    const SegmentationPostProcessingOptions &opts) {
  return (!opts.remove_labels_csl.empty());
}

static inline bool IsMergingAdjacentSegmentsToBeDone(
    const SegmentationPostProcessingOptions &opts) {
  return (opts.merge_adjacent_segments);
}

static inline bool IsSplittingSegmentsToBeDone(
    const SegmentationPostProcessingOptions &opts) {
  return (opts.max_segment_length != -1);
}


SegmentationPostProcessor::SegmentationPostProcessor(
    const SegmentationPostProcessingOptions &opts) : opts_(opts) {
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
  if (IsPaddingSegmentsToBeDone(opts_) && opts_.pad_label < 0) {
    KALDI_ERR << "Invalid value " << opts_.pad_label << " for option "
              << "--pad-label. It must be non-negative.";
  }

  if (IsPaddingSegmentsToBeDone(opts_) && opts_.pad_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.pad_length << " for option "
              << "--pad-length. It must be positive.";
  }

  if (IsShrinkingSegmentsToBeDone(opts_) && opts_.shrink_label < 0) {
    KALDI_ERR << "Invalid value " << opts_.shrink_label << " for option "
              << "--shrink-label. It must be non-negative.";
  }

  if (IsShrinkingSegmentsToBeDone(opts_) && opts_.shrink_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.shrink_length << " for option "
              << "--shrink-length. It must be positive.";
  }

  if (IsBlendingShortSegmentsToBeDone(opts_) &&
      opts_.blend_short_segments_class < 0) {
    KALDI_ERR << "Invalid value " << opts_.blend_short_segments_class
              << " for option " << "--blend-short-segments-class. "
              << "It must be non-negative.";
  }

  if (IsBlendingShortSegmentsToBeDone(opts_) && opts_.max_blend_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.max_blend_length << " for option "
              << "--max-blend-length. It must be positive.";
  }

  if (IsRemovingSegmentsToBeDone(opts_) && 
      (remove_labels_[0] < -1 || 
       (remove_labels_.size() > 1 && remove_labels_[0] == -1))) {
    KALDI_ERR << "Invalid value " << opts_.remove_labels_csl
              << " for option " << "--remove-labels. "
              << "The labels must be non-negative.";
  }

  if (IsMergingAdjacentSegmentsToBeDone(opts_) &&
      opts_.max_intersegment_length < 0) {
    KALDI_ERR << "Invalid value " << opts_.max_intersegment_length
              << " for option "
              << "--max-intersegment-length. It must be non-negative.";
  }

  if (IsSplittingSegmentsToBeDone(opts_) && opts_.max_segment_length <= 0) {
    KALDI_ERR << "Invalid value " << opts_.max_segment_length
              << " for option "
              << "--max-segment-length. It must be positive.";
  }

  if (opts_.post_process_label != -1 && opts_.post_process_label < 0) {
    KALDI_ERR << "Invalid value " << opts_.post_process_label << " for option "
              << "--post-process-label. It must be non-negative.";
  }
}

bool SegmentationPostProcessor::PostProcess(Segmentation *seg) const {
  DoMergingLabels(seg);
  DoPaddingSegments(seg);
  DoShrinkingSegments(seg);
  DoBlendingShortSegments(seg);
  DoRemovingSegments(seg);
  DoMergingAdjacentSegments(seg);
  DoSplittingSegments(seg);

  return true;
}

void SegmentationPostProcessor::DoMergingLabels(Segmentation *seg) const {
  if (!IsMergingLabelsToBeDone(opts_)) return;
  MergeLabels(merge_labels_, opts_.merge_dst_label, seg);
}

void SegmentationPostProcessor::DoPaddingSegments(Segmentation *seg) const {
  if (!IsPaddingSegmentsToBeDone(opts_)) return;
  PadSegments(opts_.pad_label, opts_.pad_length, seg);
}

void SegmentationPostProcessor::DoShrinkingSegments(Segmentation *seg) const {
  if (!IsShrinkingSegmentsToBeDone(opts_)) return;
  ShrinkSegments(opts_.shrink_label, opts_.shrink_length, seg);
}

void SegmentationPostProcessor::DoBlendingShortSegments(
    Segmentation *seg) const {
  if (!IsBlendingShortSegmentsToBeDone(opts_)) return;
  BlendShortSegmentsWithNeighbors(opts_.blend_short_segments_class,
                                  opts_.max_blend_length,
                                  opts_.max_intersegment_length, seg);
}

void SegmentationPostProcessor::DoRemovingSegments(Segmentation *seg) const {
  if (!IsRemovingSegmentsToBeDone(opts_)) return;
  RemoveSegments(remove_labels_, opts_.max_remove_length, 
                 seg);
}

void SegmentationPostProcessor::DoMergingAdjacentSegments(
    Segmentation *seg) const {
  if (!IsMergingAdjacentSegmentsToBeDone(opts_)) return;
  MergeAdjacentSegments(opts_.max_intersegment_length, seg);
}

void SegmentationPostProcessor::DoSplittingSegments(Segmentation *seg) const {
  if (!IsSplittingSegmentsToBeDone(opts_)) return;
  SplitSegments(opts_.max_segment_length,
                opts_.max_segment_length / 2,
                opts_.overlap_length,
                opts_.post_process_label, seg);
}

} // end namespace segmenter
} // end namespace kaldi
