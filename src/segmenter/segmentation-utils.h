// segmenter/segmentation-utils.h

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

#ifndef KALDI_SEGMENTER_SEGMENTATION_UTILS_H_
#define KALDI_SEGMENTER_SEGMENTATION_UTILS_H_

#include "segmenter/segmentation.h"

namespace kaldi {
namespace segmenter {

/**
 * This function is very straight forward. It just merges the labels in
 * merge_labels to the class-id dest_label. This means any segment that
 * originally had the class-id as any of the labels in merge_labels would end
 * up having the class-id dest_label.
 **/
void MergeLabels(const std::vector<int32> &merge_labels,
                 int32 dest_label, Segmentation *segmentation);

// Relabel segments using a map from old to new label.
// Some special functionality is as follows:
// 1) If segment label is not found in the map, the function exits with
// an error, unless it has a default_label defined as follows.
// 2) If label_map contains a mapping from -1 to a new label (other than -1),
// then that new label is the default_label. Any segment whose label is not
// found in the label_map is assigned the default_label.
// 3) Any segment with new label -1 is removed from the segmentation.
void RelabelSegmentsUsingMap(const unordered_map<int32, int32> &label_map,
                             Segmentation *segmentation);

// Relabel all segments to class-id label
void RelabelAllSegments(int32 label, Segmentation *segmentation);

/**
 * Scale frame shift by this factor. 
 * Scales both start_time and end_time of the segments.
 * Usually frame length is 0.01 and frame shift 0.015. But sometimes
 * the alignments are obtained using a subsampling factor of 3. This
 * function can be used to maintain consistency among different
 * alignments and segmentations.
 **/
void ScaleFrameShift(BaseFloat factor, Segmentation *segmentation);

/**
 * This is very straight forward. It removes all segments of label "label"
**/
void RemoveSegments(int32 label, Segmentation *segmentation);

/**
 * This function removes any segment whose label is
 * contained in the vector "labels" and has a length smaller than
 * max_remove_length. 
 * max_remove_length can be provided -1, which has a special meaning of +inf
 * i.e. to remove segments based on only the labels and irrespective of the
 * lengths.
**/
void RemoveSegments(const std::vector<int32> &labels,
                    int32 max_remove_length,
                    Segmentation *segmentation);

/**
 * This function removes segments that are shorter than min_length frames.
 **/
void RemoveShortSegments(int32 label, int32 min_length,
                         Segmentation *segmentation);

// Keep only segments of label "label"
void KeepSegments(int32 label, Segmentation *segmentation);

/**
 * This function splits an input segmentation in_segmentation into pieces of
 * approximately segment_length. Each piece is given the same class id as the
 * original segment.
 *
 * The way this function is written is that it first figures out the number of
 * pieces that the segment must be broken into. Then it creates that many pieces
 * of equal size (actual_segment_length). This mimics some of the approaches
 * used at script level
**/
void SplitInputSegmentation(const Segmentation &in_segmentation,
                            int32 segment_length,
                            Segmentation *out_segmentation);

/**
 * This function splits the segments in the the segmentation
 * into pieces of length segment_length.
 * But if the last remaining piece is smaller than min_remainder, then the last
 * piece is merged to the piece before it, resulting in a piece that is of
 * length < segment_length + min_remainder.
 * If overlap_length > 0, then the created pieces overlap by these many frames.
 * If segment_label == -1, then all segments are split.
 * Otherwise, only the segments with this label are split.
 *
 * The way this function works it is it looks at the current segment length and
 * checks if it is larger than segment_length + min_remainder. If it is larger,
 * then it must be split. To do this, it first modifies the start_frame of
 * the current frame to start_frame + segment_length - overlap.
 * It then creates a new segment of length segment_length from the original
 * start_frame to start_frame + segment_length - 1 and adds it just before the
 * current segment. So in the next iteration, we would actually be back to the
 * same segment, but whose start_frame had just been modified.
**/
void SplitSegments(int32 segment_length,
                   int32 min_remainder, int32 overlap_length,
                   int32 segment_label,
                   Segmentation *segmentation);

/**
 * Split this segmentation into pieces of size segment_length,
 * but only if possible by creating split points at the
 * middle of the chunk where alignment == ali_label and
 * the chunk is at least min_segment_length frames long
 *
 * min_remainder, segment_label serve the same purpose as in the
 * above SplitSegments function.
**/
void SplitSegmentsUsingAlignment(int32 segment_length,
                                 int32 segment_label,
                                 const std::vector<int32> &alignment,
                                 int32 alignment_label,
                                 int32 min_align_chunk_length,
                                 Segmentation *segmentation);

/**
 * This function is used to merge segments next to each other in the SegmentList
 * and within a distance of max_intersegment_length frames from each other,
 * provided the segments are of the same label.
 * This function requires the segmentation to be sorted before passing it.
 **/
void MergeAdjacentSegments(int32 max_intersegment_length,
                           Segmentation *segmentation,
                           bool sort = true);

/**
 * This function is used to pad segments of label "label" by "length"
 * frames on either side of the segment.
 * This is useful to pad segments of speech by a few frames.
**/
void PadSegments(int32 label, int32 length, Segmentation *segmentation);

/**
 * This function is used to shrink segments of class_id "label" by "length"
 * frames on either side of the segment.
 * If the whole segment is smaller than 2*length, then the segment is
 * removed entirely.
**/
void ShrinkSegments(int32 label, int32 length, Segmentation *segmentation);

/**
 * This function blends segments of label "label" that are shorter than
 * "max_length" frames, provided the segments before and after it are of the
 * same label "other_label" and the distance to the neighbor is less than
 * "max_intersegment_distance".
 * After blending, the three segments have the same label "other_label" and
 * hence can be merged into a composite segment.
 * An example where this is useful is when there is a short segment of silence
 * with speech segments on either sides. Then the short segment of silence is
 * removed and called speech instead. The three continguous segments of speech
 * are merged into a single composite segment.
**/
void BlendShortSegmentsWithNeighbors(int32 label, int32 max_length,
                                     int32 max_intersegment_distance,
                                     Segmentation *segmentation,
                                     bool sort = true);

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
 * length        : the number of frames required in the alignment.
 *                 If set to -1, then this length is ignored.
 *                 In most applications, the length of the alignment required is
 *                 known.  Usually it must match the length of the features
 *                 (obtained using feat-to-len). Then the alignment is resized
 *                 to this length and filled with default_label. The segments
 *                 are then read and the frames corresponding to the segments
 *                 are relabeled with the class_id of the respective segments.
 * tolerance     : the tolerance in number of frames that we allow for the
 *                 frame index corresponding to the end_frame of the last
 *                 segment. Applicable when length != -1.
 *                 Since, we use 25 ms widows with 10 ms frame shift,
 *                 it is possible that the features length is 2 frames less than
 *                 the end of the last segment. So the user can set the
 *                 tolerance to 2 in order to avoid returning with error in this
 *                 function.
 * Function returns true is successful.
**/
bool ConvertToAlignment(const Segmentation &segmentation,
                        int32 default_label, int32 length,
                        int32 tolerance,
                        std::vector<int32> *alignment);

/**
 * Insert segments created from alignment starting from frame index "start"
 * until and excluding frame index "end".
 * The inserted segments are shifted by "start_time_offset".
 * "start_time_offset" is useful when the "alignment" is per-utterance, in which
 * case the start time of the utterance can be provided as the
 * "start_time_offset"
 * The function returns the number of segments created.
 * If "frame_counts_per_class" is provided, then the number of frames per class
 * is accumulated there.
**/
int32 InsertFromAlignment(const std::vector<int32> &alignment,
                          int32 start, int32 end,
                          int32 start_time_offset,
                          Segmentation *segmentation,
                          std::map<int32, int64> *frame_counts_per_class = NULL);

/**
 * Insert segments from in_segmentation, but shift them by
 * start_time offset.
 * If sort is true, then the final segmentation is sorted.
 * It is useful in some applications to set sort to false.
 * Returns number of segments inserted.
**/
int32 InsertFromSegmentation(const Segmentation &in_segmentation,
                             int32 start_time_offset, bool sort,
                             Segmentation *segmentation,
                             std::vector<int64> *frame_counts_per_class = NULL);

/**
 * Extend a segmentation by adding another one.
 * If "sort" is set to true, then resultant segmentation would be sorted.
 * If its known that the other segmentation must all be after this segmentation,
 * then the user may set "sort" false.
**/
void ExtendSegmentation(const Segmentation &in_segmentation, bool sort,
                        Segmentation *segmentation);

/**
 * This function is used to get per-frame count of number of classes.
 * The output is in the format of a vector of maps.
 * class_counts_per_frame: A pointer to a vector of maps used to get the output.
 *                         The size of the vector is the number of frames.
 *                         For each frame, there is a map from the "class_id"
 *                         to the number of segments where the label the
 *                         corresponding "class_id".
 *                         The size of the map gives the number of unique
 *                         labels in this frame e.g. number of speakers.
 *                         The count for each "class_id" is the number
 *                         of segments with that "class_id" at that frame.
 * length        : the number of frames required in the output.
 *                 In most applications, this length is known.
 *                 Usually it must match the length of the features (obtained
 *                 using feat-to-len). Then the output is resized to this
 *                 length. The map is empty for frames where no segments are
 *                 seen.
 * tolerance     : the tolerance in number of frames that we allow for the
 *                 frame index corresponding to the end_frame of the last
 *                 segment. Since, we use 25 ms widows with 10 ms frame shift,
 *                 it is possible that the features length is 2 frames less than
 *                 the end of the last segment. So the user can set the
 *                 tolerance to 2 in order to avoid returning an error in this
 *                 function.
 * Function returns true is successful.
**/
bool GetClassCountsPerFrame(
    const Segmentation &segmentation,
    int32 length, int32 tolerance,
    std::vector<std::map<int32, int32> > *class_counts_per_frame);

// Checks if segmentation is non-overlapping
bool IsNonOverlapping(const Segmentation &segmentation);

// Check if segmentation is sorted
bool IsSorted(const Segmentation &segmentation);

// Sorts segments on start frame.
void Sort(Segmentation *segmentation);

// Truncate segmentation to "length".
// Removes any segments with "start_time" >= "length"
// and truncates any segments with "end_time" >= "length"
void TruncateToLength(int32 length, Segmentation *segmentation);

} // end namespace segmenter
} // end namespace kaldi

#endif // KALDI_SEGMENTER_SEGMENTATION_UTILS_H_
