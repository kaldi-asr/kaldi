// segmenter/segmentation.h

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

#ifndef KALDI_SEGMENTER_SEGMENTATION_H_
#define KALDI_SEGMENTER_SEGMENTATION_H_

#include <list>
#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "util/kaldi-table.h"
#include "segmenter/segment.h"

namespace kaldi {
namespace segmenter {

// Segments are stored as a doubly-linked-list. This could be changed later
// if needed. Hence defining a typedef SegmentList.
typedef std::list<Segment> SegmentList;

// Declare class
class SegmentationPostProcessor;

/**
 * The main class to store segmentation and do operations on it. The segments
 * are stored in the structure SegmentList, which is currently a doubly-linked
 * list.
 * See the .cc file for details of implementation of the different functions.
 * This file gives only a small description of the functions.
**/

class Segmentation {
 public:
  // Inserts the segment at the back of the list.
  void PushBack(const Segment &seg);

  // Inserts the segment before the segment at the position specified by the
  // iterator "it".
  SegmentList::iterator Insert(SegmentList::iterator it,
                               const Segment &seg);

  // The following function is a wrapper to the
  // emplace_back functionality of a STL list of Segments
  // and inserts a new segment to the back of the list.
  void EmplaceBack(int32 start_frame, int32 end_frame, int32 class_id);

  // The following function is a wrapper to the
  // emplace functionality of a STL list of segments
  // and inserts a segment at the position specified by the iterator "it".
  // Returns an iterator to the inserted segment.
  SegmentList::iterator Emplace(SegmentList::iterator it,
                                int32 start_frame, int32 end_frame,
                                int32 class_id);

  // Call erase operation on the SegmentList and returns the iterator pointing
  // to the next segment in the SegmentList and also decrements dim_.
  SegmentList::iterator Erase(SegmentList::iterator it);

  // Reset segmentation i.e. clear all values
  void Clear();

  // Read segmentation object from input stream
  void Read(std::istream &is, bool binary);

  // Write segmentation object to output stream
  void Write(std::ostream &os, bool binary) const;

  // Check if all segments have class_id >=0 and if dim_ matches the number of
  // segments.
  void Check() const;

  // Sort the segments on the start_frame
  void Sort();

  // Sort the segments on the length
  void SortByLength();

  // Returns an iterator to the smallest segment akin to std::min_element
  SegmentList::iterator MinElement();

  // Returns an iterator to the largest segment akin to std::max_element
  SegmentList::iterator MaxElement();

  // Generate a random segmentation for debugging purposes.
  // Arguments:
  //  max_length: The maximum length of the random segmentation to be
  //              generated.
  //  max_segment_length: Maximum length of a segment in the segmentation
  //  num_classes: Maximum number of classes in the generated segmentation
  void GenRandomSegmentation(int32 max_length, int32 max_segment_length,
                             int32 num_classes);

  // Public accessors
  inline int32 Dim() const { return dim_; }
  SegmentList::iterator Begin() { return segments_.begin(); }
  SegmentList::const_iterator Begin() const { return segments_.begin(); }
  SegmentList::iterator End() { return segments_.end(); }
  SegmentList::const_iterator End() const { return segments_.end(); }

  Segment& Back() { return segments_.back(); }
  const Segment& Back() const { return segments_.back(); }

  const SegmentList* Data() const { return &segments_; }

  // Default constructor
  Segmentation();

 private:
  // number of segments in the segmentation
  int32 dim_;

  // list of segments in the segmentation
  SegmentList segments_;

  friend class SegmentationPostProcessor;
};

typedef TableWriter<KaldiObjectHolder<Segmentation> > SegmentationWriter;
typedef SequentialTableReader<KaldiObjectHolder<Segmentation> >
  SequentialSegmentationReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Segmentation> >
  RandomAccessSegmentationReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<Segmentation> >
  RandomAccessSegmentationReaderMapped;

} // end namespace segmenter
} // end namespace kaldi

#endif  // KALDI_SEGMENTER_SEGMENTATION_H_
