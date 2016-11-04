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
    
    void PushBack(const Segment &seg);

    SegmentList::iterator Insert(SegmentList::iterator it, 
                                 const Segment &seg);

    void EmplaceBack(int32 start_frame, int32 end_frame, int32 class_id);

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

    SegmentList::iterator MinElement();
    
    SegmentList::iterator MaxElement();
  
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
typedef SequentialTableReader<KaldiObjectHolder<Segmentation> > SequentialSegmentationReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Segmentation> > RandomAccessSegmentationReader;
typedef RandomAccessTableReaderMapped<KaldiObjectHolder<Segmentation> >  RandomAccessSegmentationReaderMapped;

} // end namespace segmenter
} // end namespace kaldi

#endif // KALDI_SEGMENTER_SEGMENTATION_H_
