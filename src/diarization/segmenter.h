struct Segment {
  int32 start_frame;
  int32 end_frame;
  ClassId class_id;
};

namespace kaldi {

  namespace Segmenter {
    void SplitSegments(const list<Segment> &in_segments, 
                       list<Segment> *out_segments);
  }
}

