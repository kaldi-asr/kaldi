#include "segmenter/segment.h"

namespace kaldi {
namespace segmenter {

void Segment::Write(std::ostream &os, bool binary) const {
  if (binary) {
    os.write(reinterpret_cast<const char *>(&start_frame), sizeof(start_frame));
    os.write(reinterpret_cast<const char *>(&end_frame), sizeof(start_frame));
    os.write(reinterpret_cast<const char *>(&class_id), sizeof(class_id));
  } else {
    WriteBasicType(os, binary, start_frame);
    WriteBasicType(os, binary, end_frame);
    WriteBasicType(os, binary, Label());
  }
}

void Segment::Read(std::istream &is, bool binary) {
  if (binary) {
    is.read(reinterpret_cast<char *>(&start_frame), sizeof(start_frame));
    is.read(reinterpret_cast<char *>(&end_frame), sizeof(end_frame));
    is.read(reinterpret_cast<char *>(&class_id), sizeof(class_id));
  } else {
    ReadBasicType(is, binary, &start_frame);
    ReadBasicType(is, binary, &end_frame);
    int32 label;
    ReadBasicType(is, binary, &label);
    SetLabel(label);
  }
    
  KALDI_ASSERT(end_frame >= start_frame && start_frame >= 0);
}

std::ostream& operator<<(std::ostream& os, const Segment &seg) {
  os << "[ ";
  seg.Write(os, false);
  os << "]";
  return os;
}  

} // end namespace segmenter 
} // end namespace kaldi
