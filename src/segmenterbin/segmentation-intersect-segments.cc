// segmenterbin/segmentation-intersect-segments.cc

// Copyright 2015-16   Vimal Manohar (Johns Hopkins University)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "segmenter/segmentation-utils.h"

namespace kaldi {
namespace segmenter {

void IntersectSegmentationsNonOverlapping(
    const Segmentation &in_segmentation,
    const Segmentation &secondary_segmentation,
    int32 mismatch_label,
    Segmentation *out_segmentation) {
  KALDI_ASSERT(out_segmentation);
  KALDI_ASSERT(secondary_segmentation.Dim() > 0);

  std::vector<int32> alignment;
  ConvertToAlignment(secondary_segmentation, -1, -1, 0, &alignment);

  for (SegmentList::const_iterator it = in_segmentation.Begin();
        it != in_segmentation.End(); ++it) {
    if (it->end_frame >= alignment.size()) {
      alignment.resize(it->end_frame + 1, -1);
    }
    Segmentation filter_segmentation;
    InsertFromAlignment(alignment, it->start_frame, it->end_frame + 1,
                        0, &filter_segmentation, NULL);

    for (SegmentList::const_iterator f_it = filter_segmentation.Begin();
          f_it != filter_segmentation.End(); ++f_it) {
      int32 label = it->Label();
      if (f_it->Label() != it->Label()) {
        if (mismatch_label == -1) continue;
        label = mismatch_label;
      }

      out_segmentation->EmplaceBack(f_it->start_frame, f_it->end_frame,
                                    label);
    }
  }
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Intersect segments from two archives by retaining only regions .\n"
        "where the primary and secondary segments match on label\n"
        "\n"
        "Usage: segmentation-intersect-segments [options] "
        "<primary-segmentation-rpecifier> <secondary-segmentation-rspecifier> "
        "<segmentation-wspecifier>\n"
        " e.g.: segmentation-intersect-segments ark:foo.seg ark:bar.seg "
        "ark,t:-\n"
        "See also: segmentation-create-subsegments, "
        "segmentation-intersect-ali\n";

    int32 mismatch_label = -1;
    bool assume_non_overlapping_secondary = true;

    ParseOptions po(usage);

    po.Register("mismatch-label", &mismatch_label,
                "Intersect only where secondary segment has this label");
    po.Register("assume-non-overlapping-secondary", &
                assume_non_overlapping_secondary,
                "Assume secondary segments are non-overlapping");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string primary_rspecifier = po.GetArg(1),
      secondary_rspecifier = po.GetArg(2),
      segmentation_writer = po.GetArg(3);

    if (!assume_non_overlapping_secondary) {
      KALDI_ERR << "Secondary segment must be non-overlapping for now";
    }

    int64 num_done = 0, num_err = 0;

    SegmentationWriter writer(segmentation_writer);
    SequentialSegmentationReader primary_reader(primary_rspecifier);
    RandomAccessSegmentationReader secondary_reader(secondary_rspecifier);

    for (; !primary_reader.Done(); primary_reader.Next()) {
      const Segmentation &segmentation = primary_reader.Value();
      const std::string &key = primary_reader.Key();

      if (!secondary_reader.HasKey(key)) {
        KALDI_WARN << "Could not find segmentation for key " << key
                   << " in " << secondary_rspecifier;
        num_err++;
        continue;
      }
      const Segmentation &secondary_segmentation = secondary_reader.Value(key);

      Segmentation out_segmentation;
      IntersectSegmentationsNonOverlapping(segmentation,
                                           secondary_segmentation,
                                           mismatch_label,
                                           &out_segmentation);

      Sort(&out_segmentation);

      writer.Write(key, out_segmentation);
      num_done++;
    }

    KALDI_LOG << "Intersected " << num_done << " segmentations; failed with "
              << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

