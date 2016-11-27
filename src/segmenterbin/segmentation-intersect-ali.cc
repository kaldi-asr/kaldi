// segmenterbin/segmentation-intersect-ali.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Intersect (like sets) segmentation with an alignment and retain \n"
        "only segments where the alignment is the specified label. \n"
        "\n"
        "Usage: segmentation-intersect-alignment [options] "
        "<segmentation-rspecifier> <ali-rspecifier> "
        "<segmentation-wspecifier>\n"
        " e.g.: segmentation-intersect-alignment --binary=false ark:foo.seg "
        "ark:filter.ali ark,t:-\n"
        "See also: segmentation-combine-segments, "
        "segmentation-intersect-segments, segmentation-create-subsegments\n";

    ParseOptions po(usage);

    int32 ali_label = 0, min_alignment_chunk_length = 0;

    po.Register("ali-label", &ali_label,
                "Intersect only at this label of alignments");
    po.Register("min-alignment-chunk-length", &min_alignment_chunk_length,
                "The minimmum number of consecutive frames of ali_label in "
                "alignment at which the segments can be intersected.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_rspecifier = po.GetArg(1),
      ali_rspecifier = po.GetArg(2),
      segmentation_wspecifier = po.GetArg(3);

    int32 num_done = 0, num_err = 0;

    SegmentationWriter writer(segmentation_wspecifier);
    SequentialSegmentationReader segmentation_reader(segmentation_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(ali_rspecifier);

    for (; !segmentation_reader.Done(); segmentation_reader.Next()) {
      const Segmentation &segmentation = segmentation_reader.Value();
      const std::string &key = segmentation_reader.Key();

      if (!alignment_reader.HasKey(key)) {
        KALDI_WARN << "Could not find segmentation for key " << key
                   << " in " << ali_rspecifier;
        num_err++;
        continue;
      }
      const std::vector<int32> &ali = alignment_reader.Value(key);

      Segmentation out_segmentation;
      IntersectSegmentationAndAlignment(segmentation, ali, ali_label,
                                        min_alignment_chunk_length,
                                        &out_segmentation);
      out_segmentation.Sort();

      writer.Write(key, out_segmentation);
      num_done++;
    }

    KALDI_LOG << "Intersected " << num_done
              << " segmentations with alignments; failed with "
              << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

