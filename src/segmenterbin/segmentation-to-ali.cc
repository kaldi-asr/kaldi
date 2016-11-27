// segmenterbin/segmentation-to-ali.cc

// Copyright 2015-16    Vimal Manohar (Johns Hopkins University)

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
        "Convert segmentation to alignment\n"
        "\n"
        "Usage: segmentation-to-ali [options] <segmentation-rspecifier> "
        "<ali-wspecifier>\n"
        " e.g.: segmentation-to-ali ark:1.seg ark:1.ali\n";

    std::string lengths_rspecifier;
    int32 default_label = 0, length_tolerance = 2;

    ParseOptions po(usage);

    po.Register("lengths-rspecifier", &lengths_rspecifier,
                "Archive of frame lengths "
                "of the utterances. Fills up any extra length with "
                "the specified default-label");
    po.Register("default-label", &default_label, "Fill any extra length "
                "with this label");
    po.Register("length-tolerance", &length_tolerance, "Tolerate shortage of "
                "this many frames in the specified lengths file");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_rspecifier = po.GetArg(1);
    std::string alignment_wspecifier = po.GetArg(2);

    RandomAccessInt32Reader lengths_reader(lengths_rspecifier);

    SequentialSegmentationReader segmentation_reader(segmentation_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !segmentation_reader.Done(); segmentation_reader.Next()) {
      const Segmentation &segmentation = segmentation_reader.Value();
      const std::string &key = segmentation_reader.Key();

      int32 length = -1;
      if (lengths_rspecifier != "") {
        if (!lengths_reader.HasKey(key)) {
          KALDI_WARN << "Could not find length for utterance " << key;
          num_err++;
          continue;
        }
        length = lengths_reader.Value(key);
      }

      std::vector<int32> ali;
      if (!ConvertToAlignment(segmentation, default_label, length,
                              length_tolerance, &ali)) {
        KALDI_WARN << "Conversion failed for utterance " << key;
        num_err++;
        continue;
      }
      alignment_writer.Write(key, ali);
      num_done++;
    }

    KALDI_LOG << "Converted " << num_done << " segmentations into alignments; "
              << "failed with " << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

