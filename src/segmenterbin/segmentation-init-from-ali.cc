// segmenterbin/segmentation-init-from-ali.cc

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
        "Initialize utterance-level segmentations from alignments file. \n"
        "The user can pass this to segmentation-combine-segments to "
        "create recording-level segmentations."
        "\n"
        "Usage: segmentation-init-from-ali [options] "
        "<ali-rspecifier> <segmentation-wspecifier> \n"
        " e.g.: segmentation-init-from-ali ark:1.ali ark:-\n"
        "See also: segmentation-init-from-segments, "
        "segmentation-combine-segments\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ali_rspecifier = po.GetArg(1),
       segmentation_wspecifier = po.GetArg(2);

    SegmentationWriter segmentation_writer(segmentation_wspecifier);

    int32 num_done = 0, num_segmentations = 0;
    int64 num_segments = 0;
    int64 num_err = 0;

    std::map<int32, int64> frame_counts_per_class;

    SequentialInt32VectorReader alignment_reader(ali_rspecifier);

    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      const std::string &key = alignment_reader.Key();
      const std::vector<int32> &alignment = alignment_reader.Value();

      Segmentation segmentation;

      num_segments += InsertFromAlignment(alignment, 0, alignment.size(),
                                          0, &segmentation,
                                          &frame_counts_per_class);

      Sort(&segmentation);
      segmentation_writer.Write(key, segmentation);

      num_done++;
      num_segmentations++;
    }

    KALDI_LOG << "Processed " << num_done << " utterances; failed with "
              << num_err << " utterances; "
              << "wrote " << num_segmentations << " segmentations "
              << "with a total of " << num_segments << " segments.";
    KALDI_LOG << "Number of frames for the different classes are : ";

    std::map<int32, int64>::const_iterator it = frame_counts_per_class.begin(); 
    for (; it != frame_counts_per_class.end(); ++it) {
      KALDI_LOG << it->first << " " << it->second << " ; "; 
    }

    return ((num_done > 0 && num_err < num_done) ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

