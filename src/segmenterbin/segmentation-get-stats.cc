// segmenterbin/segmentation-get-per-frame-stats.cc

// Copyright 2016   Vimal Manohar (Johns Hopkins University)

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

#include <algorithm>
#include "base/kaldi-common.h"
#include "hmm/posterior.h"
#include "util/common-utils.h"
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Get per-frame stats from segmentation. \n"
        "Currently supported stats are \n"
        " num-overlaps: Number of overlapping segments common to this frame\n"
        " num-classes: Number of distinct classes common to this frame\n"
        "\n"
        "Usage: segmentation-get-stats [options] <segmentation-rspecifier> "
        "<num-overlaps-wspecifier> <num-classes-wspecifier> "
        "<class-counts-per-frame-wspecifier>\n"
        " e.g.: segmentation-get-stats ark:1.seg ark:/dev/null "
        "ark:num_classes.ark ark:/dev/null\n";

    ParseOptions po(usage);

    std::string lengths_rspecifier;
    int32 length_tolerance = 2;

    po.Register("lengths-rspecifier", &lengths_rspecifier,
                "Archive of frame lengths of the utterances. "
                "Fills up any extra length with zero stats.");
    po.Register("length-tolerance", &length_tolerance,
                "Tolerate shortage of this many frames in the specified "
                "lengths file");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_rspecifier = po.GetArg(1),
      num_overlaps_wspecifier = po.GetArg(2),
      num_classes_wspecifier = po.GetArg(3),
      class_counts_per_frame_wspecifier = po.GetArg(4);

    int64 num_done = 0, num_err = 0;

    SequentialSegmentationReader reader(segmentation_rspecifier);
    Int32VectorWriter num_overlaps_writer(num_overlaps_wspecifier);
    Int32VectorWriter num_classes_writer(num_classes_wspecifier);
    PosteriorWriter class_counts_per_frame_writer(
        class_counts_per_frame_wspecifier);

    RandomAccessInt32Reader lengths_reader(lengths_rspecifier);

    for (; !reader.Done(); reader.Next(), num_done++) {
      const Segmentation &segmentation = reader.Value();
      const std::string &key = reader.Key();

      int32 length = -1;
      if (!lengths_rspecifier.empty()) {
        if (!lengths_reader.HasKey(key)) {
          KALDI_WARN << "Could not find length for key " << key;
          num_err++;
          continue;
        }
        length = lengths_reader.Value(key);
      }

      std::vector<std::map<int32, int32> > class_counts_map_per_frame;
      if (!GetClassCountsPerFrame(segmentation, length,
                                  length_tolerance,
                                  &class_counts_map_per_frame)) {
        KALDI_WARN << "Failed getting stats for key " << key;
        num_err++;
        continue;
      }

      if (length == -1)
        length = class_counts_map_per_frame.size();

      std::vector<int32> num_classes_per_frame(length, 0);
      std::vector<int32> num_overlaps_per_frame(length, 0);
      Posterior class_counts_per_frame(length, 
          std::vector<std::pair<int32, BaseFloat> >());

      for (int32 i = 0; i < class_counts_map_per_frame.size(); i++) {
        std::map<int32, int32> &class_counts = class_counts_map_per_frame[i];

        for (std::map<int32, int32>::const_iterator it = class_counts.begin();
              it != class_counts.end(); ++it) {
          if (it->second > 0) {
            num_classes_per_frame[i]++;
            class_counts_per_frame[i].push_back(
                std::make_pair(it->first, it->second));
          }
          num_overlaps_per_frame[i] += it->second;
        }
        std::sort(class_counts_per_frame[i].begin(), 
                  class_counts_per_frame[i].end());
      }

      num_classes_writer.Write(key, num_classes_per_frame);
      num_overlaps_writer.Write(key, num_overlaps_per_frame);
      class_counts_per_frame_writer.Write(key, class_counts_per_frame);

      num_done++;
    }

    KALDI_LOG << "Got stats for " << num_done << " segmentations; failed with "
              << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

