// segmenterbin/segmentation-post-process.cc

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
#include "segmenter/segmentation-post-processor.h"
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Post processing of segmentation that does the following operations "
        "in order: \n"
        "1) Merge labels: Merge labels specified in --merge-labels into a "
        "single label specified by --merge-dst-label. \n"
        "2) Padding segments: Pad segments of label specified by --pad-label "
        "by a few frames as specified by --pad-length. \n"
        "3) Shrink segments: Shrink segments of label specified by "
        "--shrink-label by a few frames as specified by --shrink-length. \n"
        "4) Blend segments with neighbors: Blend short segments of class-id "
        "specified by --blend-short-segments-class that are "
        "shorter than --max-blend-length frames with their "
        "respective neighbors if both the neighbors are within "
        "a distance of --max-intersegment-length frames.\n"
        "5) Remove segments: Remove segments of class-ids contained "
        "in --remove-labels.\n"
        "6) Merge adjacent segments: Merge adjacent segments of the same "
        "label if they are within a distance of --max-intersegment-length "
        "frames.\n"
        "7) Split segments: Split segments that are longer than "
        "--max-segment-length frames into overlapping segments "
        "with an overlap of --overlap-length frames. \n"
        "Usage: segmentation-post-process [options] <segmentation-rspecifier> "
        "<segmentation-wspecifier>\n"
        "  or : segmentation-post-process [options] <segmentation-rxfilename> "
        "<segmentation-wxfilename>\n"
        " e.g.: segmentation-post-process --binary=false foo -\n"
        "       segmentation-post-process ark:foo.seg ark,t:-\n"
        "See also: segmentation-merge, segmentation-copy, "
        "segmentation-remove-segments\n";

    bool binary = true;

    ParseOptions po(usage);

    SegmentationPostProcessingOptions opts;

    po.Register("binary", &binary,
                "Write in binary mode "
                "(only relevant if output is a wxfilename)");

    opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    SegmentationPostProcessor post_processor(opts);

    std::string segmentation_in_fn = po.GetArg(1),
      segmentation_out_fn = po.GetArg(2);

    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";

    int64 num_done = 0, num_err = 0;

    if (!in_is_rspecifier) {
      Segmentation segmentation;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        segmentation.Read(ki.Stream(), binary_in);
      }
      if (post_processor.PostProcess(&segmentation)) {
        Output ko(segmentation_out_fn, binary);
        Sort(&segmentation);
        segmentation.Write(ko.Stream(), binary);
        KALDI_LOG << "Post-processed segmentation " << segmentation_in_fn
                  << " and wrote " << segmentation_out_fn;
        return 0;
      }
      KALDI_LOG << "Failed post-processing segmentation "
                << segmentation_in_fn;
      return 1;
    }

    SegmentationWriter writer(segmentation_out_fn);
    SequentialSegmentationReader reader(segmentation_in_fn);
    for (; !reader.Done(); reader.Next()) {
      Segmentation segmentation(reader.Value());
      const std::string &key = reader.Key();

      if (!post_processor.PostProcess(&segmentation)) {
        num_err++;
        continue;
      }

      Sort(&segmentation);

      writer.Write(key, segmentation);
      num_done++;
    }

    KALDI_LOG << "Successfully post-processed " << num_done
              << " segmentations; "
              << "failed with " << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

