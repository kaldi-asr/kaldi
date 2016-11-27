// segmenterbin/segmentation-split-segments.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Split long segments optionally using alignment.\n"
        "The splitting works in two possible ways:\n"
        " 1) If alignment is not provided: The segments are split if they\n"
        "    are longer than --max-segment-length frames into overlapping\n"
        "    segments with an overlap of --overlap-length frames.\n"
        " 2) If alignment is provided: The segments are split if they\n"
        "    are longer than --max-segment-length frames at the region \n"
        "    where there is a contiguous segment of --ali-label in the \n"
        "    alignment that is at least --min-alignment-chunk-length frames \n"
        "    long.\n"
        "Usage: segmentation-split-segments [options] "
        "<segmentation-rspecifier> <segmentation-wspecifier>\n"
        "  or : segmentation-split-segments [options] "
        "<segmentation-rxfilename> <segmentation-wxfilename>\n"
        " e.g.: segmentation-split-segments --binary=false foo -\n"
        "       segmentation-split-segments ark:foo.seg ark,t:-\n"
        "See also: segmentation-post-process\n";

    bool binary = true;
    int32 max_segment_length = -1;
    int32 min_remainder = -1;
    int32 overlap_length = 0;
    int32 split_label = -1;
    int32 ali_label = 0;
    int32 min_alignment_chunk_length = 2;

    std::string alignments_in_fn;

    ParseOptions po(usage);

    po.Register("binary", &binary,
                "Write in binary mode "
                "(only relevant if output is a wxfilename)");
    po.Register("max-segment-length", &max_segment_length,
                "If segment is longer than this length, split it into "
                "pieces with less than these many frames. "
                "Refer to the SplitSegments() code for details. "
                "Used in conjunction with the option --overlap-length.");
    po.Register("min-remainder", &min_remainder,
                "The minimum remainder left after splitting that will "
                "prevent a splitting from begin done. "
                "Set to max-segment-length / 2, if not specified. "
                "Applicable only when alignments is not specified.");
    po.Register("overlap-length", &overlap_length,
                "When splitting segments longer than max-segment-length, "
                "have the pieces overlap by these many frames. "
                "Refer to the SplitSegments() code for details. "
                "Used in conjunction with the option --max-segment-length.");
    po.Register("split-label", &split_label,
                "If supplied, split only segments of these labels. "
                "Otherwise, split all segments.");
    po.Register("alignments", &alignments_in_fn,
                "A single alignment file or archive of alignment used "
                "for splitting, "
                "depending on whether the input segmentation is single file "
                "or archive");
    po.Register("ali-label", &ali_label,
                "Split at this label of alignments");
    po.Register("min-alignment-chunk-length", &min_alignment_chunk_length,
                "The minimum number of frames of alignment with ali_label "
                "at which to split the segments");

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

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

    if (min_remainder == -1) {
      min_remainder = max_segment_length / 2;
    }

    int64 num_done = 0, num_err = 0;

    if (!in_is_rspecifier) {
      std::vector<int32> ali;

      Segmentation segmentation;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        segmentation.Read(ki.Stream(), binary_in);
      }

      if (!alignments_in_fn.empty()) {
        {
          bool binary_in;
          Input ki(alignments_in_fn, &binary_in);
          ReadIntegerVector(ki.Stream(), binary_in, &ali);
        }
        SplitSegmentsUsingAlignment(max_segment_length,
                                    split_label, ali, ali_label,
                                    min_alignment_chunk_length,
                                    &segmentation);
      } else {
        SplitSegments(max_segment_length, min_remainder,
                      overlap_length, split_label, &segmentation);
      }

      Sort(&segmentation);

      {
        Output ko(segmentation_out_fn, binary);
        segmentation.Write(ko.Stream(), binary);
      }

      KALDI_LOG << "Split segmentation " << segmentation_in_fn
                << " and wrote " << segmentation_out_fn;
      return 0;
    }

    SegmentationWriter writer(segmentation_out_fn);
    SequentialSegmentationReader reader(segmentation_in_fn);
    RandomAccessInt32VectorReader ali_reader(alignments_in_fn);

    for (; !reader.Done(); reader.Next()) {
      Segmentation segmentation(reader.Value());
      const std::string &key = reader.Key();

      if (!alignments_in_fn.empty()) {
        if (!ali_reader.HasKey(key)) {
          KALDI_WARN << "Could not find key " << key
                     << " in alignments " << alignments_in_fn;
          num_err++;
          continue;
        }
        SplitSegmentsUsingAlignment(max_segment_length, split_label,
                                    ali_reader.Value(key), ali_label,
                                    min_alignment_chunk_length,
                                    &segmentation);
      } else {
        SplitSegments(max_segment_length, min_remainder,
                      overlap_length, split_label,
                      &segmentation);
      }

      Sort(&segmentation);

      writer.Write(key, segmentation);
      num_done++;
    }

    KALDI_LOG << "Successfully split " << num_done
              << " segmentations; "
              << "failed with " << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

