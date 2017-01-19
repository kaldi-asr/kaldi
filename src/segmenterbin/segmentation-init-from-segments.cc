// segmenterbin/segmentation-init-from-segments.cc

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
#include "segmenter/segmentation.h"

// If segments file contains
// Alpha-001 Alpha 0.00 0.16
// Alpha-002 Alpha 1.50 4.10
// Beta-001 Beta 0.50 2.66
// Beta-002 Beta 3.50 5.20
// the output segmentation will contain
// Alpha-001 [ 0 15 1 ]
// Alpha-002 [ 0 359 1 ]
// Beta-001 [ 0 215 1 ]
// Beta-002 [ 0 169 1 ]
// If --shift-to-zero=false is provided, then the output will contain
// Alpha-001 [ 0 15 1 ]
// Alpha-002 [ 150 409 1 ]
// Beta-001 [ 50 265 1 ]
// Beta-002 [ 350 519 1 ]
//
// If the following utt2label-rspecifier was provided:
// Alpha-001 2
// Alpha-002 2
// Beta-001 4
// Beta-002 4
// then the output segmentation will contain
// Alpha-001 [ 0 15 2 ]
// Alpha-002 [ 0 359 2 ]
// Beta-001 [ 0 215 4 ]
// Beta-002 [ 0 169 4 ]

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Convert segments from segments file into utterance-level "
        "segmentation format. \n"
        "The user can convert the segmenation to recording-level using "
        "the binary segmentation-combine-segments-to-recording.\n"
        "\n"
        "Usage: segmentation-init-from-segments [options] "
        "<segments-rxfilename> <segmentation-wspecifier> \n"
        " e.g.: segmentation-init-from-segments segments ark:-\n";

    int32 segment_label = 1;
    BaseFloat frame_shift = 0.01, frame_overlap = 0.015;
    std::string utt2label_rspecifier;
    bool shift_to_zero = true;

    ParseOptions po(usage);

    po.Register("segment-label", &segment_label,
                "Label for all the segments in the segmentations");
    po.Register("utt2label-rspecifier", &utt2label_rspecifier,
                "Mapping for each utterance to an integer label. "
                "If supplied, these labels will be used as the segment "
                "labels");
    po.Register("shift-to-zero", &shift_to_zero,
                "Shift all segments to 0th frame");
    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");
    po.Register("frame-overlap", &frame_overlap, "Frame overlap in seconds");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segments_rxfilename = po.GetArg(1),
                segmentation_wspecifier = po.GetArg(2);

    SegmentationWriter writer(segmentation_wspecifier);
    RandomAccessInt32Reader utt2label_reader(utt2label_rspecifier);

    Input ki(segments_rxfilename);

    int64 num_lines = 0, num_done = 0;

    std::string line;

    while (std::getline(ki.Stream(), line)) {
      num_lines++;

      std::vector<std::string> split_line;
      // Split the line by space or tab and check the number of fields in each
      // line. There must be 4 fields--segment name , reacording wav file name,
      // start time, end time; 5th field (channel info) is optional.
      SplitStringToVector(line, " \t\r", true, &split_line);
      if (split_line.size() != 4 && split_line.size() != 5) {
        KALDI_WARN << "Invalid line in segments file: " << line;
        continue;
      }
      std::string utt = split_line[0],
                 reco = split_line[1],
            start_str = split_line[2],
              end_str = split_line[3];

      // Convert the start time and endtime to real from string. Segment is
      // ignored if start or end time cannot be converted to real.
      double start, end;
      if (!ConvertStringToReal(start_str, &start)) {
        KALDI_WARN << "Invalid line in segments file [bad start]: " << line;
        continue;
      }
      if (!ConvertStringToReal(end_str, &end)) {
        KALDI_WARN << "Invalid line in segments file [bad end]: " << line;
        continue;
      }

      // start time must not be negative; start time must not be greater than
      // end time, except if end time is -1
      if (start < 0 || (end != -1.0 && end <= 0) ||
          ((start >= end) && (end > 0))) {
        KALDI_WARN << "Invalid line in segments file "
                   << "[empty or invalid segment]: " << line;
        continue;
      }

      if (split_line.size() >= 5)
        KALDI_ERR << "Not supporting channel in segments file";

      Segmentation segmentation;

      if (!utt2label_rspecifier.empty()) {
        if (!utt2label_reader.HasKey(utt)) {
          KALDI_WARN << "Could not find utterance " << utt << " in "
                     << utt2label_rspecifier;
          continue;
        }

        segment_label = utt2label_reader.Value(utt);
      }

      if (shift_to_zero) {
        int32 last_frame = (end-frame_overlap) / frame_shift 
                           - start / frame_shift - 1;
        segmentation.EmplaceBack(0, last_frame, segment_label);
      } else {
        segmentation.EmplaceBack(
            static_cast<int32>(start / frame_shift + 0.5),
            static_cast<int32>((end-frame_overlap) / frame_shift - 0.5),
            segment_label);
      }

      writer.Write(utt, segmentation);
      num_done++;
    }

    KALDI_LOG << "Successfully processed " << num_done << " lines out of "
              << num_lines << " in the segments file";

    return (num_done > num_lines / 2 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

