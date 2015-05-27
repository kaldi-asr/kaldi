// segmenterbin/segmentation-init-from-segments.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/segmenter.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Initialize segmentations from segments file\n"
        "\n"
        "Usage: segmentation-init-from-segments [options] segments-rxfilename segmentation-out-wspecifier \n"
        " e.g.: segmentation-init-from-segments segments ark:-\n";
    
    bool binary = true, per_utt = false;
    int32 label = 1;
    BaseFloat frame_shift = 0.01;

    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("label", &label, "Label for all the segments");
    po.Register("per-utt", &per_utt, "Get segmentation per utterance instead of "
                "per file");
    po.Register("frame-shift", &frame_shift, "Frame shift");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segments_rxfilename = po.GetArg(1),
        segmentation_wspecifier = po.GetArg(2);
    
    Input ki(segments_rxfilename);
    SegmentationWriter writer(segmentation_wspecifier);

    int32 num_lines = 0, num_success = 0, num_segmentations = 0;

    std::string line, prev_recording;
    Segmentation seg;
    
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
      std::string segment = split_line[0],
          recording = split_line[1],
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
      if (start < 0 || (end != -1.0 && end <= 0) || ((start >= end) && (end > 0))) {
        KALDI_WARN << "Invalid line in segments file [empty or invalid segment]: "
                   << line;
        continue;
      }

      if (split_line.size() >= 5) 
        KALDI_ERR << "Not supporting channel in segments file";

      if (!per_utt) {
        if (prev_recording != "" && prev_recording != recording) {
          writer.Write(prev_recording, seg);
          num_segmentations++;
          seg.Clear();
        }
        seg.Emplace(std::round(start / frame_shift), 
                    std::round(end / frame_shift), label);
      }

      if (per_utt) {
        seg.Emplace(0.0, std::round((end - start)/ frame_shift), label);
        writer.Write(segment, seg);
        num_segmentations++;
        seg.Clear();
      }

      prev_recording = recording;
      num_success++;
    }
    
    if (!per_utt) {
      writer.Write(prev_recording, seg);
      num_segmentations++;
    }

    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the segments file; wrote "
              << num_segmentations << " segmentations.";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}




