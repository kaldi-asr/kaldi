// segmenterbin/segmentation-to-segments.cc

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

#include <iomanip>
#include <iostream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "segmenter/segmentation.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Convert segmentation to a segments file and utt2spk file."
        "Assumes that the input segmentations are indexed by reco-id and "
        "treats speakers from different recording as distinct speakers."
        "\n"
        "Usage: segmentation-to-segments [options] <segmentation-rspecifier> "
        "<utt2spk-wspecifier> <segments-wxfilename>\n"
        " e.g.: segmentation-to-segments ark:foo.seg ark,t:utt2spk segments\n";

    BaseFloat frame_shift = 0.01, frame_overlap = 0.015;
    bool single_speaker = false, per_utt_speaker = false;
    ParseOptions po(usage);

    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");
    po.Register("frame-overlap", &frame_overlap, "Frame overlap in seconds");
    po.Register("single-speaker", &single_speaker, "If this is set true, "
                "then all the utterances in a recording are mapped to the "
                "same speaker");
    po.Register("per-utt-speaker", &per_utt_speaker,
                "If this is set true, then each utterance is mapped to distint "
                "speaker with spkr_id = utt_id");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (frame_shift < 0.001 || frame_shift > 1) {
      KALDI_ERR << "Invalid frame-shift " << frame_shift << "; must be in "
                << "the range [0.001,1]";
    }

    if (frame_overlap < 0 || frame_overlap > 1) {
      KALDI_ERR << "Invalid frame-overlap " << frame_overlap << "; must be in "
                << "the range [0,1]";
    }

    std::string segmentation_rspecifier = po.GetArg(1),
                     utt2spk_wspecifier = po.GetArg(2),
                    segments_wxfilename = po.GetArg(3);

    SequentialSegmentationReader reader(segmentation_rspecifier);
    TokenWriter utt2spk_writer(utt2spk_wspecifier);

    Output ko(segments_wxfilename, false);

    int32 num_done = 0;
    int64 num_segments = 0;

    for (; !reader.Done(); reader.Next(), num_done++) {
      const Segmentation &segmentation = reader.Value();
      const std::string &key = reader.Key();

      for (SegmentList::const_iterator it = segmentation.Begin();
           it != segmentation.End(); ++it) {
        BaseFloat start_time = it->start_frame * frame_shift;
        BaseFloat end_time = (it->end_frame + 1) * frame_shift + frame_overlap;

        std::ostringstream oss;

        if (!single_speaker) {
          oss << key << "-" << it->Label();
        } else {
          oss << key;
        }

        std::string spk = oss.str();

        oss << "-";
        oss << std::setw(6) << std::setfill('0') << it->start_frame;
        oss << std::setw(1) << "-";
        oss << std::setw(6) << std::setfill('0')
            << it->end_frame + 1 +
                static_cast<int32>(frame_overlap / frame_shift);

        std::string utt = oss.str();

        if (per_utt_speaker)
          utt2spk_writer.Write(utt, utt);
        else
          utt2spk_writer.Write(utt, spk);

        ko.Stream() << utt << " " << key << " ";
        ko.Stream() << std::fixed << std::setprecision(3) << start_time << " ";
        ko.Stream() << std::setprecision(3) << end_time << "\n";

        num_segments++;
      }
    }

    KALDI_LOG << "Converted " << num_done << " segmentations to segments; "
              << "wrote " << num_segments << " segments";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

