// segmenterbin/segmentation-to-segments.cc

// Copyright 2015   Vimal Manohar

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "segmenter/segmenter.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Convert segmentation to segments file and utt2spk file. "
        "Assumes that the segmentations are indexed by file-id and "
        "treats speakers from different files as distinct speakers."
        "\n"
        "Usage: segmentation-to-segments [options] segmentation-rspecifier utt2spk-wspecifier segments-wxfilename\n"
        " e.g.: segmentation-to-segments ark:foo ark,t:utt2spk -\n";
    
    bool binary = true;
    BaseFloat frame_shift = 0.01;

    ParseOptions po(usage);
    
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_rspecifier = po.GetArg(1),
                utt2spk_wspecifier = po.GetArg(2),
                segments_wxfilename = po.GetArg(3);

    SequentialSegmentationReader reader(segmentation_rspecifier);
    TokenWriter utt2spk_writer(utt2spk_wspecifier);
    Output ko(segments_wxfilename, false);

    int32 num_done = 0, num_err = 0;
    for (; !reader.Done(); reader.Next(), num_done++) {
      const Segmentation &seg = reader.Value();
      std::string key = reader.Key();

      std::string file_id = key; 

      int32 i = 0;
      for (SegmentList::const_iterator it = seg.Begin(); it != seg.End(); ++it, i++) {
        BaseFloat start_time = it->start_frame * frame_shift;
        BaseFloat end_time = (it->end_frame + 1) * frame_shift;

        std::ostringstream oss; 
        oss << key << "-" << it->Label();

        std::string spk = oss.str();

        oss.str("");
        oss << spk << "-" << std::setw(4) << std::setfill('0') << i;

        std::string utt = oss.str();

        ko.Stream() << utt << " " << key << " " << std::fixed << std::setprecision(2) << start_time << " " << end_time << "\n";
        utt2spk_writer.Write(utt, spk);
      }
    }

    ko.Close();

    KALDI_LOG << "Converted" << num_done << " segmentations to segments; "
              << "failed with " << num_err << " segmentations";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

