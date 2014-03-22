// ivectorbin/create-split-from-vad.cc

// Copyright   2014   David Snyder

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
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef std::string string;
  try {
    const char *usage =
        "Create a feats-segment file specifying splits of an utterance based "
        "on a VAD file.\n"
        "The VAD file is the output of compute-vad or a similar program "
        "(a vector \n"
        "of length num-frames, containing 1.0 for voiced, 0.0 for unvoiced). "
        "Each line of the\n"
        "feats-segment file is of the following form: \n"
        "    <dst-utt> <src-utt> <first-frame> <last-frame>\n"
        "Usage: create-split-from-vad [options] <vad-rspecifier>\n"
        "<feats-segment-filename>\n"
        "E.g.: create-split-from-vad [options] scp:vad.scp feats_segment\n";
    
    ParseOptions po(usage);
    int32 max_voiced = 9000; // 90 seconds 
    po.Register("max-voiced", &max_voiced,
                "Maximum number of voiced frames in each utterance "
                "after split.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    string vad_rspecifier = po.GetArg(1),
        feat_segment_filename = po.GetArg(2);
    
    SequentialBaseFloatVectorReader vad_reader(vad_rspecifier);
    bool binary = false;
    Output feat_segment_writer(feat_segment_filename, binary);
    int32 num_err = 0,
          num_utts_done = 0,
          total_splits = 0;
    for (;!vad_reader.Done(); vad_reader.Next()) {
      string utt = vad_reader.Key();
      const Vector<BaseFloat> &vad = vad_reader.Value();
      BaseFloat sum_voiced = vad.Sum();
      if (sum_voiced == 0.0) {
        KALDI_WARN << "No features were judged as voiced for utterance "
                   << utt;
        num_err++;
        continue;
      }

      // We want num_splits to produce segments which all
      // have close to the same number of voiced frames. At the same
      // time we want each segment's number of voiced frames to be
      // as close as possible to what is specified by max_voiced.
      int32 num_splits = std::ceil(sum_voiced / max_voiced),
            curr_sum_voiced_frames = 0,
            split = 1, // We start with the 1st split
            first_frame = 0; // First frame of the first segment
                             // will start at 0.

      // actual_max_voiced will be as large or smaller than max_voiced.
      // For example, suppose we have 20000 voiced frames in an utterance
      // and max_voiced = 9000. We want to avoid the situation in which the
      // first two segments have 9000 voiced frames and the third segment has
      // only 200 voiced frames. Given the example, actual_max_voiced
      // is 6667, creating almost equal segments (the last has 6666 voiced
      // voiced frames).
      int32 actual_max_voiced = std::ceil(sum_voiced / num_splits);
      for (int32 j = 0; j < vad.Dim(); j++) {
        curr_sum_voiced_frames += vad(j);
        if (curr_sum_voiced_frames == actual_max_voiced) {
          feat_segment_writer.Stream() << utt << "-" << split << " " 
                                       << utt << " " << first_frame
                                       << " " << j << "\n";
          // If we're not at the last frame, prepare for the next
          // segment.
          if (j != vad.Dim() - 1) {
            curr_sum_voiced_frames = 0;
            first_frame = j + 1;
            split += 1;
          }
          total_splits++;
        }
      }
      num_utts_done++; 
    }

    KALDI_LOG << "Split " << num_utts_done << " utts into " 
              << total_splits << " segments. " << num_err
              << " had errors.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
