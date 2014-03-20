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
    int32 max_voiced = 60;
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

    std::vector<Vector<BaseFloat> > vads;
    std::vector<string> utt_ids;
    int32 num_err = 0;
    for (;!vad_reader.Done(); vad_reader.Next()) {
      string utt = vad_reader.Key();
      const Vector<BaseFloat> &vad = vad_reader.Value();
      if (vad.Sum() == 0.0) {
        KALDI_WARN << "No features were judged as voiced for utterance "
                   << utt;
        num_err++;
        continue;
      }
      utt_ids.push_back(utt);
      vads.push_back(vad);
    }

    int32 num_utts_done = 0;    
    for (int32 i = 0; i < vads.size(); i++) {
      const Vector<BaseFloat> &vad = vads[i];
      string utt = utt_ids[i];
      int32 curr_sum_voiced_frames = 0;
      int32 num_split = 1; // Starting with the first split
      int32 first_frame = 0; // First frame of the first segment
                             // will start at 0.
      for (int32 j = 0; j < vad.Dim(); j++) {
        curr_sum_voiced_frames += vad(j);
        if (curr_sum_voiced_frames == max_voiced) {
          feat_segment_writer.Stream() << utt << "-" << num_split << " " 
                                       << utt << " " << first_frame
                                       << " " << j << "\n";
          // If we're not at the last frame, prepare for the next
          // segment.
          if (j != vad.Dim() - 1) {
            curr_sum_voiced_frames = 0;
            first_frame = j + 1;
            num_split += 1;
          }
        }
      }
      num_utts_done++; 
    }
    KALDI_LOG << "Split " << num_utts_done << " utts, " << num_err
              << " had errors.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
