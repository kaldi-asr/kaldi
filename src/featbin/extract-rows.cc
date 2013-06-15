// featbin/extract-rows.cc

// Copyright 2013 Korbinian Riedhammer

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
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
      "Extract certain rows of feature matrices, e.g. to train classifiers to determine\n"
      "the quality of a proposed KWS hit.  The program expects a segments file in the\n"
      "form of\n"
      "  segment-name utterance-id start end\n"
      "where start/end can frame/row numbers or seconds.msec, if invoked with --frame-shift\n"
      "  e.g. extract-rows segments ark:feats-in.ark ark:feats-out.ark\n";

    ParseOptions po(usage);

    float frame_shift = 0;

    po.Register("frame-shift", &frame_shift,
    			"Frame shift in sec (typ. 0.01), if segment files contains times instead of frames");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    string seg_rspecifier = po.GetArg(1);
    string ft_rspecifier = po.GetArg(2);
    string ft_wspecifier = po.GetArg(3);

    Input ki(seg_rspecifier);
    RandomAccessBaseFloatMatrixReader *rd = new RandomAccessBaseFloatMatrixReader(ft_rspecifier);
    BaseFloatMatrixWriter writer(ft_wspecifier);

    int32 num_lines = 0, num_missing = 0;

    string line;

    /* read each line from segments file */
    while (std::getline(ki.Stream(), line)) {
      num_lines++;

      vector<string> split_line;
      SplitStringToVector(line, " \t\r", true, &split_line);
      if (split_line.size() != 4) {
        KALDI_WARN << "Invalid line in segments file: " << line;
        continue;
      }

      string segment = split_line[0],
        utt = split_line[1],
        start_str = split_line[2],
        end_str = split_line[3];

      // if the segments are in time, we need to convert them to frame numbers
      int32 start = 0;
      int32 end = 0;
      if (frame_shift > 0) {
        // Convert the start time and endtime to real from string. Segment is
        // ignored if start or end time cannot be converted to real.
        double t1, t2;
        if (!ConvertStringToReal(start_str, &t1)) {
          KALDI_WARN << "Invalid line in segments file [bad start]: " << line;
          continue;
        }
        if (!ConvertStringToReal(end_str, &t2)) {
          KALDI_WARN << "Invalid line in segments file [bad end]: " << line;
          continue;
        }

        start = (int) (t1 / frame_shift);
        end = (int) (t2 / frame_shift);
      } else {
        if (!ConvertStringToInteger(start_str, &start)) {
          KALDI_WARN<< "Invalid line in segments file [bad start]: " << line;
          continue;
        }
        if (!ConvertStringToInteger(end_str, &end)) {
          KALDI_WARN << "Invalid line in segments file [bad end]: " << line;
          continue;
        }
      }

      if (start < 0 || end - start <= 0) {
        KALDI_WARN << "Invalid line in segments file [less than one frame]: " << line;
        continue;
      }

      if (rd->HasKey(utt)) {
        Matrix<BaseFloat> feats = rd->Value(utt);
        if (feats.NumRows() < end)
          end = feats.NumRows();
        Matrix<BaseFloat> toWrite(feats.RowRange(start, (end-start)));
        writer.Write(segment, toWrite);
      } else {
        KALDI_WARN << "Missing requested utterance " << utt;
        num_missing += 1;
      }

    }

    KALDI_LOG << "processed " << num_lines << ", " << (num_lines - num_missing)
        << "successfull, " << num_missing << " missing utterances";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
