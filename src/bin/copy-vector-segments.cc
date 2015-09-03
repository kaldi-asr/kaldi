// bin/copy-vector-segments.cc

// Copyright 2013 Korbinian Riedhammer
//           2014 David Snyder

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
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Extract certain elements of vectors.  This is useful for extracting\n"
        "segments from VAD files. The program expects a segments file in the\n"
        "form of\n"
        "  segment-name vector-id start end\n"
        "The segment-name is chosen by the user to identify the segment. The\n"
        "vector-id indexes the input vectors. By default start and end are \n"
        "zero-based indexes, but if the option --frame-shift is specified\n"
        "(e.g., --frame-shift=0.01), then the columns represent start and\n"
        "end times, respectively.\n"
        "\n"
        "Usage: copy-vector-segments [options] <segments-file> \n"
        "<vectors-rspecifier> <vectors-wspecifier>\n"
        "e.g. copy-vector-segments segments ark:vad-in.ark ark:vad-out.ark\n"
        "See also extract-rows, select-feats, subset-feats, subsample-feats\n";

    ParseOptions po(usage);

    float frame_shift = 0;

    po.Register("frame-shift", &frame_shift,
                "Frame shift in sec (e.g. 0.01), if segment files "
                        "contains times instead of frames");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    string segment_rspecifier = po.GetArg(1);
    string vec_rspecifier = po.GetArg(2);
    string vec_wspecifier = po.GetArg(3);

    Input ki(segment_rspecifier);
    RandomAccessBaseFloatVectorReader reader(vec_rspecifier);
    BaseFloatVectorWriter writer(vec_wspecifier);

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
          KALDI_ERR << "Invalid line in segments file [bad start]: " << line;
          continue;
        }
        if (!ConvertStringToReal(end_str, &t2)) {
          KALDI_ERR << "Invalid line in segments file [bad end]: " << line;
          continue;
        }

        start = (int) (t1 / frame_shift);
        end = (int) (t2 / frame_shift);
      } else {
        if (!ConvertStringToInteger(start_str, &start)) {
          KALDI_ERR << "Invalid line in segments file [bad start]: " << line;
          continue;
        }
        if (!ConvertStringToInteger(end_str, &end)) {
          KALDI_ERR << "Invalid line in segments file [bad end]: " << line;
          continue;
        }
      }

      if (start < 0 || end - start <= 0) {
        KALDI_WARN << "Invalid line in segments file [less than one frame]: " << line;
        continue;
      }

      if (reader.HasKey(utt)) {
        Vector<BaseFloat> vec = reader.Value(utt);

        if (vec.Dim() < end)
          end = vec.Dim();

        SubVector<BaseFloat> to_write_sub(vec, start, end-start);
        Vector<BaseFloat> to_write(to_write_sub);
        writer.Write(segment, to_write);
      } else {
        KALDI_WARN << "Missing requested utterance " << utt;
        num_missing += 1;
      }

    }

    KALDI_LOG << "processed " << num_lines << " segments, " << (num_lines - num_missing)
              << " successful, " << num_missing << " had invalid utterances";

    return ((num_lines - num_missing) > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
