// featbin/extract-segments.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"


int main(int argc, char *argv[])
{
  try {
    using namespace kaldi;
    const char *usage =
        "Create MFCC feature files.\n"
        " Usage:  extract-segments [options...] <wav-rspecifier> <segments-file> <wav-wspecifier>\n"
        " (segments-file has lines like: spkabc_seg1 spkabc_recording1 1.10 2.36 1\n"
        " or: spkabc_seg1 spkabc_recording1 1.10 2.36\n"
        " [if channel not provided as last element, expects mono.] ";
        

    // construct all the global objects
    ParseOptions po(usage);

    BaseFloat min_segment_length = 0.1; // Minimum segment length in seconds.

    // Register the options
    po.Register("min-segment-length", &min_segment_length, "Minimum segment length in seconds (will reject shorter segments)");


    // OPTION PARSING ..........................................................
    //

    // parse options (+filling the registered variables)
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);
    std::string segments_rxfilename = po.GetArg(2);
    std::string wav_wspecifier = po.GetArg(3);

    RandomAccessTableReader<WaveHolder> reader(wav_rspecifier);
    TableWriter<WaveHolder> writer(wav_wspecifier);
    Input ki(segments_rxfilename); // no binary argment: never binary.

    int32 num_lines = 0, num_success = 0;
    
    std::string line;
    while(std::getline(ki.Stream(), line)) {
      num_lines++;
      std::vector<std::string> split_line;
      SplitStringToVector(line, " \t\r", &split_line, true);
      if (split_line.size() != 4 && split_line.size() != 5)
        KALDI_ERR << "Invalid line in segments file: " << line;
      std::string segment = split_line[0], recording = split_line[1],
          start_str = split_line[2], end_str = split_line[3];
      double start, end;
      if(!ConvertStringToReal(start_str, &start) || !ConvertStringToReal(end_str, &end))
        KALDI_ERR << "Invalid line in segments file [bad start/end]" << line;
      if(start < 0 || end < 0 || start >= end) {
        KALDI_WARN << "Invalid line in segments file [empty or invalid segment] "
                   << line;
        continue;
      }
      int32 channel = -1; // means unspecified.
      if(split_line.size() == 5) {
        if(!ConvertStringToInteger(split_line[4], &channel) || channel < 0)
          KALDI_ERR << "Invalid line in segments file [bad channel] " << line;
      }
      if(!reader.HasKey(recording)) {
        KALDI_WARN << "Could not find recording " << recording
                   << ", skipping segment " << segment;
        continue;
      }
      const WaveData &wave = reader.Value(recording);
      const Matrix<BaseFloat> &wave_data = wave.Data();
      BaseFloat samp_freq = wave.SampFreq();
      int32 start_samp = start * samp_freq,
          end_samp = end * samp_freq,
          num_samp = wave_data.NumCols(),
          num_chan = wave_data.NumRows();
      if(start_samp < 0 || start_samp >= num_samp) {
        KALDI_WARN << "Start sample out of range " << start_samp << " [length:] "
                   << num_samp << ", skipping segment " << segment;
        continue;
      }
      if(end_samp > num_samp) {
        if(end_samp > num_samp + static_cast<int32>(0.5 * samp_freq))
          KALDI_WARN << "End sample too far out of range " << end_samp
                     << " [length:] " << num_samp << ", skipping segment "
                     << segment;
        continue;
        end_samp = num_samp; // for small differences, just truncate.
      }
      if(end_samp <= start_samp +
         static_cast<int32>(min_segment_length * samp_freq)) {
        KALDI_WARN << "Segment " << segment << " too short, skipping it.\n";
        continue;
      }
      if(channel == -1) {
        if(num_chan == 1) channel = 0;
        else {
          KALDI_ERR << "If your data has multiple channels, you must specify the"
              " channel in the segments file.  Processing segment " << segment;
        }
      } else {
        if(channel >= num_chan) {
          KALDI_WARN << "Invalid channel " << channel << " >= " << num_chan
                     << ", processing segment " << segment;
          continue;
        }
      }

      SubMatrix<BaseFloat> segment_matrix(wave_data, channel, 1, start_samp, end_samp-start_samp);
      WaveData segment_wave(samp_freq, segment_matrix);
      if(!writer.Write(segment, segment_wave))
        KALDI_ERR << "Failed to write segment: processling line " << line;
      num_success++;
    }
    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the segments file. ";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

