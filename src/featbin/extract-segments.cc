// featbin/extract-segments.cc

// Copyright 2009-2011  Microsoft Corporation;  Govivace Inc.
//           2013       Arnab Ghoshal

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
#include "feat/wave-reader.h"

/*! @brief This is the main program for extracting segments from a wav file
 - usage :
     - extract-segments [options ..]  <scriptfile > <segments-file> <wav-written-specifier>
     - "scriptfile" must contain full path of the wav file.
     - "segments-file" should have the information of the segments that needs to be extracted from wav file
     - the format of the segments file : speaker_name wavfilename start_time(in secs) end_time(in secs) channel-id(0 or 1)
     - The channel-id is 0 for the left channel and 1 for the right channel.  This is not required for mono recordings.
     - "wav-written-specifier" is the output segment format
*/
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Extract segments from a large audio file in WAV format.\n"
        "Usage:  extract-segments [options] <wav-rspecifier> <segments-file> <wav-wspecifier>\n"
        "e.g. extract-segments scp:wav.scp segments ark:- | <some-other-program>\n"
        " segments-file format: each line is either\n"
        "<segment-id> <recording-id> <start-time> <end-time>\n"
        "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
        "or (less frequently, and not supported in scripts):\n"
        "<segment-id> <wav-file-name> <start-time> <end-time> <channel>\n"
        "where <channel> will normally be 0 (left) or 1 (right)\n"
        "e.g. call-861225-A-0050-0065 call-861225 5.0 6.5 1\n"
        "And <end-time> of -1 means the segment runs till the end of the WAV file\n"
        "See also: extract-feature-segments, wav-copy, wav-to-duration\n";

    ParseOptions po(usage);
    BaseFloat min_segment_length = 0.1, // Minimum segment length in seconds.
        max_overshoot = 0.5;  // max time by which last segment can overshoot
    po.Register("min-segment-length", &min_segment_length,
                "Minimum segment length in seconds (reject shorter segments)");
    po.Register("max-overshoot", &max_overshoot,
                "End segments overshooting audio by less than this (in seconds) "
                "are truncated, else rejected.");

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
    Input ki(segments_rxfilename);  // No binary argment: never binary.

    int32 num_lines = 0, num_success = 0;

    std::string line;
    // Read each line from the segments file.
    while (std::getline(ki.Stream(), line)) {
      num_lines++;
      std::vector<std::string> split_line;
      // Split the line into whitespace-separated fields and verify their
      // number. There must be 4 or 5 fields: segment name, reacording ID, start
      // time, end time, and the optional channel number.
      SplitStringToVector(line, " \t\r", true, &split_line);
      if (split_line.size() != 4 && split_line.size() != 5) {
        KALDI_WARN << "Invalid line in segments file: " << line;
        continue;
      }
      std::string segment = split_line[0],
          recording = split_line[1],
          start_str = split_line[2],
          end_str = split_line[3];

      // Parse the start and end times as float values. Segment is ignored if
      // any of end times is malformed.
      double start, end;
      if (!ConvertStringToReal(start_str, &start)) {
        KALDI_WARN << "Invalid line in segments file [bad start]: " << line;
        continue;
      }
      if (!ConvertStringToReal(end_str, &end)) {
        KALDI_WARN << "Invalid line in segments file [bad end]: " << line;
        continue;
      }
      // Start time must be non-negative and not greater than the end time,
      // except if the end time is -1.
      if (start < 0 || (end != -1.0 && end <= 0) ||
          ((start >= end) && (end > 0))) {
        KALDI_WARN << ("Invalid line in segments file "
                       "[empty or invalid segment]: ") << line;
        continue;
      }
      int32 channel = -1;  // -1 means channel is unspecified.
      // If the line has 5 elements, then the 5th element is the channel number.
      if (split_line.size() == 5) {
        if (!ConvertStringToInteger(split_line[4], &channel) || channel < 0) {
          KALDI_WARN << "Invalid line in segments file [bad channel]: " << line;
          continue;
        }
      }

      // Check whether the recording ID is in wav.scp; if not, skip the segment.
      if (!reader.HasKey(recording)) {
        KALDI_WARN << "Could not find recording " << recording
                   << ", skipping segment " << segment;
        continue;
      }

      const WaveData &wave = reader.Value(recording);
      const Matrix<BaseFloat> &wave_data = wave.Data();
      BaseFloat samp_freq = wave.SampFreq();  // Sampling fequency.
      int32 num_samp = wave_data.NumCols(),  // Number of samples in recording.
        num_chan = wave_data.NumRows();  // Number of channels in recording.
      BaseFloat file_length = num_samp / samp_freq;  // In seconds.

      // Start must be within the wave data, otherwise skip the segment.
      if (start < 0 || start > file_length) {
        KALDI_WARN << "Segment start is out of file data range [0, "
                   << file_length << "s]; skipping segment '" << line << "'";
        continue;
      }

      // End must be less than the file length adjusted for possible overshoot;
      // otherwise skip the segment. end == -1 passes the check.
      if (end > file_length + max_overshoot) {
        KALDI_WARN << "Segment end is too far out of file data range [0,"
                   << file_length << "s]; skipping segment '" << line << "'";
        continue;
      }

      // Otherwise ensure the end is not beyond the end of data, and default
      // end == -1 to the end of file data.
      if (end < 0 || end > file_length) end = file_length;

      // Skip if segment size is less than the minimum allowed.
      if (end - start < min_segment_length) {
        KALDI_WARN << "Segment " << segment << " too short, skipping it.";
        continue;
      }

      // Check that the channel is specified in the segments file for a multi-
      // channel file, and that the channel actually exists in the wave data.
      if (channel == -1) {
        if (num_chan == 1) channel = 0;
        else {
          KALDI_ERR << ("Your data has multiple channels. You must "
                        "specify the channel in the segments file. "
                        "Skipping segment ") << segment;
        }
      } else {
        if (channel >= num_chan) {
          KALDI_WARN << "Invalid channel " << channel << " >= " << num_chan
                     << ". Skipping segment " << segment;
          continue;
        }
      }

      // Convert endpoints of the segment to sample numbers. Note that the
      // conversion requires a proper rounding.
      int32 start_samp = static_cast<int32>(start * samp_freq + 0.5f),
          end_samp = static_cast<int32>(end * samp_freq + 0.5f);
    
      if (end_samp > num_samp) 
        end_samp = num_samp;
     
      // Get the range of data from the orignial wave_data matrix.
      SubMatrix<BaseFloat> segment_matrix(wave_data, channel, 1,
                                          start_samp, end_samp - start_samp);
      WaveData segment_wave(samp_freq, segment_matrix);
      writer.Write(segment, segment_wave);  // Write the range in wave format.
      num_success++;
    }
    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the segments file. ";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
