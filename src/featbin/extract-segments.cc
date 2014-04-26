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
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

/*! @brief This is the main program for extracting segments from a wav file
 - usage : 
     - extract-segments [options ..]  <scriptfile > <segments-file> <wav-written-specifier>
     - "scriptfile" must contain full path of the wav file.
     - "segments-file" should have the information of the segments that needs to be extracted from wav file
     - the format of the segments file : speaker_name wavfilename start_time(in secs) end_time(in secs) channel(1 or 2)
     - The channel information in the segfile is optional . default value is mono(1).
     - "wav-written-specifier" is the output segment format
*/
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    
    const char *usage =
        "Extract segments from a large audio file in WAV format.\n"
        "Usage:  extract-segments [options] <wav-rspecifier> <segments-file> <wav-wspecifier>\n"
        "e.g. extract-segments wav.scp segments ark:- | <some other program>\n"
        " segments-file format: segment_id wav_file_name start_time end_time [channel]\n"
        " e.g.: spkabc_seg1 spkabc_recording1 1.10 2.36 1\n"
        " If channel is not provided as last element, expects mono.\n"
        " end_time of -1 means the segment runs till the end of the WAV file.\n"
        "See also: extract-rows, which does the same thing but to feature files,\n"
        " wav-copy, wav-to-duration\n";

    ParseOptions po(usage);
    BaseFloat min_segment_length = 0.1, // Minimum segment length in seconds.
        max_overshoot = 0.5;  // max time by which last segment can overshoot
    po.Register("min-segment-length", &min_segment_length,
                "Minimum segment length in seconds (reject shorter segments)");
    po.Register("max-overshoot", &max_overshoot,
                "End segmnents overshooting by less (in seconds) are truncated,"
                " else rejected.");

    // OPTION PARSING ...
    // parse options  (+filling the registered variables)
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
    Input ki(segments_rxfilename);  // no binary argment: never binary.

    int32 num_lines = 0, num_success = 0;

    std::string line;
    /* read each line from segments file */
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
      int32 channel = -1;  // means channel info is unspecified.
      // if each line has 5 elements then 5th element must be channel identifier
      if(split_line.size() == 5) {
        if (!ConvertStringToInteger(split_line[4], &channel) || channel < 0) {
          KALDI_WARN << "Invalid line in segments file [bad channel]: " << line;
          continue;
        }
      }
      /* check whether a segment start time and end time exists in recording 
       * if fails , skips the segment.
       */ 
      if (!reader.HasKey(recording)) {
        KALDI_WARN << "Could not find recording " << recording
                   << ", skipping segment " << segment;
        continue;
      }
      
      const WaveData &wave = reader.Value(recording);
      const Matrix<BaseFloat> &wave_data = wave.Data();
      BaseFloat samp_freq = wave.SampFreq();  // read sampling fequency
      int32 num_samp = wave_data.NumCols(),  // number of samples in recording
        num_chan = wave_data.NumRows();  // number of channels in recording

      // Convert starting time of the segment to corresponding sample number.
      // If end time is -1 then use the whole file starting from start time.
      int32 start_samp = start * samp_freq,
          end_samp = (end != -1)? (end * samp_freq) : num_samp;
      KALDI_ASSERT(start_samp >= 0 && end_samp > 0 && "Invalid start or end.");

      // start sample must be less than total number of samples,
      // otherwise skip the segment
      if (start_samp < 0 || start_samp >= num_samp) {
        KALDI_WARN << "Start sample out of range " << start_samp << " [length:] "
                   << num_samp << ", skipping segment " << segment;
        continue;
      }
      /* end sample must be less than total number samples 
       * otherwise skip the segment
       */
      if (end_samp > num_samp) {
        if ((end_samp >=
             num_samp + static_cast<int32>(max_overshoot * samp_freq))) {
          KALDI_WARN << "End sample too far out of range " << end_samp
                     << " [length:] " << num_samp << ", skipping segment "
                     << segment;
          continue;
        }
        end_samp = num_samp;  // for small differences, just truncate.
      }
      // Skip if segment size is less than minimum segment length (default 0.1s)
      if (end_samp <=
          start_samp + static_cast<int32>(min_segment_length * samp_freq)) {
        KALDI_WARN << "Segment " << segment << " too short, skipping it.";
        continue;
      }
      /* check whether the wav file has more than one channel
       * if yes, specify the channel info in segments file
       * otherwise skips the segment
       */
      if (channel == -1) {
        if (num_chan == 1) channel = 0;
        else {
          KALDI_ERR << "If your data has multiple channels, you must specify the"
              " channel in the segments file.  Processing segment " << segment;
        }
      } else {
        if (channel >= num_chan) {
          KALDI_WARN << "Invalid channel " << channel << " >= " << num_chan
                     << ", processing segment " << segment;
          continue;
        }
      }
      /*
       * This function  return a portion of a wav data from the orignial wav data matrix 
       */
      SubMatrix<BaseFloat> segment_matrix(wave_data, channel, 1, start_samp, end_samp-start_samp);
      WaveData segment_wave(samp_freq, segment_matrix);
      writer.Write(segment, segment_wave); // write segment in wave format.
      num_success++;
    }
    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the segments file. ";
    /* prints number of segments processed */
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

