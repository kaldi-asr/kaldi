/**
 * @file featbin/extract-segments.cc
 * @brief extract segments from a wav file
 *
 * @copy Copyright 2009-2011  Microsoft Corporation, Govivace Inc.
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
 * MERCHANTABLITY OR NON-INFRINGEMENT.
 * @par
 * See the Apache 2 License for the specific language governing permissions and
 * limitations under the License.

 */
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
        "Create MFCC feature files.\n"
        "Usage:  extract-segments [options...] <wav-rspecifier> <segments-file> <wav-wspecifier>\n"
        " (segments-file has lines like: spkabc_seg1 spkabc_recording1 1.10 2.36 1\n"
        " or: spkabc_seg1 spkabc_recording1 1.10 2.36\n"
        " [if channel not provided as last element, expects mono.] ";
        

    // construct all the global objects
    ParseOptions po(usage);

    BaseFloat min_segment_length = 0.1; // Minimum segment length in seconds.

    // Register the options
    po.Register("min-segment-length", &min_segment_length, "Minimum segment length in seconds (will reject shorter segments)");

    // OPTION PARSING ...
    // parse options  (+filling the registered variables)
    po.Read(argc, argv);
    // number of arguments should be 3(scriptfile,segments file and outputwav write mode)
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1); // get script file
    std::string segments_rxfilename = po.GetArg(2);// get segment file
    std::string wav_wspecifier = po.GetArg(3);// get wav written format

    RandomAccessTableReader<WaveHolder> reader(wav_rspecifier); 
    TableWriter<WaveHolder> writer(wav_wspecifier);
    Input ki(segments_rxfilename); // no binary argment: never binary.

    int32 num_lines = 0, num_success = 0;
    
    std::string line;
    /* read each line from segments file */
    while(std::getline(ki.Stream(), line)) {
      num_lines++;
      std::vector<std::string> split_line;  
      SplitStringToVector(line, " \t\r", &split_line, true);// split the line by space or tab
      if (split_line.size() != 4 && split_line.size() != 5) // check the number of elements in each line. each line must have atleast 4 elements . 5th element(channel info) is optional
        KALDI_ERR << "Invalid line in segments file: " << line;
      /* each line of segment file should have segment name , reacording wav file name,start time,end time respectively  */
      std::string segment = split_line[0], recording = split_line[1],
          start_str = split_line[2], end_str = split_line[3];
      double start, end;
      /* convert the start time and endtime to real from string */
      if(!ConvertStringToReal(start_str, &start) || !ConvertStringToReal(end_str, &end))
        KALDI_ERR << "Invalid line in segments file [bad start/end]" << line;
      /* error occurs when start time or end time is not converted to
         real. they must be specified like "1234.56" in seg file */
      /* start time and end time must be greater than 0 secs and start
         time should be greater than end time */ 
      if(start < 0 || end < 0 || start >= end) {
        KALDI_WARN << "Invalid line in segments file [empty or invalid segment] "
                   << line;
        continue;
      }
      int32 channel = -1; // means channel info is unspecified. 
      if(split_line.size() == 5) { // if each line has 5 elements then
                                   // 5th element must be channel
                                   // identifier. it should be 1 or 2
        if(!ConvertStringToInteger(split_line[4], &channel) || channel < 0)
          KALDI_ERR << "Invalid line in segments file [bad channel] " << line;
      }
      /* check whether a segment start time and end time exists in recording 
       * if fails , skips the segment.
       */ 
      if(!reader.HasKey(recording)) {
        KALDI_WARN << "Could not find recording " << recording
                   << ", skipping segment " << segment;
        continue;
      }
      
      const WaveData &wave = reader.Value(recording);
      const Matrix<BaseFloat> &wave_data = wave.Data();// read wav file data to matrix format
      BaseFloat samp_freq = wave.SampFreq(); // read sampling fequency
      int32 start_samp = start * samp_freq, // convert starting time
                                            // of the segment to
                                            // corresponding sample
                                            // number
          end_samp = end * samp_freq,// convert ending time of the segment
                               // to corresponding sample number
          num_samp = wave_data.NumCols(), // total number of samples present in wav data
          num_chan = wave_data.NumRows(); // total number of channels present in wav file
      /* start sample must be less than total number of samples 
       * otherwise skip the segment
       */
      if(start_samp < 0 || start_samp >= num_samp) {
        KALDI_WARN << "Start sample out of range " << start_samp << " [length:] "
                   << num_samp << ", skipping segment " << segment;
        continue;
      }
      /* end sample must be less than total number samples 
       * otherwise skip the segment
       */
      if(end_samp > num_samp) {
        if(end_samp > num_samp + static_cast<int32>(0.5 * samp_freq)) {
          KALDI_WARN << "End sample too far out of range " << end_samp
                     << " [length:] " << num_samp << ", skipping segment "
                     << segment;
          continue;
        }
        end_samp = num_samp; // for small differences, just truncate.
      }
      /* check whether the segment size is less than minimum segment length(default 0.1 sec)
       * if yes, skip the segment
       */
      if(end_samp <= start_samp +
         static_cast<int32>(min_segment_length * samp_freq)) {
        KALDI_WARN << "Segment " << segment << " too short, skipping it.";
        continue;
      }
      /* check whether the wav file has more than one channel
       * if yes, specify the channel info in segments file
       * otherwise skips the segment
       */
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
      /*
       * This function  return a portion of a wav data from the orignial wav data matrix 
       */
      SubMatrix<BaseFloat> segment_matrix(wave_data, channel, 1, start_samp, end_samp-start_samp);
      WaveData segment_wave(samp_freq, segment_matrix);
      if(!writer.Write(segment, segment_wave)) // write segment 
        KALDI_ERR << "Failed to write segment: processling line " << line;
      num_success++;
    }
    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the segments file. ";
    /* prints number of segments processed */
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

