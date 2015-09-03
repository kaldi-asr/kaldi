// featbin/extract-feature-segments.cc

// Copyright 2009-2011  Microsoft Corporation;  Govivace Inc.
//           2012-2013  Mirko Hannemann;  Arnab Ghoshal

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
#include "matrix/kaldi-matrix.h"

/** @brief This is a program for extracting segments from feature files/archives
 - usage : 
     - extract-feature-segments [options ..]  <scriptfile/archive> <segments-file> <features-written-specifier>
     - "segments-file" should have the information of the segments that needs to be extracted from the feature files
     - the format of the segments file : speaker_name filename start_time(in secs) end_time(in secs)
     - "features-written-specifier" is the output segment format
*/
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    
    const char *usage =
        "Create feature files by segmenting input files.\n"
        "Usage:  extract-feature-segments [options...] <feats-rspecifier> <segments-file> <feats-wspecifier>\n"
        " (segments-file has lines like: output-utterance-id input-utterance-or-spk-id 1.10 2.36)\n";

    // construct all the global objects
    ParseOptions po(usage);

    BaseFloat min_segment_length = 0.1,  // Minimum segment length in seconds.
        max_overshoot = 0.0;  // max time by which last segment can overshoot
    BaseFloat samp_freq = 100;  // feature sampling frequency (assuming 10ms window shift)

    // Register the options
    po.Register("min-segment-length", &min_segment_length,
                "Minimum segment length in seconds (reject shorter segments)");
    po.Register("frame-rate", &samp_freq,
                "Feature sampling frequency (e.g. 100 for 10ms window shift)");
    po.Register("max-overshoot", &max_overshoot,
                "End segments overshooting by less (in seconds) are truncated,"
                " else rejected.");

    // OPTION PARSING ...
    // parse options  (+filling the registered variables)
    po.Read(argc, argv);
    // number of arguments should be 3(scriptfile,segments file and outputwav write mode)
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string rspecifier = po.GetArg(1); // get script file/feature archive
    std::string segments_rxfilename = po.GetArg(2);// get segment file
    std::string wspecifier = po.GetArg(3); // get written archive name

    BaseFloatMatrixWriter feat_writer(wspecifier);

    RandomAccessBaseFloatMatrixReader feat_reader(rspecifier); 

    Input ki(segments_rxfilename); // no binary argment: never binary.

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
          utterance = split_line[1],
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
      if (start < 0 || end <= 0 || start >= end) {
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

      /* check whether a segment start time and end time exists in utterance 
       * if fails , skips the segment.
       */ 
      if (!feat_reader.HasKey(utterance)) {
        KALDI_WARN << "Did not find features for utterance " << utterance
                   << ", skipping segment " << segment;
        continue;
      }
      const Matrix<BaseFloat> &feats = feat_reader.Value(utterance);
      int32 num_samp = feats.NumRows(), // total number of samples present in wav data
          num_chan = feats.NumCols(); // total number of channels present in wav file

      // Convert start & end times of the segment to corresponding sample number
      int32 start_samp = static_cast<int32>(start * samp_freq);
      int32 end_samp = static_cast<int32>(end * samp_freq);
      /* start sample must be less than total number of samples 
       * otherwise skip the segment
       */
      if (start_samp < 0 || start_samp >= num_samp) {
        KALDI_WARN << "Start sample out of range " << start_samp << " [length:] "
                   << num_samp << "x" << num_chan << ", skipping segment " << segment;
        continue;
      }
      /* end sample must be less than total number samples 
       * otherwise skip the segment
       */
      if (end_samp > num_samp) {
        if (end_samp >=
            num_samp + static_cast<int32>(max_overshoot * samp_freq)) {
          KALDI_WARN << "End sample too far out of range " << end_samp
                     << " [length:] " << num_samp << "x" << num_chan << ", skipping segment "
                     << segment;
          continue;
        }
        end_samp = num_samp; // for small differences, just truncate.
      }
      /* check whether the segment size is less than minimum segment length(default 0.1 sec)
       * if yes, skip the segment
       */
      if (end_samp <=
          start_samp + static_cast<int32>(min_segment_length * samp_freq)) {
        KALDI_WARN << "Segment " << segment << " too short, skipping it.";
        continue;
      }

      SubMatrix<BaseFloat> segment_matrix(feats, start_samp, end_samp-start_samp, 0, num_chan);
      Matrix<BaseFloat> outmatrix(segment_matrix);
      feat_writer.Write(segment, outmatrix);  // write segment in feature archive.
      num_success++;
    }
    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the segments file. ";
    /* prints number of segments processed */
    if (num_success == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

