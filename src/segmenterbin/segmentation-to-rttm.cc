// segmenterbin/segmentation-to-rttm.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "segmenter/segmenter.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Convert segmentation into RTTM\n"
        "\n"
        "Usage: segmentation-to-rttm [options] segmentation-in-rspecifier rttm-out-wxfilename\n"
        " e.g.: segmentation-to-rttm foo -\n"
        "   segmentation-to-rttm ark:1.seg -\n";
    
    bool binary = true;
    bool map_to_speech_and_sil = true;

    BaseFloat frame_shift = 0.01;
    std::string segments_rxfilename;
    std::string reco2file_and_channel_rxfilename;
    ParseOptions po(usage);
    
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");
    po.Register("segments", &segments_rxfilename, "Segments file");
    po.Register("reco2file-and-channel", &reco2file_and_channel_rxfilename, "reco2file_and_channel file");
    po.Register("map-to-speech-and-sil", &map_to_speech_and_sil, "Map all classes to SPEECH and SILENCE");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    unordered_map<std::string, std::string, StringHasher> utt2file;
    unordered_map<std::string, BaseFloat, StringHasher> utt2start_time;

    if (segments_rxfilename != "") {
      Input ki(segments_rxfilename); // no binary argment: never binary.
      int32 i = 0;
      std::string line;
      /* read each line from segments file */
      while (std::getline(ki.Stream(), line)) {
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

        utt2file.insert(std::make_pair(segment, utterance));
        utt2start_time.insert(std::make_pair(segment, start));
        i++;
      }
      KALDI_LOG << "Read " << i << " lines from " << segments_rxfilename;
    }

    std::unordered_map<std::string, std::pair<std::string, std::string> , StringHasher> reco2file_and_channel;

    if (!reco2file_and_channel_rxfilename.empty()) {
      Input ki(reco2file_and_channel_rxfilename); // no binary argment: never binary.

      int32 i = 0;
      std::string line;
      /* read each line from reco2file_and_channel file */
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> split_line;
        SplitStringToVector(line, " \t\r", true, &split_line);
        if (split_line.size() != 3) {
          KALDI_WARN << "Invalid line in reco2file_and_channel file: " << line;
          continue;
        }

        const std::string &reco_id = split_line[0];
        const std::string &file_id = split_line[1];
        const std::string &channel = split_line[2];

        reco2file_and_channel.insert(std::make_pair(reco_id, std::make_pair(file_id, channel)));
        i++;
      }

      KALDI_LOG << "Read " << i << " lines from " << reco2file_and_channel_rxfilename;
    }

    std::unordered_set<std::string, StringHasher> seen_files;

    std::string segmentation_rspecifier = po.GetArg(1),
            rttm_out_wxfilename = po.GetArg(2);

    int64  num_done = 0, num_err = 0;
    
    Output ko(rttm_out_wxfilename, false);
    SequentialSegmentationReader reader(segmentation_rspecifier);
    for (; !reader.Done(); reader.Next(), num_done++) {
      Segmentation seg(reader.Value());
      const std::string &key = reader.Key();

      std::string reco_id = key; 
      BaseFloat start_time = 0.0;
      if (!segments_rxfilename.empty()) {
        if (utt2file.count(key) == 0 || utt2start_time.count(key) == 0)
          KALDI_ERR << "Could not find key " << key << " in segments " 
                    << segments_rxfilename;
        KALDI_ASSERT(utt2file.count(key) > 0 && utt2start_time.count(key) > 0);
        reco_id = utt2file[key];
        start_time = utt2start_time[key];
      }

      std::string file_id, channel;
      if (!reco2file_and_channel_rxfilename.empty()) {
        if (reco2file_and_channel.count(reco_id) == 0) 
          KALDI_ERR << "Could not find recording " << reco_id 
                    << " in " << reco2file_and_channel_rxfilename;
        file_id = reco2file_and_channel[reco_id].first;
        channel = reco2file_and_channel[reco_id].second;
      } else {
        file_id = reco_id;
        channel = "1";
      }

      int32 largest_class = seg.WriteRttm(ko.Stream(), file_id, channel, frame_shift, start_time, map_to_speech_and_sil);

      if (map_to_speech_and_sil) {
        if (seen_files.count(reco_id) == 0) {
          ko.Stream() << "SPKR-INFO " << file_id << " " << channel << " <NA> <NA> <NA> unknown SILENCE <NA>\n";
          ko.Stream() << "SPKR-INFO " << file_id << " " << channel << " <NA> <NA> <NA> unknown SPEECH <NA>\n";
          seen_files.insert(reco_id);
        }
      } else {
        for (int32 i = 0; i < largest_class; i++) {
          ko.Stream() << "SPKR-INFO " << file_id << " " << channel << " <NA> <NA> <NA> unknown " << i << " <NA>\n";
        }
      }
    }

    KALDI_LOG << "Copied " << num_done << " segmentation; failed with "
      << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}




