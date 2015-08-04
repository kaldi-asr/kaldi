// segmenterbin/segmentation-compute-class-ctm-conf.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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
#include "hmm/posterior.h"

namespace kaldi {

struct StringPairHasher {
  size_t operator() (const std::pair<std::string, std::string> &str_pair) const {
    return StringHasher()(str_pair.first) + kPrime * StringHasher()(str_pair.second);
  }

  private:
  static const int kPrime = 7853;
};

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Computes per-class confidences for different segmentation "
        "(usually diarization) classes using word-confidences from a CTM file"
        "\n"
        "Usage: segmentation-compute-class-ctm-conf [options] <segmentation-in-rspecifier> <ctm-rxfilename> <reco2file-and-channel-rxfilename> <post-rspecifier>"
        " e.g.: segmentation-compute-class-ctm-conf ark:exp/nnet2_multicondition/diarization_dev/diarziation_results.txt ctm reco2file_and_channel ark:exp/nnet2_multicondition/diarization_dev/post.ark\n";
    
    bool binary = true;
    BaseFloat frame_shift = 0.01;

    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("frame-shift", &frame_shift, "Frame shift");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_rspecifier = po.GetArg(1),
                ctm_rxfilename = po.GetArg(2),
                reco2file_and_channel_rxfilename = po.GetArg(3),
                post_wspecifier = po.GetArg(4);
    
    RandomAccessSegmentationReader seg_reader(segmentation_rspecifier);
    Input ki(ctm_rxfilename);
    PosteriorWriter conf_writer(post_wspecifier);

    int32 num_recos = 0, num_lines = 0, num_success = 0;

    std::unordered_map<std::pair<std::string, std::string>,std::string, StringPairHasher> file_and_channel2reco_map;
    {
      Input ki(reco2file_and_channel_rxfilename);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> split_line;
        // Split the line by space or tab and check the number of fields in each
        // line. There must be 3 fields--recording name, file name, channel
        SplitStringToVector(line, " \t\r", true, &split_line);
        if (split_line.size() != 3) {
          KALDI_ERR << "Invalid line in reco2file_and_channel file: " << line;
        }
        std::string reco_id = split_line[0],
                    file_id = split_line[1],
                    channel = split_line[2];

        file_and_channel2reco_map[std::make_pair(file_id, channel)] = reco_id;
      }
    }

    std::string line, recording, prev_recording;
    segmenter::Segmentation seg;
    segmenter::SegmentList::const_iterator seg_it;
    
    std::unordered_map<std::string, std::map<int32, std::pair<BaseFloat, BaseFloat> >*, StringHasher> confidences;
    std::set<std::string> reco_list;

    while (std::getline(ki.Stream(), line)) {
      num_lines++;
      std::vector<std::string> split_line;
      // Split the line by space or tab and check the number of fields in each
      // line. There must be 6 fields--file name, channel, 
      // start time, end time, word, confidence
      SplitStringToVector(line, " \t\r", true, &split_line);
      if (split_line.size() != 6) {
        KALDI_ERR << "Invalid line in ctm file: " << line;
      }

      std::string start_str = split_line[2],
        end_str = split_line[3];

      // Convert the start time and endtime to real from string. Segment is
      // ignored if start or end time cannot be converted to real.
      double start, end;
      if (!ConvertStringToReal(start_str, &start)) {
        KALDI_WARN << "Invalid line in ctm file [bad start]: " << line;
        continue;
      }
      if (!ConvertStringToReal(end_str, &end)) {
        KALDI_WARN << "Invalid line in ctm file [bad end]: " << line;
        continue;
      }

      // start time must not be negative; start time must not be greater than
      // end time, except if end time is -1
      if (start < 0 || (end != -1.0 && end <= 0) || ((start >= end) && (end > 0))) {
        KALDI_WARN << "Invalid line in ctm file [empty or invalid segment]: "
          << line;
        continue;
      }

      double conf; 
      if (!ConvertStringToReal(split_line[5], &conf)) {
        KALDI_ERR << "Invalid line in ctm file [bad conf]: " << line;
      }

      std::string reco_id;
      try {
        reco_id = file_and_channel2reco_map.at(std::make_pair(split_line[0], split_line[1]));
      } catch (std::out_of_range &oor) {
        KALDI_ERR << "Out of range error: " << oor.what();
      }

      if (prev_recording == "" || prev_recording != reco_id) {
        if (!seg_reader.HasKey(reco_id)) {
          KALDI_ERR << "Could not find segmentation for recording " << reco_id;
        } 
        seg = seg_reader.Value(reco_id);
        seg_it = seg.Begin();

        std::map<int32, std::pair<BaseFloat, BaseFloat> > *conf_acc;

        if (confidences.count(reco_id) > 0) {
          conf_acc = confidences[reco_id];
        } else {
          conf_acc = new std::map<int32, std::pair<BaseFloat, BaseFloat> >;
          confidences.insert(std::make_pair(reco_id, conf_acc));
          reco_list.insert(reco_id);
          num_recos++;
        }

        while (seg_it != seg.End() && seg_it->end_frame * frame_shift < start) ++seg_it;
        while (seg_it != seg.Begin() && seg_it->start_frame * frame_shift > end) ++seg_it;

        KALDI_ASSERT(seg_it->start_frame * frame_shift >= start && seg_it->end_frame * frame_shift <= end);
  
        std::map<int32, std::pair<BaseFloat, BaseFloat> >::iterator it = conf_acc->find(seg_it->Label());
        if (it == conf_acc->end()) {
          (*conf_acc)[seg_it->Label()] = std::make_pair(conf, 1.0);
        } else {
          (it->second).first += conf;
          (it->second).second += 1.0;
        }
      } 
      
      prev_recording = recording;
      num_success++;
    }
    
    for (std::set<std::string>::const_iterator it = reco_list.begin(); 
          it != reco_list.end(); ++it) {
      const std::map<int32, std::pair<BaseFloat, BaseFloat> > *conf_acc = confidences[*it];
      Posterior post(1);
      for (std::map<int32, std::pair<BaseFloat, BaseFloat> >::const_iterator c_it = conf_acc->begin();
            c_it != conf_acc->end(); ++c_it) {
        post[0].push_back(std::make_pair(c_it->first, (c_it->second).first / (c_it->second).second));
      }
      conf_writer.Write(*it, post);
    }

    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the ctm file; wrote "
              << num_recos << " recordings";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}





