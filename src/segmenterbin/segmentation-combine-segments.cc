// segmenterbin/segmentation-combine-segments.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Combine utterance-level segmentations in archive to \n"
        "file-level segmentations\n"
        "\n"
        "Usage: segmentation-combine-segments [options] segmentation-in-rspecifier segments-rxfilename segmentation-out-rspecifier\n"
        " e.g.: segmentation-combine-segments ark:utt.seg data/dev/segments ark:file.seg\n";
    
    bool binary = true;
    BaseFloat frame_shift = 0.01;
    std::string segments_rxfilename;
    ParseOptions po(usage);
    
    SegmentationOptions opts;

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");

    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
  
    std::string segmentation_rspecifier = po.GetArg(1),
      segments_rxfilename = po.GetArg(2),
      segmentation_wspecifier = po.GetArg(3);

    RandomAccessSegmentationReader segmentation_reader(segmentation_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);

    Input ki(segments_rxfilename); // no binary argment: never binary.

    int32 num_lines = 0, num_success = 0;

    unordered_map<std::string, BaseFloat, StringHasher> utt2start;
    unordered_map<std::string, std::vector<std::string> > file2utts;

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

      utt2start.insert(std::make_pair(segment, start));

      unordered_map<std::string, std::vector<std::string> > iterator it = file2utts.find(utterance);
      if (it == file2utts.end()) {
        auto ret = file2utts.insert(std::make_pair(utterance, std::vector<std::string>() ) );
        it = ret.first;
      }
      (it->second).push_back(segment);

      /* check whether a segment start time and end time exists in utterance 
       * if fails , skips the segment.
       */ 

      utt
      utt2file.insert(std::make_pair(

        utt2file.insert(std::make_pair(segment, utterance));
        utt2start_time.insert(std::make_pair(segment, start));
      }
    }

    std::vector<int32> merge_labels;
    RandomAccessSegmentationReader filter_reader(opts.filter_rspecifier);

    std::unordered_set<std::string, StringHasher> seen_files;

    if (opts.merge_labels_csl != "") {
      if (!SplitStringToIntegers(opts.merge_labels_csl, ":", false,
            &merge_labels)) {
        KALDI_ERR << "Bad value for --merge-labels option: "
          << opts.merge_labels_csl;
      }
      std::sort(merge_labels.begin(), merge_labels.end());
    }

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier);

    int64  num_done = 0, num_err = 0;
    
    if (!in_is_rspecifier) {
      Segmentation seg;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        seg.Read(ki.Stream(), binary_in);
      }
      Output ko(segmentation_out_fn, binary);
      seg.Write(ko.Stream(), binary);
      KALDI_LOG << "Copied segmentation to " << segmentation_out_fn;
      return 0;
    } else {
      
      Output ko(segmentation_out_fn, false);
      SequentialSegmentationReader reader(segmentation_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++) {
        Segmentation seg(reader.Value());
        std::string key = reader.Key();

        if (opts.filter_rspecifier != "") {
          if (!filter_reader.HasKey(key)) {
            KALDI_WARN << "Could not find filter for utterance " << key;
            num_err++;
            continue;
          }
          const Segmentation &filter_segmentation = filter_reader.Value(key);
          seg.IntersectSegments(filter_segmentation, opts.filter_label);
        }
        
        if (opts.merge_labels_csl != "") {
          seg.MergeLabels(merge_labels, opts.merge_dst_label);
        }

        std::string file_id = key; 
        BaseFloat start_time = 0.0;
        if (segments_rxfilename != "") {
          KALDI_ASSERT(utt2file.count(key) > 0 && utt2start_time.count(key) > 0);
          file_id = utt2file.at(key);
          start_time = utt2start_time.at(key);
        }

        if (seen_files.count(file_id) == 0) {
          ko.Stream() << "SPKR-INFO " << file_id << " 1 <NA> <NA> <NA> unknown SILENCE <NA>\n";
          ko.Stream() << "SPKR-INFO " << file_id << " 1 <NA> <NA> <NA> unknown SPEECH <NA>\n";
          seen_files.insert(file_id);
        }
        seg.WriteRttm(ko.Stream(), file_id, frame_shift, start_time);

      }

      KALDI_LOG << "Copied " << num_done << " segmentation; failed with "
                << num_err << " segmentations";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}





