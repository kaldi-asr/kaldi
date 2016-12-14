// segmenterbin/segmentation-init-from-overlap-info.cc

// Copyright 2015-16    Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Convert overlapping segments information into segmentation\n"
        "\n"
        "Usage: segmentation-init-from-additive-signals-info [options] <reco-segmentation-rspecifier> "
        "<additive-signals-info-rspecifier> <segmentation-wspecifier>\n"
        " e.g.: segmentation-init-from-additive-signals-info --additive-signals-segmentation-rspecifier=ark:utt_segmentation.ark "
        "ark:reco_segmentation.ark ark,t:overlapped_segments_info.txt ark:-\n";
    
    BaseFloat frame_shift = 0.01;
    std::string lengths_rspecifier;
    std::string additive_signals_segmentation_rspecifier; 
    std::string unreliable_segmentation_wspecifier;

    ParseOptions po(usage);

    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");
    po.Register("lengths-rspecifier", &lengths_rspecifier, 
                "Archive of lengths for recordings; if provided, will be "
                "used to truncate the output segmentation.");
    po.Register("additive-signals-segmentation-rspecifier", 
                &additive_signals_segmentation_rspecifier,
                "Archive of segmentation of the additive signal which will used "
                "instead of an all 1 segmentation");
    po.Register("unreliable-segmentation-wspecifier",
                &unreliable_segmentation_wspecifier,
                "Applicable when additive-signals-segmentation-rspecifier is "
                "provided and some utterances in it are missing");
                
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string reco_segmentation_rspecifier = po.GetArg(1),
                additive_signals_info_rspecifier = po.GetArg(2),
                segmentation_wspecifier = po.GetArg(3);

    SequentialSegmentationReader reco_segmentation_reader(reco_segmentation_rspecifier);
    RandomAccessTokenVectorReader additive_signals_info_reader(additive_signals_info_rspecifier);
    SegmentationWriter writer(segmentation_wspecifier);
    
    RandomAccessSegmentationReader additive_signals_segmentation_reader(additive_signals_segmentation_rspecifier);
    SegmentationWriter unreliable_writer(unreliable_segmentation_wspecifier);

    RandomAccessInt32Reader lengths_reader(lengths_rspecifier);

    int32 num_done = 0, num_err = 0, num_missing = 0;
  
    for (; !reco_segmentation_reader.Done(); reco_segmentation_reader.Next()) {
      const std::string &key = reco_segmentation_reader.Key();
        
      if (!additive_signals_info_reader.HasKey(key)) {
        KALDI_WARN << "Could not find additive_signals_info for key " << key;
        num_missing++;
        continue;
      }
      const std::vector<std::string> &additive_signals_info = additive_signals_info_reader.Value(key);

      Segmentation segmentation(reco_segmentation_reader.Value());
      Segmentation unreliable_segmentation;

      for (size_t i = 0; i < additive_signals_info.size(); i++) {
        std::vector<std::string> parts;
        SplitStringToVector(additive_signals_info[i], ",:", false, &parts);

        if (parts.size() != 3) {
          KALDI_ERR << "Invalid format of overlap info " << additive_signals_info[i] 
                    << "for key " << key << " in " << additive_signals_info_rspecifier;
        }
        const std::string &utt_id = parts[0];
        double start_time;
        double duration;
        ConvertStringToReal(parts[1], &start_time);
        ConvertStringToReal(parts[2], &duration);

        int32 start_frame = round(start_time / frame_shift);

        if (!additive_signals_segmentation_reader.HasKey(utt_id)) {
          KALDI_WARN << "Could not find utterance " << utt_id << " in "
                     << "segmentation " << additive_signals_segmentation_rspecifier;
          if (duration < 0) {
            KALDI_ERR << "duration < 0 for utt_id " << utt_id << " in "
                      << "additive_signals_info " << additive_signals_info_rspecifier
                      << "; additive-signals-segmentation must be provided in such a case";
          }
          num_err++;
          unreliable_segmentation.EmplaceBack(start_frame, start_frame + duration - 1, 0);
          continue;   // Treated as non-overlapping even though there 
                      // is overlap
        }

        InsertFromSegmentation(additive_signals_segmentation_reader.Value(utt_id),
                               start_frame, false, &segmentation);
      }

      Sort(&segmentation);
      if (!lengths_rspecifier.empty()) {
        if (!lengths_reader.HasKey(key)) {
          KALDI_WARN << "Could not find length for the recording " << key
                     << "in " << lengths_rspecifier;
          continue;
        }
        TruncateToLength(lengths_reader.Value(key), &segmentation);
      }
      writer.Write(key, segmentation);

      if (!unreliable_segmentation_wspecifier.empty()) {
        Sort(&unreliable_segmentation);
        if (!lengths_rspecifier.empty()) {
          if (!lengths_reader.HasKey(key)) {
            KALDI_WARN << "Could not find length for the recording " << key
              << "in " << lengths_rspecifier;
            continue;
          }
          TruncateToLength(lengths_reader.Value(key), &unreliable_segmentation);
        }
        unreliable_writer.Write(key, unreliable_segmentation);
      }

      num_done++;
    }
                     
    KALDI_LOG << "Successfully processed " << num_done << " recordings "
              << " in additive signals info; failed for " << num_missing
              << "; could not get segmentation for " << num_err;

    return (num_done > (num_missing/ 2) ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

