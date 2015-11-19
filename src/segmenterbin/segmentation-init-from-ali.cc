// segmenterbin/segmentation-init-from-ali.cc

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
        "Initialize segmentations from alignments file\n"
        "\n"
        "Usage: segmentation-init-from-ali [options] <ali-rspecifier> <segmentation-out-wspecifier> \n"
        " e.g.: segmentation-init-from-ali ark:1.ali ark:-\n";
    
    std::string reco2utt_rspecifier;
    std::string segments_rspecifier;
    BaseFloat frame_shift = 0.01;

    ParseOptions po(usage);

    po.Register("reco2utt-rspecifier", &reco2utt_rspecifier, 
                "Use reco2utt and segments files to create file-level "
                "segmentations instead of utterance-level segmentations. "
                "Works in conjunction with --segments-rspecifier option.");
    po.Register("segments-rspecifier", &segments_rspecifier,
                "Use reco2utt and segments files to create file-level "
                "segmentations instead of utterance-level segmentations. "
                "Works in conjunction with --segments-rspecifier option.");
    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string ali_rspecifier = po.GetArg(1),
        segmentation_wspecifier = po.GetArg(2);
    
    SegmentationWriter segmentation_writer(segmentation_wspecifier);
    
    int32 num_done = 0, num_segmentations = 0;
    int64 num_segments = 0;
    int64 num_err = 0;

    std::vector<int64> frame_counts_per_class;

    if (reco2utt_rspecifier.empty() && segments_rspecifier.empty()) {
      SequentialInt32VectorReader alignment_reader(ali_rspecifier);
      
      for (; !alignment_reader.Done(); alignment_reader.Next()) {
        std::string key = alignment_reader.Key();
        const std::vector<int32> &alignment = alignment_reader.Value();
        
        Segmentation seg;

        num_segments += seg.InsertFromAlignment(alignment, 0, 
                                &frame_counts_per_class);

        segmentation_writer.Write(key, seg);
        num_done++;
        num_segmentations++;
      }
    } else {
      if (reco2utt_rspecifier.empty() || segments_rspecifier.empty()) {
        KALDI_ERR << "Require both --reco2utt-rspecifier and "
                  << "--segments-rspecifier to be non-empty";
      }
      SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
      RandomAccessUtteranceSegmentReader segments_reader(segments_rspecifier);
      RandomAccessInt32VectorReader alignment_reader(ali_rspecifier);

      for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
        const std::vector<std::string> &utts = reco2utt_reader.Value();
        const std::string &reco_id = reco2utt_reader.Key();

        Segmentation seg;
        for (std::vector<std::string>::const_iterator it = utts.begin();
              it != utts.end(); ++it) {
          if (!segments_reader.HasKey(*it)) {
            KALDI_WARN << "Could not find utterance " << *it << " in " 
                       << "segments " << segments_rspecifier;
            num_err++;
            continue;
          }

          if (!alignment_reader.HasKey(*it)) {
            KALDI_WARN << "Could not find utterance " << *it << " in " 
                       << "alignment " << ali_rspecifier;
            num_err++;
            continue;
          }

          const UtteranceSegment &segment = segments_reader.Value(*it);
          const std::vector<int32> &alignment = alignment_reader.Value(*it);
         
          num_segments += seg.InsertFromAlignment(alignment,
                                              segment.start_time / frame_shift, 
                                              &frame_counts_per_class);

          num_done++;
        }
        segmentation_writer.Write(reco_id, seg);
        num_segmentations++;
      }
    }

    KALDI_LOG << "Processed " << num_done << " utterances; failed with "
              << num_err << " utterances; "
              << "wrote " << num_segmentations << " segmentations "
              << "with a total of " << num_segments << " segments.";
    KALDI_LOG << "Number of frames for the different classes are : ";
    WriteIntegerVector(KALDI_LOG, false, frame_counts_per_class);

    return (num_err < num_segmentations ? 0 : 1); 
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

