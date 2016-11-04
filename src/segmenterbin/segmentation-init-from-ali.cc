// segmenterbin/segmentation-init-from-ali.cc

// Copyright 2015-16   Vimal Manohar (Johns Hopkins University)

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
        "Initialize segmentations from alignments file. \n"
        "If segmentation-rspecifier and reco2utt-rspecifier is specified, the \n"
        "segmentation is created to be at recording level\n"
        "\n"
        "Usage: segmentation-init-from-ali [options] <ali-rspecifier> <segmentation-wspecifier> \n"
        " e.g.: segmentation-init-from-ali ark:1.ali ark:-\n";
    
    std::string reco2utt_rspecifier;
    std::string segmentation_rspecifier;
    BaseFloat frame_shift = 0.01;

    ParseOptions po(usage);

    po.Register("reco2utt-rspecifier", &reco2utt_rspecifier, 
                "Use reco2utt and segments files to create file-level "
                "segmentations instead of utterance-level segmentations. "
                "Works in conjunction with --segmentation-rspecifier option.");
    po.Register("segmentation-rspecifier", &segmentation_rspecifier,
                "Utterance-level segmentation from segments file used to "
                "created recording-level segmentations. "
                "Works in conjunction with --reco2utt-rspecifier option.");
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

    if (reco2utt_rspecifier.empty() && segmentation_rspecifier.empty()) {
      SequentialInt32VectorReader alignment_reader(ali_rspecifier);
      
      for (; !alignment_reader.Done(); alignment_reader.Next()) {
        const std::string &key = alignment_reader.Key();
        const std::vector<int32> &alignment = alignment_reader.Value();
        
        Segmentation segmentation;

        num_segments += InsertFromAlignment(alignment, 0, alignment.size(), 
                                            0, &segmentation,
                                            &frame_counts_per_class);
      
        Sort(&segmentation);
        segmentation_writer.Write(key, segmentation);

        num_done++;
        num_segmentations++;
      }
    } else {
      if (reco2utt_rspecifier.empty() || segmentation_rspecifier.empty()) {
        KALDI_ERR << "Require both --reco2utt-rspecifier and "
                  << "--segmentation-rspecifier to be non-empty";
      }
      SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
      RandomAccessSegmentationReader segmentation_reader(segmentation_rspecifier);
      RandomAccessInt32VectorReader alignment_reader(ali_rspecifier);

      for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
        const std::vector<std::string> &utts = reco2utt_reader.Value();
        const std::string &reco_id = reco2utt_reader.Key();

        int32 this_num_segments = 0;

        Segmentation segmentation;
        for (std::vector<std::string>::const_iterator it = utts.begin();
              it != utts.end(); ++it) {
          if (!segmentation_reader.HasKey(*it)) {
            KALDI_WARN << "Could not find utterance " << *it << " in " 
                       << "segments " << segmentation_rspecifier;
            num_err++;
            continue;
          }

          if (!alignment_reader.HasKey(*it)) {
            KALDI_WARN << "Could not find utterance " << *it << " in " 
                       << "alignment " << ali_rspecifier;
            num_err++;
            continue;
          }
          const std::vector<int32> &alignment = alignment_reader.Value(*it);

          const Segmentation &in_segmentation = segmentation_reader.Value(*it);
          if (in_segmentation.Dim() != 1) {
            KALDI_ERR << "Segmentation for utt " << *it << " is not "
                      << "kaldi segment converted to segmentation format "
                      << "in " << segmentation_rspecifier;
          }
          const Segment &segment = *(in_segmentation.Begin());

          this_num_segments += InsertFromAlignment(alignment, 0,
                                                   alignment.size(),
                                                   segment.start_frame,
                                                   &segmentation,
                                                   &frame_counts_per_class);

          num_done++;
        }

        if (this_num_segments > 0) {
          Sort(&segmentation);
          segmentation_writer.Write(reco_id, segmentation);
        }

        num_segments += this_num_segments;
        num_segmentations++;
      }
    }

    KALDI_LOG << "Processed " << num_done << " utterances; failed with "
              << num_err << " utterances; "
              << "wrote " << num_segmentations << " segmentations "
              << "with a total of " << num_segments << " segments.";
    KALDI_LOG << "Number of frames for the different classes are : ";
    WriteIntegerVector(KALDI_LOG, false, frame_counts_per_class);

    return ((num_done > 0 && num_err < num_done) ? 0 : 1); 
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

