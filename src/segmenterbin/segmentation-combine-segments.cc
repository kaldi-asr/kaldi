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
        "Combine utterance-level segmentations in an archive to file-level "
        "segmentations using the kaldi segments to map utterances to "
        "file.\n"
        "\n"
        "Usage: segmentation-combine-segments [options] <segmentation-in-rspecifier> <segments-rspecifier> <reco2utt-rspecifier> <segmentation-out-rspecifier>\n"
        " e.g.: segmentation-combine-segments ark:utt.seg ark,t:data/dev/segments ark,t:data/dev/reco2utt ark:file.seg\n";
    
    bool binary = true;
    BaseFloat frame_shift = 0.01;
    ParseOptions po(usage);
    
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("frame-shift", &frame_shift, "Frame shift in seconds");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
  
    std::string segmentation_rspecifier = po.GetArg(1),
                segments_rspecifier = po.GetArg(2),
                reco2utt_rspecifier = po.GetArg(3),
                segmentation_wspecifier = po.GetArg(4);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessUtteranceSegmentReader segments_reader(segments_rspecifier);
    RandomAccessSegmentationReader  segmentation_reader(segmentation_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);

    int32 num_done = 0, num_segmentations = 0;
    int64 num_segments = 0;
    int64 num_err = 0;
    
    std::vector<int64> frame_counts_per_class;
    
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
        
        if (!segmentation_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find utterance " << *it << " in " 
                     << "segmentation " << segmentation_rspecifier;
          num_err++;
          continue;
        }
        const UtteranceSegment &segment = segments_reader.Value(*it);
        const Segmentation &utt_seg = segmentation_reader.Value(*it);

        num_segments += seg.InsertFromSegmentation(utt_seg, 
                                            segment.start_time / frame_shift, 
                                            &frame_counts_per_class);
        num_done++;
      }
      segmentation_writer.Write(reco_id, seg);
      num_segmentations++;
    }

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}





