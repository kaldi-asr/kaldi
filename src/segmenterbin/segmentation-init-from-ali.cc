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
    
    ParseOptions po(usage);
    
//SegmentationOptions opts;

    //opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    //std::vector<int32> merge_labels;
    //if (opts.merge_labels_csl != "") {
    //  if (!SplitStringToIntegers(opts.merge_labels_csl, ":", false,
    //        &merge_labels)) {
    //    KALDI_ERR << "Bad value for --merge-labels option: "
    //      << opts.merge_labels_csl;
    //  }
    //  std::sort(merge_labels.begin(), merge_labels.end());
    //}

    std::string ali_rspecifier = po.GetArg(1),
        segmentation_wspecifier = po.GetArg(2);
    
    SequentialInt32VectorReader alignment_reader(ali_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);
    
    int32 num_done = 0;
    int64 num_segmentations = 0;

    std::vector<int64> frames_count_per_class;

    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      std::string key = alignment_reader.Key();
      const std::vector<int32> &alignment = alignment_reader.Value();
      
      Segmentation seg;
      int32 state = -1, start_frame = -1; 
      for (int32 i = 0; i < alignment.size(); i++) {
        if (alignment[i] != state) {  
          // Change of state i.e. a different class id. 
          // So the previous segment has ended.
          if (state != -1) {
            seg.Emplace(start_frame, i-1, state);
            num_segmentations++;
            if (frames_count_per_class.size() <= state) {
              frames_count_per_class.resize(state + 1, 0);
            }
            frames_count_per_class[state] += i - start_frame;
          }
          start_frame = i;
          state = alignment[i];
        }
      }
      seg.Emplace(start_frame, alignment.size()-1, state);
      if (frames_count_per_class.size() <= state) {
        frames_count_per_class.resize(state + 1, 0);
      }
      frames_count_per_class[state] += alignment.size() - start_frame;
      
      num_segmentations++;
      
      segmentation_writer.Write(key, seg);
      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " utterances; "
              << "wrote "
              << num_segmentations << " segmentations.";
    KALDI_LOG << "Number of frames for the different classes are : ";
    WriteIntegerVector(KALDI_LOG, false, frames_count_per_class);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

