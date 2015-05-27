// segmenterbin/segmentation-select-top.cc

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
        "Select top segments from the segmentations and write new segmentation\n"
        "\n"
        "Usage: select-feats-from-segmentation [options] <feats-rspecifier> <segmentation-rspecifier> <feats-wspecifier> \n"
        " e.g.: select-feats-from-segmentation ark:1.feats ark:1.seg ark:-\n";
    
    ParseOptions po(usage);

    SegmentationOptions opts;
    int32 &select_label = opts.merge_dst_label;

    po.Register("select-label", &select_label, "Select frames of only this "
                "class label");
    
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::vector<int32> merge_labels;
    RandomAccessSegmentationReader filter_reader(opts.filter_rspecifier);

    if (opts.merge_labels_csl != "") {
      if (!SplitStringToIntegers(opts.merge_labels_csl, ":", false,
            &merge_labels)) {
        KALDI_ERR << "Bad value for --merge-labels option: "
          << opts.merge_labels_csl;
      }
      std::sort(merge_labels.begin(), merge_labels.end());
    }

    std::string feats_rspecifier = po.GetArg(1),
                segmentation_rspecifier = po.GetArg(2),
                feats_wspecifier = po.GetArg(3);
 
    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
    RandomAccessSegmentationReader segmentation_reader(segmentation_rspecifier);
    BaseFloatMatrixWriter feats_writer(feats_wspecifier);

    int64 num_done = 0, num_err = 0, num_frames_selected = 0, num_frames = 0;

    for (; !feats_reader.Done(); feats_reader.Next()) {
      std::string key = feats_reader.Key();
      if (!segmentation_reader.HasKey(key)) {
        KALDI_WARN << "Could not read segmentation for utterance " << key;
        num_err++;
        continue;
      }
      
      const Matrix<BaseFloat> &feats_in = feats_reader.Value();
      const Segmentation &in_seg = segmentation_reader.Value(key);

      Segmentation seg(in_seg);
      if (opts.filter_rspecifier != "" || opts.merge_labels_csl != "") {
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
      } 

      Matrix<BaseFloat> feats_out(feats_in.NumRows(), feats_in.NumCols());
      int32 j = 0;
      for (std::forward_list<Segment>::const_iterator it = seg.Begin();
            it != seg.End(); ++it) {
        if (it->Label() != select_label) continue;
        SubMatrix<BaseFloat> this_feats_in(feats_in, it->start_frame, it->end_frame - it->start_frame + 1, 0, feats_in.NumCols());
        SubMatrix<BaseFloat> this_feats_out(feats_in, j, it->end_frame - it->start_frame + 1, 0, feats_in.NumCols());
        this_feats_out.CopyFromMat(this_feats_in);
        num_frames_selected += this_feats_in.NumRows();
      }
      num_frames += feats_in.NumRows();

      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " segmentations; "
              << "selected " << num_frames_selected << " out of " 
              << num_frames << " frames";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



