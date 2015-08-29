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
    int32 selection_padding = 0;

    po.Register("select-label", &select_label, "Select frames of only this "
                "class label");
    po.Register("selection-padding", &selection_padding, "If this is > 0, then "
                "this number of frames at the boundary are not selected."
                "Similar to program select-interior-frames.");
    
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::vector<int32> merge_labels;

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
      if (opts.merge_labels_csl != "") {
        seg.MergeLabels(merge_labels, opts.merge_dst_label);
      }

      Matrix<BaseFloat> feats_out(feats_in.NumRows(), feats_in.NumCols());
      int32 j = 0;
      for (SegmentList::const_iterator it = seg.Begin();
            it != seg.End(); ++it) {
        if (it->Label() != select_label || 
            it->end_frame - it->start_frame + 1 <= 2 * selection_padding) continue;
        const SubMatrix<BaseFloat> this_feats_in(feats_in, 
            it->start_frame + selection_padding, 
            it->end_frame - it->start_frame + 1 - 2 * selection_padding, 
            0, feats_in.NumCols());
        SubMatrix<BaseFloat> this_feats_out(feats_out, j, 
            it->end_frame - it->start_frame + 1 - 2 * selection_padding, 
            0, feats_in.NumCols());
        this_feats_out.CopyFromMat(this_feats_in);
        j += this_feats_in.NumRows();
        num_frames_selected += this_feats_in.NumRows();
      }

      num_frames += feats_in.NumRows();
      // If no frames are selected, then we don't write anything
      if (j > 0) {
        feats_out.Resize(j, feats_in.NumCols(), kCopyData);
        feats_writer.Write(key, feats_out);
      }
      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " segmentations; "
              << "selected " << num_frames_selected << " out of " 
              << num_frames << " frames";

    return (num_frames > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

