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
        "Usage: segmentation-select-top [options] <segmentation-rspecifier> <score-rspecifier> <segmentation-wspecifier> \n"
        " e.g.: segmentation-select-top ark:1.seg ark:1.log_energies ark:-\n";
    
    ParseOptions po(usage);

    int32 top_select_label = 3, bottom_select_label = 1;
    int32 reject_label = 4;
    int32 num_top_frames = 10000, num_bottom_frames = 2000;
    int32 window_size = 100, min_remainder = 50;
    bool remove_rejected_frames = false; 

    SegmentationOptions opts;
    HistogramOptions hist_opts;

    int32 &src_label = opts.merge_dst_label;

    po.Register("src-label", &src_label, "Select top segments of only this "
                " class label");
    po.Register("num-top-frames", &num_top_frames, "Number of frames to "
                "select from the top half");
    po.Register("num-bottom-frames", &num_bottom_frames, "Number of frames to "
                "select from the bottom half");
    po.Register("top-select-label", &top_select_label, "The label to assign "
                "for the selected top segments");
    po.Register("bottom-select-label", &bottom_select_label, "The label to "
                "assign for the selected bottom segments");
    po.Register("reject-label", &reject_label, "The label assigned to "
                "segments that are binned in histogram but do not make it to "
                "the top or bottom");
    po.Register("window-size", &window_size, "Split segments into windows of "
                "this size");
    po.Register("min-window-remainder", &min_remainder, "Do not split segment "
                "if final piece is smaller than this size");
    po.Register("remove-rejected-frames", &remove_rejected_frames, "If true, "
                "then remove the chunks that are not selected");
    opts.Register(&po);
    hist_opts.Register(&po);

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

    std::string segmentation_rspecifier = po.GetArg(1),
                scores_rspecifier = po.GetArg(2),
                segmentation_wspecifier = po.GetArg(3);
 
    SequentialSegmentationReader segmentation_reader(segmentation_rspecifier);
    RandomAccessBaseFloatVectorReader scores_reader(scores_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);
    
    int64 num_done = 0, num_err = 0, num_selected_top = 0, num_selected_bottom = 0;

    for (; !segmentation_reader.Done(); segmentation_reader.Next()) {
      std::string key = segmentation_reader.Key();
      if (!scores_reader.HasKey(key)) {
        KALDI_WARN << "Could not read scores for utterance " << key;
        num_err++;
        continue;
      }
      
      const Segmentation &in_seg = segmentation_reader.Value();
      const Vector<BaseFloat> &scores = scores_reader.Value(key);

      Segmentation out_seg(in_seg); // Make a copy

      if (opts.filter_rspecifier != "") {
        if (!filter_reader.HasKey(key)) {
          KALDI_WARN << "Could not find filter for utterance " << key;
          num_err++;
          continue;
        }
        const Segmentation &filter_segmentation = filter_reader.Value(key);
        out_seg.IntersectSegments(filter_segmentation, opts.filter_label);
      }
      
      if (opts.merge_labels_csl != "") {
        out_seg.MergeLabels(merge_labels, opts.merge_dst_label);
      }

      out_seg.SplitSegments(window_size, min_remainder);
      
      HistogramEncoder hist_encoder;
      out_seg.CreateHistogram(src_label, scores, hist_opts, &hist_encoder);

      if (top_select_label == -1)
        num_selected_bottom += out_seg.SelectBottomBins(hist_encoder, src_label, 
                                 bottom_select_label, 
                                 reject_label, num_bottom_frames, 
                                 remove_rejected_frames);
      else if (bottom_select_label == -1)
        num_selected_top += out_seg.SelectTopBins(hist_encoder, src_label, 
                              top_select_label, reject_label, num_top_frames,
                              remove_rejected_frames);
      else {
        std::pair<int32,int32> p = out_seg.SelectTopAndBottomBins(hist_encoder, src_label,
                    top_select_label, num_top_frames, 
                    bottom_select_label, num_bottom_frames, reject_label,
                    remove_rejected_frames);
        num_selected_top += p.first;
        num_selected_bottom += p.second;
      }

      segmentation_writer.Write(key, out_seg);
      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " segmentations; "
              << "error in " << num_err << "; "
              << "Selected " << num_selected_top << " and " 
              << num_selected_bottom << " top and bottom frames respectively";

    return (num_done == 0 || num_err >= num_done ? 1 : 0);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


