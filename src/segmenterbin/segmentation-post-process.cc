// segmenterbin/segmentation-post-process.cc

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
        "Post processing of segmentation that does the following operations "
        "in order: \n"
        "1) Intersection or Filtering: Intersects the input segmentation with "
        "segments from the segmentation in 'filter-rspecifier' and retains "
        "only regions where the segment in the filter has the class-id "
        "'filter-label'. See method IntersectSegments() for details.\n"
        "2) Merge labels: Merge labels specified in 'merge-labels' into a "
        "single label 'label'. Any segment that has class_id that is contained "
        "in 'merge-labels' is assigned class_id 'label'. "
        "See method MergeLabels() for details.\n"
        "3) Widen segments: Widen segments of label 'widen-label' by "
        "'widen-length' frames on either side of the segment. This process "
        "also shrinks the adjacent segments so that it does not overlap with "
        "the widened segment or merges the adjacent segment into a composite "
        "segment if they both have the same class_id. "
        "See method WidenSegment() for details.\n"
        "4) with the \n"
        "Usage: segmentation-post-process [options] (segmentation-in-rspecifier|segmentation-in-rxfilename) (segmentation-out-wspecifier|segmentation-out-wxfilename)\n"
        " e.g.: segmentation-post-process --binary=false foo -\n"
        "       segmentation-post-process ark:1.ali ark,t:-\n"
        "See also: segmentation-merge, segmentation-copy, segmentation-remove-segments\n";
    
    bool binary = true;

    ParseOptions po(usage);
    
    SegmentationPostProcessingOptions opts;

    po.Register("binary", &binary, 
                "Write in binary mode (only relevant if output is a wxfilename)");

    opts.Register(&po);

    po.Read(argc, argv); 
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    SegmentationPostProcessor post_processor(opts);
    std::string segmentation_in_fn = po.GetArg(1),
                segmentation_out_fn = po.GetArg(2);

    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";
    
    int64 num_done = 0, num_err = 0;
    
    if (!in_is_rspecifier) {
      Segmentation seg;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        seg.Read(ki.Stream(), binary_in);
      }
      if (post_processor.PostProcess(&seg)) {
        Output ko(segmentation_out_fn, binary);
        seg.Write(ko.Stream(), binary);
        KALDI_LOG << "Post-processed segmentation " << segmentation_in_fn 
                  << " and wrote " << segmentation_out_fn;
        return 0;
      } 
      KALDI_LOG << "Failed post-processing segmentation " 
                << segmentation_in_fn ;
      return 1;
    }

    SegmentationWriter writer(segmentation_out_fn); 
    SequentialSegmentationReader reader(segmentation_in_fn);
    for (; !reader.Done(); reader.Next()){
      Segmentation seg(reader.Value());
      std::string key = reader.Key();

      if (!post_processor.FilterAndPostProcess(&seg, &key)) {
        num_err++;
        continue;
      }
      
      writer.Write(key, seg);
      num_done++;
    }

    KALDI_LOG << "Successfully post-processed " << num_done 
              << " segmentations; "
              << "failed with " << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

