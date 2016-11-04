// segmenterbin/segmentation-post-process.cc

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
#include "segmenter/segmentation-post-processor.h"
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Post processing of segmentation that does the following operations "
        "in order: \n"
        "1) Merge labels: Merge labels specified in 'merge-labels' into a "
        "single label 'label'. Any segment that has class_id that is contained "
        "in 'merge-labels' is assigned class_id 'label'. "
        "See method MergeLabels() for details.\n"
        "2) Padding segments: \n"
        "3) Shrink segments: \n"
        "4) Blend segments with neighbors: \n"
        "5) Remove segments: \n"
        "6) Merge adjacent segments: \n"
        "7) Split segments: \n"
        "Usage: segmentation-post-process [options] <segmentation-rspecifier> <segmentation-wspecifier>\n"
        "  or : segmentation-post-process [options] <segmentation-rxfilename> (segmentation-wxfilename)\n"
        " e.g.: segmentation-post-process --binary=false foo -\n"
        "       segmentation-post-process ark:foo.seg ark,t:-\n"
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
      Segmentation segmentation;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        segmentation.Read(ki.Stream(), binary_in);
      }
      if (post_processor.PostProcess(&segmentation)) {
        Output ko(segmentation_out_fn, binary);
        Sort(&segmentation);
        segmentation.Write(ko.Stream(), binary);
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
      Segmentation segmentation(reader.Value());
      const std::string &key = reader.Key();

      if (!post_processor.PostProcess(&segmentation)) {
        num_err++;
        continue;
      }
      
      Sort(&segmentation);

      writer.Write(key, segmentation);
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

