// segmenterbin/segmentation-create-subsegments.cc

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
        "Create subsegmentation of a segmentation based on another segmentation."
        "The intersection of the segmentation are assiged a new label specified "
        "by subsegment-label\n"
        "\n"
        "Usage: segmentation-create-subsegments --subsegment-label=1000 --filter-label=10 [options] (segmentation-in-rspecifier|segmentation-in-rxfilename) (filter-segmentation-in-rspecifier|filter-segmentation-out-rxfilename) (segmentation-out-wspecifier|segmentation-out-wxfilename)\n"
        " e.g.: segmentation-copy --binary=false foo -\n"
        "   segmentation-copy ark:1.ali ark,t:-\n";
    
    bool binary = true, ignore_missing = true;
    int32 filter_label = -1, subsegment_label = -1;
    ParseOptions po(usage);
    
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("filter-label", &filter_label, "The label on which the "
                "filtering is done");
    po.Register("subsegment-label", &subsegment_label, "If non-negative, "
                "change the label of "
                "the intersection of the two segmentations to this integer.");
    po.Register("ignore-missing", &ignore_missing, "Ignore missing "
                "segmentations in filter");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
                filter_segmentation_in_fn = po.GetArg(2),
                segmentation_out_fn = po.GetArg(3);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        filter_is_rspecifier = 
        (ClassifyRspecifier(filter_segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier || in_is_rspecifier != filter_is_rspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";
    
    int64  num_done = 0, num_err = 0;
    
    if (!in_is_rspecifier) {
      Segmentation seg;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        seg.Read(ki.Stream(), binary_in);
      }
      Segmentation filter_seg;
      {
        bool binary_in;
        Input ki(filter_segmentation_in_fn, &binary_in);
        filter_seg.Read(ki.Stream(), binary_in);
      }
      seg.CreateSubSegments(filter_seg, filter_label, subsegment_label);
      Output ko(segmentation_out_fn, binary);
      seg.Write(ko.Stream(), binary);
      KALDI_LOG << "Copied segmentation to " << segmentation_out_fn;
      return 0;
    } else {
      SegmentationWriter writer(segmentation_out_fn); 
      SequentialSegmentationReader reader(segmentation_in_fn);
      RandomAccessSegmentationReader filter_reader(filter_segmentation_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++) {
        Segmentation seg(reader.Value());
        std::string key = reader.Key();
        
        if (!filter_reader.HasKey(key)) {
          KALDI_WARN << "Could not find filter for utterance " << key;
          if (!ignore_missing) {
            num_err++;
          } else 
            writer.Write(key, seg);
          continue;
        }
        const Segmentation &filter_segmentation = filter_reader.Value(key);
        
        seg.CreateSubSegments(filter_segmentation, filter_label, subsegment_label);

        writer.Write(key, seg);
      }

      KALDI_LOG << "Created subsegments for " << num_done << " segmentations; failed with "
                << num_err << " segmentations";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



