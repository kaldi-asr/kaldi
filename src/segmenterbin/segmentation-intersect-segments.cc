// segmenterbin/segmentation-intersect-segments.cc

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
        "Intersect segments from two archives\n"
        "\n"
        "Usage: segmentation-intersect-segments [options] (segmentation-rpecifier1|segmentation-rxfilename1) (segmentation-rspecifier2|segmentation-rxfilename2) (segmentation-wspecifier|segmentation-wxfilename)\n"
        " e.g.: segmentation-intersect-segments --binary=false foo bar -\n"
        "   segmentation-intersect-segments ark:foo.seg ark:bar.seg ark,t:-\n"
        "See also: segmentation-merge, segmentation-copy, segmentation-post-process --merge-labels\n";
    
    bool binary = true;
    int32 mismatch_label = -1;

    ParseOptions po(usage);
    
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("mismatch-label", &mismatch_label, "Label to be added for the "
                "mismatch segments");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
                secondary_segmentation_in_fn = po.GetArg(2),
                segmentation_out_fn = po.GetArg(3);


    // all these "fn"'s are either rspecifiers or filenames.
    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != (ClassifyRspecifier(secondary_segmentation_in_fn, NULL, NULL) != kNoRspecifier) ||
        in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";
    
    int64  num_done = 0, num_err = 0;
    
    if (!in_is_rspecifier) {
      Segmentation seg;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        seg.Read(ki.Stream(), binary_in);
      }
      
      Segmentation secondary_seg;
      {
        bool binary_in;
        Input ki(secondary_segmentation_in_fn, &binary_in);
        secondary_seg.Read(ki.Stream(), binary_in);
      }

      Segmentation out_seg;
      seg.IntersectSegments(secondary_seg, &out_seg, mismatch_label);

      Output ko(segmentation_out_fn, binary);
      out_seg.Write(ko.Stream(), binary);
      KALDI_LOG << "Intersected segmentations " << segmentation_in_fn
                << " and " << secondary_segmentation_in_fn << "; wrote "
                << segmentation_out_fn;
      return 0;
    } else {
      SegmentationWriter writer(segmentation_out_fn); 
      SequentialSegmentationReader primary_reader(segmentation_in_fn);
      RandomAccessSegmentationReader secondary_reader(secondary_segmentation_in_fn);

      for (; !primary_reader.Done(); primary_reader.Next()) {
        const Segmentation &seg = primary_reader.Value();
        const std::string &key = primary_reader.Key();

        if (!secondary_reader.HasKey(key)) {
          KALDI_WARN << "Could not find segmentation for key " << key
                     << " in " << secondary_segmentation_in_fn;
          num_err++;
          continue;
        } 
        const Segmentation &secondary_seg = secondary_reader.Value(key);

        Segmentation out_seg;
        seg.IntersectSegments(secondary_seg, &out_seg, mismatch_label);
        out_seg.Sort();

        writer.Write(key, out_seg);
        num_done++;
      }

      KALDI_LOG << "Intersected " << num_done << " segmentations; failed with "
                << num_err << " segmentations";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


