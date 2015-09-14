// segmenterbin/segmentation-merge.cc

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
        "Merge corresponding segments from two archives\n"
        "\n"
        "Usage: segmentation-merge [options] (segmentation-rpecifier1|segmentation-rxfilename1) (segmentation-rspecifier2|segmentation-rxfilename2) ... (segmentation-rspecifier<n>|segmentation-rxfilename<n>) (segmentation-wspecifier|segmentation-wxfilename)\n"
        " e.g.: segmentation-merge --binary=false foo bar -\n"
        "   segmentation-merge ark:foo.seg ark:bar.seg ark,t:-\n"
        "See also: segmentation-copy, segmentation-post-process --merge-labels\n";
    
    bool binary = true;
    ParseOptions po(usage);
    
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");

    po.Read(argc, argv);

    if (po.NumArgs() <= 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
                segmentation_out_fn = po.GetArg(po.NumArgs());


    // all these "fn"'s are either rspecifiers or filenames.
    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";
    
    int64  num_done = 0, num_err = 0;
    
    if (!in_is_rspecifier) {
      Segmentation seg;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        seg.Read(ki.Stream(), binary_in);
      }

      for (int32 i = 2; i < po.NumArgs(); i++) {
        bool binary_in;
        Input ki(po.GetArg(i), &binary_in);
        Segmentation other_seg;
        other_seg.Read(ki.Stream(), binary_in);
        seg.Merge(other_seg, false);
      }

      Output ko(segmentation_out_fn, binary);
      seg.Write(ko.Stream(), binary);
      KALDI_LOG << "Copied segmentation to " << segmentation_out_fn;
      return 0;
    } else {
      SegmentationWriter writer(segmentation_out_fn); 
      SequentialSegmentationReader reader(segmentation_in_fn);
      std::vector<RandomAccessSegmentationReader*> other_readers(po.NumArgs()-2, static_cast<RandomAccessSegmentationReader*>(NULL));

      for (size_t i = 0; i < po.NumArgs()-2; i++) {
        other_readers[i] = new RandomAccessSegmentationReader(po.GetArg(i+2));
      }

      for (; !reader.Done(); reader.Next()) {
        Segmentation seg(reader.Value());
        std::string key = reader.Key();

        for (size_t i = 0; i < po.NumArgs()-2; i++) {
          if (!other_readers[i]->HasKey(key)) {
            KALDI_WARN << "Could not find segmentation for key " << key
                       << "; in " << po.GetArg(i+2);
            num_err++;
          }
          const Segmentation &other_seg = other_readers[i]->Value(key);
          seg.Merge(other_seg, false);
        }
        seg.Sort();

        writer.Write(key, seg);
        num_done++;
      }

      KALDI_LOG << "Merged " << num_done << " segmentation; failed with "
                << num_err << " segmentations";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}




