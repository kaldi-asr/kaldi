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
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Merge corresponding segments from multiple archives or files.\n"
        "i.e. for each utterance in the first segmentation, the segments "
        "from all the supplied segmentations are merged and put in a single "
        "segmentation."
        "\n"
        "Usage: segmentation-merge [options] <segmentation-rspecifier1> "
        "<segmentation-rspecifier2> ... <segmentation-rspecifier[n]> "
        "<segmentation-wspecifier>\n"
        " e.g.: segmentation-merge ark:foo.seg ark:bar.seg ark,t:-\n"
        "    or \n"
        "       segmentation-merge <segmentation-rxfilename1> "
        "<segmentation-rxfilename2> ... <segmentation-rxfilename[n]> "
        "<segmentation-wxfilename>\n"
        " e.g.: segmentation-merge --binary=false foo bar -\n"
        "See also: segmentation-copy, segmentation-merge-recordings, "
        "segmentation-post-process --merge-labels\n";

    bool binary = true;
    bool sort = true;

    ParseOptions po(usage);

    po.Register("binary", &binary,
                "Write in binary mode "
                "(only relevant if output is a wxfilename)");
    po.Register("sort", &sort, "Sort the segements after merging");

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

    int64 num_done = 0, num_err = 0;

    if (!in_is_rspecifier) {
      Segmentation segmentation;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        segmentation.Read(ki.Stream(), binary_in);
      }

      for (int32 i = 2; i < po.NumArgs(); i++) {
        bool binary_in;
        Input ki(po.GetArg(i), &binary_in);
        Segmentation other_segmentation;
        other_segmentation.Read(ki.Stream(), binary_in);
        ExtendSegmentation(other_segmentation, false,
                           &segmentation);
      }

      Sort(&segmentation);

      Output ko(segmentation_out_fn, binary);
      segmentation.Write(ko.Stream(), binary);

      KALDI_LOG << "Merged segmentations to " << segmentation_out_fn;
      return 0;
    } else {
      SegmentationWriter writer(segmentation_out_fn);
      SequentialSegmentationReader reader(segmentation_in_fn);
      std::vector<RandomAccessSegmentationReader*> other_readers(
          po.NumArgs()-2,
          static_cast<RandomAccessSegmentationReader*>(NULL));

      for (size_t i = 0; i < po.NumArgs()-2; i++) {
        other_readers[i] = new RandomAccessSegmentationReader(po.GetArg(i+2));
      }

      for (; !reader.Done(); reader.Next()) {
        Segmentation segmentation(reader.Value());
        std::string key = reader.Key();

        for (size_t i = 0; i < po.NumArgs()-2; i++) {
          if (!other_readers[i]->HasKey(key)) {
            KALDI_WARN << "Could not find segmentation for key " << key
                       << " in " << po.GetArg(i+2);
            num_err++;
          }
          const Segmentation &other_segmentation =
            other_readers[i]->Value(key);
          ExtendSegmentation(other_segmentation, false,
                             &segmentation);
        }

        Sort(&segmentation);

        writer.Write(key, segmentation);
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

