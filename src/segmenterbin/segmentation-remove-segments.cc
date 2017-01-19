// segmenterbin/segmentation-remove-segments.cc

// Copyright 2015-16   Vimal Manohar (Johns Hopkins University)

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
        "Remove segments of particular class_id (e.g silence or noise) "
        "or a set of class_ids.\n"
        "The labels to removed can be made utterance-specific by passing "
        "--remove-labels-rspecifier option.\n"
        "\n"
        "Usage: segmentation-remove-segments [options] "
        "<segmentation-rspecifier> <segmentation-wspecifier>\n"
        "  or : segmentation-remove-segments [options] "
        "<segmentation-rxfilename> <segmentation-wxfilename>\n"
        "\n"
        " e.g.: segmentation-remove-segments --remove-label=0 ark:foo.ark "
        "ark:foo.speech.ark\n"
        "See also: segmentation-post-process --remove-labels, "
        "segmentation-post-process --max-blend-length, segmentation-copy\n";

    bool binary = true;

    int32 remove_label = -1;
    int32 max_remove_length = -1;
    std::string remove_labels_rspecifier = "";

    ParseOptions po(usage);

    po.Register("binary", &binary,
                "Write in binary mode "
                "(only relevant if output is a wxfilename)");
    po.Register("remove-label", &remove_label, "Remove segments of this label");
    po.Register("remove-labels-rspecifier", &remove_labels_rspecifier,
                "Specify colon separated list of labels for each key");
    po.Register("max-remove-length", &max_remove_length,
                "If supplied, this specifies the maximum length of segments "
                "will be removed. A value of -1 specifies a length of "
                "+infinity i.e. segments will be removed based "
                "on only their labels and irrespective of their lengths.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
      segmentation_out_fn = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";

    int64 num_done = 0, num_missing = 0;

    if (!in_is_rspecifier) {
      Segmentation segmentation;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        segmentation.Read(ki.Stream(), binary_in);
      }
      if (!remove_labels_rspecifier.empty()) {
        KALDI_ERR << "It does not make sense to specify "
                  << "--remove-labels-rspecifier "
                  << "for single segmentation";
      }

      RemoveSegments(remove_label, &segmentation);

      {
        Output ko(segmentation_out_fn, binary);
        segmentation.Write(ko.Stream(), binary);
      }

      KALDI_LOG << "Removed segments and wrote segmentation to "
                << segmentation_out_fn;

      return 0;
    } else {
      SegmentationWriter writer(segmentation_out_fn);
      SequentialSegmentationReader reader(segmentation_in_fn);

      RandomAccessTokenReader remove_labels_reader(remove_labels_rspecifier);

      for (; !reader.Done(); reader.Next(), num_done++) {
        Segmentation segmentation(reader.Value());
        std::string key = reader.Key();

        if (!remove_labels_rspecifier.empty()) {
          if (!remove_labels_reader.HasKey(key)) {
            KALDI_WARN << "No remove-labels found for recording " << key;
            num_missing++;
            writer.Write(key, segmentation);
            continue;
          }

          std::vector<int32> remove_labels;
          const std::string& remove_labels_str =
            remove_labels_reader.Value(key);

          if (!SplitStringToIntegers(remove_labels_str, ":,", false,
                                     &remove_labels)) {
            KALDI_ERR << "Bad colon-separated list "
                      << remove_labels_str << " for key " << key
                      << " in " << remove_labels_rspecifier;
          }

          remove_label = remove_labels[0];

          RemoveSegments(remove_labels, max_remove_length, &segmentation);
        } else {
          RemoveSegments(remove_label, &segmentation);
        }
        writer.Write(key, segmentation);
      }

      KALDI_LOG << "Removed segments " << "from " << num_done
                << " segmentations; "
                << "remove-labels list missing for " << num_missing;
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

