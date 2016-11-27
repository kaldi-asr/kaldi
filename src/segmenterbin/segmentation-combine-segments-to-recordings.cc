// segmenterbin/segmentation-combine-segments-to-recordings.cc

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
        "Combine kaldi segments in segmentation format to "
        "recording-level segmentation\n"
        "A reco2utt file is used to specify which utterances are contained "
        "in a recording.\n"
        "This program expects the input segmentation to be a kaldi segment "
        "converted to segmentation using segmentation-init-from-segments. "
        "For other segmentations, the user can use the binary "
        "segmentation-combine-segments instead.\n"
        "\n"
        "Usage: segmentation-combine-segments-to-recording [options] "
        "<segmentation-rspecifier> <reco2utt-rspecifier> "
        "<segmentation-wspecifier>\n"
        " e.g.: segmentation-combine-segments-to-recording \\\n"
        "'ark:segmentation-init-from-segments --shift-to-zero=false "
        "data/dev/segments ark:- |' ark,t:data/dev/reco2utt ark:file.seg\n"
        "See also: segmentation-combine-segments, "
        "segmentation-merge, segmentation-merge-recordings, "
        "segmentation-post-process --merge-adjacent-segments\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_rspecifier = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      segmentation_wspecifier = po.GetArg(3);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessSegmentationReader segmentation_reader(
        segmentation_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);

    int32 num_done = 0, num_segmentations = 0, num_err = 0;

    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::vector<std::string> &utts = reco2utt_reader.Value();
      const std::string &reco_id = reco2utt_reader.Key();

      Segmentation out_segmentation;

      for (std::vector<std::string>::const_iterator it = utts.begin();
            it != utts.end(); ++it) {
        if (!segmentation_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find utterance " << *it << " in "
                     << "segments segmentation "
                     << segmentation_rspecifier;
          num_err++;
          continue;
        }

        const Segmentation &segmentation = segmentation_reader.Value(*it);
        if (segmentation.Dim() != 1) {
          KALDI_ERR << "Segments segmentation for utt " << *it << " is not "
                    << "kaldi segment converted to segmentation format "
                    << "in " << segmentation_rspecifier;
        }
        const Segment &segment = *(segmentation.Begin());

        out_segmentation.PushBack(segment);

        num_done++;
      }

      Sort(&out_segmentation);
      segmentation_writer.Write(reco_id, out_segmentation);
      num_segmentations++;
    }

    KALDI_LOG << "Combined " << num_done << " utterance-level segments "
              << "into " << num_segmentations
              << " recording-level segmentations; failed with "
              << num_err << " utterances.";

    return ((num_done > 0 && num_err < num_done) ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

