// segmenterbin/segmentation-combine-segments.cc

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
        "Combine utterance-level segmentations in an archive to "
        "recording-level segmentations using the kaldi segments to map "
        "utterances to their positions in the recordings.\n"
        "A reco2utt file is used to specify which utterances belong to each "
        "recording.\n"
        "\n"
        "Usage: segmentation-combine-segments [options] "
        "<utt-level-segmentation-rspecifier> "
        "<kaldi-segments-segmentation-rspecifier> "
        "<reco2utt-rspecifier> <segmentation-wspecifier>\n"
        " e.g.: segmentation-combine-segments ark:utt.seg "
        "'ark:segmentation-init-from-segments --shift-to-zero=false "
        "data/dev/segments ark:- |' ark,t:data/dev/reco2utt ark:file.seg\n"
        "See also: segmentation-combine-segments-to-recording, "
        "segmentation-merge, segmentatin-merge-recordings, "
        "segmentation-post-process --merge-adjacent-segments\n";

    bool include_missing = false;

    ParseOptions po(usage);

    po.Register("include-missing-utt-level-segmentations", &include_missing,
                "If true, then the segmentations missing in "
                "utt-level-segmentation-rspecifier is included in the "
                "final output with the label taken from the "
                "kaldi-segments-segmentation-rspecifier");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string utt_segmentation_rspecifier = po.GetArg(1),
                segments_segmentation_rspecifier = po.GetArg(2),
                reco2utt_rspecifier = po.GetArg(3),
                segmentation_wspecifier = po.GetArg(4);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessSegmentationReader segments_segmentation_reader(
        segments_segmentation_rspecifier);
    RandomAccessSegmentationReader utt_segmentation_reader(
        utt_segmentation_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);

    int32 num_done = 0, num_segmentations = 0, num_err = 0;
    int64 num_segments = 0;

    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::vector<std::string> &utts = reco2utt_reader.Value();
      const std::string &reco_id = reco2utt_reader.Key();

      Segmentation out_segmentation;

      for (std::vector<std::string>::const_iterator it = utts.begin();
            it != utts.end(); ++it) {
        if (!segments_segmentation_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find utterance " << *it << " in "
                     << "segments segmentation "
                     << segments_segmentation_rspecifier;
          num_err++;
          continue;
        }

        const Segmentation &segments_segmentation =
          segments_segmentation_reader.Value(*it);
        if (segments_segmentation.Dim() != 1) {
          KALDI_ERR << "Segments segmentation for utt " << *it << " is not "
                    << "kaldi segment converted to segmentation format "
                    << "in " << segments_segmentation_rspecifier;
        }
        const Segment &segment = *(segments_segmentation.Begin());

        if (!utt_segmentation_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find utterance " << *it << " in "
                     << "segmentation " << utt_segmentation_rspecifier;
          if (!include_missing) {
            num_err++;
          } else {
            out_segmentation.PushBack(segment);
            num_segments++;
          }
          continue;
        }

        const Segmentation &utt_segmentation
          = utt_segmentation_reader.Value(*it);
        num_segments += InsertFromSegmentation(utt_segmentation,
                                               segment.start_frame, false,
                                               &out_segmentation, NULL);
        num_done++;
      }

      Sort(&out_segmentation);
      segmentation_writer.Write(reco_id, out_segmentation);
      num_segmentations++;
    }

    KALDI_LOG << "Combined " << num_done << " utterance-level segmentations "
              << "into " << num_segmentations
              << " recording-level segmentations; failed with "
              << num_err << " utterances; "
              << "wrote a total of " << num_segments << " segments.";

    return ((num_done > 0 && num_err < num_done) ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

