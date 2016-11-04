// segmenterbin/segmentation-copy.cc

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
#include "segmenter/segmentation.h"
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Copy segmentation or archives of segmentation.\n"
        "If label-map is supplied, then apply the mapping to the labels when copying.\n"
        "If utt2label-rspecifier is supplied, then ignore the original labels, "
        "and map all the segments of an utterance using the supplied utt2label map.\n"
        "\n"
        "Usage: segmentation-copy [options] (segmentation-rspecifier|segmentation-rxfilename) (segmentation-wspecifier|segmentation-wxfilename)\n"
        " e.g.: segmentation-copy --binary=false foo -\n"
        "   segmentation-copy ark:1.seg ark,t:-\n";
    
    bool binary = true;
    std::string label_map_rxfilename, utt2label_rspecifier;
    int32 keep_label = -1;
    BaseFloat frame_subsampling_factor = 1;

    ParseOptions po(usage);
    
    po.Register("binary", &binary, 
                "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("label-map", &label_map_rxfilename,
                "File with mapping from old to new labels");
    po.Register("frame-subsampling-factor", &frame_subsampling_factor,
                "Change frame subsampling by this factor");
    po.Register("utt2label-rspecifier", &utt2label_rspecifier,
                "Mapping for each utterance to an integer label");
    po.Register("keep-label", &keep_label,
                "If supplied, only segments of this label are written out");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    // all these "fn"'s are either rspecifiers or filenames.

    std::string segmentation_in_fn = po.GetArg(1),
                segmentation_out_fn = po.GetArg(2);

    // Read mapping from old to new labels
    unordered_map<int32, int32> label_map;
    if (!label_map_rxfilename.empty()) {
      Input ki(label_map_rxfilename);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> splits;
        SplitStringToVector(line, " ", true, &splits);

        if (splits.size() != 2) 
          KALDI_ERR << "Invalid format of line " << line 
                    << " in " << label_map_rxfilename;

        label_map[std::atoi(splits[0].c_str())] = std::atoi(splits[1].c_str());
      }
    }

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
      Segmentation segmentation;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        segmentation.Read(ki.Stream(), binary_in);
      }

      if (!label_map_rxfilename.empty())
        RelabelSegmentsUsingMap(label_map, &segmentation);

      if (keep_label != -1)
        KeepSegments(keep_label, &segmentation);

      if (frame_subsampling_factor != 1.0) {
        ScaleFrameShift(frame_subsampling_factor, &segmentation);
      }

      if (!utt2label_rspecifier.empty()) 
        KALDI_ERR << "It makes no sense to specify utt2label-rspecifier "
                  << "when not reading segmentation archives.";

      Output ko(segmentation_out_fn, binary);
      segmentation.Write(ko.Stream(), binary);

      KALDI_LOG << "Copied segmentation to " << segmentation_out_fn;
      return 0;
    } else {

      RandomAccessInt32Reader utt2label_reader(utt2label_rspecifier);

      SegmentationWriter writer(segmentation_out_fn); 
      SequentialSegmentationReader reader(segmentation_in_fn);

      for (; !reader.Done(); reader.Next(), num_done++) {
        const std::string &key = reader.Key();

        if (label_map_rxfilename.empty() && 
            frame_subsampling_factor == 1.0 && 
            utt2label_rspecifier.empty() && keep_label == -1)
          writer.Write(key, reader.Value());
        else {
          Segmentation segmentation = reader.Value();
          if (!label_map_rxfilename.empty())
            RelabelSegmentsUsingMap(label_map, &segmentation);
          if (!utt2label_rspecifier.empty()) {
            if (!utt2label_reader.HasKey(key)) {
              KALDI_ERR << "Utterance " << key << " not found in utt2label map " 
                        << utt2label_rspecifier;
              continue;
            }

            RelabelAllSegments(utt2label_reader.Value(key), &segmentation);
          }
          if (keep_label != -1)
            KeepSegments(keep_label, &segmentation);

          if (frame_subsampling_factor != 1.0)
            ScaleFrameShift(frame_subsampling_factor, &segmentation);

          writer.Write(key, segmentation);
        }
      }

      KALDI_LOG << "Copied " << num_done << " segmentation; failed with "
                << num_err << " segmentations";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

