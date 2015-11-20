// bin/extract-int-vector-segments.cc

// Copyright 2009-2011  Microsoft Corporation;  Govivace Inc.
//           2012-2013  Mirko Hannemann;  Arnab Ghoshal
//           2015       Vimal Manohar (Johns Hopkins University)

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
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Extract integer vectors corresponding to segments from whole vectors\n"
        "Usage:  extract-int-vector-segments [options...] <ali-rspecifier> <segments-rxfilename> <ali-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);

    BaseFloat min_segment_length = 0.1,  // Minimum segment length in seconds.
              max_overshoot = 0.0;  // max time by which last segment can overshoot
    int32 frame_shift = 10;
    int32 frame_length = 25;
    bool snip_edges = true;

    // Register the options
    po.Register("min-segment-length", &min_segment_length,
                "Minimum segment length in seconds (reject shorter segments)");
    po.Register("frame-length", &frame_length, "Frame length in milliseconds");
    po.Register("frame-shift", &frame_shift, "Frame shift in milliseconds");
    po.Register("max-overshoot", &max_overshoot,
                "End segments overshooting by less (in seconds) are truncated,"
                " else rejected.");
    po.Register("snip-edges", &snip_edges,
        "If true, n_frames frames will be snipped from the beginning of each "
        "extracted feature matrix, "
        "where n_frames = ceil((frame_length - frame_shift) / frame_shift), "
        "except for the segments at the beginning of a file, where "
        "the snipping is done from the end. "
        "This ensures that only the feature vectors that "
        "completely fit in the segment are extracted. "
        "This makes the extracted segment lengths match the lengths of the "
        "features that have been extracted from already segmented audio.");


    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ali_rspecifier = po.GetArg(1); // get script file/vector archive
    std::string segments_rspecifier = po.GetArg(2);// get segment file
    std::string ali_wspecifier = po.GetArg(3); // get written archive name

    Int32VectorWriter alignment_writer(ali_wspecifier);
    SequentialUtteranceSegmentReader segments_reader(segments_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(ali_rspecifier); 


    int32 num_done = 0, num_err = 0;

    int32 snip_length = 0;
    if (snip_edges) {
      snip_length = static_cast<int32>(ceil(
          1.0 * (frame_length - frame_shift) / frame_shift));
    }
    
    for (; !segments_reader.Done(); segments_reader.Next()) {
      const std::string &seg_id = segments_reader.Key();
      const UtteranceSegment &segment = segments_reader.Value();

      if (!alignment_reader.HasKey(segment.reco_id)) {
        KALDI_WARN << "Did not find vector for utterance " << segment.reco_id 
                   << ", skipping segment " << segment.reco_id;
        num_err++;
        continue;
      }
      const std::vector<int32> &ali = alignment_reader.Value(segment.reco_id);

      // total number of samples present in alignment
      int32 num_samp = ali.size();
      // Convert start & end times of the segment to corresponding sample number
      int32 start_samp = static_cast<int32>(std::round(
          (segment.start_time * 1000.0 / frame_shift)));
      int32 end_samp = static_cast<int32>(std::round(segment.end_time * 1000.0 / frame_shift + 0.0495));

      if (snip_edges) {
        // snip the edge at the end of the segment (usually 2 frames),
        end_samp -= snip_length;
      }
      
      /* start sample must be less than total number of samples 
       * otherwise skip the segment
       */
      if (start_samp < 0 || start_samp >= num_samp) {
        KALDI_WARN << "Start sample out of range " << start_samp << " [length:] "
                   << num_samp << ", skipping segment " << seg_id;
        num_err++;
        continue;
      }

      /* end sample must be less than total number samples 
       * otherwise skip the segment
       */
      if (end_samp > num_samp) {
        if (end_samp >
            num_samp + static_cast<int32>(max_overshoot / frame_shift)) {
          KALDI_WARN << "End sample too far out of range " << end_samp
                     << " [overshooted length:] " 
                     << num_samp + static_cast<int32>(max_overshoot / frame_shift) 
                     << ", skipping segment " << seg_id;
          num_err++;
          continue;
        }
        end_samp = num_samp; // for small differences, just truncate.
      }

      /* check whether the segment size is less than minimum segment length(default 0.1 sec)
       * if yes, skip the segment
       */
      if (end_samp
          <= start_samp
              + static_cast<int32>(round(
                  (min_segment_length * 1000.0 / frame_shift)))) {
        KALDI_WARN<< "Segment " << seg_id << " too short, skipping it.";
        num_err++;
        continue;
      }

      std::vector<int32> seg_ali(ali.begin() + start_samp, 
                                 ali.begin() + end_samp);

      alignment_writer.Write(seg_id, seg_ali);
      num_done++;
    }
    KALDI_LOG << "Successfully processed " << num_done << " segments; failed "
              << num_err << " segments.";
    /* prints number of segments processed */
    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

