// ivectorbin/ivector-combine-to-recording.cc

// Copyright 2015   Vimal Manohar

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
#include "base/kaldi-extra-types.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Combine iVectors for utterances into iVector matrix for recording "
        "along with a segmentation corresponding to the utterances\n"
        "Usage:  ivector-combine-to-recording <reco2utt-rxfilename> <segments-rxfilename> <utt-ivector-rspecifier> <segmentation-wspecifier> <reco-ivector-wspecifier>\n"
        "e.g.: \n"
        " ivector-combine-to-recording data/dev_diarized/split10/1/segments ark:exp/ivectors_dev_reco/ivectors_utt.1.ark ark:exp/ivectors_dev_reco/reco_segmentation.1.ark ark:exp/ivectors_dev_reco/ivectors_seg.1.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string reco2utt_rxfilename = po.GetArg(1);
    std::string segments_rspecifier = po.GetArg(2),
                utt_ivectors_rspecifier = po.GetArg(3),
                segmentation_wspecifier = po.GetArg(4),
                ivectors_wspecifier = po.GetArg(5);
    
    SequentialTokenVectorReader reco2utt_reader(reco2utt_rxfilename);
    RandomAccessSegmentReader segment_reader(segments_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(utt_ivectors_rspecifier);
    SegmentationWriter seg_writer(segmentation_wspecifier);
    BaseFloatMatrixWriter ivector_writer(ivectors_wspecifier);
    
    int32 num_reco = 0, num_utts = 0, num_err = 0;

    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      std::string reco = reco2utt_reader.Key();
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();

      bool missing_utt = false;
      Segmentation seg;

      for (std::vector<std::string>::const_iterator it = uttlist.begin();
            it != uttlist.end(); ++it) {
        
        if (!segment_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find utterance " << *it << " in segments "
                     << "file " << segments_rspecifier;
          missing_utt = true;
        }
        if (!ivector_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find iVector for utterance " << *it;
          missing_utt = true;
        }
        
        if (missing_utt) {
          num_err++;
          break;
        }
        
        const Segment &segment = segment_reader.Value(*it);
        const Vector<BaseFloat> &ivector = ivector_reader.Value(*it);
        
        seg.Emplace(std::round(segment.start_time / frame_shift), 
                    std::round(segment.end_time / frame_shift), 1,
                    ivector);
      }

      if (missing_utt) continue;
      seg.Sort();
      segmentation_writer.Write(prev_recording, seg);
      
      Matrix<BaseFloat> ivector_out(seg.Dim(), ivector.Dim());

      size_t i = 0;
      for (SegmentList::const_iterator it = seg.Begin();
          it != seg.End(); seg.Next(), i++) {
        ivector_out.CopyRowFromVec(i, CopyFromVec(it->Value()));
      }
      ivector_writer.Write(prev_recording, ivector_out);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

