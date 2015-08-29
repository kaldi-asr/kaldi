// ivectorbin/ivector-split-to-segments.cc

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
#include "segmenter/segmenter.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Split iVectors for recording into new segments\n"
        "Usage:  ivector-split-to-segments [options] <reco-ivector-rspecifier> "
        "<reco-segmentation-rspecifier> <reco2utt-rspecifier> <segments-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        " ivector-split-to-segments "
        "ark:exp/ivectors_dev_reco/ivectors_reco.1.ark "
        "ark:exp/ivectors_dev_reco/reco_segmentation.1.ark ark:data/dev_uniformsegmented_win10_over5/split10/1/reco2utt "
        "ark:data/dev_uniformsegmented_win10_over5/split10/1/segments ark:- | "
        "paste-feats ark:feats.1.ark ark:- "
        "ark:exp/ivectors_dev_uniformsegmented_win10_over5/ivector_online.ark\n";

    ParseOptions po(usage);
    BaseFloat frame_shift = 0.01;
    int32 offset_frames = 2;

    po.Register("frame-shift", &frame_shift,
                "Frame shift in second");
    po.Register("offset-frames", &offset_frames,
                "Number of frames to reduce output iVector size by, to "
                "adjust boundary to match feature length");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivectors_rspecifier = po.GetArg(1), 
                segmentations_rspecifier = po.GetArg(2),
                reco2utt_rspecifier = po.GetArg(3),
                segments_rspecifier = po.GetArg(4),
                ivectors_wspecifier = po.GetArg(5);
    
    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessSegmentReader segment_reader(segments_rspecifier);
    segmenter::RandomAccessSegmentationReader reco_seg_reader(segmentations_rspecifier); // Corresponding to the read reco iVectors
    RandomAccessBaseFloatMatrixReader ivector_reader(ivectors_rspecifier);
    BaseFloatMatrixWriter ivector_writer(ivectors_wspecifier);

    int32 num_reco = 0, num_success = 0, num_err = 0;
    
    int32 ivector_dim = -1;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next(), num_reco++) {
      std::string reco = reco2utt_reader.Key();
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();

      if (!reco_seg_reader.HasKey(reco)) {
        KALDI_WARN << "Could not read segmentation for recording " << reco;
        continue;
      }

      if (!ivector_reader.HasKey(reco)) {
        KALDI_WARN << "Could not find iVector for recording " << reco;
        continue;
      }

      const Matrix<BaseFloat>& ivector_in = ivector_reader.Value(reco);
      segmenter::Segmentation reco_seg(reco_seg_reader.Value(reco));
     
      ivector_dim = ivector_in.NumCols();

      KALDI_ASSERT(ivector_in.NumRows() == reco_seg.Dim());
      size_t i = 0;
      for (segmenter::SegmentList::iterator it = reco_seg.Begin(); 
            it != reco_seg.End(); ++it, i++) {
        it->SetVectorValue(SubVector<BaseFloat>(ivector_in, i));
      }
      KALDI_ASSERT(i == ivector_in.NumRows());

      // reco_seg is sorted because it is written that way.
      // This can be checked if needed.
      
      // Convert the segments file to segmentation
      segmenter::Segmentation segments_seg;
      for (std::vector<std::string>::const_iterator it = uttlist.begin();
            it != uttlist.end(); ++it) {

        if (!segment_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find utterance " << *it << " in segments "
                     << "file " << segments_rspecifier;
          num_err++;
          continue;
        }
        const Segment &segment = segment_reader.Value(*it);


        segments_seg.Emplace(std::round(segment.start_time / frame_shift), 
                        std::round(segment.end_time / frame_shift), 1, *it);
      }

      segmenter::Segmentation new_seg;
      segments_seg.CreateSubSegments(reco_seg, 1, 1, &new_seg); // data_seg is no longer sorted

      segmenter::SegmentList::iterator new_it = new_seg.Begin();
      for (segmenter::SegmentList::iterator utt_it = segments_seg.Begin();
            utt_it != segments_seg.End(); ++utt_it) {
        // Effectively doing "For each segment in segments file"
        
        // start_frame and end_frame are all reco-level for both utt_it and
        // seg_it
       
        // Create iVector matrix for the segment in segments file.
        // Offset frames is to correct for the fact that feats extracted for 
        // the segment will have about 2 frames less at the end.
        Matrix<BaseFloat> ivector(utt_it->end_frame - utt_it->start_frame - offset_frames, ivector_dim);

        KALDI_ASSERT(new_it->StringValue() == utt_it->StringValue()); 
        // By the way the CreateSubSegments function is written, this
        // must be true

        for (; new_it != new_seg.End() && new_it->StringValue() == utt_it->StringValue(); ++new_it) {
          size_t num_frames = new_it->end_frame - new_it->start_frame;
          if (new_it->end_frame > utt_it->end_frame - offset_frames) {
            num_frames -= offset_frames;
          }
          // Copy iVector for the subsegment into the iVector matrix in
          // the segments file
          SubMatrix<BaseFloat> this_ivector(ivector, new_it->start_frame - utt_it->start_frame, num_frames, 0, ivector.NumCols());
          this_ivector.CopyRowsFromVec(new_it->VectorValue());
          KALDI_ASSERT(this_ivector(0,0) != 0);

          KALDI_VLOG(2) << utt_it->StringValue() << " " << new_it->start_frame << " " << new_it->end_frame << " " << new_it->VectorValue();
        } 

        ivector_writer.Write(utt_it->StringValue(), ivector);
      }
      num_success++;
    }

    KALDI_LOG << "Split iVectors for " << num_success
              << " out of " << num_reco << " recordings; " 
              << " errors in " << num_err << " segments";
    return (num_success > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


