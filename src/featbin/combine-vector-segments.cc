// featbin/combine-vector-segments.cc

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
        "Combine vectors corresponding to segments to whole vectors. "
        "Does the reverse operation of extract-vector-segments."
        "Usage:  combine-vector-segments [options...] <vecs-rspecifier> <reco2utt-rspecifier> <segments-rspecifier> <lengths-rspecifier> <vecs-wspecifier>\n";

    ParseOptions po(usage);

    BaseFloat min_segment_length = 0.1,  // Minimum segment length in seconds.
              max_overshoot = 0.0;  // max time by which last segment can overshoot
    BaseFloat frame_shift = 0.01;
    BaseFloat default_weight = 0;
    int32 overlap = 0; 
    
    po.Register("min-segment-length", &min_segment_length,
                "Minimum segment length in seconds (reject shorter segments)");
    po.Register("frame-shift", &frame_shift,
                "Frame shift in second");
    po.Register("max-overshoot", &max_overshoot,
                "End segments overshooting by less (in seconds) are truncated,"
                " else rejected.");
    po.Register("default-weight", &default_weight, "Fill any extra "
                "length with this weight");
    po.Register("overlap", &overlap, "Overlap in segments");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string vecs_rspecifier = po.GetArg(1); // input vector archive
    std::string reco2utt_rspecifier = po.GetArg(2);
    std::string segments_rspecifier = po.GetArg(3);
    std::string lengths_rspecifier = po.GetArg(4); // lengths archive
    std::string vecs_wspecifier = po.GetArg(5); // output archive
    
    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessSegmentReader segment_reader(segments_rspecifier);
    RandomAccessBaseFloatVectorReader vector_reader(vecs_rspecifier);
    RandomAccessInt32Reader length_reader(lengths_rspecifier);
    BaseFloatVectorWriter vector_writer(vecs_wspecifier);
    
    int32 num_reco = 0, num_success = 0, num_missing = 0, num_err = 0;

    for (; !reco2utt_reader.Done(); reco2utt_reader.Next(), num_reco++) {
      std::string reco = reco2utt_reader.Key();
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();

      if (!length_reader.HasKey(reco)) {
        KALDI_WARN << "Could not find length for recording " 
                   << reco;
        num_missing++;
      }
      int32 file_length = length_reader.Value(reco);

      Vector<BaseFloat> out_vector(file_length);
      segmenter::Segmentation seg;

      for (std::vector<std::string>::const_iterator it = uttlist.begin();
            it != uttlist.end(); ++it) {
        
        if (!segment_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find utterance " << *it << " in segments "
                     << "file " << segments_rspecifier;
          num_err++;
          continue;
        }
        if (!vector_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find vector for utterance " << *it;
          num_err++;
          continue;
        }
        
        const Segment &segment = segment_reader.Value(*it);
        const Vector<BaseFloat> &vector = vector_reader.Value(*it);
        seg.Emplace(std::round(segment.start_time / frame_shift), 
                    std::round(segment.end_time / frame_shift), 1,
                    vector);
      }

      seg.Sort();
      
      size_t i = 0;
      for (segmenter::SegmentList::iterator it = seg.Begin();
          it != seg.End(); ++it, i++) {
        if (i != 0) {
          it->start_frame += overlap / 2;
        }
        if (i != seg.Dim()) {
          it->end_frame -= overlap / 2;
        }

        if (it->start_frame < 0 || it->start_frame >= file_length) {
          KALDI_WARN << "start frame out of range " << it->start_frame << " [length:] "
            << file_length << ", skipping segment ";
          num_err++;
          continue;
        }

        /* end frame must be less than total number samples 
         * otherwise skip the segment
         */
        if (it->end_frame > file_length) {
          if (it->end_frame >
              file_length + static_cast<int32>(max_overshoot / frame_shift)) {
            KALDI_WARN << "end frame too far out of range " << it->end_frame
              << " [overshooted length:] " << file_length + static_cast<int32>(max_overshoot / frame_shift) << ", skipping segment";
            num_err++;
            continue;
          }
          it->end_frame = file_length; // for small differences, just truncate.
        }

        KALDI_ASSERT(it->end_frame <= out_vector.Dim());
        SubVector<BaseFloat> this_out_vec(out_vector,
            it->start_frame, it->end_frame - it->start_frame);

        const Vector<BaseFloat> &vector = it->VectorValue();
        KALDI_ASSERT(vector.Dim() >= it->end_frame - it->start_frame);

        KALDI_ASSERT(overlap / 2 + it->end_frame - it->start_frame <= vector.Dim());

        SubVector<BaseFloat> in_vec(vector, overlap / 2, it->end_frame - it->start_frame);

        KALDI_ASSERT(in_vec.Dim() == this_out_vec.Dim());
        this_out_vec.CopyFromVec(in_vec);
      }

      vector_writer.Write(reco, out_vector);

      num_success++;
    }
   
    KALDI_LOG << "Read " << num_reco << " recordings and succeeded on "
              << num_success << " recordings; " << num_missing 
              << " recording missing; " << num_err << " utterances "
              << "skipped";

    return (num_success > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


