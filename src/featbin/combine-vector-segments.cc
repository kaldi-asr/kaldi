// featbin/combine-vector-segments.cc

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
#include "matrix/kaldi-matrix.h"

namespace kaldi {
  
  struct Segment {
    std::string utt_id;
    std::string file_id;
    BaseFloat start;
    BaseFloat end;

    void Reset() {
      utt_id = "";
      file_id = "";
      start = -1;
      end = -1;
    }
  };

  bool ProcessSegmentLine(const std::string &line, Segment *seg) {
    seg->Reset();

    std::vector<std::string> split_line;

    // Split the line by space or tab and check the number of fields in each
    // line. There must be 4 fields--segment name , reacording wav file name,
    // start time, end time; 5th field (channel info) is optional.
    SplitStringToVector(line, " \t\r", true, &split_line);
    if (split_line.size() != 4 && split_line.size() != 5) {
      KALDI_WARN << "Invalid line in segments file: " << line;
      return false;
    }

    seg->utt_id = split_line[0];
    seg->file_id = split_line[1];
    
    std::string start_str = split_line[2],
      end_str = split_line[3];

    // Convert the start time and endtime to real from string. Segment is
    // ignored if start or end time cannot be converted to real.
    double start, end;
    if (!ConvertStringToReal(start_str, &start)) {
      KALDI_WARN << "Invalid line in segments file [bad start]: " << line;
      return false;
    }
    if (!ConvertStringToReal(end_str, &end)) {
      KALDI_WARN << "Invalid line in segments file [bad end]: " << line;
      return false;
    }
    // start time must not be negative; start time must not be greater than
    // end time, except if end time is -1
    if (start < 0 || end <= 0 || start >= end) {
      KALDI_WARN << "Invalid line in segments file [empty or invalid segment]: "
        << line;
      return false;
    }
    int32 channel = -1;  // means channel info is unspecified.
    // if each line has 5 elements then 5th element must be channel identifier
    if(split_line.size() == 5) {
      if (!ConvertStringToInteger(split_line[4], &channel) || channel < 0) {
        KALDI_WARN << "Invalid line in segments file [bad channel]: " << line;
        return false;
      }
    }

    seg->start = start;
    seg->end = end;
    return true;
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    
    const char *usage =
        "Combine vectors corresponding to segments to whole vectors. "
        "Does the reverse operation of extract-vector-segments."
        "Usage:  combine-vector-segments [options...] <vecs-rspecifier> <segments-rxfilename> <lengths-rspecifier> <vecs-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);

    BaseFloat min_segment_length = 0.1,  // Minimum segment length in seconds.
              max_overshoot = 0.0;  // max time by which last segment can overshoot
    BaseFloat frame_shift = 0.01;
    BaseFloat default_weight = 0;

    // Register the options
    po.Register("min-segment-length", &min_segment_length,
                "Minimum segment length in seconds (reject shorter segments)");
    po.Register("frame-shift", &frame_shift,
                "Frame shift in second");
    po.Register("max-overshoot", &max_overshoot,
                "End segments overshooting by less (in seconds) are truncated,"
                " else rejected.");
    po.Register("default-weight", &default_weight, "Fill any extra "
                "length with this weight");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string vecs_rspecifier = po.GetArg(1); // input vector archive
    std::string segments_rxfilename = po.GetArg(2);// segment file
    std::string lengths_rspecifier = po.GetArg(3); // lengths archive
    std::string vecs_wspecifier = po.GetArg(4); // output archive

    RandomAccessInt32Reader lengths_reader(lengths_rspecifier);
    BaseFloatVectorWriter vector_writer(vecs_wspecifier);
    RandomAccessBaseFloatVectorReader vector_reader(vecs_rspecifier); 

    Input ki(segments_rxfilename); // no binary argment: never binary.

    int32 num_lines = 0, num_success = 0, num_err = 0;

    std::string line, next_line;
    Segment seg, next_seg;
    int32 overlap = 0;
    
    std::getline(ki.Stream(), line);
    num_lines++;
    if (!ProcessSegmentLine(line, &seg)) {
      num_err++;
    }

    Vector<BaseFloat> *out_vector = NULL;

    /* read each line from segments file */
    bool done = false;
    while (!done) {
      if (!std::getline(ki.Stream(), next_line)) {
        done = true;
      }

      if (!done && !ProcessSegmentLine(next_line, &next_seg)) {
        num_err++;
        continue;
      }

      if (!vector_reader.HasKey(seg.utt_id)) {
        KALDI_WARN << "Did not find vector for utterance " << seg.utt_id
                   << ", skipping segment";
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &vector = vector_reader.Value(seg.utt_id);
      
      if (!lengths_reader.HasKey(seg.file_id)) {
        KALDI_WARN << "Did not find length for file " << seg.file_id
                   << ", skipping segment" << seg.utt_id;
        num_err++;
        continue;
      }

      int32 file_length = lengths_reader.Value(seg.file_id);
      // Convert start & end times of the segment to corresponding sample number
      int32 start_frame = static_cast<int32>(seg.start / frame_shift) + overlap / 2;
      int32 end_frame = static_cast<int32>(seg.end / frame_shift);
 
      if (!done) {
        if (seg.file_id == next_seg.file_id) {
          overlap = static_cast<int32>( (seg.end - next_seg.start) / frame_shift );
        } else {
          overlap = 0;
        }
        end_frame -= overlap / 2;
      } 
          
      /* start frame must be less than total number of samples 
       * otherwise skip the segment
       */
      if (start_frame < 0 || start_frame >= file_length) {
        KALDI_WARN << "start frame out of range " << start_frame << " [length:] "
                   << file_length << ", skipping segment " << seg.utt_id
                   << " of file " << seg.file_id;
        num_err++;
        continue;
      }
      /* end frame must be less than total number samples 
       * otherwise skip the segment
       */
      if (end_frame > file_length) {
        KALDI_ASSERT(done || next_seg.file_id != seg.file_id);
        if (end_frame >
            file_length + static_cast<int32>(max_overshoot / frame_shift)) {
          KALDI_WARN << "end frame too far out of range " << end_frame
                     << " [overshooted length:] " << file_length + static_cast<int32>(max_overshoot / frame_shift) << ", skipping segment "
                     << seg.utt_id << " of file " << seg.file_id;
          num_err++;
          continue;
        }
        end_frame = file_length; // for small differences, just truncate.
      }

      if (out_vector == NULL) 
        out_vector = new Vector<BaseFloat>(file_length);
      
      KALDI_ASSERT(end_frame <= out_vector->Dim());
      SubVector<BaseFloat> this_out_vec(*out_vector, start_frame, end_frame - start_frame);

      KALDI_ASSERT(vector.Dim() >= end_frame - start_frame);

      KALDI_ASSERT(overlap / 2 + end_frame - start_frame <= vector.Dim());
      SubVector<BaseFloat> in_vec(vector, overlap / 2, end_frame - start_frame);

      KALDI_ASSERT(in_vec.Dim() == this_out_vec.Dim());
      this_out_vec.CopyFromVec(in_vec);
      
      if (done || seg.file_id != next_seg.file_id) {
        vector_writer.Write(seg.file_id, *out_vector);
        delete out_vector;
        out_vector = NULL;
      }

      num_success++;

      if (done) break;
      seg = next_seg;
      num_lines++;
    }

    if (out_vector != NULL) {
      delete out_vector;
      out_vector = NULL;
    }

    KALDI_LOG << "Successfully processed " << num_success << " lines out of "
              << num_lines << " in the segments file; "
              << "failed with " << num_err << " segments";
    /* prints number of segments processed */
    if (num_success == 0) return 1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}

