// cudadecoder/cuda-pipeline-common.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <limits>
#include <random>
#if HAVE_CUDA == 1

#include "cudadecoder/cuda-pipeline-common.h"

#define KALDI_CUDA_DECODER_BIN_FLOAT_PRINT_PRECISION 2

namespace kaldi {
namespace cuda_decoder {

int NumberOfSegments(int nsamples, int seg_length, int seg_shift) {
  KALDI_ASSERT(seg_shift > 0);
  KALDI_ASSERT(seg_length >= seg_shift);
  int r = seg_length - seg_shift;
  if (nsamples <= seg_length) return 1;
  int nsegments = ((nsamples - r) + seg_shift - 1) / seg_shift;
  return nsegments;
}

void WriteLattices(std::vector<CudaPipelineResult> &results,
                   const std::string &key, bool print_offsets,
                   CompactLatticeWriter &clat_writer) {
  for (CudaPipelineResult &result : results) {
    double offset = result.GetTimeOffsetSeconds();
    if (!result.HasValidResult()) {
      KALDI_WARN << "Utterance " << key << ": "
                 << " Segment with offset " << offset
                 << " is not valid. Skipping";
    }

    std::ostringstream key_with_offset;
    key_with_offset << key;
    if (print_offsets) key_with_offset << "-" << offset;
    clat_writer.Write(key_with_offset.str(), *result.GetLatticeResult());
    if (!print_offsets) {
      if (results.size() > 1) {
        KALDI_WARN << "Utterance " << key
                   << " has multiple segments but only one is written to "
                      "output. Use print_offsets=true";
      }
      break;  // printing only one result if offsets are not used
    }
  }
}

// Reads all CTM outputs in results and merge them together
// into a single output. That output is then written as a CTM text format to
// ostream
void MergeSegmentsToCTMOutput(std::vector<CudaPipelineResult> &results,
                              const std::string &key, std::ostream &ostream,
                              fst::SymbolTable *word_syms,
                              bool use_segment_offsets) {
  size_t nresults = results.size();

  if (nresults == 0) {
    KALDI_WARN << "Utterance " << key << " has no results. Skipping";
    return;
  }

  bool all_results_valid = true;

  for (size_t iresult = 0; iresult < nresults; ++iresult)
    all_results_valid &= results[iresult].HasValidResult();

  if (!all_results_valid) {
    KALDI_WARN << "Utterance " << key
               << " has at least one segment with an error. Skipping";
    return;
  }

  ostream << std::fixed;
  ostream.precision(KALDI_CUDA_DECODER_BIN_FLOAT_PRINT_PRECISION);

  // opt: combine results into one here
  BaseFloat previous_segment_word_end = 0;
  for (size_t iresult = 0; iresult < nresults; ++iresult) {
    bool this_segment_first_word = true;
    bool is_last_segment = ((iresult + 1) == nresults);
    BaseFloat next_offset_seconds = std::numeric_limits<BaseFloat>::max();
    if (!is_last_segment) {
      next_offset_seconds = results[iresult + 1].GetTimeOffsetSeconds();
    }

    auto &result = results[iresult];
    BaseFloat offset_seconds =
        use_segment_offsets ? result.GetTimeOffsetSeconds() : 0;
    int isegment = result.GetSegmentID();
    auto &ctm = *result.GetCTMResult();
    for (size_t iword = 0; iword < ctm.times_seconds.size(); ++iword) {
      BaseFloat word_from = offset_seconds + ctm.times_seconds[iword].first;
      BaseFloat word_to = offset_seconds + ctm.times_seconds[iword].second;

      // If beginning of this segment, only keep "new" words
      // i.e. the ones that were not already in previous segment
      if (this_segment_first_word) {
        if (word_from >= previous_segment_word_end) {
          // Found the first "new" word for this segment
          this_segment_first_word = false;
        } else
          continue;  // skipping this word
      }

      // If end of this segment, skip the words which are
      // overlapping two segments
      if (!is_last_segment) {
        if (word_from >= next_offset_seconds) break;  // done with this segment
      }

      previous_segment_word_end = word_to;

      ostream << key << " " << isegment << "  " << word_from << ' '
              << (word_to - word_from) << ' ';

      int32 word_id = ctm.words[iword];
      if (word_syms)
        ostream << word_syms->Find(word_id);
      else
        ostream << word_id;

      ostream << ' ' << ctm.conf[iword] << '\n';
    }
  }
}

}  // namespace cuda_decoder
}  // namespace kaldi

#endif
