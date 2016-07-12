// nnet3/nnet-example-utils.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_EXAMPLE_UTILS_H_
#define KALDI_NNET3_NNET_EXAMPLE_UTILS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"

namespace kaldi {
namespace nnet3 {



/** Merge a set of input examples into a single example (typically the size of
    "src" will be the minibatch size).  Will crash if "src" is the empty vector.
    If "compress" is true, it will compress any non-sparse features in the output.
 */
void MergeExamples(const std::vector<NnetExample> &src,
                   bool compress,
                   NnetExample *dest);


/** Shifts the time-index t of everything in the "eg" by adding "t_offset" to
    all "t" values.  This might be useful in things like clockwork RNNs that are
    not invariant to time-shifts, to ensure that we see different shifts of each
    example during training.  "exclude_names" is a vector (not necessarily
    sorted) of names of nnet inputs that we avoid shifting the "t" values of--
    normally it will contain just the single string "ivector" because we always
    leave t=0 for any ivector. */
void ShiftExampleTimes(int32 t_offset,
                       const std::vector<std::string> &exclude_names,
                       NnetExample *eg);

/**  This function takes a NnetExample (which should already have been
     frame-selected, if desired, and merged into a minibatch) and produces a
     ComputationRequest.  It ssumes you don't want the derivatives w.r.t. the
     inputs; if you do, you can create/modify the ComputationRequest manually.
     Assumes that if need_model_derivative is true, you will be supplying
     derivatives w.r.t. all outputs.
*/
void GetComputationRequest(const Nnet &nnet,
                           const NnetExample &eg,
                           bool need_model_derivative,
                           bool store_component_stats,
                           ComputationRequest *computation_request);


// Writes as unsigned char a vector 'vec' that is required to have
// values between 0 and 1.
void WriteVectorAsChar(std::ostream &os,
                       bool binary,
                       const VectorBase<BaseFloat> &vec);

// Reads data written by WriteVectorAsChar.
void ReadVectorAsChar(std::istream &is,
                             bool binary,
                             Vector<BaseFloat> *vec);

// This function rounds up the quantities 'num_frames' and 'num_frames_overlap'
// to the nearest multiple of the frame_subsampling_factor
void RoundUpNumFrames(int32 frame_subsampling_factor,
                      int32 *num_frames,
                      int32 *num_frames_overlap);


/** Returns true if the "eg" contains just a single example, meaning
    that all the "n" values in the indexes are zero, and the example
    has NnetIo members named both "input" and "output"

    Also computes the minimum and maximum "t" values in the "input" and
    "output" NnetIo members.
**/
bool ContainsSingleExample(const NnetExample &eg,
                           int32 *min_input_t,
                           int32 *max_input_t,
                           int32 *min_output_t,
                           int32 *max_output_t);

/*
   This function filters the indexes (and associated feature rows) in a
   NnetExample, removing any index/row in an NnetIo named "input" with t <
   min_input_t or t > max_input_t and any index/row in an NnetIo named "output" with t <
   min_output_t or t > max_output_t.
   Will crash if filtering removes all Indexes of "input" or "output".
 */
void FilterExample(const NnetExample &eg,
                   int32 min_input_t,
                   int32 max_input_t,
                   int32 min_output_t,
                   int32 max_output_t,
                   NnetExample *eg_out);

/*
   This function is responsible for possibly selecting one frame from multiple
   supervised frames, and reducing the left and right context as specified.  If
   frame == "" it does not reduce the supervised frames; if frame == "random" it
   selects one random frame; otherwise it expects frame to be an integer, and
   will select only the output with that frame index (or return false if there was
   no such output).

   If left_context != -1 it removes any inputs with t < (smallest output - left_context).
      If left_context != -1 it removes any inputs with t < (smallest output - left_context).

   It returns true if it was able to select a frame.  We only anticipate it ever
   returning false in situations where frame is an integer, and the eg came from
   the end of a file and has a smaller than normal number of supervised frames.

*/
bool SelectFromExample(const NnetExample &eg,
                       std::string frame_str,
                       int32 left_context,
                       int32 right_context,
                       int32 frame_shift,
                       NnetExample *eg_out);

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_UTILS_H_
