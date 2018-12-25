// nnet3a/nnet-chaina-utils.h

// Copyright    2015-2018  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_CHAINA_UTILS_H_
#define KALDI_NNET3_NNET_CHAINA_UTILS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-training.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"

namespace kaldi {
namespace nnet3 {

/**
   This function works out certain structural information from an example for
   'chaina' (adapted chain) training.  It assumes (and spot-checks) that the eg
   has a single input, called 'input', with a regular structure where the 'n'
   has the highest stride so it's: all frames for sequence 0; all frames for
   sequence 1; and so on.  It will raise an exception if the example does not,
   in some respect, have the expected structure.

      @param [in]  The example we are getting the structural information from
      @param [out] num_sequences  The number of sequences/chunks (actually just
                            the num_sequences in the eg.supervision object).
      @param [out] chunks_per_spk  The number of chunks per speaker
                            (just eg.chunks_per_spk)
      @param [out] first_input_t   The lowest numbered 't' value in the inputs.
                            Usually will be negative.  This function requires the
                            input 't' values to be consecutive, and will crash
                            if they are not.
      @param [out] num_input_frames  The number of input frames.  The last input
                            't' value will be first_input_t + num_input_frames - 1.
      @param [out] num_output_frames  The number of output frames (which are
                             assumed to start from t=0 and to be spaced by
                             'frame_subsampling_factor.
      @param [out] frame_subsampling_factor  The spacing on the output frames,
                             equal to the amount of subsampling that happens
                             between the input and the output (this will
                             later be factorized as:
                             frame_subsampling_factor =
                                bottom_subsampling_factor * top_subsampling_factor.
      @param [out] eg_left_context  Just as a convenience, this function outputs
                             the left-context in the example, which equals
                             first_output_t - first_input_t = -first_input_t.
      @param [out] eg_right_context  Again just as a convenience, this function
                             outputs the right-context of the example, which
                             equals last_input_t - last_output_t =
                             (first_input_t + num_input_frames - 1) -
             (first_output_t + num_output_frames - 1) * frame_subsampling_factor
                             (note: first_output_t is zero).
*/
void FindChainaExampleStructure(const NnetChainExample &eg,
                                int32 *num_sequences,
                                int32 *chunks_per_spk,
                                int32 *first_input_t,
                                int32 *num_input_frames,
                                int32 *num_output_frames,
                                int32 *frame_subsampling_factor,
                                int32 *eg_left_context,
                                int32 *eg_right_context);

/**
   This function computes some info about which frames we need to compute the
   embeddings for (i.e. which frames we need to request at the output of the
   bottom nnet).  It will print a warning and return false if the egs had
   insufficient context to compute what is requested.

      @param [in] first_input_t  The first 't' value for the input that
                       is provided to the bottom nnet.
      @param [in] num_input_frames   The number of input frames provided to
                       the bottom nnet; these are assumed to be consecutive.
      @param [in] num_output_frames  The number of output frames that we
                       need to compute the output for (this will be
                       the sequence_length in the chain supervision object).
      @param [in] frame_subsampling_factor  The factor by which we
                       subsample to get the final output (includes subsampling
                       in both the bottom and top nnet).
      @param [in] bottom_subsampling_factor  The amount of subsampling
                       for getting the embeddings (i.e. the embeddings
                       are obtained at t = multiples of this value.)
                       Must be >0 and divide frame_subsampling_factor.
                       This must be provided and can't be worked out from
                       the nnets, because the top nnet uses a different frame
                       numbering-- i.e. we divide the 't' values by
                       'bottom_subsampling_factor' so that the inputs to the
                       top nnet are consecutive.  This will make it easier
                       to apply the top nnet separately from binaries.
      @param [in] bottom_left_context  The num-frames of left-context that the
                       bottom nnet requires
      @param [in] bottom_right_context  The num-frames of right-context that the
                       bottom nnet requires
      @param [in] top_left_context  The num-frames of left-context that the
                       top nnet requires.  Note: this is *after* dividing the
                       't' values by bottom_subsampling_factor, so the number
                       top_left_context * bottom_subsampling_factor can be used
                       to compute the total left-context that we need to put in
                       the egs.
      @param [in] top_right_context  The num-frames of right-context that the
                       top nnet requires.  See docs for top_left_context for more
                       info RE frame subsampling
      @param [in] keep_embedding_context  True if we want to compute as
                       many frames of the embedding as we can given the amount
                       of available left context in the input.  This will be
                       usually be set to true if the top nnet is recurrent or
                       can otherwise consume extra context.
      @param [out] first_embedding_t  First 't' value of the embedding.  CAUTION:
                       this is in the original frame numbering (the one we use
                       for the bottom nnet), and will be a multiple of
                       'bottom_subsampling_factor'.  You need to divide by
                       'bottom_subsampling_factor' to get the 't' value used
                       at the input of the top nnet.
      @param [out] num_embedding_frames  The number of embedding frames that
                       we are computing.
      @return          Returns true if it could successfully compute the output,
                       and false if it could not because of insufficient input
                       context.
 */
bool ComputeEmbeddingTimes(int32 first_input_t,
                           int32 num_input_frames,
                           int32 num_output_frames,
                           int32 frame_subsampling_factor,
                           int32 bottom_subsampling_factor,
                           int32 bottom_left_context,
                           int32 bottom_right_context,
                           int32 top_left_context,
                           int32 top_right_context,
                           bool keep_embedding_context,
                           int32 *first_embedding_t,
                           int32 *num_embedding_frames);


/**
   This function parses a string value from a 'url-like' string (which is probably actually
   a key value from an scp file).  The general format this function handles is:
       iiiiiiiiiiiiiiiiiii?aaa=xxxx&bbb=yyyy
   where the only 'special characters' are '?' and '&'.  This is modeled after a query
   string in HTML.  This function searches for a key name with the value 'key_name',
   (e.g. 'aaa' or 'bbb' in the example), and if it exists, sets `value` to that value
   (e.g. 'xxxx' or 'yyyy' in the example.  If the string `string` has no '?' in it,
   or the key name `key_name` is not present, this function returns false; otherwise,
   it returns true and sets `value` to that value.

*/
bool ParseFromQueryString(const std::string &string,
                          const std::string &key_name,
                          std::string *value);


// This overloaded version of ParseFromQueryString()is for where a float value
// is required.  If the key is present but cannot be turned into a float, it
// will raise an error.
bool ParseFromQueryString(const std::string &string,
                          const std::string &key,
                          BaseFloat *f);



} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAINA_UTILS_H_
