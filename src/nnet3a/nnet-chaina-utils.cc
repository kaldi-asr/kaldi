// nnet3/nnet-chaina-utils.cc

// Copyright      2018    Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-utils.h"
#include "nnet3a/nnet-chaina-utils.h"

namespace kaldi {
namespace nnet3 {

void FindChainaExampleStructure(const NnetChainExample &eg,
                                int32 *num_sequences,
                                int32 *chunks_per_spk,
                                int32 *first_input_t,
                                int32 *num_input_frames,
                                int32 *num_output_frames,
                                int32 *frame_subsampling_factor,
                                int32 *eg_left_context,
                                int32 *eg_right_context) {
  if (eg.inputs.size() != 1 ||
      eg.inputs[0].name != "input")
    KALDI_ERR << "Expected eg to have exactly one input, named 'input'";

  if (eg.outputs.size() != 1 ||
      eg.outputs[0].name != "output")
        KALDI_ERR << "Expected eg to have exactly one output, named 'output'";


  const NnetChainSupervision &supervision = eg.outputs[0];
  *num_sequences = supervision.supervision.num_sequences;
  *chunks_per_spk = supervision.chunks_per_spk;

  KALDI_ASSERT(supervision.indexes.size() % *num_sequences == 0 &&
               !supervision.indexes.empty());
  KALDI_ASSERT(supervision.indexes[0] == Index() &&
               "Expected first index to have t=0,n=0,x=0");
  // We expect t to have the larger stride.
  KALDI_ASSERT(supervision.indexes[1].n == 1 &&
               "Supervision is in an unexpected order");
  Index last_output_index = supervision.indexes.back();
  KALDI_ASSERT(last_output_index.n == *num_sequences - 1);
  *num_output_frames = int32(supervision.indexes.size()) / *num_sequences;
  int32 last_output_t = last_output_index.t;
  KALDI_ASSERT(last_output_t % (*num_output_frames - 1) == 0);
  *frame_subsampling_factor = last_output_t / (*num_output_frames - 1);


  const NnetIo &input_io = eg.inputs[0];
  *first_input_t = - input_io.indexes[0].t;
  if (input_io.indexes[0].t != *first_input_t + 1) {
    KALDI_ERR << "Input indexes are in the wrong order or not consecutive: "
              << input_io.indexes[0].t << " != " << (*first_input_t + 1);
  }
  Index last_input_index = input_io.indexes.back();
  KALDI_ASSERT(last_input_index.n == *num_sequences - 1);
  int32 last_input_t = last_input_index.t;
  *num_input_frames = last_input_t + 1 - *first_input_t;

  *eg_left_context = -(*first_input_t);
  *eg_right_context = last_input_t - last_output_t;
}


bool ParseFromQueryString(const std::string &string,
                          const std::string &key_name,
                          std::string *value) {
  size_t question_mark_location = string.find_last_of("?");
  if (question_mark_location == std::string::npos)
    return false;
  std::string key_name_plus_equals = key_name + "=";
  // the following do/while and the initialization of key_name_location is a
  // little convoluted.  We want to find "key_name_plus_equals" but if we find
  // it and it's not preceded by '?' or '&' then it's part of a longer key and we
  // need to ignore it and see if there's a next one.
  size_t key_name_location = question_mark_location;
  do {
    key_name_location = string.find(key_name_plus_equals,
                                    key_name_location + 1);
  } while (key_name_location != std::string::npos &&
           key_name_location != question_mark_location + 1 &&
           string[key_name_location - 1] != '&');

  if (key_name_location == std::string::npos)
    return false;
  size_t value_location = key_name_location + key_name_plus_equals.length();
  size_t next_ampersand = string.find_first_of("&", value_location);
  size_t value_len;
  if (next_ampersand == std::string::npos)
    value_len = std::string::npos;  // will mean "rest of string"
  else
    value_len = next_ampersand - value_location;
  *value = string.substr(value_location, value_len);
  return true;
}


bool ParseFromQueryString(const std::string &string,
                          const std::string &key_name,
                          BaseFloat *value) {
  std::string s;
  if (!ParseFromQueryString(string, key_name, &s))
    return false;
  bool ans = ConvertStringToReal(s, value);
  if (!ans)
    KALDI_ERR << "For key " << key_name << ", expected float but found '"
              << s << "', in string: " << string;
  return true;
}


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
                           int32 *num_embedding_frames) {
  KALDI_ASSERT(num_input_frames > 0 && num_output_frames > 0 &&
               first_input_t <= 0 && frame_subsampling_factor > 0);
  KALDI_ASSERT(bottom_subsampling_factor > 0 &&
                frame_subsampling_factor % bottom_subsampling_factor == 0);
  KALDI_ASSERT(bottom_left_context >= 0 && bottom_right_context >= 0 &&
               top_left_context >= 0 && top_right_context >= 0);

  // below '_subsampled' means after dividing the 't' values by
  // 'bottom_subsampling_factor'.
  // Note: implicitly, the first frame required at the output is t=0.
  int32 first_required_embedding_t_subsampled = -top_left_context,
      last_required_embedding_t_subsampled =
      num_output_frames - 1 + top_right_context;

  int32 first_computable_embedding_t = first_input_t + bottom_left_context,
      last_computable_embedding_t =
      first_input_t + num_input_frames - 1 - bottom_right_context;

  int32 b = bottom_subsampling_factor;

  // By adding b - 1 and doing division that rounds down (towards negative
  // infinity, we effectively round up when computing
  // first_computable_embedding_t / b, which is appropriate because
  // we need the first multiple of b that's actually computable.
  int32 first_computable_embedding_t_subsampled =
      DivideRoundingDown(first_computable_embedding_t + b - 1, b),
      last_computable_embedding_t_subsampled =
      DivideRoundingDown(last_computable_embedding_t, b);
  if (first_computable_embedding_t_subsampled > first_required_embedding_t_subsampled ||
      last_computable_embedding_t_subsampled < last_required_embedding_t_subsampled) {
    KALDI_WARN << "The training examples have insufficient context vs. the models.";
    return false;
  }
  if (keep_embedding_context) {
    *first_embedding_t = first_computable_embedding_t_subsampled * b;
    *num_embedding_frames = 1 + last_computable_embedding_t_subsampled -
        first_computable_embedding_t_subsampled;
  } else {
    *first_embedding_t = first_required_embedding_t_subsampled * b;
    *num_embedding_frames = 1 + last_required_embedding_t_subsampled -
        first_required_embedding_t_subsampled;
  }
  return true;
}



} // namespace nnet3
} // namespace kaldi
