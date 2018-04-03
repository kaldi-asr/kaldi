// nnet3/nnet-compile-looped.cc

// Copyright      2016  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-compile-looped.h"
#include "nnet3/nnet-optimize-utils.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {


void ModifyNnetIvectorPeriod(int32 ivector_period,
                             Nnet *nnet) {
  KALDI_ASSERT(ivector_period > 0);
  std::vector<std::string> config_lines;
  nnet->GetConfigLines(false, &config_lines);
  std::ostringstream config_to_read;
  for (size_t i = 0; i < config_lines.size(); i++) {
    std::string s = config_lines[i];
    ConfigLine config_line;
    bool b = config_line.ParseLine(config_lines[i]);
    KALDI_ASSERT(b && "Could not parse config line.");
    if (config_line.FirstToken() == "component-node") {
      std::string whole_line = config_lines[i];
      std::string to_search_for = "ReplaceIndex(";
      std::string::size_type to_search_for_size = to_search_for.size();
      std::string::size_type pos = whole_line.find(to_search_for);
      if (pos != std::string::npos) {
        std::string::size_type comma_pos = whole_line.find(',', pos);
        if (comma_pos != std::string::npos) {
          // if the line contained ReplaceIndex(ivector, t, 0),
          // descriptor_name would now be 'ivector'.
          std::string descriptor_name =
              whole_line.substr(pos + to_search_for_size,
                                comma_pos - (pos + to_search_for_size));
          std::string::size_type end_pos = whole_line.find(')', pos);
          std::string::size_type expr_size = end_pos + 1 - pos;
          // e.g. expr_size would be strlen("ReplaceIndex(ivector, t, 0)").
          std::ostringstream to_replace_with;
          to_replace_with << "Round(" << descriptor_name << ", " << ivector_period << ")";
          whole_line.replace(pos, expr_size, to_replace_with.str());
          config_to_read << whole_line << "\n";
        }
      }
    }
  }
  if (!config_to_read.str().empty()) {
    std::istringstream is(config_to_read.str());
    nnet->ReadConfig(is);
  }
}


int32 GetChunkSize(const Nnet &nnet,
                   int32 frame_subsampling_factor,
                   int32 advised_chunk_size) {
  int32 modulus = nnet.Modulus();
  KALDI_ASSERT(modulus > 0 && frame_subsampling_factor > 0 &&
               advised_chunk_size > 0);
  int32 chunk_size = advised_chunk_size;
  while (1) {
    if (chunk_size % modulus == 0 &&
        chunk_size % frame_subsampling_factor == 0)
      return chunk_size;
    chunk_size++;
  }
}


/// Mod(m, n), defined for integers m and n where n > 0, returns
/// the modulus m % n, defined as the integer 0 <= i < n
/// such that i and m are congruent modulo n; for instance,
/// Mod(13, 10) = 3.
/// This is like the % operation in C/C++, except that it always returns a
/// positive value even for negative m; in 99% of cases where it makes a
/// difference, this is what you want.  In the C/C++ standard, the sign of a % b
/// for negative a is not specified (except by relation with the division '/'
/// operator), but in practice it would be <= 0 for almost all implementations.
template<class I> I  Mod(I m, I n) {
  I ans = m % n;
  if (ans < 0) ans += n;
  return ans;
}


static void CreateComputationRequestInternal(
    int32 begin_input_t, int32 end_input_t,
    int32 begin_output_t, int32 end_output_t,
    int32 num_sequences,
    int32 frame_subsampling_factor,
    const std::set<int32> &ivector_times,
    ComputationRequest *request) {
  request->inputs.reserve(2);
  request->inputs.clear();
  request->inputs.resize(1 + (ivector_times.empty() ? 0 : 1));
  request->inputs[0].name = "input";
  request->inputs[0].has_deriv = false;
  request->outputs.clear();
  request->outputs.resize(1);
  request->outputs[0].name = "output";
  request->outputs[0].has_deriv = false;
  if (!ivector_times.empty()) {
    request->inputs[1].name = "ivector";
    request->inputs[1].has_deriv = false;
  }

  // in the computation request the 'n' indexes (the sequence/utterance indexes)
  // have the larger stride than 't', although this is opposite to the way it's
  // done inside the computation.  This is for user convenience where it may be
  // easier to deal with submatrixes per sequence.
  for (int32 n = 0; n < num_sequences; n++) {
    int32 x = 0;
    for (int32 t = begin_input_t; t < end_input_t; t++) {
      request->inputs[0].indexes.push_back(Index(n, t, x));
    }
    for (int32 t = begin_output_t;
         t < end_output_t;
         t += frame_subsampling_factor)
      request->outputs[0].indexes.push_back(Index(n, t, x));
  }
  if (!ivector_times.empty()) {
    request->inputs.resize(2);
    request->inputs[1].name = "ivector";
    request->inputs[1].has_deriv = false;
    for (int32 n = 0; n < num_sequences; n++) {
      // note: std::sets store things in sorted order.
      for (std::set<int32>::const_iterator iter = ivector_times.begin();
           iter != ivector_times.end(); ++iter) {
        int32 t = *iter, x = 0;
        request->inputs[1].indexes.push_back(Index(n, t, x));
      }
    }
  }
}


void CreateLoopedComputationRequest(const Nnet &nnet,
                                    int32 chunk_size,
                                    int32 frame_subsampling_factor,
                                    int32 ivector_period,
                                    int32 left_context_begin,
                                    int32 right_context,
                                    int32 num_sequences,
                                    ComputationRequest *request1,
                                    ComputationRequest *request2,
                                    ComputationRequest *request3) {
  bool has_ivector = (nnet.InputDim("ivector") > 0);
  KALDI_ASSERT(chunk_size % frame_subsampling_factor == 0 &&
               chunk_size % nnet.Modulus() == 0 &&
               chunk_size % ivector_period == 0);
  KALDI_ASSERT(left_context_begin >= 0 && right_context >= 0);
  // note, 'end' is one past the last one.
  int32 chunk1_input_begin_t = - left_context_begin,
      chunk1_input_end_t = chunk_size + right_context,
      chunk2_input_begin_t = chunk1_input_end_t,
      chunk2_input_end_t = chunk2_input_begin_t + chunk_size,
      chunk3_input_begin_t = chunk2_input_end_t,
      chunk3_input_end_t = chunk3_input_begin_t + chunk_size;


  // work out the times at which i-vectors are required.
  std::set<int32> ivector_times1, ivector_times2, ivector_times3;
  if (has_ivector) {
    for (int32 t = chunk1_input_begin_t; t < chunk1_input_end_t; t++) {
      int32 ivector_t = t - Mod(t, ivector_period);
      ivector_times1.insert(ivector_t);
    }
    for (int32 t = chunk2_input_begin_t; t < chunk2_input_end_t; t++) {
      int32 ivector_t = t - Mod(t, ivector_period);
      if (ivector_times2.count(ivector_t) == 0 &&
	  ivector_times1.count(ivector_t) == 0)
        ivector_times2.insert(ivector_t);
    }
    for (int32 t = chunk3_input_begin_t; t < chunk3_input_end_t; t++) {
      int32 ivector_t = t - Mod(t, ivector_period);
      if (ivector_times3.count(ivector_t) == 0 &&
          ivector_times2.count(ivector_t) == 0 &&
	  ivector_times1.count(ivector_t) == 0)
        ivector_times3.insert(ivector_t);
    }
  }

  CreateComputationRequestInternal(
      chunk1_input_begin_t, chunk1_input_end_t,
      0, chunk_size,
      num_sequences, frame_subsampling_factor,
      ivector_times1,
      request1);

  CreateComputationRequestInternal(
      chunk2_input_begin_t, chunk2_input_end_t,
      chunk_size, chunk_size * 2,
      num_sequences, frame_subsampling_factor,
      ivector_times2,
      request2);

  CreateComputationRequestInternal(
      chunk3_input_begin_t, chunk3_input_end_t,
      chunk_size * 2, chunk_size * 3,
      num_sequences, frame_subsampling_factor,
      ivector_times3,
      request3);

}



void AddTimeOffsetToComputationRequest(int32 t_offset,
                                       ComputationRequest *request) {
  for (size_t i = 0; i < request->inputs.size(); i++) {
    size_t size = request->inputs[i].indexes.size();
    for (size_t j = 0; j < size; j++)
      request->inputs[i].indexes[j].t += t_offset;
  }
  for (size_t i = 0; i < request->outputs.size(); i++) {
    size_t size = request->outputs[i].indexes.size();
    for (size_t j = 0; j < size; j++)
      request->outputs[i].indexes[j].t += t_offset;
  }
}



static bool ExtrapolateComputationRequest(
    const ComputationRequest &request1,
    const ComputationRequest &request2,
    ComputationRequest *request3) {
  // accepts two computation requests 'request1' and 'request2' that
  // must be identical except for a time offset, and creates 'request3'
  // that is the extrapolation of the next term in sequence.
  *request3 = request2;
  KALDI_ASSERT(!request1.inputs.empty() && !request1.inputs[0].indexes.empty() &&
               !request2.inputs.empty() && !request2.inputs[0].indexes.empty());
  int32 t_offset = request2.inputs[0].indexes[0].t -
      request1.inputs[0].indexes[0].t;
  // the following is just to make sure that the inputs are structurally
  // equivalent.
  AddTimeOffsetToComputationRequest(-t_offset, request3);
  if (!(*request3 == request1))
    return false;  // there is somse structural difference, or
                   // the time offset is not consistent.
  // the following reverses the last call to AddTimeOffsetToComputationRequest,
  // then adds the offset we want.
  AddTimeOffsetToComputationRequest(2 * t_offset, request3);
  return true;
}


/* Internal version of CompileLooped where
   you specify the the number of computation requests (must be >= 3).
   Returns true on success.
   It's possible for the optimization to fail if you give too small
   a value of 'num_requests' (this depends on the network topology),
   and in that case this function will return false and you should re-try
   with a higher value of num_requests.
 */
static bool CompileLoopedInternal(
    const Nnet &nnet,
    NnetOptimizeOptions optimize_opts,
    const ComputationRequest &request1,
    const ComputationRequest &request2,
    const ComputationRequest &request3,
    int32 num_requests,
    NnetComputation *computation) {

  KALDI_ASSERT(num_requests >= 3);
  std::vector<ComputationRequest> extra_requests(num_requests - 3);
  const ComputationRequest *prev_request = &request2;
  const ComputationRequest *cur_request = &request3;
  for (int32 i = 0; i < num_requests - 3; i++) {
    if (!ExtrapolateComputationRequest(*prev_request, *cur_request,
                                       &(extra_requests[i]))) {
      KALDI_LOG << "prev_request is:";
      prev_request->Print(std::cerr);
      KALDI_LOG << "cur_request is:";
      cur_request->Print(std::cerr);
      KALDI_ERR << "Computation requests do not have the right relationship";
    }
    prev_request = cur_request;
    cur_request = &(extra_requests[i]);
  }

  std::vector<const ComputationRequest*> requests;
  requests.push_back(&request1);
  requests.push_back(&request2);
  requests.push_back(&request3);
  for (int32 i = 0; i < num_requests - 3; i++)
    requests.push_back(&(extra_requests[i]));
  Compiler compiler(requests, nnet);
  CompilerOptions compiler_opts;
  compiler.CreateComputation(compiler_opts, computation);
  optimize_opts.optimize_looped_computation = true;

  int32 dont_really_care = MaxOutputTimeInRequest(request3);
  Optimize(optimize_opts, nnet,
           dont_really_care, computation);

  return computation->commands.size() != 0 &&
      computation->commands.back().command_type == kGotoLabel;
}

void CompileLooped(const Nnet &nnet,
                   const NnetOptimizeOptions &optimize_opts,
                   const ComputationRequest &request1,
                   const ComputationRequest &request2,
                   const ComputationRequest &request3,
                   NnetComputation *computation) {
  int32 num_requests1 = 5, factor = 2, max_requests = 100,
      num_requests;

  Timer timer;

  for (num_requests = num_requests1; num_requests <= max_requests;
       num_requests *= factor) {
    if (CompileLoopedInternal(nnet, optimize_opts,
                             request1, request2, request3,
                             num_requests, computation)) {
      KALDI_LOG << "Spent " << timer.Elapsed()
                << " seconds in looped compilation.";
      return;
    } else {
      KALDI_VLOG(2) << "Looped compilation failed with "
                    << num_requests << " requests, trying "
                    << (num_requests * factor);
    }
  }
  KALDI_ERR << "Looped compilation failed with "
            << (num_requests/factor) << " requests, which "
            << "we expect should be enough... something "
            << "went wrong.";
}


void CreateLoopedComputationRequestSimple(const Nnet &nnet,
                                          int32 chunk_size,
                                          int32 frame_subsampling_factor,
                                          int32 ivector_period,
                                          int32 extra_left_context_begin,
                                          int32 extra_right_context,
                                          int32 num_sequences,
                                          ComputationRequest *request1,
                                          ComputationRequest *request2,
                                          ComputationRequest *request3) {
  int32 left_context, right_context;
  ComputeSimpleNnetContext(nnet, &left_context, &right_context);

  CreateLoopedComputationRequest(nnet, chunk_size, frame_subsampling_factor,
                                 ivector_period,
                                 extra_left_context_begin + left_context,
                                 extra_right_context + right_context,
                                 num_sequences, request1, request2, request3);
}

} // namespace nnet3
} // namespace kaldi
