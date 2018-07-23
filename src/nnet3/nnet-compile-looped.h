// nnet3/nnet-compile-looped.h

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

#ifndef KALDI_NNET3_NNET_COMPILE_LOOPED_H_
#define KALDI_NNET3_NNET_COMPILE_LOOPED_H_

#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-utils.h"

#include <list>

namespace kaldi {
namespace nnet3 {


/**
   CompileLooped() provides an internal interface for 'looped' computation.
   It's usable for inference only (not training), meaning that backprop is
   not supported (for now, at least).  CompileLooped() allows you to do the
   neural net computation for small chunks with increasing 't' values, and
   naturally cache the intermediate activations (rather than recomputing them
   every time you see new input data).

   This function does both compilation and optimization, so it's like a combination of
   Compiler::CreateComputation() [nnet-compile.h] and Optimize() [nnet-optimize.h].

   You provide 3 computation requests.  request1 is the first computation
   request of an utterance (or other type of segment) that contains any required
   extra left context in the input.  request2 and request3 are the second and
   third computation request, and must have exactly the same structure, except
   for a fixed time offset (change in 't' index) between them.  This will be
   extrapolated to an infinite sequence of further requests (request4,
   request5, etc.).  In practice the way it's done is that we extrapolate
   to a small finite number of requests (like 10), and then attempt to
   identify a common structure in the computation where, after processing,
   as an example, the 3nd computation request, the active variables can
   be identified with those present at, say, the 7th computation request, and
   we then cut and splice the computation together at this points, like
   making a tape loop, by adding a goto statement that jumps from the end of
   the 7th computation request to the end of the 3rd computation request.
   We also have to identify the variables with each other (merge variables).

   That's done in the optimization code.
 */
void CompileLooped(const Nnet &nnet,
                   const NnetOptimizeOptions &optimize_opts,
                   const ComputationRequest &request1,
                   const ComputationRequest &request2,
                   const ComputationRequest &request3,
                   NnetComputation *computation);

/*
  This function gives you a suitable chunk size, which is the smallest number >=
  'advised_chunk_size' that is an exact multiple of nnet.Modulus() and
  frame_subsampling_factor.  This will ensure that all the chunks have the same
  structure, which makes compiling the looped computation a little more
  straightforward.
 */
int32 GetChunkSize(const Nnet &nnet,
                   int32 frame_subsampling_factor,
                   int32 advised_chunk_size);

/**
   This function modifies the descriptors in the neural network to change the
   periodicity with which it expects to read an iVector at its input.

   We normally train neural networks that expect to see an iVector at frame zero
   only; this is because we train on fixed-size chunks and the iVector doesn't
   change that much within each chunk.  However, expecting just one iVector
   isn't that convenient for looped recognition because it changes with time, so
   we modify the iVector input period in the network by replacing expressions
   like ReplaceIndex(ivector, t, 0) with Round(ivector, 10) [assuming
   ivector_period == 10].  The descriptor doesn't have to be named "ivector", it
   would work for ReplaceIndex(foo, t, 0).  This won't work in every conceivable
   network, but it does do what you want in the cases of interest.

   It does this in a rather simple way, by getting the config lines that
   correspond to descriptors, and doing a search-and-replace.  It's
   maybe not ideal, but it was the easiest way to do it.

 */
void ModifyNnetIvectorPeriod(int32 ivector_period,
                             Nnet *nnet);

/**
  This function creates computation request suitable for giving to ComputeLooped().
  It's intended for use with a 'simple' nnet (one satisfying IsSimpleNnet()), and this
  basically means that the inputs must be named "input" and possibly "ivector",
  and that there is an output named "output", and that those are the ones you
  care about (it won't generate any other outputs or use any other inputs).

  If you want to use looped computation for different types of neural net, you
  should use the deeper interface, CompileLooped().

   @param [in] nnet   The neural net this computation request is to be used with.
               This is used to check whether the neural net accepts iVectors,
               and to work out the left-context and right-context required
               by the network.
   @param [in] chunk_size  The number of frames of output that will be generated
               for each chunk (note: this is the shift in the t-index, which will not
               equal the number of output frames if frame_subsampling_factor != 1).
               Note: it is required that chunk_size be a multiple of ivector_period,
               frame_subsampling_factor, and nnet.Modulus().  You should use
               GetChunkSize() to compute the chunk size, giving it an advisory/
               minimum chunksize, to make sure it satisfies these properties.
   @param [in] frame_subsampling_factor  This will normally be 1, but may be
               more than 1 (e.g. 3) in chain systems; it determines the frame-skipping
               on the output, so we evaluate the output with 't' at multiples of
               this value.
   @param [in] ivector_period The period with which iVectors are to be supplied
               to the network (if you're using iVectors).  Not necessarily the
               same as the period with which the ivectors are extracted or
               stored on disk (--online-ivector-period).  You will normally set
               this to the chunk size.  It must divide the chunk size (if you're
               using iVectors) Note: you should call ModifyNnetIvectorPeriod on
               'nnet' before calling this function; otherwise the neural net
               will most likely not actually be able to consume the iVector with
               this frequency.
   @param [in] left_context_begin This should be the left-context of the network
               plus any additional left-context (provided via the option
               --extra-left-context-begin) that should be supplied to the
               network on top of the minimum that the network requires.  We call
               this left_context_begin because this only relates to the
               start of the utterance (t=0).
   @param [in] right_context This should be the right-context of the network,
               plus any additional right-context ("extra-right-context") that
               should be supplied to the network on top of the minimum that the
               network requires (currently extra-right-context != 0 is is not
               supported at the command-line level).
   @param [in] num_sequences  The number of separate 'n' values to put in the computation;
               normally this will be just 1, but it can be increased to allow
               simultaneous operation on multiple streams of input.
   @param [out] request1 The first of the 3 requests that this function
               generates, that the user should then supply to CompileLooped().
               Note: this will tend to be the largest computation request in
               terms of input, because we have to provide enough left and right
               context that it can evaluate the first chunk.  Note: as
               elsewhere, the job of duplicating first and last frames enough to
               provide the required left/right context to the network, is left
               to the caller (at runtime, not during compilation).
   @param [out] request2  The second of the 3 requests that this function generates.
               Caution: none of the inputs and outputs should overlap.
   @param [out] request3  The third of the 3 requests that this function generates.
                It will be the same as request2, except for a time offset.
*/
void CreateLoopedComputationRequest(const Nnet &nnet,
                                    int32 chunk_size,
                                    int32 frame_subsampling_factor,
                                    int32 ivector_period,
                                    int32 left_context_begin,
                                    int32 right_context,
                                    int32 num_sequences,
                                    ComputationRequest *request1,
                                    ComputationRequest *request2,
                                    ComputationRequest *request3);


/**
   This function is deprecated.  It has the same interface as
   CreateLoopedComputationRequest(), except that the left and right context are
   specified in a different way (as just the 'extra' part).  It is deprecated because
   this function has to work out the left and right context of the network, which
   turns out to be quite slow if it's done after you call ModifyNnetIvectorPeriod().
*/
void CreateLoopedComputationRequestSimple(const Nnet &nnet,
                                          int32 chunk_size,
                                          int32 frame_subsampling_factor,
                                          int32 ivector_period,
                                          int32 extra_left_context_begin,
                                          int32 extra_right_context,
                                          int32 num_sequences,
                                          ComputationRequest *request1,
                                          ComputationRequest *request2,
                                          ComputationRequest *request3);





} // namespace nnet3
} // namespace kaldi


#endif
