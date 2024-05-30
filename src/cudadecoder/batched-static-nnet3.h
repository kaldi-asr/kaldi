// cudadecoder/batched-static-nnet3.h
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

#if HAVE_CUDA == 1

#ifndef KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_H_
#define KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_H_

// Following define is NOT an upper bound for max_batch_size
// It only concerns the nnet3 compiled computation
// If we use a batch size > MAX_COMPUTE_BATCH_SIZE, we will run nnet3
// multiple times, each computing minibatches of size MAX_COMPUTE_BATCH_SIZE
// MAX_COMPUTE_BATCH_SIZE is defined to be big enough to hide kernel launch
// latency and increase the arithmetic intensity of the GEMMs
// not not bigger so that running partial batches is faster
// (e.g. running a batch size = 72 with max_batch_size_=512)
#define MAX_COMPUTE_BATCH_SIZE 64

#include "cudadecoder/batched-static-nnet3-kernels.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace cuda_decoder {

struct BatchedStaticNnet3Config {
  BatchedStaticNnet3Config()
      : max_batch_size(200), nchannels(-1), has_ivector(false) {}
  nnet3::NnetSimpleComputationOptions compute_opts;
  int max_batch_size;
  int nchannels;
  bool has_ivector;  // probably can be deducted from am_nnet?
};

// Light driver for Nnet3. Compiles the nnet only once and reuse it.
// It is cheaper to waste some computation by adding partial chunks to a batch
// than recompiling a nnet3 computation just for that chunk (and running smaller
// batches, because each batch would be specialized to a specific chunk
// size/batch size)
// Also takes care of storing/restoring left/right context, generating initial
// context/final context, flushing this context.
// Supports context switch with ivectors
class BatchedStaticNnet3 {
 public:
  BatchedStaticNnet3(const BatchedStaticNnet3Config &config,
                     const nnet3::AmNnetSimple &am_nnet)
      : config_(config),
        am_nnet_(am_nnet),
        max_batch_size_(config.max_batch_size),
        has_ivector_(config.has_ivector),
        log_priors_(am_nnet.Priors()) {
    nchannels_ = (config.nchannels != -1) ? config.nchannels : max_batch_size_;
    KALDI_ASSERT(max_batch_size_ > 0);
    nnet3_batch_size_ = std::min(max_batch_size_, MAX_COMPUTE_BATCH_SIZE);
    KALDI_ASSERT(nchannels_ >= max_batch_size_);
    ReadParametersFromModelAndConfig();
    CompileNnet3();
    Allocate();
  }

  virtual ~BatchedStaticNnet3() { Deallocate(); }

  // Receives a batch with a set of chunks (at most one chunk per channel).
  // Restore contextes, run nnet3, save the context for next RunBatch.
  // Pointers to the output frames are set in all_frames_log_posteriors
  //
  // For each batch slot i:
  // 	- channels[i] is the associated channel.
  // 	- d_features[i] points to a submatrix of features. It is made of
  // mfcc_dim*n_input_frames_valid[i] BaseFloats
  // 	- d_ivectors[i] is the ivector to use for this nnet3 run, if ivectors
  // are available.
  // 	- n_input_frames_valid[i] how many frames can be read from d_features.
  // It can be strictly less than frames_per_chunk, for instance for the last
  // chunk
  // 	- is_first_chunk[i] set <=> first chunk for that channel. Will reset
  // left context
  // 	- is_last_chunk[i] set <=> last chunk for that channel. Will flush right
  // context
  //    - d_all_log_posteriors where to store the output frames. Could be owned
  //    by that class (the decoder is supposed to access those frames through
  //    all_frames_log_posteriors
  //    - all_frames_log_posteriors. For each output frame index (dim1), list
  //    all the channels which have a valid frame, and the corresponding pointer
  //    in memory.
  //
  //    E.g.: We called RunBatch with channels = {4,7} Channels 4 has
  //    2 valid output frames, channel 7 has 3 valid output frames.
  //    all_frames_log_posteriors = [
  //    [[4,ptr0,4],[7,ptr0,7]],
  //    [[4,ptr1,4],[7,ptr1,7]],
  //    [[7,ptr2,7]],
  //    ]
  //    with ptri,j the pointer to the output frame i for channel j.
  //    frame i is a local indexing: the first frame for channel j for this
  //    RunBatch call will always be 0, even if other output frames have already
  //    been generated for that channel in previous RunBatch calls.
  void RunBatch(const std::vector<int> &channels,
                const std::vector<BaseFloat *> &d_features,
                const int features_stride,
                const std::vector<BaseFloat *> &d_ivectors,
                const std::vector<int> &n_input_frames_valid,
                const std::vector<bool> &is_first_chunk,
                const std::vector<bool> &is_last_chunk,
                CuMatrix<BaseFloat> *d_all_log_posteriors,
                std::vector<std::vector<std::pair<int, const BaseFloat *>>>
                    *all_frames_log_posteriors);

  // Nnet3 puts the output frames in the matrix all_frames_log_posteriors_ptrs
  // However, we still have to only consider "valid" output frames.
  // See RunBatch comments for a description of the output
  // n_output_frames_valid_offset describes how many valid output frames we
  // already have in all_frames_log_posteriors_ptrs for each channel
  void FormatOutputPtrs(
      const std::vector<int> &channels,
      CuMatrix<BaseFloat> *d_all_log_posteriors,
      std::vector<std::vector<std::pair<int, const BaseFloat *>>>
          *all_frames_log_posteriors_ptrs,
      const std::vector<int> &n_output_frames_valid,
      const std::vector<int> *n_output_frames_valid_offset = NULL);

  int GetNOutputFramesPerChunk() { return output_frames_per_chunk_; }
  int GetTotalNnet3RightContext() { return total_nnet_right_context_; }

 private:
  // Compiling nnet3 using that computation request
  void ReadParametersFromModelAndConfig();
  // Define the computation request for nnet3 based on parameters
  void SetComputationRequest();
  void Allocate();
  void PresetKernelParams();
  void Deallocate();
  void CompileNnet3();
  // Run Nnet3 itself. Divides the execution batch into smaller nnet3 batches
  // That nnet3 batch size is choosen so that we saturate the GPU, but we still
  // keep the smallest batch size possible to have a better granularity with
  // partial batches
  void RunNnet3(CuMatrix<BaseFloat> *d_all_log_posteriors, int batch_size);
  void BatchContextSwitch(const std::vector<int> &channels,
                          const std::vector<BaseFloat *> &d_features,
                          const int features_stride,
                          const std::vector<BaseFloat *> &d_ivectors,
                          const std::vector<int> &n_input_frames_valid,
                          bool flush_eos_context,
                          std::vector<int> *n_output_frames_valid);
  void InitChannel(int32 ichannel) {
    KALDI_ASSERT(ichannel < nchannels_);
    channel_n_frames_in_context_[ichannel] = 0;
  }

  BatchedStaticNnet3Config config_;
  cudaStream_t st_;
  nnet3::AmNnetSimple am_nnet_;
  int max_batch_size_;
  int nnet3_batch_size_;  // Cf RunNnet3. Batch size for the execution for nnet3
  int nchannels_;  // Number of possible channels. Each channel owns a context.
  bool has_ivector_;
  CuVector<BaseFloat> log_priors_;

  // Extracted from config or models
  int input_dim_;    // mfcc dim
  int ivector_dim_;  // ivector dim
  int input_frames_per_chunk_;
  int input_frames_per_chunk_with_context_;  // input_frames_per_chunk_ with
                                             // left and right context
  int total_nnet_left_context_;
  int total_nnet_right_context_;
  int total_nnet_context_;
  int output_frames_per_chunk_;
  int subsampling_factor_;

  // Storing frames which will be used in future context
  // If the channel has just been resetted, those frames are empty.
  // Otherwise, it contains at most total_nnet_context_ frames
  CuMatrix<BaseFloat> d_all_context_frames_;
  CuMatrix<BaseFloat> d_batch_with_context_;
  CuMatrix<BaseFloat> d_nnet3_input_;
  CuMatrix<BaseFloat> d_nnet3_ivectors_;
  CuMatrix<BaseFloat> d_nnet3_output_;
  CuMatrix<BaseFloat> d_batch_ivectors_;
  CuMatrix<BaseFloat> d_all_log_posteriors_;
  CuMatrix<BaseFloat> d_all_eos_log_posteriors_;
  // batch slot assignement. Size [max_batch_size]
  BatchSlotAssignment *d_batch_slot_assignement_;
  BatchSlotAssignment *h_batch_slot_assignement_;
  BatchedStaticNnet3KernelParams context_switch_kernel_params_;
  cudaEvent_t batch_slot_assignement_copy_evt_;
  // Number of frames already stored in context
  // Size [nchannels]
  // If channel not initialized, equals to -1
  std::vector<int> channel_n_frames_in_context_;
  std::vector<int> n_output_frames_valid_;

  // Used to flush context at eos (end of sequence)
  std::vector<int> eos_channels_;
  std::vector<BaseFloat *> d_eos_features_;
  std::vector<BaseFloat *> d_eos_ivectors_;
  std::vector<int> eos_n_input_frames_valid_;
  std::vector<int> eos_n_output_frames_valid_;
  std::vector<int> eos_n_output_frames_offset_;

  std::unique_ptr<nnet3::CachingOptimizingCompiler> compiler_;
  std::shared_ptr<const nnet3::NnetComputation>
      computation_;  // shared because returned as shared by compiler
  nnet3::ComputationRequest request_;
};
}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_H_
#endif  // HAVE_CUDA
