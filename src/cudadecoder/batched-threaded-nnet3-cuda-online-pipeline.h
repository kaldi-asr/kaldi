// cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h
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

#ifndef KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_ONLINE_PIPELINE_H_
#define KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_ONLINE_PIPELINE_H_

#define KALDI_CUDA_DECODER_MIN_NCHANNELS_FACTOR 2

#include <atomic>
#include <thread>

#include "base/kaldi-utils.h"
#include "cudadecoder/batched-static-nnet3.h"
#include "cudadecoder/cuda-decoder.h"
#include "cudadecoder/thread-pool-light.h"
#include "cudafeat/online-batched-feature-pipeline-cuda.h"
#include "feat/wave-reader.h"
#include "lat/determinize-lattice-pruned.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "online2/online-nnet2-feature-pipeline.h"

namespace kaldi {
namespace cuda_decoder {

//
// Online Streaming Batched Pipeline calling feature extraction, CUDA light
// Nnet3 driver and CUDA decoder. Can handle up to num_channels streaming audio
// channels in parallel. Each channel is externally identified by a correlation
// id (corr_id). Receives chunks of audio (up to max_batch_size per DecodeBatch
// call). Will call a callback with the final lattice once the processing of the
// final chunk is done.
//
// For an example on how to use that pipeline, see
// cudadecoderbin/batched-threaded-wav-nnet3-online.cc
//
// Feature extraction can be CUDA or CPU
// (multithreaded).
// Internally reuses the concept of channels and lanes from the CUDA decoder
//

struct BatchedThreadedNnet3CudaOnlinePipelineConfig {
  BatchedThreadedNnet3CudaOnlinePipelineConfig()
      : max_batch_size(400),
        num_channels(600),
        num_worker_threads(-1),
        determinize_lattice(true),
        num_decoder_copy_threads(2),
        use_gpu_feature_extraction(true) {}
  void Register(OptionsItf *po) {
    po->Register("max-batch-size", &max_batch_size,
                 "The maximum execution batch size. "
                 "Larger = Better throughput slower latency.");
    po->Register("num-channels", &num_channels,
                 "The number of parallel audio channels. This is the maximum "
                 "number of parallel audio channels supported by the pipeline"
                 ". This should be larger "
                 "than max_batch_size.");
    po->Register("cuda-worker-threads", &num_worker_threads,
                 "(optional) The total number of CPU threads launched to "
                 "process CPU tasks. -1 = use std::hardware_concurrency()");
    po->Register("determinize-lattice", &determinize_lattice,
                 "Determinize the lattice before output.");
    po->Register("cuda-decoder-copy-threads", &num_decoder_copy_threads,
                 "Advanced - Number of worker threads used in the "
                 "decoder for "
                 "the host to host copies.");
    po->Register("gpu-feature-extract", &use_gpu_feature_extraction,
                 "Use GPU feature extraction");

    feature_opts.Register(po);
    decoder_opts.Register(po);
    det_opts.Register(po);
    compute_opts.Register(po);
  }
  int max_batch_size;
  int num_channels;
  int num_worker_threads;
  bool determinize_lattice;
  int num_decoder_copy_threads;
  bool use_gpu_feature_extraction;

  OnlineNnet2FeaturePipelineConfig feature_opts;
  CudaDecoderConfig decoder_opts;
  fst::DeterminizeLatticePhonePrunedOptions det_opts;
  nnet3::NnetSimpleComputationOptions compute_opts;

  void CheckAndFixConfigs() {
    KALDI_ASSERT(max_batch_size > 0);
    // Lower bound on nchannels.
    // Using strictly more than max_batch_size because channels are still used
    // when the lattice postprocessing is running. We still want to run full
    // max_batch_size batches in the meantime
    int min_nchannels =
        max_batch_size * KALDI_CUDA_DECODER_MIN_NCHANNELS_FACTOR;
    num_channels = std::max(num_channels, min_nchannels);

    // If not set use number of physical threads
    num_worker_threads = (num_worker_threads > 0)
                             ? num_worker_threads
                             : std::thread::hardware_concurrency();
  }
};

class BatchedThreadedNnet3CudaOnlinePipeline {
 public:
  using CorrelationID = uint64_t;
  typedef std::function<void(const string &, bool, bool)> BestPathCallback;
  typedef std::function<void(CompactLattice &)> LatticeCallback;

  BatchedThreadedNnet3CudaOnlinePipeline(
      const BatchedThreadedNnet3CudaOnlinePipelineConfig &config,
      const fst::Fst<fst::StdArc> &decode_fst,
      const nnet3::AmNnetSimple &am_nnet, const TransitionModel &trans_model)
      : config_(config),
        max_batch_size_(config.max_batch_size),
        trans_model_(&trans_model),
        am_nnet_(&am_nnet),
        partial_hypotheses_(NULL),
        end_points_(NULL),
        word_syms_(NULL) {
    config_.compute_opts.CheckAndFixConfigs(am_nnet_->GetNnet().Modulus());
    config_.CheckAndFixConfigs();
    int num_worker_threads = config_.num_worker_threads;
    thread_pool_.reset(new ThreadPoolLight(num_worker_threads));

    Initialize(decode_fst);
  }

  const BatchedThreadedNnet3CudaOnlinePipelineConfig &GetConfig() {
    return config_;
  }

  // Called when a new utterance will be decoded w/ correlation id corr_id
  // When this utterance will be done (when it will receive a chunk with
  // last_chunk=true)
  // If no channels are available, will wait for "wait_for" microseconds
  // Returns true if a channel was available (eventually after waiting for
  // up to wait_for seconds)
  bool TryInitCorrID(CorrelationID corr_id, int wait_for = 0);

  void SetBestPathCallback(CorrelationID corr_id,
                           const BestPathCallback &callback);

  // Set the callback function to call with the final lattice for a given
  // corr_id
  void SetLatticeCallback(CorrelationID corr_id,
                          const LatticeCallback &callback);

  // Chunk of one utterance. We receive batches of those chunks through
  // DecodeBatch
  // Contains pointers to that chunk, the corresponding correlation ID,
  // and whether that chunk is the last one for that utterance
  struct UtteranceChunk {
    CorrelationID corr_id;
    SubVector<BaseFloat> wave_samples;
    bool last_chunk;  // sets to true if last chunk for that
                      // utterance
  };

  // Receive a batch of chunks. Will decode them, then return.
  // If it contains some last chunks for given utterances, it will call
  // FinalizeDecoding (building the final lattice, determinize it, etc.)
  // asynchronously. The callback for that utterance will then be called
  //
  // If partial_hypotheses is not null, generate and set the current partial
  // hypotheses in partial_hypotheses The pointers in partial_hypotheses are
  // only valid until the next DecodeBatch call - perform a deep copy if
  // necessary
  void DecodeBatch(
      const std::vector<CorrelationID> &corr_ids,
      const std::vector<SubVector<BaseFloat>> &wave_samples,
      const std::vector<bool> &is_first_chunk,
      const std::vector<bool> &is_last_chunk,
      std::vector<const std::string *> *partial_hypotheses = nullptr,
      std::vector<bool> *end_point = nullptr);

  // Version providing directly the features. Only runs nnet3 & decoder
  // Used when we want to provide the final ivectors (offline case)
  // channels can be provided if they are known (internal use)
  void DecodeBatch(const std::vector<CorrelationID> &corr_ids,
                   const std::vector<BaseFloat *> &d_features,
                   const int features_frame_stride,
                   const std::vector<int> &n_input_frames_valid,
                   const std::vector<BaseFloat *> &d_ivectors,
                   const std::vector<bool> &is_first_chunk,
                   const std::vector<bool> &is_last_chunk,
                   std::vector<int> *channels = NULL);

  void ComputeGPUFeatureExtraction(
      const std::vector<int> &channels,
      const std::vector<SubVector<BaseFloat>> &wave_samples,
      const std::vector<bool> &is_first_chunk,
      const std::vector<bool> &is_last_chunk);

  void ComputeCPUFeatureExtraction(
      const std::vector<int> &channels,
      const std::vector<SubVector<BaseFloat>> &wave_samples,
      const std::vector<bool> &is_last_chunk);

  // Maximum number of samples per chunk
  int32 GetNSampsPerChunk() { return samples_per_chunk_; }
  int32 GetNInputFramesPerChunk() { return input_frames_per_chunk_; }
  float GetModelFrequency() { return model_frequency_; }
  int GetTotalNnet3RightContext() {
    return cuda_nnet3_->GetTotalNnet3RightContext();
  }
  // Maximum number of seconds per chunk
  BaseFloat GetSecondsPerChunk() { return seconds_per_chunk_; }

  // Used for partial hypotheses
  void SetSymbolTable(const fst::SymbolTable &word_syms) {
    word_syms_ = &word_syms;
    KALDI_ASSERT(cuda_decoder_);
    cuda_decoder_->SetSymbolTable(word_syms);
  }

  // Wait for all lattice callbacks to complete
  // Can be called after DecodeBatch
  void WaitForLatticeCallbacks();

 private:
  // Initiliaze this object
  void Initialize(const fst::Fst<fst::StdArc> &decode_fst);

  // Allocate and initialize data that will be used for computation
  void AllocateAndInitializeData(const fst::Fst<fst::StdArc> &decode_fst);

  // Reads what's needed from models, such as left and right context
  void ReadParametersFromModel();

  // Following functions are DecodeBatch's helpers

  // Filling  curr_batch_ichannels_
  void ListIChannelsInBatch(const std::vector<CorrelationID> &corr_ids,
                            std::vector<int> *channels);
  void CPUFeatureExtraction(
      const std::vector<int> &channels,
      const std::vector<SubVector<BaseFloat>> &wave_samples);

  // Compute features and ivectors for the chunk
  // curr_batch[element]
  // CPU function
  void ComputeOneFeature(int element);
  static void ComputeOneFeatureWrapper(void *obj, uint64_t element,
                                       void *ignored) {
    static_cast<BatchedThreadedNnet3CudaOnlinePipeline *>(obj)
        ->ComputeOneFeature(element);
  }
  void RunNnet3(const std::vector<int> &channels,
                const std::vector<BaseFloat *> &d_features,
                const int feature_stride,
                const std::vector<int> &n_input_frames_valid,
                const std::vector<bool> &is_first_chunk,
                const std::vector<bool> &is_last_chunk,
                const std::vector<BaseFloat *> &d_ivectors);

  void RunDecoder(const std::vector<int> &channels);

  void RunCallbacksAndFinalize(const std::vector<CorrelationID> &corr_ids,
                               const std::vector<int> &channels,
                               const std::vector<bool> &is_last_chunk);

  // If an utterance is done, we call FinalizeDecoding async on
  // the threadpool
  // it will call the utterance's callback when done
  void FinalizeDecoding(int32 ichannel, const LatticeCallback *callback);
  // static wrapper for thread pool
  static void FinalizeDecodingWrapper(void *obj, uint64_t ichannel64,
                                      void *callback_ptr) {
    int32 ichannel = static_cast<int32>(ichannel64);
    const LatticeCallback *callback =
        static_cast<const LatticeCallback *>(callback_ptr);
    static_cast<BatchedThreadedNnet3CudaOnlinePipeline *>(obj)
        ->FinalizeDecoding(ichannel, callback);
  }
  // Data members

  BatchedThreadedNnet3CudaOnlinePipelineConfig config_;
  int32 max_batch_size_;  // extracted from config_
  // Models
  const TransitionModel *trans_model_;
  const nnet3::AmNnetSimple *am_nnet_;
  std::unique_ptr<OnlineNnet2FeaturePipelineInfo> feature_info_;

  // Decoder channels currently available, w/ mutex
  std::vector<int32> available_channels_;
  std::mutex available_channels_m_;

  // corr_id -> decoder channel map
  std::unordered_map<CorrelationID, int32> corr_id2channel_;

  // Where to store partial_hypotheses_ and end_points_ if available
  std::vector<const std::string *> *partial_hypotheses_;
  std::vector<bool> *end_points_;

  // Used when none were provided by the API but we still need to generate
  // partial hyp and endp
  std::vector<const std::string *> partial_hypotheses_buf_;
  std::vector<bool> end_points_buf_;

  // The callback is called once the final lattice is ready
  std::unordered_map<CorrelationID, const LatticeCallback> lattice_callbacks_;
  // Used for both final and partial best paths
  std::unordered_map<CorrelationID, const BestPathCallback>
      best_path_callbacks_;
  // Lock for callbacks
  std::mutex map_callbacks_m_;

  // New channels in the current batch. We've just received
  // their first batch
  std::vector<int32> list_channels_first_chunk_;

  std::vector<int> n_samples_valid_, n_input_frames_valid_;

  std::vector<std::vector<std::pair<int, BaseFloat *>>>
      all_frames_log_posteriors_;

  // Channels done after current batch. We've just received
  // their last chunk
  std::vector<int> list_channels_last_chunk_;
  std::vector<CorrelationID> list_corr_id_last_chunk_;
  std::vector<LatticeCallback *> list_lattice_callbacks_last_chunk_;

  // Number of frames already computed in channel (before
  // curr_batch_)
  std::vector<int32> channel_frame_offset_;

  // Parameters extracted from the models
  int input_frames_per_chunk_;
  int output_frames_per_chunk_;
  BaseFloat seconds_per_chunk_;
  BaseFloat samples_per_chunk_;
  BaseFloat model_frequency_;
  int32 ivector_dim_, input_dim_;

  // Buffers used during computation
  Matrix<BaseFloat> h_all_features_;
  Matrix<BaseFloat> h_all_waveform_;
  CuMatrix<BaseFloat> d_all_waveform_;
  CuMatrix<BaseFloat> d_all_features_;
  Matrix<BaseFloat> h_all_ivectors_;
  CuVector<BaseFloat> d_all_ivectors_;  // gpu pipeline uses a meta vector
  CuMatrix<BaseFloat> d_all_log_posteriors_;

  bool use_ivectors_;
  // Used with CPU features extraction. Contains the number of CPU FE tasks
  // still running
  std::atomic<int> n_compute_features_not_done_;
  // Number of CPU lattice postprocessing tasks still running
  std::atomic<int> n_lattice_callbacks_not_done_;

  // Current assignement buffers, when DecodeBatch is running
  std::vector<int> channels_;
  std::vector<BaseFloat *> d_features_ptrs_;
  std::vector<BaseFloat *> d_ivectors_ptrs_;

  // Used by CPU FE threads. Could be merged with channels_
  const std::vector<int> *fe_threads_channels_;
  const std::vector<SubVector<BaseFloat>> *fe_threads_wave_samples_;

  std::unique_ptr<OnlineBatchedFeaturePipelineCuda> gpu_feature_pipeline_;
  std::unique_ptr<BatchedStaticNnet3> cuda_nnet3_;

  // Feature pipelines, associated to a channel
  // Only used if feature extraction is run on the CPU
  std::vector<std::unique_ptr<OnlineNnet2FeaturePipeline>> feature_pipelines_;

  // HCLG graph : CudaFst object is a host object, but contains
  // data stored in
  // GPU memory
  std::shared_ptr<CudaFst> cuda_fst_;
  std::unique_ptr<CudaDecoder> cuda_decoder_;

  std::unique_ptr<ThreadPoolLight> thread_pool_;

  // Used for debugging
  const fst::SymbolTable *word_syms_;
  // Used when printing to stdout for debugging purposes
  std::mutex stdout_m_;
};

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_ONLINE_PIPELINE_H_
#endif  // HAVE_CUDA
