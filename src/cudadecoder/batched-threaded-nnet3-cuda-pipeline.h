// cudadecoder/batched-threaded-nnet3-cuda-pipeline.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
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

#ifndef KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_DECODER_H_
#define KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_DECODER_H_

#include <atomic>
#include <thread>

#include "cudadecoder/cuda-decoder.h"
#include "decodable-cumatrix.h"
#include "feat/wave-reader.h"
#include "lat/determinize-lattice-pruned.h"
#include "nnet3/nnet-batch-compute.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "thread-pool.h"

namespace kaldi {
namespace cuda_decoder {

/* BatchedThreadedNnet3CudaPipelineConfig
 * This class is a common configuration class for the various components
 * of a batched cuda multi-threaded pipeline.  It defines a single place
 * to control all operations and ensures that the various componets
 * match configurations
 */
// configuration options common to the BatchedThreadedNnet3CudaPipeline and
// BatchedThreadedNnet3CudaPipeline
struct BatchedThreadedNnet3CudaPipelineConfig {
  BatchedThreadedNnet3CudaPipelineConfig()
      : max_batch_size(100),
        batch_drain_size(10),
        num_control_threads(2),
        num_worker_threads(20),
        determinize_lattice(true),
        max_pending_tasks(4000){};
  void Register(OptionsItf *po) {
    po->Register("max-batch-size", &max_batch_size,
                 "The maximum batch size to be used by the decoder. "
                 "Higher->Faster, more GPU memory used");
    po->Register("batch-drain-size", &batch_drain_size,
                 "How far to drain the batch before refilling work. This "
                 "batches pre/post decode work");
    po->Register("cuda-control-threads", &num_control_threads,
                 "The number of pipeline control threads for the CUDA work. "
                 "e.g. 2 control threads -> 2 independent CUDA pipeline (nnet3 "
                 "and decoder)");
    po->Register(
        "cuda-worker-threads", &num_worker_threads,
        "The total number of CPU threads launched to process CPU tasks");
    po->Register("determinize-lattice", &determinize_lattice,
                 "Determinize the lattice before output");
    po->Register("max-outstanding-queue-length", &max_pending_tasks,
                 "Number of files to allow to be outstanding at a time. When "
                 "the number of files is larger than this handles will be "
                 "closed before opening new ones in FIFO order");

    decoder_opts.nlanes = max_batch_size;
    decoder_opts.nchannels = max_batch_size;

    feature_opts.Register(po);
    decoder_opts.Register(po);
    det_opts.Register(po);
    compute_opts.Register(po);
  }
  int max_batch_size;
  int batch_drain_size;
  int num_control_threads;
  int num_worker_threads;
  bool determinize_lattice;
  int max_pending_tasks;

  OnlineNnet2FeaturePipelineConfig feature_opts;      // constant readonly
  CudaDecoderConfig decoder_opts;                     // constant readonly
  fst::DeterminizeLatticePhonePrunedOptions det_opts; // constant readonly
  nnet3::NnetBatchComputerOptions compute_opts;       // constant readonly
};

/*
 * BatchedThreadedNnet3CudaPipeline uses multiple levels of parallelism in order to
 * decode quickly on CUDA GPUs. This is the primary interface for cuda decoding.
 * For examples of how to use this decoder see cudadecoder/README and
 * cudadecoderbin/batched-wav-nnet3-cuda.cc
 */
class BatchedThreadedNnet3CudaPipeline {
public:
  BatchedThreadedNnet3CudaPipeline(const BatchedThreadedNnet3CudaPipelineConfig &config)
      : config_(config){};

  // TODO should this take an nnet instead of a string?
  // allocates reusable objects that are common across all decodings
  void Initialize(const fst::Fst<fst::StdArc> &decode_fst,
                  const nnet3::AmNnetSimple &nnet,
                  const TransitionModel &trans_model);

  // deallocates reusable objects
  void Finalize();

  // query a specific key to see if compute on it is complete
  bool isFinished(const std::string &key);

  // remove an audio file from the decoding and clean up resources
  void CloseDecodeHandle(const std::string &key);

  // Adds a decoding task to the decoder
  void OpenDecodeHandle(const std::string &key, const WaveData &wave_data);
  // When passing in a vector of data, the caller must ensure the data exists
  // until the CloseDecodeHandle is called
  void OpenDecodeHandle(const std::string &key,
                        const VectorBase<BaseFloat> &wave_data,
                        float sample_rate);

  // Copies the raw lattice for decoded handle "key" into lat
  bool GetRawLattice(const std::string &key, Lattice *lat);
  // Determinizes raw lattice and returns a compact lattice
  bool GetLattice(const std::string &key, CompactLattice *lat);

  inline int NumPendingTasks() {
    return (tasks_back_ - tasks_front_ + config_.max_pending_tasks + 1) %
           (config_.max_pending_tasks + 1);
  };

private:
  // State needed for each decode task.
  // This state can be passed around by reference or pointer safely
  // and provides a convieniet way to store all decoding state.
  struct TaskState {
    Vector<BaseFloat> raw_data; // Wave input data when wave_reader passed
    SubVector<BaseFloat> *wave_samples; // Used as a pointer to either the raw
                                        // data or the samples passed
    std::string key;
    float sample_frequency;
    bool error;
    std::string error_string;

    Lattice lat;                // Raw Lattice output
    CompactLattice dlat;        // Determinized lattice output.  Only set if
                                // determinize-lattice=true
    std::atomic<bool> finished; // Tells master thread if task has finished
                                // execution

    bool determinized;

    Vector<BaseFloat> ivector_features;
    Matrix<BaseFloat> input_features;
    CuMatrix<BaseFloat> posteriors;

    TaskState()
        : wave_samples(NULL), sample_frequency(0), error(false),
          finished(false), determinized(false) {}
    ~TaskState() {
      if (wave_samples)
        delete wave_samples;
    }

    // Init when wave data is passed directly in.  This data is deep copied.
    void Init(const std::string &key_in, const WaveData &wave_data_in) {
      raw_data.Resize(wave_data_in.Data().NumRows() *
                          wave_data_in.Data().NumCols(),
                      kUndefined);
      memcpy(raw_data.Data(), wave_data_in.Data().Data(),
             raw_data.Dim() * sizeof(BaseFloat));
      wave_samples = new SubVector<BaseFloat>(raw_data, 0, raw_data.Dim());
      sample_frequency = wave_data_in.SampFreq();
      determinized = false;
      finished = false;
      key = key_in;
    };
    // Init when raw data is passed in.  This data is shallow copied.
    void Init(const std::string &key_in,
              const VectorBase<BaseFloat> &wave_data_in, float sample_rate) {
      wave_samples =
          new SubVector<BaseFloat>(wave_data_in, 0, wave_data_in.Dim());
      sample_frequency = sample_rate;
      determinized = false;
      finished = false;
      key = key_in;
    }
  };

  // Holds the current channel state for a worker
  struct ChannelState {
    std::vector<ChannelId> channels;
    std::vector<ChannelId> free_channels;
    std::vector<ChannelId> completed_channels;
  };

  // Adds task to the PendingTaskQueue
  void AddTaskToPendingTaskQueue(TaskState *task);

  // Attempts to fill the batch from the task queue.  May not fully fill the
  // batch.
  void AquireAdditionalTasks(CudaDecoder &cuda_decoder,
                             ChannelState &channel_state,
                             std::vector<TaskState *> &tasks);

  // Computes Features for a single decode instance.
  void ComputeOneFeature(TaskState *task);

  // Computes Nnet across the current decode batch
  void ComputeBatchNnet(nnet3::NnetBatchComputer &computer, int32 first,
                        std::vector<TaskState *> &tasks);

  // Allocates decodables for tasks in the range of
  // dstates[first,dstates.size())
  void AllocateDecodables(int32 first, std::vector<TaskState *> &tasks,
                          std::vector<CudaDecodableInterface *> &decodables);

  // Removes all completed channels from the channel list.
  // Also enqueues up work for post processing
  void
  RemoveCompletedChannels(CudaDecoder &cuda_decoder,
                          ChannelState &channel_state,
                          std::vector<CudaDecodableInterface *> &decodables,
                          std::vector<TaskState *> &tasks);

  // For each completed decode perform post processing work and clean up
  void PostDecodeProcessing(CudaDecoder &cuda_decoder,
                            ChannelState &channel_state,
                            std::vector<CudaDecodableInterface *> &decodables,
                            std::vector<TaskState *> &tasks);

  void DeterminizeOneLattice(TaskState *state);

  // Thread execution function.  This is a single worker thread which processes
  // input.
  void ExecuteWorker(int threadId);

  const BatchedThreadedNnet3CudaPipelineConfig &config_;

  CudaFst cuda_fst_;
  const TransitionModel *trans_model_;
  const nnet3::AmNnetSimple *am_nnet_;
  nnet3::DecodableNnetSimpleLoopedInfo *decodable_info_;
  OnlineNnet2FeaturePipelineInfo *feature_info_;

  std::mutex tasks_mutex_; // protects tasks_front_ and pending_task_queue_ for
                           // workers
  std::mutex tasks_add_mutex_; // protect OpenDecodeHandle if multiple threads
                               // access
  std::mutex tasks_lookup_mutex_; // protext tasks_lookup map
  std::atomic<int> tasks_front_, tasks_back_;
  TaskState **pending_task_queue_;

  std::atomic<bool> exit_;      // signals threads to exit
  std::atomic<int> numStarted_; // signals master how many threads have started

  ThreadPool *work_pool_; // thread pool for CPU work

  std::map<std::string, TaskState> tasks_lookup_; // Contains a map of
                                                  // utterance to TaskState
  std::vector<std::thread> thread_contexts_;      // A list of thread contexts
};

}  // end namespace cuda_decoder
} // end namespace kaldi.

#endif  // KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_DECODER_H_
