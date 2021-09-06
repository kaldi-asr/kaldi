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

#ifndef KALDI_CUDA_DECODER_BATCHED_THREADED_NNET3_CUDA_PIPELINE_H_
#define KALDI_CUDA_DECODER_BATCHED_THREADED_NNET3_CUDA_PIPELINE_H_

#include <atomic>
#include <memory>
#include <thread>

#include "cudadecoder/cuda-decoder.h"
#include "cudadecoder/decodable-cumatrix.h"
#include "cudadecoder/thread-pool.h"
#include "cudafeat/online-cuda-feature-pipeline.h"
#include "feat/wave-reader.h"
#include "lat/determinize-lattice-pruned.h"
#include "nnet3/nnet-batch-compute.h"
#include "nnet3/decodable-simple-looped.h"
#include "online2/online-nnet2-feature-pipeline.h"

// This pipeline is deprecated and will be removed. Please switch to
// batched-threaded-nnet3-cuda-pipeline2

// If num_channels sets to automatic,
// num_channels = [this define] * max_batch_size
#define KALDI_CUDA_DECODER_CHANNELS_BATCH_SIZE_RATIO 1.3

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
struct [[deprecated]] BatchedThreadedNnet3CudaPipelineConfig {
  BatchedThreadedNnet3CudaPipelineConfig()
      : max_batch_size(200),
        num_channels(-1),
        batch_drain_size(10),
        num_control_threads(2),
        num_worker_threads(20),
        determinize_lattice(true),
        max_pending_tasks(4000),
        pending_queue_padding(10),
        num_decoder_copy_threads(2),
        gpu_feature_extract(true){};
  void Register(OptionsItf *po) {
    po->Register("max-batch-size", &max_batch_size,
                 "The maximum batch size to be used by the decoder. "
                 "This is also the number of lanes in the CudaDecoder. "
                 "Larger = Faster and more GPU memory used.");
    std::ostringstream num_channels_desc;
    num_channels_desc
        << "The number of channels "
           "allocated to the cuda decoder.  This should be larger "
           "than max_batch_size.  Each channel consumes a small "
           "amount of memory but also allows us to better overlap "
           "computation"
           " (-1 = set to "
        << KALDI_CUDA_DECODER_CHANNELS_BATCH_SIZE_RATIO << "*max-batch-size).";
    po->Register("num-channels", &num_channels, num_channels_desc.str());
    po->Register("batch-drain-size", &batch_drain_size,
                 "How far to drain the batch before refilling work. This "
                 "batches pre/post decode work.");
    po->Register("cuda-control-threads", &num_control_threads,
                 "The number of pipeline control threads for the CUDA work. "
                 "e.g. 2 control threads -> 2 independent CUDA pipeline "
                 "(nnet3 "
                 "and decoder).");
    po->Register("cuda-worker-threads", &num_worker_threads,
                 "The total number of CPU threads launched to "
                 "process CPU tasks.");
    po->Register("determinize-lattice", &determinize_lattice,
                 "Determinize the lattice before output.");
    po->Register("max-outstanding-queue-length", &max_pending_tasks,
                 "Number of files to allow to be outstanding at a time. "
                 "When "
                 "the number of files is larger than this handles will be "
                 "closed before opening new ones in FIFO order.");
    po->Register("cuda-decoder-copy-threads", &num_decoder_copy_threads,
                 "Advanced - Number of worker threads used in the "
                 "decoder for "
                 "the host to host copies.");
    po->Register("gpu-feature-extract", &gpu_feature_extract,
                 "Extract features on the GPU.  This reduces CPU overhead "
                 "leading to better scalability but may reduce overall "
                 "performance for a single GPU.");

    feature_opts.Register(po);
    decoder_opts.Register(po);
    det_opts.Register(po);
    compute_opts.Register(po);
  }
  int max_batch_size;
  int num_channels;
  int batch_drain_size;
  int num_control_threads;
  int num_worker_threads;
  bool determinize_lattice;
  int max_pending_tasks;
  int pending_queue_padding;
  int num_decoder_copy_threads;
  bool gpu_feature_extract;

  void ComputeConfig() {
    if (num_channels == -1)
      num_channels =
          max_batch_size * KALDI_CUDA_DECODER_CHANNELS_BATCH_SIZE_RATIO;
  }

  OnlineNnet2FeaturePipelineConfig feature_opts;       // constant readonly
  CudaDecoderConfig decoder_opts;                      // constant readonly
  fst::DeterminizeLatticePhonePrunedOptions det_opts;  // constant readonly
  nnet3::NnetBatchComputerOptions compute_opts;        // constant readonly
};

/*
 * BatchedThreadedNnet3CudaPipeline uses multiple levels of parallelism in order
 * to decode quickly on CUDA GPUs. This is the primary interface for cuda
 * decoding. For examples of how to use this decoder see cudadecoder/README and
 * cudadecoderbin/batched-wav-nnet3-cuda.cc
 */
class [[deprecated]] BatchedThreadedNnet3CudaPipeline {
 public:
  BatchedThreadedNnet3CudaPipeline(
      const BatchedThreadedNnet3CudaPipelineConfig &config)
      : config_(config), all_group_tasks_not_done_(0) {
    config_.ComputeConfig();
  };

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
  void CloseAllDecodeHandlesForGroup(const std::string &group);
  void CloseAllDecodeHandles();

  // Adds a decoding task to the decoder
  // When passing in a vector of data, the caller must ensure the data
  // exists until the CloseDecodeHandle/WaitForAllTasks is called callback
  // is called once task is done and we pass it the final lattice callback
  // can be used to compute lattice rescoring, find best path in lattice,
  // writing lattice to disk, etc. Important: callback is launched in the
  // threadpool. It must be threadsafe. For instance, if writing to disk,
  // or to stdout, use a lock: e.g. :
  // {
  // 	std::lock_guard<std::mutex> lock(global_mutex);
  // 	// write lattice to disk
  //    // lock is released in the destructor of lock_guard<>
  // }
  void OpenDecodeHandle(
      const std::string &key, const WaveData &wave_data,
      const std::string &group = std::string(),
      const std::function<void(CompactLattice &clat)> &callback =
          std::function<void(CompactLattice &clat)>());
  // When passing in a vector of data, the caller must ensure the data
  // exists until the CloseDecodeHandle is called
  void OpenDecodeHandle(
      const std::string &key, const VectorBase<BaseFloat> &wave_data,
      float sample_rate, const std::string &group = std::string(),
      const std::function<void(CompactLattice &clat)> &callback =
          std::function<void(CompactLattice &clat)>());

  // Copies the raw lattice for decoded handle "key" into lat
  bool GetRawLattice(const std::string &key, Lattice *lat);
  // Determinizes raw lattice and returns a compact lattice
  bool GetLattice(const std::string &key, CompactLattice *lat);

  int32 GetNumberOfTasksPending();

  // Wait for all tasks to complete
  void WaitForAllTasks();
  // Wait for all tasks in the group to complete
  void WaitForGroup(const std::string &group);
  // Check if a group is available. Returns if not.
  bool IsGroupCompleted(const std::string &group);
  // Wait for any group to complete, then returns which group completed
  std::string WaitForAnyGroup();
  // Check if any group is available. If one is available, set its name in
  // *group
  bool IsAnyGroupCompleted(std::string *group);
  inline int NumPendingTasks() {
    return (tasks_back_ - tasks_front_ + config_.max_pending_tasks +
            config_.pending_queue_padding) %
           (config_.max_pending_tasks + config_.pending_queue_padding);
  };

 private:
  // Task data used during computation
  // Is cleared when task is completed
  struct TaskData {
    Vector<BaseFloat> raw_data;  // Wave input data when wave_reader passed
    std::shared_ptr<SubVector<BaseFloat>>
        wave_samples;  // Used as a pointer to either the raw
                       // data or the samples passed
    float sample_frequency;
    Vector<BaseFloat> ivector_features_cpu;
    Matrix<BaseFloat> input_features_cpu;
    CuVector<BaseFloat> ivector_features;
    CuMatrix<BaseFloat> input_features;
    CuMatrix<BaseFloat> posteriors;

    TaskData(const WaveData &wave_data_in)
        : wave_samples(NULL), sample_frequency(0) {
      int rows = wave_data_in.Data().NumRows();
      int cols = wave_data_in.Data().NumCols();
      int stride = wave_data_in.Data().Stride();

      raw_data.Resize(rows * cols, kUndefined);

      if (stride == cols) {
        // contigious so use one large memory copy
        memcpy(raw_data.Data(), wave_data_in.Data().Data(),
               rows * cols * sizeof(BaseFloat));
      } else {
        // data is not contigious so we need to copy one
        // row at a time
        for (int i = 0; i < rows; i++) {
          memcpy(raw_data.Data() + i * cols, wave_data_in.Data().RowData(i),
                 cols * sizeof(BaseFloat));
        }
      }
      wave_samples =
          std::make_shared<SubVector<BaseFloat>>(raw_data, 0, raw_data.Dim());
      sample_frequency = wave_data_in.SampFreq();
    };

    // Init when raw data is passed in.  This data is shallow
    // copied.
    TaskData(const VectorBase<BaseFloat> &wave_data_in, float sample_rate) {
      wave_samples = std::make_shared<SubVector<BaseFloat>>(wave_data_in, 0,
                                                            wave_data_in.Dim());
      sample_frequency = sample_rate;
    }
  };

  // State needed for each decode task.
  // This state can be passed around by reference or pointer safely
  // and provides a convieniet way to store all decoding state.
  struct TaskState {
    std::string key;
    std::string group;  // group for that task. "" is default
    bool error;
    std::string error_string;

    std::unique_ptr<TaskData> task_data;

    int32 ichannel;              // associated CudaDecoder channel
    Lattice lat;                 // Raw Lattice output
    CompactLattice dlat;         // Determinized lattice output.  Only set
                                 // if determinize-lattice=true
    std::atomic<bool> finished;  // Tells master thread if task has
                                 // finished execution

    bool determinized;

    // (optional) callback is called task is finished and we have a
    // lattice ready that way we can compute all CPU tasks in the
    // threadpool (lattice rescoring, find best path in lattice,
    // etc.)
    std::function<void(CompactLattice &clat)> callback;

    TaskState() : error(false), finished(false), determinized(false) {}

    // Init when wave data is passed directly in.  This data is deep
    // copied.
    void Init(const std::string &key_in, const WaveData &wave_data_in) {
      task_data.reset(new TaskData(wave_data_in));
      key = key_in;
    };
    // Init when raw data is passed in.  This data is shallow
    // copied.
    void Init(const std::string &key_in,
              const VectorBase<BaseFloat> &wave_data_in, float sample_rate) {
      task_data.reset(new TaskData(wave_data_in, sample_rate));
      key = key_in;
    }
  };

  // Creating a new task in the hashmaps
  TaskState *AddTask(const std::string &key, const std::string &group);

  // Holds the current channel state for a worker
  struct ChannelState {
    std::vector<ChannelId> channels;
    std::vector<ChannelId> free_channels;
    std::vector<ChannelId> completed_channels;
    std::mutex free_channels_mutex;
  };

  // Adds task to the PendingTaskQueue
  void AddTaskToPendingTaskQueue(TaskState *task);

  // Attempts to fill the batch from the task queue.  May not fully fill
  // the batch.
  void AquireAdditionalTasks(CudaDecoder &cuda_decoder,
                             ChannelState &channel_state,
                             std::vector<TaskState *> &tasks);

  // Computes Features for a single decode instance.
  void ComputeOneFeatureCPU(TaskState *task);

  // Computes features across the tasks[first,tasks.size()
  void ComputeBatchFeatures(int32 first, std::vector<TaskState *> &tasks,
                            OnlineCudaFeaturePipeline &feature_pipeline);

  // Computes Nnet across the current decode batch
  void ComputeBatchNnet(nnet3::NnetBatchComputer &computer, int32 first,
                        std::vector<TaskState *> &tasks);

  // Allocates decodables for tasks in the range of
  // dstates[first,dstates.size())
  void AllocateDecodables(int32 first, std::vector<TaskState *> &tasks,
                          std::vector<CudaDecodableInterface *> &decodables);

  // Removes all completed channels from the channel list.
  // Also enqueues up work for post processing
  void RemoveCompletedChannels(
      CudaDecoder &cuda_decoder, ChannelState &channel_state,
      std::vector<CudaDecodableInterface *> &decodables,
      std::vector<TaskState *> &tasks);

  // For each completed decode perform post processing work and clean up
  void PostDecodeProcessing(CudaDecoder &cuda_decoder,
                            ChannelState &channel_state,
                            std::vector<CudaDecodableInterface *> &decodables,
                            std::vector<TaskState *> &tasks);

  // Calls ConcurrentGetRawLatticeSingleChannel and Determinize
  // on a dedicated CPU worker thread at the end of the decode
  void CompleteTask(CudaDecoder *cuda_decoder, ChannelState *channel_state,
                    TaskState *state);

  // Determinize one lattice
  void DeterminizeOneLattice(TaskState *task);
  // Thread execution function.  This is a single worker thread which
  // processes input.
  void ExecuteWorker(int threadId);

  BatchedThreadedNnet3CudaPipelineConfig config_;

  std::unique_ptr<CudaFst> cuda_fst_;
  const TransitionModel *trans_model_;
  const nnet3::AmNnetSimple *am_nnet_;
  OnlineNnet2FeaturePipelineInfo *feature_info_;

  std::mutex tasks_mutex_;         // protects tasks_front_ and
                                   // pending_task_queue_ for workers
  std::mutex tasks_add_mutex_;     // protect OpenDecodeHandle if multiple
                                   // threads access
  std::mutex tasks_lookup_mutex_;  // protext tasks_lookup map
  std::condition_variable tasks_lookup_cv_;
  std::atomic<int> tasks_front_, tasks_back_;
  TaskState **pending_task_queue_;

  std::atomic<bool> exit_;       // signals threads to exit
  std::atomic<int> numStarted_;  // signals master how many threads have started

  ThreadPool *work_pool_;  // thread pool for CPU work
  std::map<std::string, int32> group_tasks_not_done_;
  int32 all_group_tasks_not_done_;
  std::mutex group_tasks_mutex_;
  std::condition_variable group_done_cv_;
  std::unordered_multimap<std::string, TaskState *>
      tasks_group_lookup_;  // group -> list of tasks
  std::unordered_map<std::string, TaskState>
      tasks_lookup_;                          // Contains a map of
                                              // utterance to TaskState
  std::vector<std::thread> thread_contexts_;  // A list of thread contexts
};

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // KALDI_CUDA_DECODER_BATCHED_THREADED_CUDA_DECODER_H_
