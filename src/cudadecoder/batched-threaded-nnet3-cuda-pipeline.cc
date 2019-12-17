// cudadecoder/batched-threaded-nnet3-cuda-pipeline.cc
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

#define SLEEP_BACKOFF_NS 500
#define SLEEP_BACKOFF_S ((double)SLEEP_BACKOFF_NS / 1e9)
#if HAVE_CUDA == 1

#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline.h"
#include <nvToolsExt.h>
#include "base/kaldi-utils.h"

namespace kaldi {
namespace cuda_decoder {

void BatchedThreadedNnet3CudaPipeline::Initialize(
    const fst::Fst<fst::StdArc> &decode_fst, const nnet3::AmNnetSimple &am_nnet,
    const TransitionModel &trans_model) {
  KALDI_LOG << "BatchedThreadedNnet3CudaPipeline Initialize with "
            << config_.num_control_threads << " control threads, "
            << config_.num_worker_threads << " worker threads"
            << " and batch size " << config_.max_batch_size;

  am_nnet_ = &am_nnet;
  trans_model_ = &trans_model;
  cuda_fst_.Initialize(decode_fst, trans_model_);

  feature_info_ = new OnlineNnet2FeaturePipelineInfo(config_.feature_opts);
  feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
  feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;

  // initialize threads and save their contexts so we can join them later
  thread_contexts_.resize(config_.num_control_threads);

  // create work queue, padding so that we can better detect if this
  // overflows. this should not happen and is just there as a sanity check
  pending_task_queue_ = new TaskState *[config_.max_pending_tasks + 
                                        config_.pending_queue_padding];
  tasks_front_ = 0;
  tasks_back_ = 0;

  // ensure all allocations/kernels above are complete before launching threads
  // in different streams.
  cudaStreamSynchronize(cudaStreamPerThread);

  // Create threadpool for CPU work
  work_pool_ = new ThreadPool(config_.num_worker_threads);

  exit_ = false;
  numStarted_ = 0;

  // start workers
  for (int i = 0; i < config_.num_control_threads; i++) {
    thread_contexts_[i] =
        std::thread(&BatchedThreadedNnet3CudaPipeline::ExecuteWorker, this, i);
  }

  // wait for threads to start to ensure allocation time isn't in the timings
  while (numStarted_ < config_.num_control_threads)
    kaldi::Sleep(SLEEP_BACKOFF_S);
}
void BatchedThreadedNnet3CudaPipeline::Finalize() {
  // Tell threads to exit and join them
  exit_ = true;

  for (int i = 0; i < config_.num_control_threads; i++) {
    thread_contexts_[i].join();
  }

  cuda_fst_.Finalize();

  delete feature_info_;
  delete work_pool_;
  delete[] pending_task_queue_;
}

// query a specific key to see if compute on it is complete
bool BatchedThreadedNnet3CudaPipeline::isFinished(const std::string &key) {
  bool finished;
  {
    std::lock_guard<std::mutex> lock(tasks_lookup_mutex_);
    auto it = tasks_lookup_.find(key);
    KALDI_ASSERT(it != tasks_lookup_.end());
    finished = it->second.finished;
  }
  return finished;
}

// remove an audio file from the decoding and clean up resources
void BatchedThreadedNnet3CudaPipeline::CloseDecodeHandle(
    const std::string &key) {
  TaskState *task;
  decltype(tasks_lookup_.end()) it;
  {
    std::lock_guard<std::mutex> lock(tasks_lookup_mutex_);
    it = tasks_lookup_.find(key);
    KALDI_ASSERT(it != tasks_lookup_.end());
    task = &it->second;
  }

  // wait for task to finish processing
  while (task->finished != true) kaldi::Sleep(SLEEP_BACKOFF_S);

  // Delete the group counter if necessary
  std::lock_guard<std::mutex> lk1(group_tasks_mutex_);
  if (group_tasks_not_done_[task->group] == 0)
    group_tasks_not_done_.erase(task->group);

  // remove it
  {
    std::lock_guard<std::mutex> lock(tasks_lookup_mutex_);
    std::string &group = task->group;
    auto p = tasks_group_lookup_.equal_range(group);
    bool found = false;
    for (auto it = p.first; it != p.second; ++it) {
      if (it->second == task) {
        tasks_group_lookup_.erase(it);
        found = true;
        break;
      }
    }
    KALDI_ASSERT(found);
    tasks_lookup_.erase(it);

    if (tasks_lookup_.empty()) tasks_lookup_cv_.notify_all();
  }
}

void BatchedThreadedNnet3CudaPipeline::WaitForAllTasks() {
  std::unique_lock<std::mutex> lk(group_tasks_mutex_);
  group_done_cv_.wait(lk, [this] { return all_group_tasks_not_done_ == 0; });
}

void BatchedThreadedNnet3CudaPipeline::WaitForGroup(const std::string &group) {
  std::unique_lock<std::mutex> lk(group_tasks_mutex_);
  group_done_cv_.wait(
      lk, [this, &group] { return group_tasks_not_done_[group] == 0; });
  // Safe to delete entry from the map now. If the user creates new task in that
  // group,
  // the entry will be created once more
  group_tasks_not_done_.erase(group);
}

bool BatchedThreadedNnet3CudaPipeline::IsGroupCompleted(
    const std::string &group) {
  std::unique_lock<std::mutex> lk(group_tasks_mutex_);
  return (group_tasks_not_done_[group] == 0);  // will unlock in destructor
}

std::string BatchedThreadedNnet3CudaPipeline::WaitForAnyGroup() {
  std::unique_lock<std::mutex> lk(group_tasks_mutex_);
  // Waiting for any group to be done.
  const string *group_done;
  auto predicate = [this, &group_done] {
    for (auto it : group_tasks_not_done_) {
      if (it.second == 0) {
        group_done = &it.first;
        return true;
      }
    }
    return false;
  };
  group_done_cv_.wait(lk, predicate);
  return *group_done;
}

bool BatchedThreadedNnet3CudaPipeline::IsAnyGroupCompleted(std::string *group) {
  std::lock_guard<std::mutex> lk(group_tasks_mutex_);
  for (auto it : group_tasks_not_done_) {
    if (it.second == 0) {
      *group = it.first;
      return true;
    }
  }
  return false;  // will unlock in destructor
}

void BatchedThreadedNnet3CudaPipeline::CloseAllDecodeHandlesForGroup(
    const std::string &group) {
  WaitForGroup(group);
  std::lock_guard<std::mutex> lk1(tasks_lookup_mutex_);
  auto p = tasks_group_lookup_.equal_range(group);
  for (auto it = p.first; it != p.second; ++it) {
    KALDI_ASSERT(it->second->finished==true);
    tasks_lookup_.erase(it->second->key);
  }
  tasks_group_lookup_.erase(p.first, p.second);
  std::lock_guard<std::mutex> lk2(group_tasks_mutex_);
  group_tasks_not_done_.erase(group);
}

void BatchedThreadedNnet3CudaPipeline::CloseAllDecodeHandles() {
  WaitForAllTasks();
  std::lock_guard<std::mutex> lk1(tasks_lookup_mutex_);
  tasks_lookup_.clear();
  tasks_group_lookup_.clear();
  std::lock_guard<std::mutex> lk2(group_tasks_mutex_);
  group_tasks_not_done_.clear();
}

int32 BatchedThreadedNnet3CudaPipeline::GetNumberOfTasksPending() {
  int size;
  {
    std::lock_guard<std::mutex> lk(group_tasks_mutex_);
    size = all_group_tasks_not_done_;
  }
  return size;
}

BatchedThreadedNnet3CudaPipeline::TaskState *
BatchedThreadedNnet3CudaPipeline::AddTask(const std::string &key,
                                          const std::string &group) {
  TaskState *task;
  {
    std::lock_guard<std::mutex> lock(tasks_lookup_mutex_);
    // ensure key is unique
    KALDI_ASSERT(tasks_lookup_.end() == tasks_lookup_.find(key));

    // Create a new task in lookup map
    task = &tasks_lookup_[key];
    tasks_group_lookup_.insert({group, task});
  }
  task->group = group;

  // Add the task to its group
  {
    std::lock_guard<std::mutex> lk(group_tasks_mutex_);
    ++all_group_tasks_not_done_;
    ++group_tasks_not_done_[task->group];
  }
  return task;
}

// Adds a decoding task to the decoder
void BatchedThreadedNnet3CudaPipeline::OpenDecodeHandle(
    const std::string &key, const WaveData &wave_data, const std::string &group,
    const std::function<void(CompactLattice &clat)> &callback) {
  TaskState *task = AddTask(key, group);
  task->callback = std::move(callback);
  task->Init(key, wave_data);

  if (config_.gpu_feature_extract) {
    // Feature extraction done on device
    AddTaskToPendingTaskQueue(task);
  } else {
    // Feature extraction done on host thread
    work_pool_->enqueue(THREAD_POOL_LOW_PRIORITY,
                        &BatchedThreadedNnet3CudaPipeline::ComputeOneFeatureCPU,
                        this, task);
  }
}

void BatchedThreadedNnet3CudaPipeline::OpenDecodeHandle(
    const std::string &key, const VectorBase<BaseFloat> &wave_data,
    float sample_rate, const std::string &group,
    const std::function<void(CompactLattice &clat)> &callback) {
  TaskState *task = AddTask(key, group);
  task->Init(key, wave_data, sample_rate);
  task->callback = std::move(callback);

  if (config_.gpu_feature_extract) {
    // Feature extraction done on device
    AddTaskToPendingTaskQueue(task);
  } else {
    // Feature extraction done on host thread
    work_pool_->enqueue(THREAD_POOL_LOW_PRIORITY,
                        &BatchedThreadedNnet3CudaPipeline::ComputeOneFeatureCPU,
                        this, task);
  }
}

bool BatchedThreadedNnet3CudaPipeline::GetRawLattice(const std::string &key,
                                                     Lattice *lat) {
  nvtxRangePushA("GetRawLattice");
  TaskState *task;
  {
    std::lock_guard<std::mutex> lock(tasks_lookup_mutex_);
    auto it = tasks_lookup_.find(key);
    KALDI_ASSERT(it != tasks_lookup_.end());
    task = &it->second;
  }

  // wait for task to finish.  This should happens automatically without
  // intervention from the master thread.
  while (task->finished == false) kaldi::Sleep(SLEEP_BACKOFF_S);

  // GetRawLattice on a determinized lattice is not supported (Per email from
  // DanP)
  KALDI_ASSERT(task->determinized == false);

  if (task->error) {
    nvtxRangePop();
    return false;
  }
  // Store off the lattice
  *lat = task->lat;
  nvtxRangePop();
  return true;
}

bool BatchedThreadedNnet3CudaPipeline::GetLattice(const std::string &key,
                                                  CompactLattice *clat) {
  nvtxRangePushA("GetLattice");
  TaskState *task;
  {
    std::lock_guard<std::mutex> lock(tasks_lookup_mutex_);

    auto it = tasks_lookup_.find(key);
    KALDI_ASSERT(it != tasks_lookup_.end());
    task = &it->second;
  }
  // wait for task to finish.  This should happens automatically without
  // intervention from the master thread.
  while (!task->finished) kaldi::Sleep(SLEEP_BACKOFF_S);

  if (task->error) {
    nvtxRangePop();
    return false;
  }

  // if user has not requested a determinized lattice from the decoder then we
  // must
  // determinize it here since it was done done already.
  if (!config_.determinize_lattice && !task->determinized) {
    // Determinzation was not done by worker threads so do it here
    DeterminizeOneLattice(task);
  }

  *clat = task->dlat;  // grab compact lattice
  nvtxRangePop();
  return true;
}

// Adds task to the PendingTaskQueue
void BatchedThreadedNnet3CudaPipeline::AddTaskToPendingTaskQueue(
    TaskState *task) {
  std::lock_guard<std::mutex> lk(tasks_add_mutex_);
  if (NumPendingTasks() == config_.max_pending_tasks) {
    // task queue is full launch a new thread to add this task and exit to make
    // room for other work
    work_pool_->enqueue(
        THREAD_POOL_LOW_PRIORITY,
        &BatchedThreadedNnet3CudaPipeline::AddTaskToPendingTaskQueue, this,
        task);
  } else {
    // there is room so let's add it
    // insert into pending task queue
    pending_task_queue_[tasks_back_] = task;
    // (int)tasks_back_);
    tasks_back_ = (tasks_back_ + 1) % (config_.max_pending_tasks + 
        config_.pending_queue_padding);
    KALDI_ASSERT(NumPendingTasks() <= config_.max_pending_tasks);
  }
}

// Attempts to fill the batch from the task queue.  May not fully fill the
// batch.
void BatchedThreadedNnet3CudaPipeline::AquireAdditionalTasks(
    CudaDecoder &cuda_decoder, ChannelState &channel_state,
    std::vector<TaskState *> &tasks) {
  std::vector<ChannelId> &channels = channel_state.channels;
  std::vector<ChannelId> &free_channels = channel_state.free_channels;

  int tasksRequested =
      std::min(free_channels.size(), config_.max_batch_size - channels.size());
  int tasksAssigned = 0;
  int firstTask = channels.size();

  {
    // lock required because front might change from other
    // workers
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    {
      // compute number of tasks to grab
      int tasksAvailable = NumPendingTasks();
      tasksAssigned = std::min(tasksAvailable, tasksRequested);

      // grab tasks
      for (int i = 0; i < tasksAssigned; i++) {
        // pending_task_queue_[tasks_front_]);
        KALDI_ASSERT(NumPendingTasks() > 0);
        tasks.push_back(pending_task_queue_[tasks_front_]);
        tasks_front_ = (tasks_front_ + 1) % (config_.max_pending_tasks
            + config_.pending_queue_padding);
      }
    }
  }

  if (tasksAssigned > 0) {
    // for each assigned tasks we have to do a little bookkeeping

    // list of channels that need initialization
    std::vector<ChannelId> init_channels(tasksAssigned);

    for (int i = 0; i < tasksAssigned; i++) {
      // assign a free channel
      ChannelId channel;
      {
        std::lock_guard<std::mutex> lk(channel_state.free_channels_mutex);
        KALDI_ASSERT(free_channels.size() >
                     0);  // it should always be true (cf std::min above)
        channel = free_channels.back();
        free_channels.pop_back();
      }
      // assign channel to task
      tasks[i + firstTask]->ichannel = channel;
      // add channel to processing list
      channels.push_back(channel);
      // add new channel to initialization list
      init_channels[i] = channel;
    }

    // Setup cuda_decoder channels
    cuda_decoder.InitDecoding(init_channels);
  }
}

// Computes NNET3 across the tasks[first,tasks.size())
void BatchedThreadedNnet3CudaPipeline::ComputeBatchNnet(
    nnet3::NnetBatchComputer &computer, int32 first,
    std::vector<TaskState *> &tasks) {
  nvtxRangePushA("ComputeBatchNnet");

  bool output_to_cpu = false;
  int32 online_ivector_period = 0;
  int max_pending_minibatches =
      0;  // zero means unlimited.  This API call should not block then.

  // list of nnet tasks for each batch
  std::vector<std::vector<nnet3::NnetInferenceTask>> nnet_tasks(tasks.size());

  // for all new batches enqueue up nnet work.
  for (int i = first; i < tasks.size(); i++) {
    TaskState &task = *tasks[i];
    std::unique_ptr<TaskData> &task_data = task.task_data;
    std::vector<nnet3::NnetInferenceTask> &ntasks = nnet_tasks[i];

    if (config_.gpu_feature_extract) {
      CuVector<BaseFloat> &ivector_features = task_data->ivector_features;
      CuMatrix<BaseFloat> &input_features = task_data->input_features;

      CuVector<BaseFloat> *ifeat = NULL;
      if (ivector_features.Dim() > 0) {
        ifeat = &ivector_features;
      }
      // create task list
      computer.SplitUtteranceIntoTasks(output_to_cpu, input_features, ifeat,
                                       NULL, online_ivector_period, &ntasks);
    } else {
      Vector<BaseFloat> &ivector_features = task_data->ivector_features_cpu;
      Matrix<BaseFloat> &input_features = task_data->input_features_cpu;

      Vector<BaseFloat> *ifeat = NULL;
      if (ivector_features.Dim() > 0) {
        ifeat = &ivector_features;
      }
      // create task list
      computer.SplitUtteranceIntoTasks(output_to_cpu, input_features, ifeat,
                                       NULL, online_ivector_period, &ntasks);
    }

    // Add tasks to computer
    for (size_t j = 0; j < ntasks.size(); j++) {
      computer.AcceptTask(&ntasks[j], max_pending_minibatches);
    }
  }

  // process all minibatches, we allow partial minibatches but this should only
  // occur on the last iteration
  bool allow_partial_minibatch = true;
  while (computer.Compute(allow_partial_minibatch))
    ;

  // Extract Posteriors
  for (int i = first; i < tasks.size(); i++) {
    TaskState &task = *tasks[i];
    std::unique_ptr<TaskData> &task_data = task.task_data;
    CuMatrix<BaseFloat> &posteriors = task_data->posteriors;
    MergeTaskOutput(nnet_tasks[i], &posteriors);

    // nnet output is no longer necessary as we have copied the output out
    nnet_tasks[i].resize(0);

    // features are no longer needed so free memory here
    task_data->ivector_features.Resize(0);
    task_data->input_features.Resize(0,0);
  }

  nvtxRangePop();
}

// Computes Features for a single decode instance.
void BatchedThreadedNnet3CudaPipeline::ComputeOneFeatureCPU(TaskState *task_) {
  nvtxRangePushA("ComputeOneFeatureCPU");
  TaskState &task = *task_;
  std::unique_ptr<TaskData> &task_data = task.task_data;
  Vector<BaseFloat> &ivector_features = task_data->ivector_features_cpu;
  Matrix<BaseFloat> &input_features = task_data->input_features_cpu;

  // create decoding state
  OnlineNnet2FeaturePipeline feature(*feature_info_);

  // Accept waveforms
  feature.AcceptWaveform(task_data->sample_frequency,
                         SubVector<BaseFloat>(*task_data->wave_samples, 0,
                                              task_data->wave_samples->Dim()));
  feature.InputFinished();
  // All frames should be ready here
  int32 numFrames = feature.NumFramesReady();
  // If we don't have anything to do, we must return now
  if (numFrames == 0) {
    task_->finished = true;
    return;
  }
  int32 input_dim = feature.InputFeature()->Dim();

  std::vector<int> frames(numFrames);
  // create list of frames
  for (int j = 0; j < numFrames; j++) frames[j] = j;

  // Copy Features
  input_features.Resize(numFrames, input_dim);
  feature.InputFeature()->GetFrames(frames, &input_features);

  // Ivectors are optional, if they were not provided skip this step
  if (feature.IvectorFeature() != NULL) {
    int32 ivector_dim = feature.IvectorFeature()->Dim();
    ivector_features.Resize(ivector_dim);

    // Copy Features
    feature.IvectorFeature()->GetFrame(numFrames - 1, &ivector_features);
  }

  AddTaskToPendingTaskQueue(task_);

  nvtxRangePop();
}

// Computes features across the tasks[first,tasks.size()
void BatchedThreadedNnet3CudaPipeline::ComputeBatchFeatures(
    int32 first, std::vector<TaskState *> &tasks,
    OnlineCudaFeaturePipeline &feature_pipeline) {
  KALDI_ASSERT(config_.gpu_feature_extract == true);
  nvtxRangePushA("CopyBatchWaves");
  // below we will pack waves into a single buffer for efficient transfer across
  // device

  // first count the total number of elements and create a single large vector
  int count = 0;
  for (int i = first; i < tasks.size(); i++) {
    count += tasks[i]->task_data->wave_samples->Dim();
  }

  // creating a thread local vector of pinned memory.
  // wave data will be stagged through this memory to get
  // more efficient non-blocking transfers to the device.
  thread_local Vector<BaseFloat> pinned_vector;

  if (pinned_vector.Dim() < count) {
    // WAR:  Not pinning memory because it seems to impact correctness
    // we are continuing to look into a fix but want to commit this workaround
    // as a temporary measure.
    if (pinned_vector.Dim() != 0) {
      cudaHostUnregister(pinned_vector.Data());
    }

    // allocated array 2x size
    pinned_vector.Resize(count * 2, kUndefined);
    cudaHostRegister(pinned_vector.Data(),
        pinned_vector.Dim() * sizeof(BaseFloat), 0);
  }

  // We will launch a thread for each task in order to get better host memory
  // bandwidth
  std::vector<std::future<void>> futures;  // for syncing

  // vector copy function for threading below.
  auto copy_vec = [](SubVector<BaseFloat> dst, const SubVector<BaseFloat> src) {
    nvtxRangePushA("CopyVec");
    dst.CopyFromVec(src);
    nvtxRangePop();
  };

  // next launch threads to copy all waves for each task in parallel
  count = 0;
  for (int i = first; i < tasks.size(); i++) {
    std::unique_ptr<TaskData> &task_data = tasks[i]->task_data;
    SubVector<BaseFloat> wave(pinned_vector, count,
                              task_data->wave_samples->Dim());
    count += task_data->wave_samples->Dim();
    futures.push_back(
        work_pool_->enqueue(copy_vec, wave, *(task_data->wave_samples)));
  }

  // wait for waves to be copied into place
  for (int i = 0; i < futures.size(); i++) {
    futures[i].get();
  }

  CuVector<BaseFloat> cu_waves(count, kUndefined);
  // copy memory down asynchronously.  Vector copy functions are synchronous so
  // we do it manually.
  // It is important for this to happen asynchrously to help hide launch latency
  // of smaller kernels
  // that come in the future.
  cudaMemcpyAsync(cu_waves.Data(), pinned_vector.Data(),
                  cu_waves.Dim() * sizeof(BaseFloat), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);
  nvtxRangePop();

  nvtxRangePushA("ComputeBatchFeatures");
  // extract features for each wave
  count = 0;
  for (int i = first; i < tasks.size(); i++) {
    TaskState &task = *tasks[i];
    std::unique_ptr<TaskData> &task_data = task.task_data;

    CuSubVector<BaseFloat> cu_wave(cu_waves, count,
                                   task_data->wave_samples->Dim());
    count += task_data->wave_samples->Dim();
    feature_pipeline.ComputeFeatures(cu_wave, task_data->sample_frequency,
                                     &task_data->input_features,
                                     &task_data->ivector_features);

    int32 numFrames = task_data->input_features.NumRows();

    if (numFrames == 0) {
      // Make this a warning for now.  Need to check how this is handled
      KALDI_WARN << "Warning empty audio file";
    }
  }
  nvtxRangePop();
}

// Allocates decodables for tasks in the range of tasks[first,tasks.size())
void BatchedThreadedNnet3CudaPipeline::AllocateDecodables(
    int32 first, std::vector<TaskState *> &tasks,
    std::vector<CudaDecodableInterface *> &decodables) {
  // Create mapped decodable here
  for (int i = first; i < tasks.size(); i++) {
    std::unique_ptr<TaskData> &task_data = tasks[i]->task_data;
    CuMatrix<BaseFloat> &posteriors = task_data->posteriors;
    decodables.push_back(
        new DecodableCuMatrixMapped(*trans_model_, posteriors, 0));
  }
}

// Removes all completed channels from the channel list.
// Also enqueues up work for post processing
void BatchedThreadedNnet3CudaPipeline::RemoveCompletedChannels(
    CudaDecoder &cuda_decoder, ChannelState &channel_state,
    std::vector<CudaDecodableInterface *> &decodables,
    std::vector<TaskState *> &tasks) {
  std::vector<ChannelId> &channels = channel_state.channels;
  std::vector<ChannelId> &completed_channels = channel_state.completed_channels;

  // Here we will reorder arrays to put finished decodes at the end
  int cur = 0;  // points to the current unchecked decode
  int back = tasks.size() - completed_channels.size() -
             1;  // points to the last unchecked decode

  // for each active channel
  // scan channels to find finished decodes
  // move finished decodes to the end
  for (int i = 0; i < channels.size(); i++) {
    ChannelId channel = channels[cur];
    int numDecoded = cuda_decoder.NumFramesDecoded(channel);
    int toDecode = decodables[cur]->NumFramesReady();

    if (toDecode == numDecoded) {  // if current task is completed
      // add channel to free and completed queues
      completed_channels.push_back(channel);

      // this was assigned earlier just making sure it is still consistent
      KALDI_ASSERT(tasks[cur]->ichannel == channel);

      // Rearrange queues,
      // move this element to end and end to this spot
      std::swap(tasks[cur], tasks[back]);
      std::swap(channels[cur], channels[back]);
      std::swap(decodables[cur], decodables[back]);

      // back is a completed decode so decrement it
      back--;
    } else {
      // not completed move to next task
      cur++;
    }  // end if completed[cur]
  }    // end for loop

  // removing finished channels from list
  channels.resize(cur);
}

// Post decode some channels will be complete
// For those channels we need to
//  free up the channel
//  get and determinize the lattice
//
void BatchedThreadedNnet3CudaPipeline::PostDecodeProcessing(
    CudaDecoder &cuda_decoder, ChannelState &channel_state,
    std::vector<CudaDecodableInterface *> &decodables,
    std::vector<TaskState *> &tasks) {
  std::vector<ChannelId> &channels = channel_state.channels;
  std::vector<ChannelId> &completed_channels = channel_state.completed_channels;

  // consistency check
  KALDI_ASSERT(tasks.size() == channels.size() + completed_channels.size());

  // Prepare data for GetRawLattice
  cuda_decoder.PrepareForGetRawLattice(completed_channels, true);
  // clean up datastructures for completed tasks
  for (int i = channels.size(); i < tasks.size(); i++) {
    tasks[i]->task_data->posteriors.Resize(0,0);
    delete decodables[i];
  }

  // Calling GetRawLattice + Determinize (optional) on a CPU worker thread
  for (int i = channels.size(); i < tasks.size(); i++) {
    // checking that this channel is actually in the completed channels list
    // order is reversed because we used push_back into completed_channel list
    KALDI_ASSERT(tasks[i]->ichannel ==
                 completed_channels[channels.size() +
                                    completed_channels.size() - i - 1]);
    // enqueue task completion on a worker thread.  We do not need to wait 
    // for sychronization on this thread as the parameters passed to this
    // thread are persistent and that thread will return resources to the
    // system when they free up.
    work_pool_->enqueue(THREAD_POOL_NORMAL_PRIORITY,
                            &BatchedThreadedNnet3CudaPipeline::CompleteTask,
                            this, &cuda_decoder, &channel_state, tasks[i]);
  }

  tasks.resize(channels.size());
  decodables.resize(channels.size());
  completed_channels.resize(0);
}

void BatchedThreadedNnet3CudaPipeline::CompleteTask(CudaDecoder *cuda_decoder,
                                                    ChannelState *channel_state,
                                                    TaskState *task) {
  // Calling GetRawLattice for that channel. PrepareForGetRawLattice was already
  // called
  cuda_decoder->ConcurrentGetRawLatticeSingleChannel(task->ichannel,
                                                     &task->lat);
  // We are done using that channel. Putting it back into the free channels
  {
    std::lock_guard<std::mutex> lk(channel_state->free_channels_mutex);
    channel_state->free_channels.push_back(task->ichannel);
  }

  // If necessary, determinize the lattice
  if (config_.determinize_lattice) DeterminizeOneLattice(task);

  if (!config_.determinize_lattice) {
    ConvertLattice(task->lat, &task->dlat);
  }

  if (task->callback)  // if callable
    task->callback(task->dlat);

  task->finished = true;

  {
    std::lock_guard<std::mutex> lk(group_tasks_mutex_);
    --all_group_tasks_not_done_;
    int32 left_in_group = --group_tasks_not_done_[task->group];
    //    std::cout << "left in group " << task->group << " " << left_in_group
    //    << std::endl;
    if (left_in_group == 0) group_done_cv_.notify_all();
  }
}

void BatchedThreadedNnet3CudaPipeline::DeterminizeOneLattice(TaskState *task) {
  nvtxRangePushA("DeterminizeOneLattice");
  // Note this destroys the original raw lattice
  DeterminizeLatticePhonePrunedWrapper(*trans_model_, &task->lat,
                                       config_.decoder_opts.lattice_beam,
                                       &(task->dlat), config_.det_opts);
  task->determinized = true;
  nvtxRangePop();
}

void BatchedThreadedNnet3CudaPipeline::ExecuteWorker(int threadId) {
  // Initialize this threads device
  CuDevice::Instantiate();

  KALDI_LOG << "CudaDecoder batch_size=" << config_.max_batch_size
            << " num_channels=" << config_.num_channels;
  // Data structures that are reusable across decodes but unique to each thread
  CudaDecoder cuda_decoder(cuda_fst_, config_.decoder_opts,
                           config_.max_batch_size, config_.num_channels);
  if (config_.num_decoder_copy_threads > 0)
    cuda_decoder.SetThreadPoolAndStartCPUWorkers(
        work_pool_, config_.num_decoder_copy_threads);
  nnet3::NnetBatchComputer computer(config_.compute_opts, am_nnet_->GetNnet(),
                                    am_nnet_->Priors());

  OnlineCudaFeaturePipeline feature_pipeline(config_.feature_opts);

  ChannelState channel_state;

  std::vector<TaskState *> tasks;  // The state for each decode
  std::vector<CudaDecodableInterface *> decodables;

  // Initialize reuseable data structures
  {
    channel_state.channels.reserve(config_.max_batch_size);
    channel_state.completed_channels.reserve(config_.max_batch_size);
    tasks.reserve(config_.max_batch_size);
    decodables.reserve(config_.max_batch_size);
    {
      std::lock_guard<std::mutex> lk(channel_state.free_channels_mutex);
      channel_state.free_channels.reserve(config_.num_channels);
      // add all channels to free channel list
      for (int i = 0; i < config_.num_channels; i++) {
        channel_state.free_channels.push_back(i);
      }
    }
  }

  numStarted_++;  // Tell master I have started

  // main control loop.  At each iteration a thread will see if it has been
  // asked to shut
  // down.  If it has it will exit.  This loop condition will only be processed
  // if all
  // other work assigned to this thread has been processed.
  while (!exit_) {
    // main processing loop.  At each iteration the thread will do the
    // following:
    // 1) Attempt to grab more work.
    // 2) Initialize any new work
    // do
    // 3) Process work in a batch
    // while(free lanes < drain_count)
    // 4) Postprocess any completed work
    do {
      // 1) attempt to fill the batch
      if (tasks_front_ != tasks_back_) {  // if work is available grab more work

        int start = tasks.size();  // Save the current assigned tasks size

        AquireAdditionalTasks(cuda_decoder, channel_state, tasks);
        // New tasks are now in the in tasks[start,tasks.size())
        if (start != tasks.size()) {  // if there are new tasks
          if (config_.gpu_feature_extract)
            ComputeBatchFeatures(start, tasks, feature_pipeline);
          ComputeBatchNnet(computer, start, tasks);
          AllocateDecodables(start, tasks, decodables);
        }
      }  // end if (tasks_front_!=tasks_back_)

      // check if there is no active work on this thread.
      // This can happen if another thread was assigned the work.
      if (tasks.size() == 0) {
        // Thread is spinning waiting for work.  Backoff.
        kaldi::Sleep(SLEEP_BACKOFF_S);
        break;
      }

      // try/catch to catch and report errors inside decoder.
      // errors can be recoverable or non-recoverable
      // unrecoverable errors will assert
      // recoverable errors will cancel the batch (output empty lattice)
      // and print a warning.
      // There should be no errors and this is just a sanity check
      try {
        // This is in a loop in case we want to drain the batch a little.
        // Draining the batch will cause initialization tasks to be batched.
        do {
          // 3) Process outstanding work in a batch
          // Advance decoding on all open channels
          cuda_decoder.AdvanceDecoding(channel_state.channels, decodables);

          // Adjust channel state for all completed decodes
          RemoveCompletedChannels(cuda_decoder, channel_state, decodables,
                                  tasks);
          // do loop repeates until we meet drain size or run out of work
        } while (config_.max_batch_size - channel_state.channels.size() <
                     config_.batch_drain_size &&
                 channel_state.channels.size() > 0);
        // 4) Post process work.  This reorders completed work to the end,
        // copies results outs, and cleans up data structures
        PostDecodeProcessing(cuda_decoder, channel_state, decodables, tasks);

      } catch (CudaDecoderException e) {
        // Code to catch errors.  Most errors are unrecoverable but a user can
        // mark them
        // recoverable which will cancel the entire batch but keep processing.
        if (!e.recoverable) {
          bool UNRECOVERABLE_EXCEPTION = false;
          KALDI_LOG << "Error unrecoverable cuda decoder error '" << e.what()
                    << "'\n";
          KALDI_ASSERT(UNRECOVERABLE_EXCEPTION);
        } else {
          KALDI_LOG << "Error recoverable cuda decoder error '" << e.what()
                    << "'\n";
          KALDI_LOG << "    Aborting batch for recovery.  Canceling the "
                       "following decodes:\n";
          // Cancel all outstanding tasks
          for (int i = 0; i < tasks.size(); i++) {
            // move all channels to free channel queue
            ChannelId channel = channel_state.channels[i];
            {
              std::lock_guard<std::mutex> lk(channel_state.free_channels_mutex);
              channel_state.free_channels.push_back(channel);
            }
            TaskState &task = *(tasks[i]);
            KALDI_LOG << "      Canceled: " << task.key << "\n";

            // set error flag
            task.error = true;
            task.error_string = e.what();

            // cleanup memory
            delete decodables[i];

            // notifiy master decode is finished
            task.finished = true;
          }
          tasks.resize(0);
          channel_state.channels.resize(0);
          decodables.resize(0);
        }
      }
    } while (tasks.size() > 0);  // more work don't check exit condition
  }                              // end while(!exit_)
}  // end ExecuteWorker

}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // HAVE_CUDA == 1
