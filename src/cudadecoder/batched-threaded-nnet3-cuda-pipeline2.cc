// cudadecoder/batched-threaded-nnet3-cuda-pipeline2.cc
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

#if !HAVE_CUDA
#error CUDA support is required to compile this library.
#endif

#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline2.h"

#include <atomic>

#include <nvToolsExt.h>

namespace kaldi {
namespace cuda_decoder {

const float kSleepForTaskComplete = 10e-3;
const float kSleepForNewTask = 100e-6;

BatchedThreadedNnet3CudaPipeline2::BatchedThreadedNnet3CudaPipeline2(
    const BatchedThreadedNnet3CudaPipeline2Config &config,
    const fst::Fst<fst::StdArc> &decode_fst, const nnet3::AmNnetSimple &am_nnet,
    const TransitionModel &trans_model)
    : config_(config),
      cuda_online_pipeline_(config.cuda_online_pipeline_opts, decode_fst,
                            am_nnet, trans_model),
      use_online_features_(config_.use_online_features),
      corr_id_cnt_(0),
      max_batch_size_(config_.cuda_online_pipeline_opts.max_batch_size),
      threads_running_(true),
      online_pipeline_control_thread_(
          &BatchedThreadedNnet3CudaPipeline2::ComputeTasks, this),
      n_tasks_not_done_(0),
      lattice_postprocessor_(NULL) {
  config_.Check();  // Verifying config
  KALDI_ASSERT(
      "CPU feature extraction is only available when "
      "use-online-features is set" &&
      (config_.cuda_online_pipeline_opts.use_gpu_feature_extraction ||
       config_.use_online_features));
  batch_corr_ids_.reserve(max_batch_size_);
  batch_wave_samples_.reserve(max_batch_size_);
  batch_is_last_chunk_.reserve(max_batch_size_);
  batch_is_first_chunk_.reserve(max_batch_size_);
  tasks_last_chunk_.reserve(max_batch_size_);
  if (use_online_features_) {
    n_input_per_chunk_ = cuda_online_pipeline_.GetNSampsPerChunk();
  } else {
    n_input_per_chunk_ = cuda_online_pipeline_.GetNInputFramesPerChunk();
    cuda_features_.reset(new OnlineCudaFeaturePipeline(
        config_.cuda_online_pipeline_opts.feature_opts));
    wave_buffer_.reset(new HostDeviceVector());
    next_wave_buffer_.reset(new HostDeviceVector());
  }

  model_freq_ = cuda_online_pipeline_.GetModelFrequency();
  segment_length_nsamples_ = config_.seg_opts.segment_length_s * model_freq_;
  segment_shift_nsamples_ =
      (config_.seg_opts.segment_length_s - config_.seg_opts.segment_overlap_s) *
      model_freq_;
  min_segment_length_nsamples_ =
      config_.seg_opts.min_segment_length_s * model_freq_;
}

BatchedThreadedNnet3CudaPipeline2::~BatchedThreadedNnet3CudaPipeline2() {
  threads_running_ = false;
  online_pipeline_control_thread_.join();
}

void BatchedThreadedNnet3CudaPipeline2::BuildBatchFromCurrentTasks() {
  batch_corr_ids_.clear();
  batch_is_last_chunk_.clear();
  batch_is_first_chunk_.clear();
  if (use_online_features_) {
    batch_wave_samples_.clear();
  } else {
    batch_features_.clear();
    batch_ivectors_.clear();
    batch_n_input_frames_valid_.clear();
  }
  for (size_t task_id = 0; task_id < current_tasks_.size();) {
    UtteranceTask &task = current_tasks_[task_id];
    int32 total_n_input;
    if (use_online_features_) {
      KALDI_ASSERT(task.h_wave);
      SubVector<BaseFloat> &h_wave = *task.h_wave;
      total_n_input = h_wave.Dim();
    } else {
      total_n_input = task.d_features->NumRows();
    }

    int32 samp_offset = task.samp_offset;
    int32 samp_remaining = total_n_input - samp_offset;
    int32 num_samp = std::min(n_input_per_chunk_, samp_remaining);
    KALDI_ASSERT(num_samp > 0);
    bool is_last_chunk = (samp_remaining == num_samp);
    bool is_first_chunk = (task.samp_offset == 0);
    CorrelationID corr_id = task.corr_id;
    task.samp_offset += num_samp;

    batch_corr_ids_.push_back(corr_id);
    batch_is_last_chunk_.push_back(is_last_chunk);
    batch_is_first_chunk_.push_back(is_first_chunk);

    if (use_online_features_) {
      SubVector<BaseFloat> &h_wave = *task.h_wave;
      SubVector<BaseFloat> wave_part(h_wave, samp_offset, num_samp);
      batch_wave_samples_.push_back(wave_part);
    } else {
      batch_features_.push_back(task.d_features->Data() +
                                samp_offset * task.d_features->Stride());
      if (task_id == 0)
        batch_features_frame_stride_ = task.d_features->Stride();
      else
        KALDI_ASSERT(batch_features_frame_stride_ == task.d_features->Stride());
      batch_ivectors_.push_back(task.d_ivectors->Data());
      batch_n_input_frames_valid_.push_back(num_samp);
    }

    // If last chunk, moving the task to tasks_last_chunk_
    if (is_last_chunk) {
      tasks_last_chunk_.push_back(std::move(task));
      size_t last_task_id = current_tasks_.size() - 1;
      current_tasks_[task_id] = std::move(current_tasks_[last_task_id]);
      current_tasks_.pop_back();
    } else {
      // If it was the last chunk, we replaced the current
      // task with another one we must process that task_id
      // again (because it is now another task) If it was not
      // the last chunk, then we must take care of the next
      // task_id
      ++task_id;
    }
  }
}

void BatchedThreadedNnet3CudaPipeline2::WaitForAllTasks() {
  while (n_tasks_not_done_.load() != 0) {
    Sleep(kSleepForTaskComplete);
  }
}

void BatchedThreadedNnet3CudaPipeline2::CreateTaskGroup(
    const std::string &group) {
  std::lock_guard<std::mutex> lk(n_group_tasks_not_done_m_);
  bool inserted;
  std::unique_ptr<std::atomic<int>> group_cnt;
  group_cnt.reset(new std::atomic<int>(0));
  std::tie(std::ignore, inserted) =
      n_group_tasks_not_done_.emplace(group, std::move(group_cnt));
  KALDI_ASSERT("Group is already in use" && inserted);
}

void BatchedThreadedNnet3CudaPipeline2::DestroyTaskGroup(
    const std::string &group) {
  std::lock_guard<std::mutex> lk(n_group_tasks_not_done_m_);
  int nerased = n_group_tasks_not_done_.erase(group);
  KALDI_ASSERT("Group does not exist" && (nerased == 1));
}

void BatchedThreadedNnet3CudaPipeline2::WaitForGroup(const std::string &group) {
  std::atomic<int> *n_not_done;
  {
    std::lock_guard<std::mutex> lk(n_group_tasks_not_done_m_);
    auto it = n_group_tasks_not_done_.find(group);
    KALDI_ASSERT("Group does not exist. Call CreateTaskGroup() first" &&
                 (it != n_group_tasks_not_done_.end()));
    n_not_done = it->second.get();
  }

  while (n_not_done->load(std::memory_order_consume) != 0)
    Sleep(kSleepForTaskComplete);
}

void BatchedThreadedNnet3CudaPipeline2::CreateTask(
    const std::string &key, const std::shared_ptr<WaveData> &wave_data,
    std::unique_ptr<SubVector<BaseFloat>> &&h_wave,
    const std::function<void(CompactLattice &)> *callback,
    const SegmentedResultsCallback *segmented_callback,
    const std::string &group) {
  if (wave_data) {
    KALDI_ASSERT(
        "Mismatch in model and utt frequency" &&
        (wave_data->SampFreq() == cuda_online_pipeline_.GetModelFrequency()));
  }
  UtteranceTask task;

  // If that task depends on a wave_data, then "list"
  // that task as one of the owners of wave_data (shared_ptr)
  if (wave_data) task.wave_data = wave_data;

  if (h_wave) {
    task.h_wave = std::move(h_wave);
  } else {
    KALDI_ASSERT(wave_data);
    task.h_wave.reset(new SubVector<BaseFloat>(wave_data->Data(), 0));
  }
  if (task.h_wave->Dim() == 0) return;  // nothing to do

  task.key = key;
  task.samp_offset = 0;
  task.corr_id = corr_id_cnt_.fetch_add(
      1);  // at 5000 files/s, expected to overflow in ~116 million years

  if (callback) task.callback = *callback;
  if (segmented_callback) task.segmented_callback = *segmented_callback;

  n_tasks_not_done_.fetch_add(1);

  if (!group.empty()) {
    // Need to add it to group
    std::lock_guard<std::mutex> lk(n_group_tasks_not_done_m_);
    auto it = n_group_tasks_not_done_.find(group);
    KALDI_ASSERT("Group does not exist. Call CreateTaskGroup() first" &&
                 (it != n_group_tasks_not_done_.end()));
    it->second->fetch_add(1);           // adding current task
    task.group_cnt = it->second.get();  // will be used to --cnt
  } else {
    task.group_cnt = NULL;
  }

  // Adding task to the relevant task queue
  if (use_online_features_) {
    // If we use online ivectors, we can just add it to the
    // outstanding queue. ivectors and mfcc will be computed in the
    // online pipeline
    std::lock_guard<std::mutex> lk(outstanding_utt_m_);
    outstanding_utt_.push(std::move(task));
  } else {
    // Otherwise we first need to compute ivectors and mfcc for the
    // full audio file Adding it to the preprocessing queue
    std::lock_guard<std::mutex> lk(preprocessing_utt_queue_m_);
    preprocessing_utt_queue_.push(std::move(task));
  }
}

void BatchedThreadedNnet3CudaPipeline2::DecodeWithCallback(
    const std::string &key, const std::shared_ptr<WaveData> &wave_data,
    std::unique_ptr<SubVector<BaseFloat>> &&h_wave,
    const std::function<void(CompactLattice &)> &callback,
    const std::string &group) {
  CreateTask(key, wave_data, std::move(h_wave), &callback, nullptr, group);
}

void BatchedThreadedNnet3CudaPipeline2::SegmentedDecodeWithCallback(
    const std::shared_ptr<WaveData> &wave_data,
    const SegmentedResultsCallback &segmented_callback, const int result_type) {
  KALDI_ASSERT(result_type && "You must define at least one result type");
  SubVector<BaseFloat> h_wave(wave_data->Data(), 0);
  int total_nsamples = h_wave.Dim();

  // If utterance is empty, call some empty callback
  if (total_nsamples == 0)
    return CallEmptySegmentedCallback(segmented_callback);

  // Number of segments for an utterance of size segment_length_nsamples_
  int nsegments = NumberOfSegments(total_nsamples, segment_length_nsamples_,
                                   segment_shift_nsamples_);

  // We'll store results from each segment in this array
  auto segmented_results =
      std::make_shared<std::vector<CudaPipelineResult>>(nsegments);

  // That atomic is used to know that all segments result are ready
  auto n_segments_callbacks_not_done_ =
      std::make_shared<std::atomic<std::int32_t>>(nsegments);
  int isegment = 0;

  // Those tasks have no key. Task key is a deprecated feature
  std::string no_key;
  for (int offset = 0; /* break in loop */; offset += segment_shift_nsamples_) {
    int nsamples = std::min(total_nsamples - offset, segment_length_nsamples_);

    if (nsamples < min_segment_length_nsamples_) {
      // This segment is too short, skipping it
      // Adjusting the # of segments
      --nsegments;
      n_segments_callbacks_not_done_->fetch_sub(1, std::memory_order_acq_rel);
    } else {
      std::unique_ptr<SubVector<BaseFloat>> h_wave_segment(
          new SubVector<BaseFloat>(h_wave.Data() + offset, nsamples));
      BaseFloat offset_seconds =
          std::floor(static_cast<BaseFloat>(offset) / model_freq_);

      // Saving this segment offset in result for later use
      (*segmented_results)[isegment].SetTimeOffsetSeconds(offset_seconds);

      // One callback per segment will generate the segment result,
      // add it to the vector of results,
      // and if it is the last segment to complete,
      // call the segmented callback with the vector of results
      LatticeCallback callback = [=](CompactLattice &clat) {
        CudaPipelineResult &result = (*segmented_results)[isegment];

        SetResultUsingLattice(clat, result_type, lattice_postprocessor_,
                              &result);

        int n_not_done = n_segments_callbacks_not_done_->fetch_sub(1);
        if (n_not_done == 1 && segmented_callback) {
          SegmentedLatticeCallbackParams params;
          params.results = std::move(*segmented_results);
          segmented_callback(params);
        }
      };

      // Create task for this segment
      CreateTask(no_key, wave_data, std::move(h_wave_segment), &callback,
                 &segmented_callback);
      ++isegment;
    }

    // If last segment, done
    if ((offset + nsamples) >= total_nsamples) break;
  }

  KALDI_ASSERT(nsegments == isegment);
}

void BatchedThreadedNnet3CudaPipeline2::CallEmptySegmentedCallback(
    const SegmentedResultsCallback &segmented_callback) {
  // Calling the segmented callback with one empty lattice
  SegmentedLatticeCallbackParams params;
  params.results.emplace_back();
  CompactLattice clat;
  params.results.back().SetLatticeResult(std::move(clat));
  segmented_callback(params);
  return;
}

void BatchedThreadedNnet3CudaPipeline2::ComputeOfflineFeatures() {
  bool iterate = true;
  do {
    UtteranceTask task;
    {
      std::lock_guard<std::mutex> lk(preprocessing_utt_queue_m_);
      if (preprocessing_utt_queue_.empty()) {
        iterate = false;
        break;
      }

      task = std::move(preprocessing_utt_queue_.front());
      preprocessing_utt_queue_.pop();
    }
    KALDI_ASSERT(task.h_wave);
    SubVector<BaseFloat> &h_wave = *task.h_wave;
    int32 nsamp = h_wave.Dim();

    cudaEventSynchronize(wave_buffer_->evt);
    if (nsamp > wave_buffer_->size) {
      wave_buffer_->Reallocate(nsamp);
    }

    std::memcpy(wave_buffer_->h_data, h_wave.Data(), nsamp * sizeof(BaseFloat));
    cudaMemcpyAsync(wave_buffer_->d_data, wave_buffer_->h_data,
                    nsamp * sizeof(BaseFloat), cudaMemcpyHostToDevice,
                    cudaStreamPerThread);

    task.d_features.reset(new CuMatrix<BaseFloat>());
    task.d_ivectors.reset(new CuVector<BaseFloat>());

    CuSubVector<BaseFloat> wrapper(wave_buffer_->d_data, nsamp);

    cuda_features_->ComputeFeatures(
        wrapper, cuda_online_pipeline_.GetModelFrequency(),
        task.d_features.get(), task.d_ivectors.get());

    cudaEventRecord(wave_buffer_->evt, cudaStreamPerThread);

    std::swap(wave_buffer_, next_wave_buffer_);

    if (task.wave_data) task.wave_data.reset();  // delete wave samples on host

    {
      std::lock_guard<std::mutex> lk(outstanding_utt_m_);
      outstanding_utt_.push(std::move(task));
      // We dont want to have too many files ready in
      // outstanding_utt_ (using device memory) using
      // max_batch_size_ as an arbitrary (large enough) value
      iterate = (outstanding_utt_.size() < max_batch_size_);
    }
  } while (iterate);
  cudaStreamSynchronize(cudaStreamPerThread);  // to keep CuVector in scope
}

void BatchedThreadedNnet3CudaPipeline2::AcquireTasks() {
  // Trying to get new tasks
  std::unique_lock<std::mutex> lk(outstanding_utt_m_);
  while (current_tasks_.size() < max_batch_size_) {
    // If use_online_features_ is false, we have to fill
    // outstanding_utt_ by computing features
    if (!use_online_features_ && outstanding_utt_.size() == 0) {
      lk.unlock();
      ComputeOfflineFeatures();
      lk.lock();
    }
    // If still empty, break
    if (outstanding_utt_.size() == 0) break;
    UtteranceTask &task = outstanding_utt_.front();
    bool was_created = cuda_online_pipeline_.TryInitCorrID(task.corr_id);
    // No channel was available. Breaking for now
    if (!was_created) break;

    auto &callback = task.callback;
    auto &key = task.key;
    std::atomic<int> *group_cnt = task.group_cnt;
    cuda_online_pipeline_.SetLatticeCallback(
        task.corr_id, [this, callback, key, group_cnt](CompactLattice &clat) {
          if (callback) callback(clat);
          n_tasks_not_done_.fetch_sub(1, std::memory_order_release);
          if (group_cnt) group_cnt->fetch_sub(1, std::memory_order_release);
          // if last segmented callback, segmented callback...
        });
    current_tasks_.push_back(std::move(task));
    outstanding_utt_.pop();
  }
}

void BatchedThreadedNnet3CudaPipeline2::ComputeTasks() {
  while (threads_running_) {
    if (current_tasks_.size() < max_batch_size_) AcquireTasks();
    if (current_tasks_.empty()) {
      // If we still have nothing to do, let's sleep a bit
      Sleep(kSleepForNewTask);
      continue;
    }
    BuildBatchFromCurrentTasks();

    if (use_online_features_)
      cuda_online_pipeline_.DecodeBatch(batch_corr_ids_, batch_wave_samples_,
                                        batch_is_first_chunk_,
                                        batch_is_last_chunk_);
    else
      cuda_online_pipeline_.DecodeBatch(
          batch_corr_ids_, batch_features_, batch_features_frame_stride_,
          batch_n_input_frames_valid_, batch_ivectors_, batch_is_first_chunk_,
          batch_is_last_chunk_);
    // Calling the destructors, freeing memory
    tasks_last_chunk_.clear();
  }
}

void BatchedThreadedNnet3CudaPipeline2::SetLatticePostprocessor(
    const std::shared_ptr<LatticePostprocessor> &lattice_postprocessor) {
  lattice_postprocessor_ = lattice_postprocessor;
  lattice_postprocessor_->SetDecoderFrameShift(
      cuda_online_pipeline_.GetDecoderFrameShiftSeconds());
  lattice_postprocessor_->SetTransitionInformation(
      &cuda_online_pipeline_.GetTransitionModel());
}

}  // namespace cuda_decoder
}  // namespace kaldi
