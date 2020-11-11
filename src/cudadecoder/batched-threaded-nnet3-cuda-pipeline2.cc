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

#if HAVE_CUDA == 1

#define KALDI_CUDA_DECODER_WAIT_FOR_TASKS_US 10000
#define KALDI_CUDA_DECODER_WAIT_FOR_NEW_TASKS_US 100

#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline2.h"
#include <nvToolsExt.h>

namespace kaldi {
namespace cuda_decoder {

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
    usleep(KALDI_CUDA_DECODER_WAIT_FOR_TASKS_US);
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
    usleep(KALDI_CUDA_DECODER_WAIT_FOR_TASKS_US);
}

void BatchedThreadedNnet3CudaPipeline2::DecodeWithCallback(
    const std::string &key, const std::shared_ptr<WaveData> &wave_data,
    std::unique_ptr<SubVector<BaseFloat>> &&h_wave,
    const std::function<void(CompactLattice &)> &callback,
    const std::string &group) {
  if (wave_data) {
    KALDI_ASSERT(
        "Mismatch in model and utt frequency" &&
        (wave_data->SampFreq() == cuda_online_pipeline_.GetModelFrequency()));
  }

  UtteranceTask task;
  if (wave_data) task.wave_data = wave_data;
  if (h_wave) {
    task.h_wave = std::move(h_wave);
  } else {
    KALDI_ASSERT(wave_data);
    task.h_wave.reset(new SubVector<BaseFloat>(wave_data->Data(), 0));
  }

  if (task.h_wave->Dim() == 0) return;  // nothing to do
  n_tasks_not_done_.fetch_add(1);
  task.key = key;
  task.samp_offset = 0;
  task.corr_id = corr_id_cnt_.fetch_add(
      1);  // at 5000 files/s, expected to overflow in ~116 million years
  task.callback = callback;

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
    cudaMemcpyAsync(wave_buffer_->d_data, wave_buffer_->h_data, nsamp * sizeof(BaseFloat), cudaMemcpyHostToDevice, cudaStreamPerThread);

    task.d_features.reset(new CuMatrix<BaseFloat>());
    task.d_ivectors.reset(new CuVector<BaseFloat>());

    CuSubVector<BaseFloat> wrapper(wave_buffer_->d_data, nsamp);

    cuda_features_->ComputeFeatures(wrapper, cuda_online_pipeline_.GetModelFrequency(), task.d_features.get(), task.d_ivectors.get());

    cudaEventRecord(wave_buffer_->evt, cudaStreamPerThread);

    std::swap(wave_buffer_, next_wave_buffer_);

    if (task.wave_data)
	task.wave_data.reset();  // delete wave samples on host

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
      usleep(KALDI_CUDA_DECODER_WAIT_FOR_NEW_TASKS_US);
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

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // HAVE_CUDA
