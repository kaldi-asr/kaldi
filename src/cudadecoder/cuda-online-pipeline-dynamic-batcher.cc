// cudadecoder/cuda-online-pipeline-dynamic-batcher.cc
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
#error CUDA support must be configured to compile this library.
#endif

#include "cudadecoder/cuda-online-pipeline-dynamic-batcher.h"

namespace kaldi {
namespace cuda_decoder {

// Delay between iterations of the batcher thread loop.
const double kDynamicBatcherLoopTick = 100e-6;

CudaOnlinePipelineDynamicBatcher::CudaOnlinePipelineDynamicBatcher(
    CudaOnlinePipelineDynamicBatcherConfig config,
    BatchedThreadedNnet3CudaOnlinePipeline &cuda_pipeline)
    : config_(config),
      run_batcher_thread_(true),
      cuda_pipeline_(cuda_pipeline),
      n_chunks_not_done_(0),
      max_batch_size_(cuda_pipeline.GetConfig().max_batch_size),
      num_channels_(cuda_pipeline.GetConfig().num_channels) {
  curr_batch_.reset(
      new Batch(max_batch_size_, cuda_pipeline.GetNSampsPerChunk()));

  next_batch_.reset(
      new Batch(max_batch_size_, cuda_pipeline.GetNSampsPerChunk()));

  batcher_thread_.reset(new std::thread(
      &CudaOnlinePipelineDynamicBatcher::BatcherThreadLoop, this));
}

CudaOnlinePipelineDynamicBatcher::~CudaOnlinePipelineDynamicBatcher() {
  run_batcher_thread_ = false;
  batcher_thread_->join();
}

void CudaOnlinePipelineDynamicBatcher::Push(
    CorrelationID corr_id, bool is_first_chunk, bool is_last_chunk,
    const SubVector<BaseFloat> &wave_samples_subv) {
  std::lock_guard<std::mutex> lk(next_batch_and_backlog_m_);

  // Try to add the chunk is the batch right away
  if (!TryAddChunkToNextBatchDeepCopy(corr_id, is_first_chunk, is_last_chunk,
                                      wave_samples_subv)) {
    // If we cannot add the chunk to the next batch, put it in the backlog.
    Vector<BaseFloat> wave_samples(wave_samples_subv);  // Make a deep copy.
    backlog_.push_back(
        {corr_id, is_first_chunk, is_last_chunk, std::move(wave_samples)});
  }
  n_chunks_not_done_.fetch_add(1, std::memory_order_release);
}

// CONTRACT: The executing thread must own the next_batch_and_backlog_m_ mutex.
bool CudaOnlinePipelineDynamicBatcher::TryAddChunkToNextBatchDeepCopy(
    CorrelationID corr_id, bool is_first_chunk, bool is_last_chunk,
    const VectorBase<BaseFloat> &wave_samples) {

  if (next_batch_->Size() == max_batch_size_) {
    return false;  // The batch is full.
  }

  bool corr_id_not_in_batch;
  decltype(next_batch_->is_corr_id_in_batch.end()) is_corr_id_in_batch_it;
  std::tie(is_corr_id_in_batch_it, corr_id_not_in_batch) =
      next_batch_->is_corr_id_in_batch.insert(corr_id);
  if (!corr_id_not_in_batch) {
    // We already have that corr_id in batch. Not adding it for now.
    return false;
  }

  if (is_first_chunk) {
    // Initialize this stream
    if (!cuda_pipeline_.TryInitCorrID(corr_id, 0)) {
      KALDI_LOG << "All decoding channels are in use. Consider "
                   "increasing --num-channels";
      next_batch_->is_corr_id_in_batch.erase(
          is_corr_id_in_batch_it);  // This elt won't be in the batch.
      return false;
    }
  }

  // If everything looks good, skip the backlog_ and add a deep copy of
  // wave_samples directly to the next batch.
  next_batch_->PushBack(corr_id, is_first_chunk, is_last_chunk, wave_samples);

  return true;
}

// CONTRACT: The executing thread must own the next_batch_and_backlog_m_ mutex.
void CudaOnlinePipelineDynamicBatcher::FillNextBatchWithBacklog() {
  for (auto it = backlog_.begin(); it != backlog_.end(); /* nothing */) {
    if (next_batch_->Size() == max_batch_size_) break;

    if (TryAddChunkToNextBatchDeepCopy(it->corr_id, it->is_first_chunk,
                                       it->is_last_chunk, it->wave_samples)) {
      // This chunk made it into the batch. Remove it from the backlog.
      it = backlog_.erase(it);
    } else {
      // Cannot add chunk to batch. Will retry later.
      ++it;
    }
  }
}

void CudaOnlinePipelineDynamicBatcher::BatcherThreadLoop() {
  Timer timer;
  double next_timeout_at = timer.Elapsed() + config_.dynamic_batcher_timeout;
  while (run_batcher_thread_) {
    if (n_chunks_not_done_.load() >= max_batch_size_ ||
        timer.Elapsed() >= next_timeout_at) {
      // Time to run the batch.
      {
        // Take ownership of the lock protecting next_batch_, backlog_.
        std::lock_guard<std::mutex> lk(next_batch_and_backlog_m_);
        std::swap(curr_batch_, next_batch_);
        // We will soon run curr_batch_.
        // Before releasing the lock, we'll add the backlog to next_batch_.
        // Once the lock is released, Push() will start adding things in
        // next_batch_.
        // We want to preserve FIFO property, so considering the backlog first.
        FillNextBatchWithBacklog();
      }  // Release the lock.

      // Batch created, push batch.
      if (curr_batch_->Size() > 0) {
        cuda_pipeline_.DecodeBatch(
            curr_batch_->corr_ids, curr_batch_->h_all_waveform,
            curr_batch_->n_samples_valid, curr_batch_->is_first_chunk,
            curr_batch_->is_last_chunk);
        n_chunks_not_done_.fetch_sub(curr_batch_->Size(),
                                     std::memory_order_release);

        curr_batch_->Clear();
      }

      timer.Reset();  // to avoid some kind of overflow..
      next_timeout_at = timer.Elapsed() + config_.dynamic_batcher_timeout;
    } else {
      Sleep(kDynamicBatcherLoopTick);
    }
  }
}

void CudaOnlinePipelineDynamicBatcher::WaitForCompletion() {
  // Waiting for the batcher to be done sending work to pipeline
  while (n_chunks_not_done_.load() > 0) {
    Sleep(kDynamicBatcherLoopTick);
  }
  // Waiting for pipeline to complete
  cuda_pipeline_.WaitForLatticeCallbacks();
}

}  // namespace cuda_decoder
}  // namespace kaldi
