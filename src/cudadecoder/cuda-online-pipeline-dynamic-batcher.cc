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

#include "cudadecoder/cuda-online-pipeline-dynamic-batcher.h"
#include <unistd.h>

// Tries to generate a batch every n us
#define KALDI_CUDA_DECODER_DYNAMIC_BATCHER_LOOP_US 100

namespace kaldi {
namespace cuda_decoder {

CudaOnlinePipelineDynamicBatcher::CudaOnlinePipelineDynamicBatcher(
    CudaOnlinePipelineDynamicBatcherConfig config,
    BatchedThreadedNnet3CudaOnlinePipeline &cuda_pipeline)
    : config_(config),
      n_chunks_not_done_(0),
      run_batcher_thread_(true),
      cuda_pipeline_(cuda_pipeline),
      max_batch_size_(cuda_pipeline.GetConfig().max_batch_size),
      num_channels_(cuda_pipeline.GetConfig().num_channels) {
  batcher_thread_.reset(new std::thread(
      &CudaOnlinePipelineDynamicBatcher::BatcherThreadLoop, this));
}

CudaOnlinePipelineDynamicBatcher::~CudaOnlinePipelineDynamicBatcher() {
  run_batcher_thread_ = false;
  batcher_thread_->join();
}

void CudaOnlinePipelineDynamicBatcher::Push(
    CorrelationID corr_id, bool is_first_chunk, bool is_last_chunk,
    const SubVector<BaseFloat> &wave_samples) {
  std::lock_guard<std::mutex> lk(chunks_m_);
  chunks_.push_back({corr_id, is_first_chunk, is_last_chunk, wave_samples});
  n_chunks_not_done_.fetch_add(1);
}

void CudaOnlinePipelineDynamicBatcher::BatcherThreadLoop() {
  Timer timer;
  double next_timeout_at = timer.Elapsed() + config_.dynamic_batcher_timeout;
  while (run_batcher_thread_) {
    if (n_chunks_not_done_.load() >= max_batch_size_ ||
        timer.Elapsed() >= next_timeout_at) {
      int curr_batch_size = 0;
      // create batch
      {
        std::lock_guard<std::mutex> lk(chunks_m_);
        // Following assert would not be valid if we had multiple consumer
        // threads (equality not verified while DecodeBatch is running)
        KALDI_ASSERT(n_chunks_not_done_.load() == chunks_.size());
        is_corr_id_in_batch_.clear();
        batch_corr_ids_.clear();
        batch_is_first_chunk_.clear();
        batch_is_last_chunk_.clear();
        batch_wave_samples_.clear();
        for (auto it = chunks_.begin(); it != chunks_.end();) {
          CorrelationID corr_id = it->corr_id;
          bool corr_id_not_in_batch;
          decltype(is_corr_id_in_batch_.end()) is_corr_id_in_batch_it;
          std::tie(is_corr_id_in_batch_it, corr_id_not_in_batch) =
              is_corr_id_in_batch_.insert(corr_id);
          if (corr_id_not_in_batch) {
            if (it->is_first_chunk) {
              if (!cuda_pipeline_.TryInitCorrID(corr_id, 0)) {
                KALDI_LOG << "All decoding channels are in use. Consider "
                             "increasing --num-channels";
                is_corr_id_in_batch_.erase(
                    is_corr_id_in_batch_it);  // this elt won't be in the batch
                ++it;  // Ignoring this elt, we'll retry later
                continue;
              }
            }
            // first chunk with this corr_id in batch
            ++curr_batch_size;

            batch_corr_ids_.push_back(it->corr_id);
            batch_is_first_chunk_.push_back(it->is_first_chunk);
            batch_is_last_chunk_.push_back(it->is_last_chunk);
            batch_wave_samples_.push_back(it->wave_samples);

            it = chunks_.erase(it);
            if (curr_batch_size == max_batch_size_) break;
          } else {
            // Ignoring this element, we already have this corr_id in batch
            ++it;
          }
        }
      }

      // Batch created, push batch
      if (curr_batch_size > 0) {
        // KALDI_LOG << "Online Cuda pipeline decoding batch of size "
        //          << curr_batch_size;
        cuda_pipeline_.DecodeBatch(batch_corr_ids_, batch_wave_samples_,
                                   batch_is_first_chunk_, batch_is_last_chunk_);
        n_chunks_not_done_.fetch_sub(curr_batch_size,
                                     std::memory_order_release);
      }
      timer.Reset();  // to avoid some kind of overflow..
      next_timeout_at = timer.Elapsed() + config_.dynamic_batcher_timeout;

      // process callbacks here
    } else {
      usleep(KALDI_CUDA_DECODER_DYNAMIC_BATCHER_LOOP_US);
    }
  }
}

void CudaOnlinePipelineDynamicBatcher::WaitForCompletion() {
  // Waiting for the batcher to be done sending work to pipeline
  while (n_chunks_not_done_.load() > 0) {
    usleep(KALDI_CUDA_DECODER_DYNAMIC_BATCHER_LOOP_US);
  }
  // Waiting for pipeline to complete
  cuda_pipeline_.WaitForLatticeCallbacks();
}

}  // namespace cuda_decoder
}  // end namespace kaldi.
