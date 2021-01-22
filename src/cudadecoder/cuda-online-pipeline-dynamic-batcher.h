// cudadecoder/cuda-online-pipeline-dynamic-batcher.h
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

#include <unordered_map>
#if HAVE_CUDA == 1

#ifndef KALDI_CUDA_DECODER_DYNAMIC_BATCHER_H_
#define KALDI_CUDA_DECODER_DYNAMIC_BATCHER_H_

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"

namespace kaldi {
namespace cuda_decoder {

struct CudaOnlinePipelineDynamicBatcherConfig {
  double dynamic_batcher_timeout = 2e-3;
};

class CudaOnlinePipelineDynamicBatcher {
 public:
  CudaOnlinePipelineDynamicBatcher(
      CudaOnlinePipelineDynamicBatcherConfig config,
      BatchedThreadedNnet3CudaOnlinePipeline &cuda_pipeline);
  typedef BatchedThreadedNnet3CudaOnlinePipeline::CorrelationID CorrelationID;

  virtual ~CudaOnlinePipelineDynamicBatcher();
  void Push(CorrelationID corr_id, bool is_first_chunk, bool is_last_chunk,
            const SubVector<BaseFloat> &wave_samples);

  void WaitForCompletion();

 private:
  CudaOnlinePipelineDynamicBatcherConfig config_;
  struct Chunk {
    CorrelationID corr_id;
    bool is_first_chunk;
    bool is_last_chunk;
    SubVector<BaseFloat> wave_samples;
  };

  void BatcherThreadLoop();
  std::list<Chunk> chunks_;
  std::unordered_set<CorrelationID> is_corr_id_in_batch_;
  std::unordered_set<CorrelationID> is_corr_id_in_use_;
  std::mutex chunks_m_;
  std::atomic<std::uint32_t> n_chunks_not_done_;  // chunks not yet computed
  bool run_batcher_thread_;
  std::unique_ptr<std::thread> batcher_thread_;
  BatchedThreadedNnet3CudaOnlinePipeline &cuda_pipeline_;

  std::vector<CorrelationID> batch_corr_ids_;
  std::vector<bool> batch_is_first_chunk_;
  std::vector<bool> batch_is_last_chunk_;
  std::vector<SubVector<BaseFloat>> batch_wave_samples_;

  std::vector<const std::string *> partial_hypotheses_;
  std::vector<bool> end_points_;

  int max_batch_size_;
  int num_channels_;
};

}  // end namespace cuda_decoder
}  // end namespace kaldi.

#endif  // KALDI_CUDA_DECODER_DYNAMIC_BATCHER_H_
#endif  // HAVE_CUDA
