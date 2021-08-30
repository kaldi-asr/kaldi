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

#ifndef KALDI_CUDADECODER_CUDA_ONLINE_PIPELINE_DYNAMIC_BATCHER_H_
#define KALDI_CUDADECODER_CUDA_ONLINE_PIPELINE_DYNAMIC_BATCHER_H_

#if HAVE_CUDA

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"

namespace kaldi {
namespace cuda_decoder {

struct CudaOnlinePipelineDynamicBatcherConfig {
  double dynamic_batcher_timeout = 2e-3;
};

class CudaOnlinePipelineDynamicBatcher {
 public:
  typedef BatchedThreadedNnet3CudaOnlinePipeline::CorrelationID CorrelationID;

  CudaOnlinePipelineDynamicBatcher(
      CudaOnlinePipelineDynamicBatcherConfig config,
      BatchedThreadedNnet3CudaOnlinePipeline &cuda_pipeline);

  virtual ~CudaOnlinePipelineDynamicBatcher();

  // Push a new chunk to the dynamic batcher
  // the wave_samples will be deep copied,
  // the caller can safely reuse or free wave_samples's storage after Push's
  // return
  void Push(CorrelationID corr_id, bool is_first_chunk, bool is_last_chunk,
            const SubVector<BaseFloat> &wave_samples);
  void WaitForCompletion();

 private:
  // Batches created by this Batcher
  struct Batch {
    std::vector<CorrelationID> corr_ids;
    std::vector<bool> is_first_chunk;
    std::vector<bool> is_last_chunk;
    Matrix<BaseFloat> h_all_waveform;
    std::vector<int> n_samples_valid;
    std::unordered_set<CorrelationID> is_corr_id_in_batch;

    Batch(int max_batch_size, int max_samps_per_chunk) {
      h_all_waveform.Resize(max_batch_size, max_samps_per_chunk, kUndefined,
                            kStrideEqualNumCols);
      // TODO use cudaHostRegister, check cudaDevAttrHostRegisterSupported
    }

    void Clear() {
      corr_ids.clear();
      is_first_chunk.clear();
      is_last_chunk.clear();
      n_samples_valid.clear();
      is_corr_id_in_batch.clear();
    }

    // Adding chunk to the batch. Deep copy of samples
    void PushBack(CorrelationID corr_id, bool is_first, bool is_last,
                  const VectorBase<BaseFloat> &samples) {
      size_t idx = corr_ids.size();
      KALDI_ASSERT(idx < h_all_waveform.NumRows());
      corr_ids.push_back(corr_id);
      is_first_chunk.push_back(is_first);
      is_last_chunk.push_back(is_last);
      int nsamples = samples.Dim();
      KALDI_ASSERT(nsamples <= h_all_waveform.NumCols());
      const BaseFloat *wave_src = samples.Data();
      BaseFloat *wave_dst = h_all_waveform.RowData(idx);
      std::memcpy(wave_dst, wave_src, nsamples * sizeof(BaseFloat));
      n_samples_valid.push_back(nsamples);
    }

    size_t Size() { return corr_ids.size(); }
  };

  struct Chunk {
    CorrelationID corr_id;
    bool is_first_chunk;
    bool is_last_chunk;
    Vector<BaseFloat> wave_samples;  // deep copy, owns data
  };

  // Either add to next_batch_ or to backlog_
  bool TryAddChunkToNextBatchDeepCopy(
      CorrelationID corr_id, bool is_first_chunk, bool is_last_chunk,
      const VectorBase<BaseFloat> &wave_samples);

  // When we start building a fresh (empty) batch,
  // we'll first add what we can from the backlog_
  // (FIFO)
  void FillNextBatchWithBacklog();

  CudaOnlinePipelineDynamicBatcherConfig config_;

  void BatcherThreadLoop();
  std::list<Chunk> backlog_;
  // Protects both next_batch_ and backlog_
  std::mutex next_batch_and_backlog_m_;
  bool run_batcher_thread_;
  std::unique_ptr<std::thread> batcher_thread_;
  BatchedThreadedNnet3CudaOnlinePipeline &cuda_pipeline_;

  std::vector<const std::string *> partial_hypotheses_;
  std::vector<bool> end_points_;
  std::atomic<std::uint32_t> n_chunks_not_done_;

  int max_batch_size_;
  int num_channels_;

  std::unique_ptr<Batch> curr_batch_, next_batch_;
};

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // HAVE_CUDA
#endif  // KALDI_CUDADECODER_CUDA_ONLINE_PIPELINE_DYNAMIC_BATCHER_H_
