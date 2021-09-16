// cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.cc
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

#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"

#include <nvToolsExt.h>

#include <mutex>
#include <numeric>
#include <tuple>

#include "cudamatrix/cu-common.h"
#include "feat/feature-window.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace cuda_decoder {

const double kSleepForCallBack = 10e-3;
const double kSleepForCpuFeatures = 1e-3;
const double kSleepForChannelAvailable = 1e-3;

BatchedThreadedNnet3CudaOnlinePipeline::~BatchedThreadedNnet3CudaOnlinePipeline(
) {
  // The destructor races with callback completion. Even if all callbacks have
  // finished, the counter may (non-deterministically) lag behind by a few ms.
  // Deleting the object when all callbacks had been called is UB: the variable
  // n_lattice_callbacks_not_done_ is accessed after a callback has returned.
  WaitForLatticeCallbacks();
  KALDI_ASSERT(n_lattice_callbacks_not_done_ == 0);
  KALDI_ASSERT(available_channels_.empty() ||
               available_channels_.size() == num_channels_);

  if (h_all_waveform_.SizeInBytes() > 0) {
    CU_SAFE_CALL(::cudaHostUnregister(h_all_waveform_.Data()));
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::Initialize(
    const fst::Fst<fst::StdArc> &decode_fst) {
  ReadParametersFromModel();
  AllocateAndInitializeData(decode_fst);
}

void BatchedThreadedNnet3CudaOnlinePipeline::AllocateAndInitializeData(
    const fst::Fst<fst::StdArc> &decode_fst) {
  d_all_features_.Resize(max_batch_size_ * input_frames_per_chunk_, input_dim_,
                         kUndefined, kStrideEqualNumCols);

  if (config_.use_gpu_feature_extraction) {
    h_all_waveform_.Resize(max_batch_size_, samples_per_chunk_, kUndefined,
                           kStrideEqualNumCols);
    cudaHostRegister(h_all_waveform_.Data(), h_all_waveform_.SizeInBytes(),
                     cudaHostRegisterDefault);
    d_all_waveform_.Resize(max_batch_size_, samples_per_chunk_, kUndefined,
                           kStrideEqualNumCols);
  } else {
    h_all_features_.Resize(max_batch_size_ * input_frames_per_chunk_,
                           input_dim_, kUndefined, kStrideEqualNumCols);
  }

  if (use_ivectors_) {
    d_all_ivectors_.Resize(max_batch_size_ * ivector_dim_, kSetZero);
    h_all_ivectors_.Resize(max_batch_size_, ivector_dim_, kSetZero,
                           kStrideEqualNumCols);
  }

  // Set d_features_ptrs_, d_ivectors_ptrs_, features_frame_stride_
  // to be used with DecodeBatch
  SetFeaturesPtrs();

  d_all_log_posteriors_.Resize(max_batch_size_ * output_frames_per_chunk_,
                               trans_model_->NumPdfs(), kUndefined);
  lattice_callbacks_.reserve(num_channels_);
  best_path_callbacks_.reserve(num_channels_);
  std::iota(available_channels_.begin(), available_channels_.end(),
            0);  // 0,1,2,3..
  corr_id2channel_.reserve(num_channels_);

  // Feature extraction
  if (config_.use_gpu_feature_extraction) {
    gpu_feature_pipeline_.reset(new OnlineBatchedFeaturePipelineCuda(
        config_.feature_opts, samples_per_chunk_, config_.max_batch_size,
        num_channels_));
  } else {
    feature_pipelines_.resize(num_channels_);
  }

  // Decoder.
  cuda_fst_ = std::make_unique<CudaFst>(decode_fst, trans_model_);
  cuda_decoder_.reset(new CudaDecoder(*cuda_fst_, config_.decoder_opts,
                                      max_batch_size_, num_channels_));
  if (config_.num_decoder_copy_threads > 0) {
    cuda_decoder_->SetThreadPoolAndStartCPUWorkers(
        thread_pool_.get(), config_.num_decoder_copy_threads);
  }

  decoder_frame_shift_seconds_ = feature_info_->FrameShiftInSeconds() *
                                 config_.compute_opts.frame_subsampling_factor;
  cuda_decoder_->SetOutputFrameShiftInSeconds(decoder_frame_shift_seconds_);

  n_samples_valid_.resize(max_batch_size_);
  n_input_frames_valid_.resize(max_batch_size_);
  n_lattice_callbacks_not_done_.store(0);
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetLatticeCallback(
    CorrelationID corr_id, LatticeCallback &&callback) {
  SegmentedResultsCallback segmented_callback =
      [callback = std::move(callback)](SegmentedLatticeCallbackParams& params) {
        if (params.results.empty()) {
          KALDI_WARN << "Empty result for callback";
          return;
        }
        CompactLattice *clat = params.results[0].GetLatticeResult();
        callback(*clat);
      };

  SetLatticeCallback(corr_id, std::move(segmented_callback),
                     CudaPipelineResult::RESULT_TYPE_LATTICE);
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetLatticeCallback(
    CorrelationID corr_id, const LatticeCallback &callback_) {
  auto callback = callback_;
  SetLatticeCallback(corr_id, std::move(callback));
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetBestPathCallback(
    CorrelationID corr_id, BestPathCallback &&callback) {
  std::lock_guard<std::mutex> lk(map_callbacks_m_);
  best_path_callbacks_.erase(corr_id);
  best_path_callbacks_.insert({corr_id, std::move(callback)});
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetBestPathCallback(
    CorrelationID corr_id, const BestPathCallback &callback_) {
  auto callback = callback_;
  SetBestPathCallback(corr_id, std::move(callback));
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetLatticeCallback(
    CorrelationID corr_id, SegmentedResultsCallback &&callback,
    const int result_type) {
  std::lock_guard<std::mutex> lk(map_callbacks_m_);
  lattice_callbacks_.erase(corr_id);
  lattice_callbacks_.emplace(
    std::piecewise_construct,
    std::forward_as_tuple(corr_id),
    std::forward_as_tuple(std::move(callback), result_type));
}


void BatchedThreadedNnet3CudaOnlinePipeline::SetLatticeCallback(
    CorrelationID corr_id, const SegmentedResultsCallback &callback_,
    const int result_type) {
  auto callback = callback_;
  SetLatticeCallback(corr_id, std::move(callback), result_type);
}

bool BatchedThreadedNnet3CudaOnlinePipeline::TryInitCorrID(
    CorrelationID corr_id, int32 wait_for_us) {
  bool inserted;
  decltype(corr_id2channel_.end()) it;
  std::tie(it, inserted) = corr_id2channel_.insert({corr_id, -1});
  int32 ichannel;
  if (inserted) {
    // The corr_id was not in use
    std::unique_lock<std::mutex> lk(available_channels_m_);
    bool channel_available = (available_channels_.size() > 0);
    if (!channel_available) {
      // We cannot use that corr_id
      int waited_for = 0;
      while (waited_for < wait_for_us) {
        lk.unlock();
        Sleep(kSleepForChannelAvailable);
        waited_for += int32(kSleepForChannelAvailable * 1e6);
        lk.lock();
        channel_available = (available_channels_.size() > 0);
        if (channel_available) break;
      }

      // If still not available return
      if (!channel_available) {
        corr_id2channel_.erase(it);
        return false;
      }
    }

    ichannel = available_channels_.back();
    available_channels_.pop_back();
    it->second = ichannel;
  } else {
    // This corr id was already in use but not closed
    // It can happen if for instance a channel lost connection and
    // did not send its last chunk Cleaning up
    KALDI_WARN << "This corr_id was already in use - resetting channel";
    ichannel = it->second;
  }

  if (!config_.use_gpu_feature_extraction) {
    KALDI_ASSERT(!feature_pipelines_[ichannel]);
    feature_pipelines_[ichannel].reset(
        new OnlineNnet2FeaturePipeline(*feature_info_));
  }

  channels_info_[ichannel].Reset();

  return true;
}

void BatchedThreadedNnet3CudaOnlinePipeline::CompactWavesToMatrix(
    const std::vector<SubVector<BaseFloat>> &wave_samples) {
  for (int i = 0; i < wave_samples.size(); ++i) {
    const SubVector<BaseFloat> &src = wave_samples[i];
    int size = src.Dim();
    n_samples_valid_[i] = size;
    const BaseFloat *wave_src = src.Data();
    BaseFloat *wave_dst = h_all_waveform_.RowData(i);
    std::memcpy(wave_dst, wave_src, size * sizeof(BaseFloat));
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::ComputeGPUFeatureExtraction(
    const std::vector<int> &channels, const Matrix<BaseFloat> &h_all_waveform,
    const std::vector<int> &n_samples_valid,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk) {
  // CopyFromMat syncs, avoiding it
  KALDI_ASSERT(d_all_waveform_.SizeInBytes() == h_all_waveform.SizeInBytes());
  // Note : we could have smaller copies using the actual channels.size()
  cudaMemcpyAsync(d_all_waveform_.Data(), h_all_waveform.Data(),
                  h_all_waveform.SizeInBytes(), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);

  KALDI_ASSERT(channels.size() == is_last_chunk.size());
  KALDI_ASSERT(channels.size() == is_first_chunk.size());

  KALDI_ASSERT(gpu_feature_pipeline_);
  gpu_feature_pipeline_->ComputeFeaturesBatched(
      channels.size(), channels, n_samples_valid, is_first_chunk, is_last_chunk,
      model_frequency_, d_all_waveform_, &d_all_features_, &d_all_ivectors_,
      &n_input_frames_valid_);
}

void BatchedThreadedNnet3CudaOnlinePipeline::ComputeCPUFeatureExtraction(
    const std::vector<int> &channels, const Matrix<BaseFloat> &h_all_waveform,
    const std::vector<int> &n_samples_valid,
    const std::vector<bool> &is_last_chunk) {
  // Will be used by worker threads to grab work

  fe_threads_channels_ = &channels;
  fe_threads_h_all_waveform_ = &h_all_waveform;
  fe_threads_n_samples_valid_ = &n_samples_valid;

  n_compute_features_not_done_.store(channels.size());

  for (size_t i = 0; i < channels.size(); ++i) {
    thread_pool_->Push(
        {&BatchedThreadedNnet3CudaOnlinePipeline::ComputeOneFeatureWrapper,
         this, i, 0});  // second argument "0" is not used
  }

  while (n_compute_features_not_done_.load(std::memory_order_acquire))
    Sleep(kSleepForCpuFeatures);

  KALDI_ASSERT(d_all_features_.NumRows() == h_all_features_.NumRows() &&
               d_all_features_.NumCols() == h_all_features_.NumCols());
  cudaMemcpyAsync(d_all_features_.Data(), h_all_features_.Data(),
                  h_all_features_.SizeInBytes(), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);
  if (use_ivectors_) {
    KALDI_ASSERT(d_all_ivectors_.Dim() >=
                 (h_all_ivectors_.NumRows() * h_all_ivectors_.NumCols()));
    cudaMemcpyAsync(d_all_ivectors_.Data(), h_all_ivectors_.Data(),
                    h_all_ivectors_.SizeInBytes(), cudaMemcpyHostToDevice,
                    cudaStreamPerThread);
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::DecodeBatch(
    const std::vector<CorrelationID> &corr_ids,
    const Matrix<BaseFloat> &h_all_waveform,
    const std::vector<int> &n_samples_valid,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk,
    std::vector<const std::string *> *in_partial_hypotheses,
    std::vector<bool> *in_end_points) {
  KALDI_ASSERT(corr_ids.size() > 0);
  KALDI_ASSERT(corr_ids.size() <= h_all_waveform.NumRows());
  KALDI_ASSERT(corr_ids.size() == is_last_chunk.size());
  ListIChannelsInBatch(corr_ids, &channels_);

  // Feature extraction
  if (config_.use_gpu_feature_extraction) {
    ComputeGPUFeatureExtraction(channels_, h_all_waveform, n_samples_valid,
                                is_first_chunk, is_last_chunk);
  } else {
    ComputeCPUFeatureExtraction(channels_, h_all_waveform, n_samples_valid,
                                is_last_chunk);
  }

  // Calling DecodeBatch with computed features
  DecodeBatch(corr_ids, d_features_ptrs_, features_frame_stride_,
              n_input_frames_valid_, d_ivectors_ptrs_, is_first_chunk,
              is_last_chunk, &channels_, in_partial_hypotheses, in_end_points);
}

void BatchedThreadedNnet3CudaOnlinePipeline::DecodeBatch(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<SubVector<BaseFloat>> &wave_samples,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk,
    std::vector<const std::string *> *in_partial_hypotheses,
    std::vector<bool> *in_end_points) {
  KALDI_ASSERT(corr_ids.size() > 0);
  KALDI_ASSERT(corr_ids.size() == wave_samples.size());
  KALDI_ASSERT(corr_ids.size() == is_last_chunk.size());
  ListIChannelsInBatch(corr_ids, &channels_);

  // Compact in h_all_waveform_ to use the main DecodeBatch version
  CompactWavesToMatrix(wave_samples);

  DecodeBatch(corr_ids, h_all_waveform_, n_samples_valid_, is_first_chunk,
              is_last_chunk, in_partial_hypotheses, in_end_points);
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetFeaturesPtrs() {
  d_features_ptrs_.clear();
  d_ivectors_ptrs_.clear();
  for (int i = 0; i < max_batch_size_; ++i) {
    d_features_ptrs_.push_back(d_all_features_.Data() +
                               i * input_frames_per_chunk_ *
                                   d_all_features_.Stride());
    if (use_ivectors_) {
      d_ivectors_ptrs_.push_back(d_all_ivectors_.Data() + i * ivector_dim_);
    }
  }
  features_frame_stride_ = d_all_features_.Stride();
}

void BatchedThreadedNnet3CudaOnlinePipeline::DecodeBatch(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<BaseFloat *> &d_features, const int features_frame_stride,
    const std::vector<int> &n_input_frames_valid,
    const std::vector<BaseFloat *> &d_ivectors,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk, std::vector<int> *channels,
    std::vector<const std::string *> *in_partial_hypotheses,
    std::vector<bool> *in_end_points) {
  nvtxRangePushA("DecodeBatch");
  if (!channels) {
    channels = &channels_;
    ListIChannelsInBatch(corr_ids, channels);
  }

  partial_hypotheses_ = in_partial_hypotheses;
  end_points_ = in_end_points;
  {
    std::lock_guard<std::mutex> lk(map_callbacks_m_);
    if (!best_path_callbacks_.empty()) {
      // If best path callbacks are in use, always activate partial hyp and endp
      if (!partial_hypotheses_) partial_hypotheses_ = &partial_hypotheses_buf_;
      if (!end_points_) end_points_ = &end_points_buf_;
    }
    if (config_.reset_on_endpoint) {
      if (!end_points_) end_points_ = &end_points_buf_;
    }
  }

  RunNnet3(*channels, d_features, features_frame_stride, n_input_frames_valid,
           is_first_chunk, is_last_chunk, d_ivectors);

  RunDecoder(*channels, is_first_chunk);

  RunCallbacksAndFinalize(corr_ids, *channels, is_last_chunk);
  nvtxRangePop();
}

void BatchedThreadedNnet3CudaOnlinePipeline::ComputeOneFeature(int element) {
  const int nsamples = (*fe_threads_n_samples_valid_)[element];
  const SubVector<BaseFloat> wave_samples(
      (*fe_threads_h_all_waveform_).RowData(element), nsamples);

  const int ichannel = (*fe_threads_channels_)[element];
  OnlineNnet2FeaturePipeline &feature_pipeline = *feature_pipelines_[ichannel];
  // KALDI_ASSERT("Mismatch sample frequency/model frequency" &&
  //             (model_frequency_ ==
  //             utt_chunk.sample_frequency_));
  KALDI_ASSERT(
      "Too many samples for one chunk. Must be <= "
      "this.GetNSampsPerChunk()" &&
      wave_samples.Dim() <= samples_per_chunk_);
  int32 start_iframe = feature_pipeline.NumFramesReady();

  feature_pipeline.AcceptWaveform(model_frequency_, wave_samples);

  // All frames should be ready here
  int32 end_iframe = feature_pipeline.NumFramesReady();
  int32 nframes = end_iframe - start_iframe;
  if (nframes > 0) {
    SubMatrix<BaseFloat> utt_features =
        h_all_features_.RowRange(element * input_frames_per_chunk_, nframes);
    std::vector<int> frames(nframes);
    for (int j = start_iframe; j < end_iframe; ++j)
      frames[j - start_iframe] = j;
    //
    // Copy Features
    feature_pipeline.InputFeature()->GetFrames(frames, &utt_features);

    // If available, copy ivectors
    if (use_ivectors_) {
      SubVector<BaseFloat> utt_ivector = h_all_ivectors_.Row(element);
      feature_pipeline.IvectorFeature()->GetFrame(end_iframe - 1, &utt_ivector);
    }
  }
  n_input_frames_valid_[element] = nframes;

  n_compute_features_not_done_.fetch_sub(1, std::memory_order_release);
}

void BatchedThreadedNnet3CudaOnlinePipeline::RunBestPathCallbacks(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<int> &channels) {
  std::lock_guard<std::mutex> lk(map_callbacks_m_);
  if (!best_path_callbacks_.empty() && partial_hypotheses_ && end_points_) {
    for (int i = 0; i < corr_ids.size(); ++i) {
      CorrelationID corr_id = corr_ids[i];
      auto it_callback = best_path_callbacks_.find(corr_id);
      if (it_callback != best_path_callbacks_.end()) {
        // We have a best path callback for this corr_id
        const std::string &best_path = *((*partial_hypotheses_)[i]);
        bool partial = !is_end_of_segment_[i];
        bool endpoint_detected = (*end_points_)[i];
        // Run them on main thread - We could move the best path callbacks on
        // the threadpool
        it_callback->second(best_path, partial, endpoint_detected);

        // If end of stream, clean up
        if (is_end_of_stream_[i]) best_path_callbacks_.erase(it_callback);
      }
    }
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::RunLatticeCallbacks(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<int> &channels, const std::vector<bool> &is_last_chunk) {
  list_channels_last_chunk_.clear();
  list_corr_id_last_chunk_.clear();
  list_lattice_callbacks_last_chunk_.clear();
  {
    std::lock_guard<std::mutex> lk_callbacks(map_callbacks_m_);
    std::lock_guard<std::mutex> lk_channels(available_channels_m_);
    for (int i = 0; i < is_last_chunk.size(); ++i) {
      // Only generating a lattice at end of segments
      if (!is_end_of_segment_[i]) continue;

      ChannelId ichannel = channels[i];
      CorrelationID corr_id = corr_ids[i];
      ChannelInfo &channel_info = channels_info_[ichannel];

      // End of segment, so we'll reset the decoder
      // We can only do it after the lattice has been generated
      // We'll check can_reset_decoder before resetting this channel
      // In practice we are decoding batches multiple times faster than
      // realtime, so we shouldn't have to wait on can_reset_decoder
      channel_info.must_reset_decoder = true;
      channel_info.can_reset_decoder.store(false);
      ++channel_info.segmentid;

      // Used by FinalizeDecoding to know if we should cleanup
      auto it_lattice_callback = lattice_callbacks_.find(corr_id);
      bool has_lattice_callback =
          (it_lattice_callback != lattice_callbacks_.end());
      if (has_lattice_callback) {
        std::unique_ptr<CallbackWithOptions> lattice_callback(
            new CallbackWithOptions(it_lattice_callback->second));
        // We will trigger this callback
        lattice_callback->is_last_segment = is_end_of_stream_[i];
        lattice_callback->segment_id = channel_info.segmentid;
        list_channels_last_chunk_.push_back(ichannel);
        list_corr_id_last_chunk_.push_back(corr_id);
        list_lattice_callbacks_last_chunk_.push_back(
            std::move(lattice_callback));
      }

      // If we are end of stream (last segment)
      // we need to do some cleanup
      if (is_end_of_stream_[i]) {
        // We will not use the corr_id anymore releasing it
        int32 ndeleted = corr_id2channel_.erase(corr_id);
        KALDI_ASSERT(ndeleted == 1);
        if (has_lattice_callback) {
          // We need to generate a lattice, so we cannot free the channel right
          // away indicating that this is the last segment so that we know that
          // we need to free the channel also erasing the callback
          lattice_callbacks_.erase(it_lattice_callback);
        } else {
          // We don't have any callback to run.
          // So freeing up the channel now
          // All done with this corr_ids. Cleaning up
          // We already own the available_channels_m_ lock
          available_channels_.push_back(ichannel);
        }

        if (!config_.use_gpu_feature_extraction) {
          // Done with this CPU FE pipeline
          KALDI_ASSERT(feature_pipelines_[ichannel]);
          feature_pipelines_[ichannel].reset();
        }
      }
    }
  }

  // If zero channels require lattice gen, skip it
  if (list_channels_last_chunk_.empty()) return;

  cuda_decoder_->PrepareForGetRawLattice(list_channels_last_chunk_, true);
  // Storing number of callbacks not done. Used if
  // WaitForLatticeCallbacks() is called
  n_lattice_callbacks_not_done_.fetch_add(list_channels_last_chunk_.size(),
                                          std::memory_order_acquire);

  // delete data used for decoding that corr_id
  for (int32 i = 0; i < list_channels_last_chunk_.size(); ++i) {
    uint64_t ichannel = list_channels_last_chunk_[i];
    ChannelInfo &channel_info = channels_info_[ichannel];
    bool q_was_empty;
    {
      std::lock_guard<std::mutex> lk(channel_info.mutex);
      q_was_empty = channel_info.queue.empty();
      channel_info.queue.push(std::move(list_lattice_callbacks_last_chunk_[i]));
    }
    if (q_was_empty) {
      // If q is not empty, it means we already have a task in the threadpool
      // for that channel it is important to run those task in FIFO order if
      // empty, run a new task
      thread_pool_->Push(
          {&BatchedThreadedNnet3CudaOnlinePipeline::FinalizeDecodingWrapper,
           this, ichannel, /* ignored */ nullptr});
    }
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::RunCallbacksAndFinalize(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<int> &channels, const std::vector<bool> &is_last_chunk) {
  // Reading endpoints, figuring out is_end_of_segment_
  for (size_t i = 0; i < is_last_chunk.size(); ++i) {
    bool endpoint_detected = false;
    if (config_.reset_on_endpoint) {
      KALDI_ASSERT(end_points_ && i < end_points_->size());
      endpoint_detected = (*end_points_)[i];
    }
    is_end_of_segment_[i] = endpoint_detected || is_last_chunk[i];
    is_end_of_stream_[i] = is_last_chunk[i];
  }

  RunBestPathCallbacks(corr_ids, channels);

  RunLatticeCallbacks(corr_ids, channels, is_last_chunk);
}

void BatchedThreadedNnet3CudaOnlinePipeline::ListIChannelsInBatch(
    const std::vector<CorrelationID> &corr_ids, std::vector<int> *channels) {
  channels->clear();
  list_channels_last_chunk_.clear();
  list_corr_id_last_chunk_.clear();
  for (int i = 0; i < corr_ids.size(); ++i) {
    int corr_id = corr_ids[i];
    auto it = corr_id2channel_.find(corr_id);
    KALDI_ASSERT(it != corr_id2channel_.end());
    int ichannel = it->second;
    channels->push_back(ichannel);
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::RunNnet3(
    const std::vector<int> &channels,
    const std::vector<BaseFloat *> &d_features, const int features_stride,
    const std::vector<int> &n_input_frames_valid,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk,
    const std::vector<BaseFloat *> &d_ivectors) {
  cuda_nnet3_->RunBatch(channels, d_features, features_stride, d_ivectors,
                        n_input_frames_valid, is_first_chunk, is_last_chunk,
                        &d_all_log_posteriors_, &all_frames_log_posteriors_);
}

void BatchedThreadedNnet3CudaOnlinePipeline::InitDecoding(
    const std::vector<int> &channels, const std::vector<bool> &is_first_chunk) {
  init_decoding_list_channels_.clear();
  for (size_t i = 0; i < is_first_chunk.size(); ++i) {
    int ichannel = channels[i];
    ChannelInfo &channel_info = channels_info_[ichannel];

    bool should_reset_decoder = is_first_chunk[i];

    // If reset_on_endpoint is set, we might need to reset channels even if
    // is_first_chunk is false
    if (channel_info.must_reset_decoder) {
      // Making sure the last ConcurrentGetRawLatticeSingleChannel has completed
      // on this channel
      // It shouldn't trigger in practice - pipeline runs multiple time faster
      // than realtime
      while (!channel_info.can_reset_decoder.load()) {
        usleep(kSleepForChannelAvailable);  // TODO
      }
      should_reset_decoder = true;
      channel_info.must_reset_decoder = false;

      // Before resetting the channel, saving the offset of the next segment
      channel_info.segment_offset_seconds +=
          cuda_decoder_->NumFramesDecoded(ichannel) *
          GetDecoderFrameShiftSeconds();
    }

    if (should_reset_decoder)
      init_decoding_list_channels_.push_back((channels)[i]);
  }

  if (!init_decoding_list_channels_.empty())
    cuda_decoder_->InitDecoding(init_decoding_list_channels_);
}

void BatchedThreadedNnet3CudaOnlinePipeline::RunDecoder(
    const std::vector<int> &channels, const std::vector<bool> &is_first_chunk) {
  if (partial_hypotheses_) {
    // We're going to have to generate the partial hypotheses
    if (word_syms_ == nullptr) {
      KALDI_ERR << "You need to set --word-symbol-table to use "
                << "partial hypotheses";
    }
    cuda_decoder_->AllowPartialHypotheses();
  }
  if (end_points_) cuda_decoder_->AllowEndpointing();

  // Will check which channels needs to be init (or reset),
  // and call the decoder's InitDecoding
  InitDecoding(channels, is_first_chunk);

  for (int iframe = 0; iframe < all_frames_log_posteriors_.size(); ++iframe) {
    cuda_decoder_->AdvanceDecoding(all_frames_log_posteriors_[iframe]);
  }

  if (partial_hypotheses_) {
    partial_hypotheses_->resize(channels_.size());
    for (size_t i = 0; i < channels_.size(); ++i) {
      PartialHypothesis *partial_hypothesis;
      ChannelId ichannel = channels_[i];
      cuda_decoder_->GetPartialHypothesis(ichannel, &partial_hypothesis);
      (*partial_hypotheses_)[i] = &partial_hypothesis->out_str;
    }
  }

  if (end_points_) {
    end_points_->resize(channels_.size());
    for (size_t i = 0; i < channels_.size(); ++i) {
      ChannelId ichannel = channels_[i];
      (*end_points_)[i] = cuda_decoder_->EndpointDetected(ichannel);
    }
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::ReadParametersFromModel() {
  feature_info_.reset(new OnlineNnet2FeaturePipelineInfo(config_.feature_opts));
  feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
  feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;

  OnlineNnet2FeaturePipeline feature(*feature_info_);
  use_ivectors_ = (feature.IvectorFeature() != NULL);
  input_dim_ = feature.InputFeature()->Dim();
  if (use_ivectors_) ivector_dim_ = feature.IvectorFeature()->Dim();
  model_frequency_ = feature_info_->GetSamplingFrequency();
  BaseFloat frame_shift_seconds = feature_info_->FrameShiftInSeconds();
  input_frames_per_chunk_ = config_.compute_opts.frames_per_chunk;
  seconds_per_chunk_ = input_frames_per_chunk_ * frame_shift_seconds;
  int32 samp_per_frame =
      static_cast<int>(model_frequency_ * frame_shift_seconds);
  samples_per_chunk_ = input_frames_per_chunk_ * samp_per_frame;
  BatchedStaticNnet3Config nnet3_config;
  nnet3_config.compute_opts = config_.compute_opts;
  nnet3_config.max_batch_size = max_batch_size_;
  nnet3_config.nchannels = num_channels_;
  nnet3_config.has_ivector = (feature.IvectorFeature() != NULL);

  cuda_nnet3_.reset(new BatchedStaticNnet3(nnet3_config, *am_nnet_));
  output_frames_per_chunk_ = cuda_nnet3_->GetNOutputFramesPerChunk();
}

void BatchedThreadedNnet3CudaOnlinePipeline::FinalizeDecoding(int32 ichannel) {
  ChannelInfo &channel_info = channels_info_[ichannel];

  while (true) {
    std::unique_ptr<CallbackWithOptions> callback_w_options;
    {
      std::lock_guard<std::mutex> lk(channel_info.mutex);
      // This is either the first iter of the loop, or we have tested for
      // empty() at end of previous iter
      KALDI_ASSERT(!channel_info.queue.empty());
      callback_w_options = std::move(channel_info.queue.front());
      // we'll pop when done. this is used to track when we have a worker
      // running in thread pool
    }

    Lattice lat;
    cuda_decoder_->ConcurrentGetRawLatticeSingleChannel(ichannel, &lat);

    BaseFloat segment_offset_seconds = channel_info.segment_offset_seconds;

    if (callback_w_options->is_last_segment) {
      // If this is the last segment, we can make that channel available again
      std::lock_guard<std::mutex> lk(available_channels_m_);
      available_channels_.push_back(ichannel);
    } else {
      // If this is the end of a segment but not end of stream, we keep the
      // channel open, but we will reset the decoder. Saying that we can reset
      // it now.
      channels_info_[ichannel].can_reset_decoder.store(
          true, std::memory_order_release);
    }

    // If necessary, determinize the lattice
    CompactLattice dlat;
    if (config_.determinize_lattice) {
      DeterminizeLatticePhonePrunedWrapper(*trans_model_, &lat,
                                           config_.decoder_opts.lattice_beam,
                                           &dlat, config_.det_opts);
    } else {
      ConvertLattice(lat, &dlat);
    }

    if (dlat.NumStates() > 0) {
      // Used for debugging
      if (false && word_syms_) {
        CompactLattice best_path_clat;
        CompactLatticeShortestPath(dlat, &best_path_clat);

        Lattice best_path_lat;
        ConvertLattice(best_path_clat, &best_path_lat);

        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
        std::ostringstream oss;
        for (size_t i = 0; i < words.size(); i++) {
          std::string s = word_syms_->Find(words[i]);
          if (s == "") oss << "Word-id " << words[i] << " not in symbol table.";
          oss << s << " ";
        }
        {
          std::lock_guard<std::mutex> lk(stdout_m_);
          KALDI_LOG << "OUTPUT: " << oss.str();
        }
      }
    }

    // if ptr set and if callback func callable
    if (callback_w_options) {
      const SegmentedResultsCallback &callback = callback_w_options->callback;
      if (callback) {
        SegmentedLatticeCallbackParams params;
        params.results.emplace_back();
        CudaPipelineResult &result = params.results[0];
        if (callback_w_options->is_last_segment) result.SetAsLastSegment();
        result.SetSegmentID(callback_w_options->segment_id);
        result.SetTimeOffsetSeconds(segment_offset_seconds);

        SetResultUsingLattice(dlat, callback_w_options->result_type,
                              lattice_postprocessor_, &result);
        (callback)(params);
      }
    }

    // Callback has been run
    n_lattice_callbacks_not_done_.fetch_sub(1, std::memory_order_release);

    // pop. This marks task as done
    {
      std::lock_guard<std::mutex> lk(channel_info.mutex);
      channel_info.queue.pop();
      // Need to stop now. If the queue is seen as empty, we'll assume we have
      // no worker running in threadpool the mutex will get unlocked in
      // destructor
      if (channel_info.queue.empty()) break;
    }
  }
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetLatticePostprocessor(
    const std::shared_ptr<LatticePostprocessor> &lattice_postprocessor) {
  lattice_postprocessor_ = lattice_postprocessor;
  lattice_postprocessor_->SetDecoderFrameShift(GetDecoderFrameShiftSeconds());
  lattice_postprocessor_->SetTransitionInformation(&GetTransitionModel());
}

void BatchedThreadedNnet3CudaOnlinePipeline::WaitForLatticeCallbacks()
    noexcept {
  while (n_lattice_callbacks_not_done_.load() != 0)
    Sleep(kSleepForCallBack);
}

}  // namespace cuda_decoder
}  // namespace kaldi
