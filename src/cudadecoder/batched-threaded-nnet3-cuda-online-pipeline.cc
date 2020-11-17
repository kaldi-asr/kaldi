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

#if HAVE_CUDA == 1

#define KALDI_CUDA_DECODER_WAIT_FOR_CALLBACKS_US 10000
#define KALDI_CUDA_DECODER_WAIT_FOR_CPU_FEATURES_THREADS_US 1000
#define KALDI_CUDA_DECODER_WAIT_FOR_AVAILABLE_CHANNEL_US 1000

#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include <nvToolsExt.h>
#include "feat/feature-window.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace cuda_decoder {
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

  d_all_log_posteriors_.Resize(max_batch_size_ * output_frames_per_chunk_,
                               trans_model_->NumPdfs(), kUndefined);
  available_channels_.resize(config_.num_channels);
  lattice_callbacks_.reserve(config_.num_channels);
  best_path_callbacks_.reserve(config_.num_channels);
  std::iota(available_channels_.begin(), available_channels_.end(),
            0);  // 0,1,2,3..
  corr_id2channel_.reserve(config_.num_channels);
  channel_frame_offset_.resize(config_.num_channels, 0);

  // Feature extraction
  if (config_.use_gpu_feature_extraction) {
    gpu_feature_pipeline_.reset(new OnlineBatchedFeaturePipelineCuda(
        config_.feature_opts, samples_per_chunk_, config_.max_batch_size,
        config_.num_channels));
  } else {
    feature_pipelines_.resize(config_.num_channels);
  }

  // Decoder
  cuda_fst_ = std::make_shared<CudaFst>();
  cuda_fst_->Initialize(decode_fst, trans_model_);
  cuda_decoder_.reset(new CudaDecoder(*cuda_fst_, config_.decoder_opts,
                                      max_batch_size_, config_.num_channels));
  if (config_.num_decoder_copy_threads > 0) {
    cuda_decoder_->SetThreadPoolAndStartCPUWorkers(
        thread_pool_.get(), config_.num_decoder_copy_threads);
  }

  cuda_decoder_->SetOutputFrameShiftInSeconds(
      feature_info_->FrameShiftInSeconds() *
      config_.compute_opts.frame_subsampling_factor);

  n_samples_valid_.resize(max_batch_size_);
  n_input_frames_valid_.resize(max_batch_size_);
  n_lattice_callbacks_not_done_.store(0);
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetLatticeCallback(
    CorrelationID corr_id, const LatticeCallback &callback) {
  std::lock_guard<std::mutex> lk(map_callbacks_m_);
  lattice_callbacks_.insert({corr_id, callback});
}

void BatchedThreadedNnet3CudaOnlinePipeline::SetBestPathCallback(
    CorrelationID corr_id, const BestPathCallback &callback) {
  std::lock_guard<std::mutex> lk(map_callbacks_m_);
  best_path_callbacks_.insert({corr_id, callback});
}

bool BatchedThreadedNnet3CudaOnlinePipeline::TryInitCorrID(
    CorrelationID corr_id, int wait_for) {
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
      while (waited_for < wait_for) {
        lk.unlock();
        usleep(KALDI_CUDA_DECODER_WAIT_FOR_AVAILABLE_CHANNEL_US);
        waited_for += KALDI_CUDA_DECODER_WAIT_FOR_AVAILABLE_CHANNEL_US;
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
    KALDI_WARN << "This corr_id was already in use";
    ichannel = it->second;
  }

  if (!config_.use_gpu_feature_extraction) {
    KALDI_ASSERT(!feature_pipelines_[ichannel]);
    feature_pipelines_[ichannel].reset(
        new OnlineNnet2FeaturePipeline(*feature_info_));
  }

  channel_frame_offset_[ichannel] = 0;
  return true;
}  // namespace cuda_decoder

void BatchedThreadedNnet3CudaOnlinePipeline::ComputeGPUFeatureExtraction(
    const std::vector<int> &channels,
    const std::vector<SubVector<BaseFloat>> &wave_samples,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk) {
  for (int i = 0; i < wave_samples.size(); ++i) {
    const SubVector<BaseFloat> &src = wave_samples[i];
    int size = src.Dim();
    n_samples_valid_[i] = size;
    const BaseFloat *wave_src = src.Data();
    BaseFloat *wave_dst = h_all_waveform_.RowData(i);
    std::memcpy(wave_dst, wave_src, size * sizeof(BaseFloat));
  }
  // CopyFromMat syncs, avoiding it
  KALDI_ASSERT(d_all_waveform_.SizeInBytes() == h_all_waveform_.SizeInBytes());
  cudaMemcpyAsync(d_all_waveform_.Data(), h_all_waveform_.Data(),
                  h_all_waveform_.SizeInBytes(), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);

  KALDI_ASSERT(channels.size() == is_last_chunk.size());
  KALDI_ASSERT(channels.size() == is_first_chunk.size());

  KALDI_ASSERT(gpu_feature_pipeline_);
  gpu_feature_pipeline_->ComputeFeaturesBatched(
      channels.size(), channels, n_samples_valid_, is_first_chunk,
      is_last_chunk, model_frequency_, d_all_waveform_, &d_all_features_,
      &d_all_ivectors_, &n_input_frames_valid_);
}

void BatchedThreadedNnet3CudaOnlinePipeline::ComputeCPUFeatureExtraction(
    const std::vector<int> &channels,
    const std::vector<SubVector<BaseFloat>> &wave_samples,
    const std::vector<bool> &is_last_chunk) {
  // Will be used by worker threads to grab work
  fe_threads_channels_ = &channels;
  fe_threads_wave_samples_ = &wave_samples;

  n_compute_features_not_done_.store(channels.size());

  for (size_t i = 0; i < channels.size(); ++i) {
    thread_pool_->Push(
        {&BatchedThreadedNnet3CudaOnlinePipeline::ComputeOneFeatureWrapper,
         this, i, 0});  // second argument "0" is not used
  }

  while (n_compute_features_not_done_.load(std::memory_order_acquire))
    usleep(KALDI_CUDA_DECODER_WAIT_FOR_CPU_FEATURES_THREADS_US);

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
    const std::vector<SubVector<BaseFloat>> &wave_samples,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk,
    std::vector<const std::string *> *in_partial_hypotheses,
    std::vector<bool> *in_end_points) {
  nvtxRangePushA("DecodeBatch");
  KALDI_ASSERT(corr_ids.size() > 0);
  KALDI_ASSERT(corr_ids.size() == wave_samples.size());
  KALDI_ASSERT(corr_ids.size() == is_last_chunk.size());

  partial_hypotheses_ = in_partial_hypotheses;
  end_points_ = in_end_points;
  {
    std::lock_guard<std::mutex> lk(map_callbacks_m_);
    if (!best_path_callbacks_.empty()) {
      // If best path callbacks are in use, always activate partial hyp and endp
      if (!partial_hypotheses_) partial_hypotheses_ = &partial_hypotheses_buf_;
      if (!end_points_) end_points_ = &end_points_buf_;
    }
  }

  ListIChannelsInBatch(corr_ids, &channels_);

  if (config_.use_gpu_feature_extraction)
    ComputeGPUFeatureExtraction(channels_, wave_samples, is_first_chunk,
                                is_last_chunk);
  else
    ComputeCPUFeatureExtraction(channels_, wave_samples, is_last_chunk);

  d_features_ptrs_.clear();
  d_ivectors_ptrs_.clear();
  for (int i = 0; i < channels_.size(); ++i) {
    d_features_ptrs_.push_back(d_all_features_.Data() +
                               i * input_frames_per_chunk_ *
                                   d_all_features_.Stride());
    if (use_ivectors_) {
      d_ivectors_ptrs_.push_back(d_all_ivectors_.Data() + i * ivector_dim_);
    }
  }
  int features_frame_stride = d_all_features_.Stride();
  DecodeBatch(corr_ids, d_features_ptrs_, features_frame_stride,
              n_input_frames_valid_, d_ivectors_ptrs_, is_first_chunk,
              is_last_chunk, &channels_);
}

void BatchedThreadedNnet3CudaOnlinePipeline::DecodeBatch(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<BaseFloat *> &d_features, const int features_frame_stride,
    const std::vector<int> &n_input_frames_valid,
    const std::vector<BaseFloat *> &d_ivectors,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk, std::vector<int> *channels) {
  nvtxRangePushA("DecodeBatch");
  if (!channels) {
    channels = &channels_;
    ListIChannelsInBatch(corr_ids, channels);
  }

  list_channels_first_chunk_.clear();
  for (size_t i = 0; i < is_first_chunk.size(); ++i) {
    if (is_first_chunk[i]) list_channels_first_chunk_.push_back((*channels)[i]);
  }
  if (!list_channels_first_chunk_.empty())
    cuda_decoder_->InitDecoding(list_channels_first_chunk_);

  RunNnet3(*channels, d_features, features_frame_stride, n_input_frames_valid,
           is_first_chunk, is_last_chunk, d_ivectors);
  if (partial_hypotheses_) {
    // We're going to have to generate the partial hypotheses
    if (word_syms_ == nullptr) {
      KALDI_ERR << "You need to set --word-symbol-table to use "
                << "partial hypotheses";
    }
    cuda_decoder_->AllowPartialHypotheses();
  }
  if (end_points_) cuda_decoder_->AllowEndpointing();

  RunDecoder(*channels);

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
  RunCallbacksAndFinalize(corr_ids, *channels, is_last_chunk);
  nvtxRangePop();
}

void BatchedThreadedNnet3CudaOnlinePipeline::ComputeOneFeature(int element) {
  const SubVector<BaseFloat> &wave_samples =
      (*fe_threads_wave_samples_)[element];
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

void BatchedThreadedNnet3CudaOnlinePipeline::RunCallbacksAndFinalize(
    const std::vector<CorrelationID> &corr_ids,
    const std::vector<int> &channels, const std::vector<bool> &is_last_chunk) {
  // Best path callbacks
  {
    std::lock_guard<std::mutex> lk(map_callbacks_m_);
    if (!best_path_callbacks_.empty() && partial_hypotheses_ && end_points_) {
      for (int i = 0; i < corr_ids.size(); ++i) {
        CorrelationID corr_id = corr_ids[i];
        auto it_callback = best_path_callbacks_.find(corr_id);
        if (it_callback != best_path_callbacks_.end()) {
          // We have a best path callback for this corr_id
          const std::string &best_path = *((*partial_hypotheses_)[i]);
          bool partial = !is_last_chunk[i];
          bool endpoint_detected = (*end_points_)[i];
          // Run them on main thread - We could move the best path callbacks on
          // the threadpool
          it_callback->second(best_path, partial, endpoint_detected);
          if (is_last_chunk[i]) best_path_callbacks_.erase(it_callback);
        }
      }
    }
  }

  list_channels_last_chunk_.clear();
  list_corr_id_last_chunk_.clear();
  list_lattice_callbacks_last_chunk_.clear();
  {
    std::lock_guard<std::mutex> lk_callbacks(map_callbacks_m_);
    std::lock_guard<std::mutex> lk_channels(available_channels_m_);
    for (int i = 0; i < is_last_chunk.size(); ++i) {
      if (is_last_chunk[i]) {
        ChannelId ichannel = channels[i];
        CorrelationID corr_id = corr_ids[i];

        bool has_lattice_callback = false;
        decltype(lattice_callbacks_.end()) it_lattice_callback;
        if (!lattice_callbacks_.empty()) {
          it_lattice_callback = lattice_callbacks_.find(corr_id);
          has_lattice_callback =
              (it_lattice_callback != lattice_callbacks_.end());
        }
        if (has_lattice_callback) {
          LatticeCallback *lattice_callback =
              new LatticeCallback(std::move(it_lattice_callback->second));
          lattice_callbacks_.erase(it_lattice_callback);
          list_channels_last_chunk_.push_back(ichannel);
          list_corr_id_last_chunk_.push_back(corr_id);
          list_lattice_callbacks_last_chunk_.push_back(lattice_callback);
        } else {
          // All done with this corr_ids. Cleaning up
          available_channels_.push_back(ichannel);
          int32 ndeleted = corr_id2channel_.erase(corr_id);
          KALDI_ASSERT(ndeleted == 1);
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
    LatticeCallback *lattice_callback = list_lattice_callbacks_last_chunk_[i];
    thread_pool_->Push(
        {&BatchedThreadedNnet3CudaOnlinePipeline::FinalizeDecodingWrapper, this,
         ichannel, static_cast<void *>(lattice_callback)});
  }
}  // namespace cuda_decoder

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

void BatchedThreadedNnet3CudaOnlinePipeline::RunDecoder(
    const std::vector<int> &channels) {
  for (int iframe = 0; iframe < all_frames_log_posteriors_.size(); ++iframe) {
    cuda_decoder_->AdvanceDecoding(all_frames_log_posteriors_[iframe]);
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
  nnet3_config.nchannels = config_.num_channels;
  nnet3_config.has_ivector = (feature.IvectorFeature() != NULL);

  cuda_nnet3_.reset(new BatchedStaticNnet3(nnet3_config, *am_nnet_));
  output_frames_per_chunk_ = cuda_nnet3_->GetNOutputFramesPerChunk();
}

void BatchedThreadedNnet3CudaOnlinePipeline::FinalizeDecoding(
    int32 ichannel, const LatticeCallback *callback) {
  Lattice lat;
  cuda_decoder_->ConcurrentGetRawLatticeSingleChannel(ichannel, &lat);

  // Getting the channel callback now, we're going to free that channel
  // Done with this channel. Making it available again
  {
    std::lock_guard<std::mutex> lk(available_channels_m_);
    available_channels_.push_back(ichannel);
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
  if (callback && *callback) {
    (*callback)(dlat);
    delete callback;
  }

  n_lattice_callbacks_not_done_.fetch_sub(1, std::memory_order_release);
}

void BatchedThreadedNnet3CudaOnlinePipeline::WaitForLatticeCallbacks() {
  while (n_lattice_callbacks_not_done_.load() != 0)
    usleep(KALDI_CUDA_DECODER_WAIT_FOR_CALLBACKS_US);
}
}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // HAVE_CUDA
