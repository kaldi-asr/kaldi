// cudafeat/compute-online-feats-batched-cuda.cc
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
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

#if HAVE_CUDA
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#endif

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "cudafeat/online-batched-feature-pipeline-cuda.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

using namespace kaldi;

// This class stores data for input and output for this binary.
// We will read/write slices of this input/output in an online
// fashion.
struct UtteranceDataHandle {
  std::string utt;
  WaveData wave_data_in;
  Matrix<BaseFloat> feats_out;
  Vector<BaseFloat> ivector_out;
  int32_t num_samples;
  int32_t current_sample;
  int32_t current_frame;

  UtteranceDataHandle(const std::string &utt, WaveData &wave_data,
                      const FrameExtractionOptions &opts, int32_t feat_dim,
                      int32_t ivector_dim)
      : utt(utt) {
    current_sample = 0;
    current_frame = 0;
    num_samples = wave_data.Data().NumCols();

    wave_data_in = wave_data;

    int32_t num_frames = NumFrames(num_samples, opts, true);
    feats_out.Resize(num_frames, feat_dim);
    ivector_out.Resize(ivector_dim);
  }
};

struct CallbackState {
  int32_t current_frame;
  int32_t num_chunk_frames;
  UtteranceDataHandle *handle;
  Matrix<BaseFloat> *features;
  Vector<BaseFloat> *ivectors;
};

int32_t feat_dim, ivector_dim, ldf;

void CUDART_CB CopySlicesCallback(void *cb_state_p) {
  nvtxRangePushA("CopySlices");
  std::vector<CallbackState> &cb_state =
      *reinterpret_cast<std::vector<CallbackState> *>(cb_state_p);
  // At this time the batch is computed.  We now need to copy each slice
  // into the appropriate output buffer
  for (int32_t lane = 0; lane < cb_state.size(); lane++) {
    CallbackState &state = cb_state[lane];
    UtteranceDataHandle &handle = *state.handle;

    int32_t current_frame = state.current_frame;
    int32_t num_chunk_frames = state.num_chunk_frames;

    if (num_chunk_frames > 0) {
      // Copy slice
      SubMatrix<BaseFloat> p_feats(
          state.features->Range(lane * ldf, num_chunk_frames, 0, feat_dim));
      SubMatrix<BaseFloat> h_feats(
          handle.feats_out.Range(current_frame, num_chunk_frames, 0, feat_dim));
      h_feats.CopyFromMat(p_feats);
    }
    // This overwrites the old ivector at every chunk.
    // Essentially after each chunk we have an estimate for the ivector.
    SubVector<BaseFloat> p_ivector(
        state.ivectors->Range(ivector_dim * lane, ivector_dim));
    handle.ivector_out.CopyFromVec(p_ivector);
  }  // end copy slices loop
  nvtxRangePop();
}  // end callback

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Compute online features and ivector features.\n\n"
        "This binary processes the audio in chunks of samples. "
        "In addition, the computation is batched and done in CUDA. "
        "This binary is not intended to demonstrate how to achieve "
        "maximum perfomrance.  Instead it is intended to demonstrate "
        "how to use the class OnlineCudaFeaturePipeline and provide "
        "a mechanism to test this class independently.\n\n"
        "Usage: ./compute-online-feats-batched-cuda --batch-size=100 "
        "<wave-rspecifier> "
        "<ivector-wspecifier> "
        "<feature-wspecifier> \n";

    int32_t num_channels = 50;
    int32_t num_lanes = 10;
    int32_t max_chunk_length_samples = 10000;
    BaseFloat sample_freq = -1;

    ParseOptions po(usage);
    OnlineNnet2FeaturePipelineConfig feature_opts;
    feature_opts.Register(&po);

    po.Register("num-channels", &num_channels,
                "The number of"
                " channels used for compute");
    po.Register("batch-size", &num_lanes,
                "The number of chunks from"
                " audio cuts processed in a single batch");
    po.Register("chunk-length", &max_chunk_length_samples,
                "The length of a chunk"
                " of audio in terms of samples.");

    CuDevice::RegisterDeviceOptions(&po);
    RegisterCuAllocatorOptions(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(num_channels >= num_lanes);

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    std::string wav_rspecifier = po.GetArg(1),
                ivector_wspecifier = po.GetArg(2),
                feature_wspecifier = po.GetArg(3);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatVectorWriter ivector_writer;
    BaseFloatMatrixWriter feature_writer;

    if (!ivector_writer.Open(ivector_wspecifier)) {
      KALDI_ERR << "Could not initialize ivector_writer with wspecifier "
                << ivector_wspecifier;
      exit(1);
    }
    if (!feature_writer.Open(feature_wspecifier)) {
      KALDI_ERR << "Could not initialize feature_writer with wspecifier "
                << feature_wspecifier;
      exit(1);
    }

    std::vector<ChannelId> free_channels;

    // list of audio handles to be processed
    std::vector<UtteranceDataHandle> data_handles;
    // maps currently active channels to their handle index
    std::map<int32_t, int32_t> channel_to_handle_idx;
    // Index of next unprocessed audio file
    int32_t not_done_idx = 0;
    int32_t num_done = 0, tot_t = 0;

    OnlineBatchedFeaturePipelineCuda feature_pipeline(
        feature_opts, max_chunk_length_samples, num_lanes, num_channels);

    feat_dim = feature_pipeline.FeatureDim();
    ivector_dim = feature_pipeline.IvectorDim();
    // span between consective features in output
    ldf = feature_pipeline.GetMaxChunkFrames();

    // compute the maximum chunk length in frames
    const FrameExtractionOptions &frame_opts =
        feature_pipeline.GetFrameOptions();

    CU_SAFE_CALL(cudaGetLastError());

    // create synchronization event and copy streams
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    cudaStream_t dtoh, htod;
    cudaStreamCreateWithFlags(&dtoh, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&htod, cudaStreamNonBlocking);

    // This binary is pipelined to allow concurrent memory copies and compute.
    // State exists for each pipeline and successive chunks go to different
    // pipelines in a modular fashion.  The calling thread will synchronize with
    // a pipeline prior to launching work in that pipeline.  2 should be enough
    // to get concurrency on current hardware.
    const int num_pipelines = 3;

    cudaEvent_t pipeline_events[num_pipelines];
    for (int p = 0; p < num_pipelines; p++) {
      cudaEventCreateWithFlags(&pipeline_events[p], cudaEventDisableTiming);
    }

    // This state must be replicated for each pipeline stage to avoid race
    // conditions
    std::vector<CallbackState> cb_state[num_pipelines];
    CuMatrix<BaseFloat> d_batch_wav_in[num_pipelines],
        d_batch_feats_out[num_pipelines];
    // pinned matrix to stage slices in
    Matrix<BaseFloat> h_batch_wav_in[num_pipelines],
        h_batch_feats_out[num_pipelines];
    CuVector<BaseFloat> d_batch_ivector_out[num_pipelines];
    // pinned vector to stage slices in
    Vector<BaseFloat> h_batch_ivector_out[num_pipelines];

    // size pipeline state
    for (int p = 0; p < num_pipelines; p++) {
      d_batch_wav_in[p].Resize(num_lanes, max_chunk_length_samples, kUndefined,
                               kStrideEqualNumCols);
      h_batch_wav_in[p].Resize(num_lanes, max_chunk_length_samples, kUndefined,
                               kStrideEqualNumCols);

      d_batch_feats_out[p].Resize(num_lanes * ldf, feat_dim, kUndefined,
                                  kStrideEqualNumCols);
      h_batch_feats_out[p].Resize(num_lanes * ldf, feat_dim, kUndefined,
                                  kStrideEqualNumCols);

      d_batch_ivector_out[p].Resize(num_lanes * ivector_dim);
      h_batch_ivector_out[p].Resize(num_lanes * ivector_dim);
    }

    size_t wave_in_size =
        num_lanes * max_chunk_length_samples * sizeof(BaseFloat);
    size_t feats_out_size = num_lanes * ldf * feat_dim * sizeof(BaseFloat);
    size_t ivector_out_size = num_lanes * ivector_dim * sizeof(BaseFloat);

    // pin memory for faster and asynchronous copies
    for (int p = 0; p < num_pipelines; p++) {
      cudaHostRegister(h_batch_wav_in[p].Data(), wave_in_size, 0);
      cudaHostRegister(h_batch_feats_out[p].Data(), feats_out_size, 0);
      if (ivector_dim > 0) {
        cudaHostRegister(h_batch_ivector_out[p].Data(), ivector_out_size, 0);
      }
    }

    std::vector<bool> first, last;
    std::vector<ChannelId> channels;
    std::vector<int32_t> num_chunk_samples;

    std::vector<int32_t> num_frames_computed(num_lanes);

    for (int32_t i = 0; i < num_channels; i++) {
      free_channels.push_back(i);
    }

    sample_freq = frame_opts.samp_freq;

    double duration = 0.0;
    double samples = 0.0;
    // preload data for batching
    for (; !reader.Done(); reader.Next()) {
      std::string utt = reader.Key();
      WaveData &wave_data = reader.Value();
      duration += wave_data.Duration();
      samples += wave_data.Data().NumCols();
      KALDI_ASSERT(wave_data.SampFreq() == sample_freq);

      data_handles.emplace_back(utt, wave_data, frame_opts, feat_dim,
                                ivector_dim);
    }

    // Timing just compute, we don't want to include
    // disc I/O in this timer.
    Timer timer;

    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double allocated = (total_byte - free_byte) / 1024 / 1024;
    // current pipeline
    int p = -1;

    int chunk = 0;
    // A single pass through this loop will fill the
    // batch with new work if any is available.
    // Then process a single iteration of batched cmvn.
    // At exit each process handle should have valid data
    // in feats_out.
    while (true) {
      chunk++;

      // advance the pipeline
      p = (p + 1) % num_pipelines;

      // ensure previous work in this pipeline is complete
      cudaEventSynchronize(pipeline_events[p]);

      // This loop will fill the batch by pulling from the
      // data_handles vector for new work
      while (channels.size() < num_lanes &&
             not_done_idx < data_handles.size()) {
        UtteranceDataHandle &handle = data_handles[not_done_idx];
        int32_t num_samples = handle.num_samples;
        num_samples = std::min(max_chunk_length_samples, num_samples);

        // grab a free channel
        int32_t channel = free_channels.back();
        free_channels.pop_back();

        channels.push_back(channel);
        first.push_back(true);
        last.push_back(num_samples == handle.num_samples);
        num_chunk_samples.push_back(num_samples);

        channel_to_handle_idx[channel] = not_done_idx;
        not_done_idx++;
      }

      // No work in lanes, this means corpus is finished
      if (channels.size() == 0) break;

      // This loop copies a slice from each active audio cut
      // down to the device for processing
      for (int32_t lane = 0; lane < channels.size(); lane++) {
        int32_t channel = channels[lane];
        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        int32_t current_sample = handle.current_sample;
        int32_t num_samples = num_chunk_samples[lane];

        // Create a subvector for this slice of data
        SubVector<BaseFloat> p_wave(
            h_batch_wav_in[p].Row(lane).Range(0, num_samples));

        SubVector<BaseFloat> h_wave(handle.wave_data_in.Data().Row(0).Range(
            current_sample, num_samples));

        // Copy slice into pinned memory
        p_wave.CopyFromVec(h_wave);
      }

      // use a memcpy here to avoid a possible 2D memcpy which is very slow
      cudaMemcpyAsync(d_batch_wav_in[p].Data(), h_batch_wav_in[p].Data(),
                      wave_in_size, cudaMemcpyHostToDevice, htod);
      CU_SAFE_CALL(cudaGetLastError());

      // ensure computation doesn't begin till copy is done
      cudaEventRecord(event, htod);
      cudaStreamWaitEvent(cudaStreamPerThread, event, 0);

      // process batch
      feature_pipeline.ComputeFeaturesBatched(
          channels.size(), channels, num_chunk_samples, first, last,
          sample_freq, d_batch_wav_in[p], &d_batch_feats_out[p],
          &d_batch_ivector_out[p], &num_frames_computed);

      // ensure copies don't begin until compute is done
      cudaEventRecord(event, cudaStreamPerThread);
      cudaStreamWaitEvent(dtoh, event, 0);

      // copy feats to host
      cudaMemcpyAsync(h_batch_feats_out[p].Data(), d_batch_feats_out[p].Data(),
                      feats_out_size, cudaMemcpyDeviceToHost, dtoh);
      CU_SAFE_CALL(cudaGetLastError());

      if (ivector_dim > 0) {
        // copy ivectors to host
        cudaMemcpyAsync(h_batch_ivector_out[p].Data(),
                        d_batch_ivector_out[p].Data(), ivector_out_size,
                        cudaMemcpyDeviceToHost, dtoh);
        CU_SAFE_CALL(cudaGetLastError());
      }

      // reset callback state vector
      cb_state[p].resize(0);
      // construct callback state
      for (int32_t lane = 0; lane < channels.size(); lane++) {
        ChannelId channel = channels[lane];
        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        int32_t current_frame = handle.current_frame;
        int32_t num_chunk_frames = num_frames_computed[lane];

        handle.current_frame += num_chunk_frames;

        CallbackState state;
        state.current_frame = current_frame;
        state.num_chunk_frames = num_chunk_frames;
        state.handle = &handle;
        state.features = &h_batch_feats_out[p];
        state.ivectors = &h_batch_ivector_out[p];

        cb_state[p].push_back(state);
      }

      // enqueue copy slices callback
#if CUDA_VERSION >= 10000
      cudaLaunchHostFunc(dtoh, CopySlicesCallback, (void *)&cb_state[p]);
#else
      KALDI_ERR << "Cuda 10.0 or newer required to run this binary";
#endif
      // mark the end of this chunk
      cudaEventRecord(pipeline_events[p], dtoh);

      // For each lane check if compute is done.
      // If completed, remove from channel list and
      // free the channel. If not then schedule the
      // next chunk for that lane.
      for (int32_t lane = 0; lane < channels.size();) {
        ChannelId channel = channels[lane];
        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        // read if we said last chunk was last
        bool finished = last[lane];

        int32_t &chunk_samples = num_chunk_samples[lane];
        // advance by samples processed in last chunk
        handle.current_sample += chunk_samples;

        // compute next batch of samples
        int32_t num_remaining_samples =
            handle.num_samples - handle.current_sample;
        chunk_samples =
            std::min(max_chunk_length_samples, num_remaining_samples);

        int32_t num_total_samples = handle.current_sample + chunk_samples;

        num_chunk_samples[lane] = chunk_samples;
        last[lane] = num_total_samples == handle.num_samples;
        first[lane] = false;

        if (finished) {
          // free this channel
          free_channels.push_back(channel);
          // Move last lane to this lane
          channels[lane] = channels.back();
          num_chunk_samples[lane] = num_chunk_samples.back();
          first[lane] = first.back();
          last[lane] = last.back();

          // Remove last element from lists
          channels.pop_back();
          num_chunk_samples.pop_back();
          first.pop_back();
          last.pop_back();

          num_done++;
        } else {
          lane++;
        }
      }  // end check if done loop
    }    // end while(true)
    double total_time = timer.Elapsed();

    // output all utterances.  In an efficeint implementation
    // this would be done on demand in a threaded manner.  This
    // binary is purely for checking correctness and demonstrating
    // usage and thus this type of optimization is not done.
    for (int i = 0; i < data_handles.size(); i++) {
      UtteranceDataHandle &handle = data_handles[i];

      tot_t += handle.feats_out.NumRows();
      feature_writer.Write(handle.utt, handle.feats_out);
      if (ivector_dim > 0) {
        ivector_writer.Write(handle.utt, handle.ivector_out);
      }
    }

    KALDI_LOG << "Computed Online Features for  " << num_done << " files, and "
              << tot_t << " frames.";

    KALDI_LOG << "Total Audio: " << duration << " seconds";
    KALDI_LOG << "Total Time: " << total_time << " seconds";
    KALDI_LOG << "RTFX: " << duration / total_time;
    KALDI_LOG << "Avg Chunk Latency: " << total_time / chunk * 1e6 << " us";
    KALDI_LOG << "Samples: " << samples;
    KALDI_LOG << "Samples/Second: " << samples / total_time;
    KALDI_LOG << "Memory Usage: " << allocated << " MB";
    for (int p = 0; p < num_pipelines; p++) {
      cudaHostUnregister(h_batch_wav_in[p].Data());
      cudaHostUnregister(h_batch_feats_out[p].Data());
      if (ivector_dim > 0) {
        cudaHostUnregister(h_batch_ivector_out[p].Data());
      }
    }

    cudaEventDestroy(event);
    for (int p = 0; p < num_pipelines; p++) {
      cudaEventDestroy(pipeline_events[p]);
    }
    cudaStreamDestroy(dtoh);
    cudaStreamDestroy(htod);

    cudaDeviceSynchronize();
    cudaProfilerStop();
    CU_SAFE_CALL(cudaGetLastError());

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
