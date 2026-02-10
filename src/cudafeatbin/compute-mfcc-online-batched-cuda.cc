// cudafeat/compute-mfcc-online-batched-cuda.cc
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens, Levi Barnes
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
#include <cuda_profiler_api.h>
#endif

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "cudafeat/feature-online-batched-spectral-cuda.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/feature-window.h"
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
  int32_t num_frames;
  int32_t current_frame;

  UtteranceDataHandle(const std::string &utt, WaveData &wave_data,
                      const FrameExtractionOptions &opts, int32_t feat_dim)
      : utt(utt) {
    current_sample = 0;
    current_frame = 0;
    num_samples = wave_data.Data().NumCols();

    wave_data_in = wave_data;

    num_frames = NumFrames(num_samples, opts, true);
    feats_out.Resize(num_frames, feat_dim);
  }
};

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Compute online mfcc features.\n\n"
        "This binary processes the audio in chunks of samples. "
        "In addition, the computation is batched and done in CUDA. "
        "This binary is not intended to demonstrate how to achieve "
        "maximum performance.  Instead it is intended to demonstrate "
        "how to use the class CudaOnlineBatchedSpectralFeatures and provide "
        "a mechanism to test this class independently.\n\n"
        "Usage: ./compute-mfcc-batched-cuda --batch-size=50 "
        "<wave-rspecifier> "
        "<feature-wspecifier> \n";

    int32_t num_channels = 50;
    int32_t num_lanes = 10;
    int32_t max_chunk_length_samples = 10000;
    BaseFloat sample_freq = -1;
    BaseFloat vtln_warp = 1.0;

    ParseOptions po(usage);
    MfccOptions feature_opts;
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

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(num_channels >= num_lanes);

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    LaneDesc *d_lanes = (LaneDesc *)CuDevice::Instantiate().Malloc(
        sizeof(LaneDesc) * num_lanes);

    std::string wav_rspecifier = po.GetArg(1),
                feature_wspecifier = po.GetArg(2);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter feature_writer;

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

    int32_t feat_dim = feature_opts.num_ceps;

    // compute the maximum chunk length in frames
    const FrameExtractionOptions &frame_opts = feature_opts.frame_opts;

    // span between consective features in output
    int32_t shift = frame_opts.WindowShift();
    int32_t max_chunk_frames = (max_chunk_length_samples + shift - 1) / shift;

    int32_t ldf = max_chunk_frames;

    CudaOnlineBatchedSpectralFeatures mfcc(feature_opts, max_chunk_frames,
                                           num_channels, num_lanes);

    CuMatrix<BaseFloat> d_batch_wav_in(num_lanes, max_chunk_length_samples,
                                       kUndefined, kStrideEqualNumCols);
    CuMatrix<BaseFloat> d_batch_feats_out(num_lanes * ldf, feat_dim, kUndefined,
                                          kStrideEqualNumCols);

    // host matrices for staging data in pinned memory before copying down
    Matrix<BaseFloat> h_batch_wav_in(num_lanes, max_chunk_length_samples,
                                     kUndefined, kStrideEqualNumCols);
    Matrix<BaseFloat> h_batch_feats_out(num_lanes * ldf, feat_dim, kUndefined,
                                        kStrideEqualNumCols);

    size_t wave_in_size =
        num_lanes * max_chunk_length_samples * sizeof(BaseFloat);
    size_t feats_out_size = num_lanes * ldf * feat_dim * sizeof(BaseFloat);
    ;

    cudaHostRegister(h_batch_wav_in.Data(), wave_in_size, 0);
    cudaHostRegister(h_batch_feats_out.Data(), feats_out_size, 0);

    CU_SAFE_CALL(cudaGetLastError());

    std::vector<int32_t> num_frames_computed(num_lanes);

    std::vector<LaneDesc> lanes;

    for (int32_t i = 0; i < num_channels; i++) {
      free_channels.push_back(i);
    }

    sample_freq = frame_opts.samp_freq;

    double duration = 0.0;
    // preload data for batching
    for (; !reader.Done(); reader.Next()) {
      std::string utt = reader.Key();
      WaveData &wave_data = reader.Value();
      if (wave_data.SampFreq() != feature_opts.frame_opts.samp_freq) {
        KALDI_ERR << "File: " << utt << " has an mismatched sampling "
          << "rate (config= " << feature_opts.frame_opts.samp_freq
          << " vs file=" << wave_data.SampFreq() << ".";
      }

      duration += wave_data.Duration();
      data_handles.emplace_back(utt, wave_data, frame_opts, feat_dim);
    }

    // Timing just compute, we don't want to include
    // disc I/O in this timer.
    Timer timer;
    // A single pass through this loop will fill the
    // batch with new work if any is available.
    // Then process a single iteration of batched cmvn.
    // At exit each process handle should have valid data
    // in feats_out.
    while (true) {
      // This loop will fill the batch by pulling from the
      // data_handles vector for new work
      while (lanes.size() < num_lanes && not_done_idx < data_handles.size()) {
        UtteranceDataHandle &handle = data_handles[not_done_idx];
        int32_t num_samples = handle.num_samples;
        num_samples = std::min(max_chunk_length_samples, num_samples);

        // grab a free channel
        int32_t channel = free_channels.back();
        free_channels.pop_back();

        LaneDesc desc;
        desc.channel = channel;
        desc.current_sample = 0;
        desc.num_chunk_samples = num_samples;
        desc.first = true;
        desc.last = num_samples == handle.num_samples;
        desc.current_frame = 0;
        desc.num_chunk_frames = NumFrames(num_samples, frame_opts, desc.last);
        lanes.push_back(desc);

        channel_to_handle_idx[channel] = not_done_idx;
        not_done_idx++;
      }

      // No work in lanes, this means corpus is finished
      if (lanes.size() == 0) break;

      cudaMemcpyAsync(d_lanes, &lanes[0], sizeof(LaneDesc) * lanes.size(),
                      cudaMemcpyHostToDevice, cudaStreamPerThread);

      // This loop copies a slice from each active audio cut
      // down to the device for processing
      for (int32_t lane = 0; lane < lanes.size(); lane++) {
        LaneDesc &desc = lanes[lane];
        int32_t channel = desc.channel;
        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        int32_t current_sample = handle.current_sample;
        int32_t num_samples = desc.num_chunk_samples;

        // Create a subvector for this slice of data
        SubVector<BaseFloat> p_wave(
            h_batch_wav_in.Row(lane).Range(0, num_samples));

        SubVector<BaseFloat> h_wave(handle.wave_data_in.Data().Row(0).Range(
            current_sample, num_samples));

        // Copy slice into pinned memory
        p_wave.CopyFromVec(h_wave);
      }

      // use a memcpy here to avoid a possible 2D memcpy which is very slow
      cudaMemcpyAsync(d_batch_wav_in.Data(), h_batch_wav_in.Data(),
                      wave_in_size, cudaMemcpyHostToDevice,
                      cudaStreamPerThread);
      CU_SAFE_CALL(cudaGetLastError());

      // process batch
      mfcc.ComputeFeaturesBatched(d_lanes, lanes.size(), d_batch_wav_in,
                                  sample_freq, vtln_warp, &d_batch_feats_out);

      // copy feats to host
      cudaMemcpyAsync(h_batch_feats_out.Data(), d_batch_feats_out.Data(),
                      feats_out_size, cudaMemcpyDeviceToHost,
                      cudaStreamPerThread);
      CU_SAFE_CALL(cudaGetLastError());

      // wait for copy to host to complete before copying to final
      // location.  For additional optimization you should double buffer
      // h_batch_* arrays so that the GPU isn't idle while the CPU
      // is copying data into final destination.  We don't envision
      // people using this binary directly and thus won't do that
      // here to keep the API example more concise.
      cudaStreamSynchronize(cudaStreamPerThread);

      // At this time the batch is computed.  We now need to copy each slice
      // into the appropriate output buffer
      for (int lane = 0; lane < lanes.size(); lane++) {
        LaneDesc &desc = lanes[lane];
        ChannelId channel = desc.channel;

        int32_t current_frame = desc.current_frame;
        int32_t num_chunk_frames = desc.num_chunk_frames;
        if (num_chunk_frames == 0) continue;

        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        // Copy slice back up
        CuSubMatrix<BaseFloat> A(d_batch_feats_out.Range(
            lane * max_chunk_frames, num_chunk_frames, 0, feat_dim));
        SubMatrix<BaseFloat> B(handle.feats_out.Range(
            current_frame, num_chunk_frames, 0, feat_dim));

        B.CopyFromMat(A);
      }  // end copy to host loop

      // For each lane check if compute is done.
      // If completed, remove from channel list and
      // free the channel.
      for (int32_t lane = 0; lane < lanes.size();) {
        LaneDesc &desc = lanes[lane];
        ChannelId channel = desc.channel;
        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        int32_t &chunk_samples = desc.num_chunk_samples;
        // advance by samples processed in last chunk
        handle.current_sample += chunk_samples;

        desc.current_sample += desc.num_chunk_samples;
        desc.num_chunk_samples = std::min(
            max_chunk_length_samples, handle.num_samples - desc.current_sample);
        desc.current_frame = NumFrames(desc.current_sample, frame_opts, false);
        int32_t num_samples = desc.current_sample + desc.num_chunk_samples;
        int32_t num_frames = NumFrames(num_samples, frame_opts, desc.last);
        desc.num_chunk_frames =
            std::min(max_chunk_frames, num_frames - desc.current_frame);
        // read if we said last chunk was last
        bool finished = desc.last;

        // compute next batch of samples
        int32_t num_remaining_samples =
            handle.num_samples - handle.current_sample;
        chunk_samples =
            std::min(max_chunk_length_samples, num_remaining_samples);

        int32_t num_total_samples = handle.current_sample + chunk_samples;

        desc.last = num_total_samples == handle.num_samples;
        desc.first = false;

        if (finished) {
          // free this channel
          free_channels.push_back(channel);
          // Move last lane to this lane
          lanes[lane] = lanes.back();
          lanes.pop_back();

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
    }

    KALDI_LOG << "Computed Online Features for  " << num_done << " files, and "
              << tot_t << " frames.";

    KALDI_LOG << "Total Audio: " << duration
              << " seconds, Total Time: " << total_time
              << " seconds, RTFX: " << duration / total_time;

    cudaHostUnregister(h_batch_wav_in.Data());
    cudaHostUnregister(h_batch_feats_out.Data());

    cudaDeviceSynchronize();
#if HAVE_CUDA == 1
    cudaProfilerStop();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
