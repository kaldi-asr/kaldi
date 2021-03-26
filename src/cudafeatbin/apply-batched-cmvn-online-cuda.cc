// online2bin/apply-batched-cmvn-online.cc

// cudafeat/online-cuda-batched-feature-pipeline-kernels.h
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

#if HAVE_CUDA == 1
#include <cuda_profiler_api.h>
#endif

#include <string>
#include <vector>
#include "base/kaldi-common.h"
#include "cudafeat/feature-online-batched-cmvn-cuda.h"
#include "feat/online-feature.h"
#include "util/common-utils.h"

using namespace kaldi;

// This class stores data for input and output of this binary.
struct UtteranceDataHandle {
  std::string utt;
  Matrix<BaseFloat> feats_in;
  Matrix<BaseFloat> feats_out;
  int32_t num_frames;

  UtteranceDataHandle(const std::string &utt, Matrix<float> &feats)
      : utt(utt), num_frames(feats.NumRows()) {
    feats_out.Resize(feats.NumRows(), feats.NumCols(), kUndefined);
    feats_in.Swap(&feats);
  }
};

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Apply online cepstral mean (and possibly variance) computation "
        "online,\n"
        "using the same code as used for online decoding in the 'new' setup "
        "in\n"
        "online2/ and online2bin/.'\n"
        "The computation is done on the device in chunks that are batched. "
        "spk2utt is not supported.\n"
        "\n"
        "Usage: apply-batched-cmvn-online-cuda [options] <global-cmvn-stats> "
        "<feature-rspecifier> "
        "<feature-wspecifier>\n"
        "e.g. apply-batched-cmvn-online-cuda 'matrix-sum "
        "scp:data/train/cmvn.scp -|' "
        "data/train/split8/1/feats.scp ark:-\n";

    int32_t num_channels = 200;
    int32_t batch_size = 100;
    int32_t chunk_length = 10000;
    int32_t stats_coarsening_factor = 1;

    int32_t feat_dim = -1;

    ParseOptions po(usage);

    po.Register("num-channels", &num_channels,
                "The number of"
                " channels used for compute");
    po.Register("batch-size", &batch_size,
                "The number of chunks from"
                " audio cuts processed in a single batch");
    po.Register("chunk-length", &chunk_length,
                "The length of a chunk"
                " of audio in frames that is processed at one time");
    po.Register(
        "stats-coarsening-factor", &stats_coarsening_factor,
        " Coarsen CMVN stats by this factor.  This reduces memory and time. "
        " But comes at the potential loss of accuracy.");

    OnlineCmvnOptions cmvn_opts;

    std::string spk2utt_rspecifier;
    cmvn_opts.Register(&po);
    CuDevice::RegisterDeviceOptions(&po);
    RegisterCuAllocatorOptions(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(num_channels >= batch_size);

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    LaneDesc *d_lanes = (LaneDesc *)CuDevice::Instantiate().Malloc(
        sizeof(LaneDesc) * batch_size);

    std::string global_stats_rxfilename = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                feature_wspecifier = po.GetArg(3);

    // global_cmvn_stats helps us initialize to online CMVN to
    // reasonable values at the beginning of the utterance.
    Matrix<double> global_cmvn_stats;
    ReadKaldiObject(global_stats_rxfilename, &global_cmvn_stats);

    BaseFloatMatrixWriter feature_writer(feature_wspecifier);
    int32 num_done = 0;
    int64 tot_t = 0;

    OnlineCmvnState cmvn_state(global_cmvn_stats);
    CudaOnlineCmvnState cu_cmvn_state(cmvn_state);

    std::vector<ChannelId> free_channels;

    // list of audio handles to be processed
    std::vector<UtteranceDataHandle> data_handles;
    // maps currently active channels to their handle index
    std::map<int, int> channel_to_handle_idx;
    // Index of next unprocessed audio file
    int not_done_idx = 0;

    bool first = true;
    // preload data for batching
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      Matrix<BaseFloat> &feats = feature_reader.Value();
      if (first == true) {
        feat_dim = feats.NumCols();
        first = false;
      }

      data_handles.emplace_back(utt, feats);
    }

    CudaOnlineBatchedCmvn cuda_cmvn(cmvn_opts, cu_cmvn_state, feat_dim,
                                    chunk_length, num_channels,
                                    stats_coarsening_factor);

    CuMatrix<BaseFloat> d_batch_in(batch_size * chunk_length, feat_dim);
    CuMatrix<BaseFloat> d_batch_out(batch_size * chunk_length, feat_dim);

    std::vector<LaneDesc> lanes;

    for (int i = 0; i < num_channels; i++) {
      free_channels.push_back(i);
    }

    // A single pass through this loop will fill the
    // batch with new work if any is available.
    // Then process a single iteration of batched cmvn.
    // At exit each process handle should have valid data
    // in feats_out.
    while (true) {
      // This loop will fill the batch by pulling from the
      // data_handles vector for new work
      while (lanes.size() < batch_size && not_done_idx < data_handles.size()) {
        UtteranceDataHandle &handle = data_handles[not_done_idx];
        int32_t current_frame = 0;
        int32_t num_frames = handle.num_frames;
        int32_t num_chunk_frames = std::min(chunk_length, num_frames);

        // grab a free channel
        int32_t channel = free_channels.back();
        free_channels.pop_back();

        LaneDesc desc;
        desc.channel = channel;
        desc.current_frame = current_frame;
        desc.num_chunk_frames = num_chunk_frames;
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
      for (int lane = 0; lane < lanes.size(); lane++) {
        LaneDesc &desc = lanes[lane];
        int32_t channel = desc.channel;
        int32_t current_frame = desc.current_frame;
        int32_t num_chunk_frames = desc.num_chunk_frames;

        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        // Create a submatrix for this slice of data
        CuSubMatrix<BaseFloat> A(d_batch_in.Range(
            lane * chunk_length, num_chunk_frames, 0, feat_dim));
        SubMatrix<BaseFloat> B(handle.feats_in.Range(
            current_frame, num_chunk_frames, 0, feat_dim));

        // Copy slice down to the device
        A.CopyFromMat(B);
      }

      // process batch
      cuda_cmvn.ComputeFeaturesBatched(lanes.size(), d_lanes, d_batch_in,
                                       &d_batch_out);

      // At this time the batch is computed.  We now need to copy each slice
      // into the appropriate output buffer
      for (int lane = 0; lane < lanes.size(); lane++) {
        LaneDesc &desc = lanes[lane];
        ChannelId channel = desc.channel;

        int32_t current_frame = desc.current_frame;
        int32_t num_chunk_frames = desc.num_chunk_frames;

        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        // Copy slice back up
        CuSubMatrix<BaseFloat> A(d_batch_out.Range(
            lane * chunk_length, num_chunk_frames, 0, feat_dim));
        SubMatrix<BaseFloat> B(handle.feats_out.Range(
            current_frame, num_chunk_frames, 0, feat_dim));

        B.CopyFromMat(A);
      }  // end copy to host loop

      // For each lane check if compute is done.
      // If completed, remove from channel list and
      // free the channel.
      for (int lane = 0; lane < lanes.size();) {
        LaneDesc &desc = lanes[lane];
        ChannelId channel = desc.channel;

        UtteranceDataHandle &handle =
            data_handles[channel_to_handle_idx[channel]];

        // advance channel
        desc.current_frame += desc.num_chunk_frames;
        desc.num_chunk_frames =
            std::min(chunk_length, handle.num_frames - desc.current_frame);

        if (desc.current_frame == handle.num_frames) {
          // free this channel
          free_channels.push_back(channel);
          // Move last lane to this lane
          lanes[lane] = lanes.back();
          lanes.pop_back();
        } else {
          // This lane is not finished so leave it alone
          lane++;
        }
      }  // end check if done loop
    }    // end while(true)

    // output all utterances.  In an efficeint implementation
    // this would be done on demand in a threaded manner.  This
    // binary is purely for checking correctness and demonstrating
    // usage and thus this type of optimization is not done.
    for (int i = 0; i < data_handles.size(); i++) {
      UtteranceDataHandle &handle = data_handles[i];

      num_done++;
      tot_t += handle.feats_out.NumRows();
      feature_writer.Write(handle.utt, handle.feats_out);
    }

    CuDevice::Instantiate().Free(d_lanes);

    KALDI_LOG << "Applied online CMVN to " << num_done << " files, or " << tot_t
              << " frames.";

    cudaDeviceSynchronize();
    cudaProfilerStop();

    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
