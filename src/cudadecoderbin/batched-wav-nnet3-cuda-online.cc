// cudadecoderbin/batched-wav-nnet3-cuda-online.cc
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

// Binary for the online pipeline BatchedThreadedNnet3CudaOnlinePipeline.
// Can serve both as a benchmarking tool and an example on how to call
// BatchedThreadedNnet3CudaOnlinePipeline.

#if !HAVE_CUDA
#error CUDA support must be configured to compile this binary.
#endif

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <iomanip>
#include <queue>
#include <sstream>

#include "cudadecoder/cuda-online-pipeline-dynamic-batcher.h"
#include "cudadecoderbin/cuda-bin-tools.h"
#include "cudamatrix/cu-allocator.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

using namespace kaldi;
using namespace kaldi::cuda_decoder;
using namespace fst;
using CorrelationID = CudaOnlinePipelineDynamicBatcher::CorrelationID;

typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

struct Stream {
  std::shared_ptr<WaveData> wav;
  CorrelationID corr_id;
  int offset;
  double send_next_chunk_at;
  double *latency_ptr;

  Stream(const std::shared_ptr<WaveData> &wav, CorrelationID corr_id,
         double send_next_chunk_at, double *latency_ptr)
      : wav(wav),
        corr_id(corr_id),
        offset(0),
        send_next_chunk_at(send_next_chunk_at),
        latency_ptr(latency_ptr) {}

  bool operator<(const Stream &other) const {
    return (send_next_chunk_at > other.send_next_chunk_at);
  }
};

int main(int argc, char *argv[]) {
  try {
    CudaOnlineBinaryOptions opts;
    if (SetUpAndReadCmdLineOptions(argc, argv, &opts) != 0) return 1;

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    fst::Fst<fst::StdArc> *decode_fst;
    fst::SymbolTable *word_syms;
    ReadModels(opts, &trans_model, &am_nnet, &decode_fst, &word_syms);
    BatchedThreadedNnet3CudaOnlinePipeline cuda_pipeline(
        opts.batched_decoder_config, *decode_fst, am_nnet, trans_model);
    delete decode_fst;
    if (word_syms) cuda_pipeline.SetSymbolTable(*word_syms);

    std::unique_ptr<CompactLatticeWriter> clat_writer;
    std::unique_ptr<Output> ctm_writer;
    OpenOutputHandles(opts.clat_wspecifier, &clat_writer, &ctm_writer);

    std::mutex output_writer_m_;
    if (!opts.write_lattice) {
      KALDI_LOG
          << ("If you want to write lattices to disk, please set "
              "--write-lattice=true");
      clat_writer.reset();
    }
    // result_type is used by the pipeline to know what to generate
    int result_type = 0;
    if (ctm_writer) result_type |= CudaPipelineResult::RESULT_TYPE_CTM;
    if (clat_writer) result_type |= CudaPipelineResult::RESULT_TYPE_LATTICE;

    // Lattice postprocessor
    if (opts.lattice_postprocessor_config_rxfilename.empty()) {
      if (ctm_writer) {
        KALDI_ERR << "You must configure the lattice postprocessor with "
                     "--lattice-postprocessor-rxfilename to use CTM output";
      }
    } else {
      LoadAndSetLatticePostprocessor(
          opts.lattice_postprocessor_config_rxfilename, &cuda_pipeline);
    }

    int chunk_length = cuda_pipeline.GetNSampsPerChunk();
    double chunk_seconds = cuda_pipeline.GetSecondsPerChunk();
    double seconds_per_sample = chunk_seconds / chunk_length;

    std::vector<std::shared_ptr<WaveData>> all_wav;
    std::vector<std::string> all_wav_keys;
    ReadDataset(opts, &all_wav, &all_wav_keys);
    KALDI_ASSERT(all_wav.size() > 0);
    KALDI_ASSERT(all_wav.size() == all_wav_keys.size());
    KALDI_LOG << "Loaded " << all_wav.size() << "files.";
    for (int i = 0; i < all_wav.size(); ++i) {
      if (all_wav[i]->Data().NumRows() <= 0) {
        KALDI_ERR << "Bad file, 0 channels at index [" << i
                  << "], id=" << all_wav_keys[i];
      }
    }

    CorrelationID correlation_id_cnt = 0;

    CudaOnlinePipelineDynamicBatcherConfig dynamic_batcher_config;
    CudaOnlinePipelineDynamicBatcher dynamic_batcher(dynamic_batcher_config,
                                                     cuda_pipeline);
    // Streaming code
    // Wav reader index
    size_t all_wav_i = 0;
    size_t all_wav_max = all_wav.size() * opts.niterations;
    std::vector<double> latencies(all_wav_max);

    // Start all utterances at a random position within the first second.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::priority_queue<Stream> streams;
    nvtxRangePush("Global Timer");
    Timer timer;

    // Initial set of streams will start randomly within the first second of
    // streaming. That's to simulate something more realistic than a large
    // spike of all channels starting at the same time.
    bool add_random_offset = true;
    KALDI_LOG << "Inferencing...";
    while (true) {
      while (streams.size() < opts.num_streaming_channels) {
        int wav_i = all_wav_i++;
        if (wav_i >= all_wav_max) {
          break;
        }
        uint64_t corr_id = correlation_id_cnt++;
        size_t all_wav_i_modulo = wav_i % (all_wav.size());
        double *latency_ptr = &latencies[corr_id];

        // Utterance starts at stream_will_start_at.
        double stream_will_start_at = timer.Elapsed();  // "now"
        if (add_random_offset) stream_will_start_at += dis(gen);
        // Utterance continues for stream_duration seconds.
        double stream_duration = all_wav[all_wav_i_modulo]->Duration();
        // The first chunk will be available whenever the first chunk was
        // "spoken" e.g. if the first chunk is made of 0.5s for audio, we have
        // to wait 0.5s after stream_will_start_at
        double first_chunk_available_at =
            stream_will_start_at + std::min(stream_duration, chunk_seconds);

        // stream_will_stop_at is used for latency computation.
        // Streaming starts at t0 = stream_will_start_at and ends at
        // t1 = stream_will_start_at + duration. We will get the result at t2,
        // therefore the latency = t2 - t1.
        double stream_will_stop_at = stream_will_start_at + stream_duration;
        // tmp storing in lat_ptr
        // we'll do lat_ptr = now - lat_ptr in callback
        *latency_ptr = stream_will_stop_at;

        // Define the callback for results
        bool use_bestpath_callback = !result_type ||
                                     opts.print_partial_hypotheses ||
                                     opts.print_endpoints;
        if (use_bestpath_callback) {
          cuda_pipeline.SetBestPathCallback(
              corr_id, [latency_ptr, &timer, &opts, corr_id, result_type](
                           const std::string &str, bool partial,
                           bool endpoint_detected) {
                if (partial && opts.print_partial_hypotheses) {
                  KALDI_LOG << "corr_id #" << corr_id << " [partial] : " << str;
		}

                if (endpoint_detected && opts.print_endpoints) {
                  KALDI_LOG << "corr_id #" << corr_id << " [endpoint detected]";
		}

                if (!partial) {
                  // *latency_ptr currently contains t1="stream_will_start_at +
                  // duration" where stream_will_start_at is when this stream
                  // started and duration is the duration of this audio file so
                  // t1 is the time when the virtual user is done talking
                  // timer.Elapsed() now contains t2, i.e. when the result is
                  // ready latency = t2 - t1
                  if (!result_type) {
                    // If we need to gen a lattice, latency will take the
                    // lattice gen into account
                    *latency_ptr = timer.Elapsed() - *latency_ptr;
                  }
                  if (opts.print_hypotheses) {
                    KALDI_LOG << "corr_id #" << corr_id << " : " << str;
		  }
                }
              });
        }

        if (result_type) {
          // Setting a callback will indicate the pipeline to generate a
          // lattice
          int iter = all_wav_i / all_wav.size();
          std::string key = all_wav_keys[all_wav_i_modulo];
          if (iter > 0) key += std::to_string(iter);
          SegmentedResultsCallback segmented_callback =
              [&clat_writer, latency_ptr, &timer, &ctm_writer,
               &output_writer_m_, key,
               word_syms](SegmentedLatticeCallbackParams &params) {
                if (params.results.empty()) {
                  KALDI_WARN << "Empty result for callback for utterance " << key;
                  return;
                }
                if (params.results[0].IsLastSegment()) {
                  *latency_ptr = timer.Elapsed() - *latency_ptr;
                }

                if (clat_writer) {
                  std::lock_guard<std::mutex> lk(output_writer_m_);
                  clat_writer->Write(key, *params.results[0].GetLatticeResult());
                }

                if (ctm_writer) {
                  std::lock_guard<std::mutex> lk(output_writer_m_);
                  MergeSegmentsToCTMOutput(params.results, key,
                                           ctm_writer->Stream(), word_syms,
                                           /* use segment offset */ false);
                }
              };

          cuda_pipeline.SetLatticeCallback(corr_id, segmented_callback,
                                           result_type);
        }

        // Adding that stream to our simulation stream pool
        streams.emplace(all_wav[all_wav_i_modulo], corr_id,
                        first_chunk_available_at, latency_ptr);
      }
      // The next streams will just start whenever a spot is available.
      add_random_offset = false;

      // If we reach this, we're just done with all streams
      if (streams.empty()) break;

      Stream chunk = streams.top();
      streams.pop();
      double wait_for = chunk.send_next_chunk_at - timer.Elapsed();
      if (wait_for > 0) kaldi::Sleep(wait_for);

      KALDI_ASSERT(chunk.wav->Data().NumRows() > 0);
      SubVector<BaseFloat> data(chunk.wav->Data(), 0);

      // Current chunk
      int32 total_num_samp = data.Dim();
      int this_chunk_num_samp =
          std::min(total_num_samp - chunk.offset, chunk_length);
      bool is_last_chunk =
          ((chunk.offset + this_chunk_num_samp) == total_num_samp);
      bool is_first_chunk = (chunk.offset == 0);
      SubVector<BaseFloat> wave_part(data, chunk.offset, this_chunk_num_samp);

      // Giving current chunk to the dynamic batcher for processing
      dynamic_batcher.Push(chunk.corr_id, is_first_chunk, is_last_chunk,
                           wave_part);

      // Current chunk was sent, done
      chunk.offset += this_chunk_num_samp;

      // Streaming simulation:
      // We need to know the duration of the next chunk
      // The next time we will "wake up" the stream will be at
      // send_next_chunk_at with send_next_chunk_at = send_current_chunk_at +
      // next_chunk_duration
      int next_chunk_num_samp =
          std::min(total_num_samp - chunk.offset, chunk_length);
      double next_chunk_seconds = next_chunk_num_samp * seconds_per_sample;
      chunk.send_next_chunk_at += next_chunk_seconds;

      // If there is a next chunk, add it to the list of streams tasks
      if (!is_last_chunk) streams.push(chunk);
    }

    dynamic_batcher.WaitForCompletion();
    KALDI_LOG << "Done.";
    nvtxRangePop();

    KALDI_LOG << "Latency stats:";
    PrintLatencyStats(latencies);
    delete word_syms;

    if (clat_writer) clat_writer->Close();
    cudaDeviceSynchronize();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}  // main()
