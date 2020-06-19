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

#if HAVE_CUDA == 1

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <iomanip>
#include <sstream>
#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include "cudamatrix/cu-allocator.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

using namespace kaldi;
using namespace cuda_decoder;

//
// Binary for the online pipeline BatchedThreadedNnet3CudaOnlinePipeline
// Can serve both as a benchmarking tool and an example on how to call
// BatchedThreadedNnet3CudaOnlinePipeline
//

// Prints some statistics based on latencies stored in latencies
void PrintLatencyStats(std::vector<double> &latencies) {
  if (latencies.empty()) return;
  double total = std::accumulate(latencies.begin(), latencies.end(), 0.);
  double avg = total / latencies.size();
  std::sort(latencies.begin(), latencies.end());

  double nresultsf = static_cast<double>(latencies.size());
  size_t per90i = static_cast<size_t>(std::floor(90. * nresultsf / 100.));
  size_t per95i = static_cast<size_t>(std::floor(95. * nresultsf / 100.));
  size_t per99i = static_cast<size_t>(std::floor(99. * nresultsf / 100.));

  double lat_90 = latencies[per90i];
  double lat_95 = latencies[per95i];
  double lat_99 = latencies[per99i];

  KALDI_LOG << "Latencies (s):\tAvg\t\t90%\t\t95%\t\t99%";
  KALDI_LOG << std::fixed << std::setprecision(3) << "\t\t\t" << avg << "\t\t"
            << lat_90 << "\t\t" << lat_95 << "\t\t" << lat_99;
}

// time with arbitrary reference
double inline gettime_monotonic() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double time = ts.tv_sec;
  time += (double)(ts.tv_nsec) / 1e9;
  return time;
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online "
        "decoding with "
        "neural nets\n"
        "(nnet3 setup).  Note: some configuration values "
        "and inputs "
        "are\n"
        "set via config files whose filenames are passed "
        "as "
        "options\n"
        "\n"
        "Usage: batched-wav-nnet3-cuda [options] "
        "<nnet3-in> "
        "<fst-in> "
        "<wav-rspecifier> <lattice-wspecifier>\n";

    std::string word_syms_rxfilename;

    bool write_lattice = true;
    int num_todo = -1;
    int niterations = 3;
    int num_streaming_channels = 2000;
    ParseOptions po(usage);
    po.Register("write-lattice", &write_lattice,
                "Output lattice to a file. Setting to "
                "false is useful when "
                "benchmarking");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("file-limit", &num_todo,
                "Limits the number of files that are processed by "
                "this driver. "
                "After N files are processed the remaining files "
                "are ignored. "
                "Useful for profiling");
    po.Register("iterations", &niterations,
                "Number of times to decode the corpus. Output will "
                "be written "
                "only once.");
    po.Register("num-parallel-streaming-channels", &num_streaming_channels,
                "Number of channels streaming in parallel");

    // Multi-threaded CPU and batched GPU decoder
    BatchedThreadedNnet3CudaOnlinePipelineConfig batched_decoder_config;
    CuDevice::RegisterDeviceOptions(&po);
    RegisterCuAllocatorOptions(&po);
    batched_decoder_config.Register(&po);

    po.Read(argc, argv);
    batched_decoder_config.num_channels = std::max(
        batched_decoder_config.num_channels, 2 * num_streaming_channels);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    std::string nnet3_rxfilename = po.GetArg(1), fst_rxfilename = po.GetArg(2),
                wav_rspecifier = po.GetArg(3), clat_wspecifier = po.GetArg(4);
    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;

    // read transition model and nnet
    bool binary;
    Input ki(nnet3_rxfilename, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
    SetDropoutTestMode(true, &(am_nnet.GetNnet()));
    nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));

    CompactLatticeWriter clat_writer(clat_wspecifier);
    std::mutex clat_writer_m;

    fst::Fst<fst::StdArc> *decode_fst =
        fst::ReadFstKaldiGeneric(fst_rxfilename);

    BatchedThreadedNnet3CudaOnlinePipeline cuda_pipeline(
        batched_decoder_config, *decode_fst, am_nnet, trans_model);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "") {
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol "
                     "table from file "
                  << word_syms_rxfilename;
      else {
        //        cuda_pipeline.SetSymbolTable(word_syms);
      }
    }

    int32 num_task_submitted = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;
    double total_audio_not_starved = 0;
    double total_compute_time_not_starved = 0;

    int chunk_length = cuda_pipeline.GetNSampsPerChunk();
    double chunk_seconds = cuda_pipeline.GetSecondsPerChunk();
    double seconds_per_sample = chunk_seconds / chunk_length;

    // pre-loading data
    // we don't want to measure I/O
    double total_audio = 0;
    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    std::vector<std::shared_ptr<WaveData>> all_wav;
    std::vector<std::string> all_wav_keys;
    {
      std::cout << "Loading eval dataset..." << std::flush;
      for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        std::shared_ptr<WaveData> wave_data = std::make_shared<WaveData>();
        wave_data->Swap(&wav_reader.Value());
        all_wav.push_back(wave_data);
        all_wav_keys.push_back(utt);
        total_audio += wave_data->Duration();
      }
      std::cout << "done" << std::endl;
    }
    total_audio *= niterations;

    struct Stream {
      std::shared_ptr<WaveData> wav;
      BatchedThreadedNnet3CudaOnlinePipeline::CorrelationID corr_id;
      int offset;
      double send_next_chunk_at;
      double *latency_ptr;

      Stream(const std::shared_ptr<WaveData> &_wav,
             BatchedThreadedNnet3CudaOnlinePipeline::CorrelationID _corr_id,
             double *_latency_ptr)
          : wav(_wav), corr_id(_corr_id), offset(0), latency_ptr(_latency_ptr) {
        send_next_chunk_at = gettime_monotonic();
      }

      bool operator<(const Stream &other) {
        return (send_next_chunk_at < other.send_next_chunk_at);
      }
    };
    nvtxRangePush("Global Timer");
    // starting timer here so we
    // can measure throughput
    // without allocation
    // overheads
    // using kaldi timer, which starts counting in the
    // constructor
    Timer timer;
    double this_iteration_timer = timer.Elapsed();
    std::vector<double> iteration_timer;
    std::vector<std::unique_ptr<Stream>> curr_tasks, next_tasks;
    curr_tasks.reserve(num_streaming_channels);
    next_tasks.reserve(num_streaming_channels);
    size_t all_wav_i = 0;
    size_t all_wav_max = all_wav.size() * niterations;
    std::vector<double> latencies(all_wav_max);
    BatchedThreadedNnet3CudaOnlinePipeline::CorrelationID correlation_id_cnt =
        0;
    // Batch sent to online pipeline
    std::vector<BatchedThreadedNnet3CudaOnlinePipeline::CorrelationID>
        batch_corr_ids;
    std::vector<bool> batch_is_first_chunk;
    std::vector<bool> batch_is_last_chunk;
    // Used when use_online_ivectors_
    std::vector<SubVector<BaseFloat>> batch_wave_samples;

    double batch_valid_at = gettime_monotonic();
    bool pipeline_starved_warning_printed = false;
    while (true) {
      int this_iteration_total_samples = 0;
      batch_valid_at = 0.;
      while (curr_tasks.size() < num_streaming_channels &&
             all_wav_i < all_wav_max) {
        // Creating new tasks
        uint64_t corr_id = correlation_id_cnt++;
        size_t all_wav_i_modulo = all_wav_i % (all_wav.size());
        double *latency_ptr = &latencies[all_wav_i];
        std::unique_ptr<Stream> ptr(
            new Stream(all_wav[all_wav_i_modulo], corr_id, latency_ptr));
        curr_tasks.emplace_back(std::move(ptr));

        // If no channels are available, we will wait up
        // to INT_MAX microseconds for a channel to
        // become available. The reason why we can in
        // theory have no channel available is because a
        // channel is still in used when the last chunk
        // has been processed but the lattice is still
        // being generated This is why we set
        // batched_decoder_config.num_channels strictly
        // higher than num_streaming_channels
        // If we want to ensure that we are never using
        // more channels than num_streaming_channels, we
        // can call WaitForLatticeCallbacks after each
        // DecodeBatch. That way, we know TryInitCorrID
        // will always have a channel available right
        // away if batched_decoder_config.num_channels
        // >= num_streaming_channels
        KALDI_ASSERT(cuda_pipeline.TryInitCorrID(corr_id, INT_MAX));
        const std::string &utt = all_wav_keys[all_wav_i_modulo];
        size_t iteration = all_wav_i / all_wav.size();
        std::string key =
            (iteration == 0) ? utt : (std::to_string(iteration) + "-" + utt);
        cuda_pipeline.SetLatticeCallback(
            corr_id, [&clat_writer, &clat_writer_m, key, write_lattice,
                      latency_ptr](CompactLattice &clat) {
              if (write_lattice) {
                std::lock_guard<std::mutex> lk(clat_writer_m);
                clat_writer.Write(key, clat);
              }
              double now = gettime_monotonic();
              *latency_ptr = now - *latency_ptr;
            });
        ++all_wav_i;
        ++num_task_submitted;
      }
      // If still empty, done
      if (curr_tasks.empty()) break;

      std::sort(curr_tasks.begin(), curr_tasks.end());

      for (size_t itask = 0; itask < curr_tasks.size(); ++itask) {
        Stream &task = *(curr_tasks[itask]);

        SubVector<BaseFloat> data(task.wav->Data(), 0);
        int32 samp_offset = task.offset;
        int32 nsamp = data.Dim();
        int32 samp_remaining = nsamp - samp_offset;
        int32 num_samp =
            chunk_length < samp_remaining ? chunk_length : samp_remaining;
        bool is_last_chunk = (chunk_length >= samp_remaining);
        SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
        bool is_first_chunk = (samp_offset == 0);

        task.offset += num_samp;
        batch_valid_at = std::max(task.send_next_chunk_at, batch_valid_at);
        this_iteration_total_samples += num_samp;

        batch_corr_ids.push_back(task.corr_id);
        batch_is_first_chunk.push_back(is_first_chunk);
        batch_is_last_chunk.push_back(is_last_chunk);
        batch_wave_samples.push_back(wave_part);

        if (!is_last_chunk) {
          next_tasks.push_back(std::move(curr_tasks[itask]));
        } else {
          *task.latency_ptr = task.send_next_chunk_at;
        }

        task.send_next_chunk_at += chunk_seconds;
        if (batch_corr_ids.size() == batched_decoder_config.max_batch_size ||
            (itask == (curr_tasks.size() - 1))) {
          // Wait for batch to be valid
          double now = gettime_monotonic();
          double wait_for = batch_valid_at - now;
          if (wait_for > 0) usleep(wait_for * 1e6);

          cuda_pipeline.DecodeBatch(batch_corr_ids, batch_wave_samples,
                                    batch_is_first_chunk, batch_is_last_chunk);
          batch_corr_ids.clear();
          batch_is_first_chunk.clear();
          batch_is_last_chunk.clear();
          batch_wave_samples.clear();
        }
      }
      bool pipeline_starved = (curr_tasks.size() < num_streaming_channels);
      if (pipeline_starved && !pipeline_starved_warning_printed) {
        std::cout << "\nNote: Streaming the end of the "
                     "last "
                     "utterances. "
                     "Not enough unprocessed "
                     "utterances available to stream "
                  << num_streaming_channels
                  << " channels in parallel. The "
                     "pipeline is starved. Will now "
                     "stream partial batches while "
                     "still limiting I/O at realtime "
                     "speed. RTFX will drop. \n"
                  << std::endl;
        pipeline_starved_warning_printed = true;
      }
      double curr_timer = timer.Elapsed();
      double diff = curr_timer - this_iteration_timer;
      this_iteration_timer = curr_timer;
      double this_iteration_total_seconds =
          this_iteration_total_samples * seconds_per_sample;
      if (!pipeline_starved) {
        total_audio_not_starved += this_iteration_total_seconds;
        total_compute_time_not_starved += diff;
      }
      double this_iteration_rtfx = this_iteration_total_seconds / diff;
      if (pipeline_starved) std::cout << "STARVED: ";
      std::cout << "Number of active streaming channels: " << std::setw(5)
                << curr_tasks.size() << "\tInstant RTFX: " << std::setw(6)
                << std::fixed << std::setprecision(1) << this_iteration_rtfx
                << std::endl;

      curr_tasks.swap(next_tasks);
      next_tasks.clear();
    }
    cuda_pipeline.WaitForLatticeCallbacks();
    nvtxRangePop();

    KALDI_LOG << "Decoded " << num_task_submitted << " utterances, " << num_err
              << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";

    KALDI_LOG << "NON-STARVED:";
    KALDI_LOG << "\tThis section only concerns the part of the "
                 "computation "
                 "where we had enough active utterances to simulate "
              << num_streaming_channels << " parallel clients. ";
    KALDI_LOG << "\tIt corresponds to the throughput an online instance "
                 "can handle with all channels in use.";
    KALDI_LOG << "\tTotal Compute Time: " << total_compute_time_not_starved;
    KALDI_LOG << "\tTotal Audio Decoded: " << total_audio_not_starved;
    KALDI_LOG << "\tRealTimeX: "
              << total_audio_not_starved / total_compute_time_not_starved;

    KALDI_LOG << "OVERALL:";
    KALDI_LOG << "\tTotal Utterances Decoded: " << num_task_submitted;
    KALDI_LOG << "\tTotal Audio Decoded: " << total_audio << " seconds";
    KALDI_LOG << "\tLatency stats:";
    PrintLatencyStats(latencies);

    delete word_syms;  // will delete if non-NULL.

    clat_writer.Close();

    cudaDeviceSynchronize();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}  // main()

#endif  // if HAVE_CUDA == 1
