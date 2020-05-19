// cudadecoderbin/batched-wav-nnet3-cuda.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
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
#include <sstream>
#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline.h"
#include "cudamatrix/cu-allocator.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"
using namespace kaldi;
using namespace cuda_decoder;

// When the pipeline is full, wait for
// KALDI_CUDA_DECODER_BIN_PIPELINE_FULL_SLEEP
// Not using a semaphore because it is usually not necessary to wait
#define KALDI_CUDA_DECODER_BIN_PIPELINE_FULL_SLEEP ((double)1 / 1e5)

// This pipeline is deprecated and will be removed. Please switch to
// batched-wav-nnet3-cuda2

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  std::mutex *stdout_mutex,
                                  int64 *tot_num_frames, double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  {
    *tot_num_frames += num_frames;
    *tot_like += likelihood;
    std::lock_guard<std::mutex> lk(*stdout_mutex);
    KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                  << (likelihood / num_frames) << " over " << num_frames
                  << " frames.";

    if (word_syms != NULL) {
      std::ostringstream oss_warn;
      oss_warn << utt << " ";
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          oss_warn << "Word-id " << words[i] << " not in symbol table.";
        oss_warn << s << " ";
      }
      KALDI_WARN << oss_warn.str();
    }
  }
}

// Called when a task is complete. Will be called by different threads
// concurrently,
// so it must be threadsafe
void FinishOneDecode(const std::string &utt, const std::string &key,
                     const fst::SymbolTable *word_syms,
                     BatchedThreadedNnet3CudaPipeline *cuda_pipeline,
                     int64 *num_frames, double *tot_like,
                     CompactLatticeWriter *clat_writer,
                     std::mutex *clat_writer_mutex, std::mutex *stdout_mutex,
                     const bool write_lattice, CompactLattice &clat) {
  nvtxRangePushA("FinishOneDecode");
  GetDiagnosticsAndPrintOutput(utt, word_syms, clat, stdout_mutex, num_frames,
                               tot_like);
  if (write_lattice) {
    std::lock_guard<std::mutex> lk(*clat_writer_mutex);
    clat_writer->Write(key, clat);
  }

  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with "
        "neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker "
        "adaptation and\n"
        "optional endpointing.  Note: some configuration values "
        "and inputs "
        "are\n"
        "set via config files whose filenames are passed as "
        "options\n"
        "\n"
        "Usage: batched-wav-nnet3-cuda [options] <nnet3-in> "
        "<fst-in> "
        "<wav-rspecifier> <lattice-wspecifier>\n";

    std::string word_syms_rxfilename;

    bool write_lattice = true;
    int num_todo = -1;
    int iterations = 1;
    ParseOptions po(usage);
    std::mutex stdout_mutex;
    int pipeline_length = 4000;  // length of pipeline of outstanding requests,
    // this is independent of queue lengths in
    // decoder

    po.Register("write-lattice", &write_lattice,
                "Output lattice to a file. Setting to false is useful when "
                "benchmarking");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("file-limit", &num_todo,
                "Limits the number of files that are processed by "
                "this driver. "
                "After N files are processed the remaining files "
                "are ignored. "
                "Useful for profiling");
    po.Register("iterations", &iterations,
                "Number of times to decode the corpus.");

    // Multi-threaded CPU and batched GPU decoder
    BatchedThreadedNnet3CudaPipelineConfig batched_decoder_config;

    CuDevice::RegisterDeviceOptions(&po);
    RegisterCuAllocatorOptions(&po);
    batched_decoder_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    BatchedThreadedNnet3CudaPipeline cuda_pipeline(batched_decoder_config);

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

    CompactLatticeWriter clat_writer;
    std::mutex clat_write_mutex;

    fst::Fst<fst::StdArc> *decode_fst =
        fst::ReadFstKaldiGeneric(fst_rxfilename);

    cuda_pipeline.Initialize(*decode_fst, am_nnet, trans_model);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_task_submitted = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;
    double total_audio = 0;

    nvtxRangePush("Global Timer");

    int num_groups_done = 0;

    clat_writer.Open(clat_wspecifier);
    // starting timer here so we
    // can measure throughput
    // without allocation
    // overheads
    // using kaldi timer, which starts counting in the constructor
    Timer timer;
    std::vector<double> iteration_timer;
    for (int iter = 0; iter < iterations; iter++) {
      std::string task_group = std::to_string(iter);
      num_task_submitted = 0;
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);

      for (; !wav_reader.Done(); wav_reader.Next()) {
        nvtxRangePushA("Utterance Iteration");

        while (cuda_pipeline.GetNumberOfTasksPending() >= pipeline_length) {
          kaldi::Sleep(KALDI_CUDA_DECODER_BIN_PIPELINE_FULL_SLEEP);
        }

        std::string utt = wav_reader.Key();
        std::string key = utt;

        if (iter > 0) {
          // make key unique for each iteration
          key = std::to_string(iter) + "-" + key;
        }

        const WaveData &wave_data = wav_reader.Value();

        if (iter == 0) {
          // calculating number of utterances per
          // iteration calculating total audio
          // time per iteration
          total_audio += wave_data.Duration();
        }

        // Creating a function alias for the callback
        // function of that utterance
        auto finish_one_decode_lamba =
            [
                // Capturing the arguments that will
                // change by copy
                utt, key,
                // Capturing the const/global args by
                // reference
                &word_syms, &cuda_pipeline, &stdout_mutex, &num_frames,
                &clat_write_mutex, &clat_writer, &write_lattice, &tot_like]
            // The callback function receive the compact
            // lattice as argument if
            // determinize_lattice is true, it is a
            // determinized lattice otherwise, it is a
            // raw lattice converted to compact format
            // through ConvertLattice
            (CompactLattice & clat_in) {
              // Content of our callback function.
              // Calling the general
              // FinishOneDecode function with the
              // proper arguments
              FinishOneDecode(
                  // Captured arguments used to
                  // specialize FinishOneDecode
                  // for this task
                  utt, key, word_syms, &cuda_pipeline, &num_frames, &tot_like,
                  &clat_writer, &clat_write_mutex, &stdout_mutex, write_lattice,
                  // Generated lattice that will
                  // be passed once the task is
                  // complete
                  clat_in);
            };
        // Adding a new task. Once the output lattice is
        // ready, it will call finish_one_decode_lamba
        // Important : finish_one_decode_lamba is called
        // in the threadpool. We need it to be
        // threadsafe (use locks around relevant parts,
        // like writing to I/O)
        cuda_pipeline.OpenDecodeHandle(key, wave_data, task_group,
                                       finish_one_decode_lamba);
        num_task_submitted++;

        nvtxRangePop();
        if (num_todo != -1 && num_task_submitted >= num_todo) break;
      }  // end utterance loop

      std::string group_done;
      // Non-blocking way to check if a group is done
      // returns false if zero groups are ready
      while (cuda_pipeline.IsAnyGroupCompleted(&group_done)) {
        cuda_pipeline.CloseAllDecodeHandlesForGroup(group_done);
        double total_time = timer.Elapsed();
        int32 iter = std::atoi(group_done.c_str());
        KALDI_LOG << "~Group " << group_done << " completed"
                  << " Aggregate Total Time: " << total_time
                  << " Audio: " << total_audio * (iter + 1)
                  << " RealTimeX: " << total_audio * (iter + 1) / total_time;
        num_groups_done++;
      }
    }  // end iterations loop

    // We've submitted all tasks. Now waiting for them to complete
    // We could also have called WaitForAllTasks and
    // CloseAllDecodeHandles
    while (num_groups_done < iterations) {
      // WaitForAnyGroup is blocking. It will hold until one
      // group is ready
      std::string group_done = cuda_pipeline.WaitForAnyGroup();
      cuda_pipeline.CloseAllDecodeHandlesForGroup(group_done);
      double total_time = timer.Elapsed();
      int32 iter = std::atoi(group_done.c_str());
      KALDI_LOG << "~Group " << group_done << " completed"
                << " Aggregate Total Time: " << total_time
                << " Audio: " << total_audio * (iter + 1)
                << " RealTimeX: " << total_audio * (iter + 1) / total_time;
      num_groups_done++;
    }

    // number of seconds elapsed since the creation of timer
    double total_time = timer.Elapsed();
    nvtxRangePop();

    KALDI_LOG << "Decoded " << num_task_submitted << " utterances, " << num_err
              << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";

    KALDI_LOG << "Overall: "
              << " Aggregate Total Time: " << total_time
              << " Total Audio: " << total_audio * iterations
              << " RealTimeX: " << total_audio * iterations / total_time;

    cuda_pipeline.Finalize();
    cudaDeviceSynchronize();

    delete word_syms;  // will delete if non-NULL.

    return 0;

  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}  // main()

#endif  // if HAVE_CUDA == 1
