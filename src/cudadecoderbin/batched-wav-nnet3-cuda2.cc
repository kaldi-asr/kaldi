// cudadecoderbin/batched-wav-nnet3-cuda2.cc
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

#include <atomic>
#if HAVE_CUDA == 1

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <sstream>
#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline2.h"
#include "cudamatrix/cu-allocator.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

using namespace kaldi;
using namespace cuda_decoder;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and decodes them with "
        "neural nets\n"
        "(nnet3 setup).  Note: some configuration values "
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
                "Number of times to decode the corpus. Output will "
                "be written "
                "only once.");

    // Multi-threaded CPU and batched GPU decoder
    BatchedThreadedNnet3CudaPipeline2Config batched_decoder_config;
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

    BatchedThreadedNnet3CudaPipeline2 cuda_pipeline(
        batched_decoder_config, *decode_fst, am_nnet, trans_model);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "") {
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;
      cuda_pipeline.SetSymbolTable(*word_syms);
    }

    int32 num_task_submitted = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;
    double total_audio = 0;

    nvtxRangePush("Global Timer");
    // starting timer here so we
    // can measure throughput
    // without allocation
    // overheads
    // using kaldi timer, which starts counting in the constructor
    Timer timer;
    std::vector<double> iteration_timer;
    for (int iter = 0; iter < iterations; iter++) {
      num_task_submitted = 0;
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
      for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string utt = wav_reader.Key();
        std::string key = utt;
        if (iter > 0) key = std::to_string(iter) + "-" + key;
        std::shared_ptr<WaveData> wave_data = std::make_shared<WaveData>();
        wave_data->Swap(&wav_reader.Value());
        if (iter == 0) {
          // calculating number of utterances per
          // iteration calculating total audio
          // time per iteration
          total_audio += wave_data->Duration();
        }

        cuda_pipeline.DecodeWithCallback(
            wave_data, [&clat_writer, &clat_writer_m, key,
                        write_lattice](CompactLattice &clat) {
              if (write_lattice) {
                std::lock_guard<std::mutex> lk(clat_writer_m);
                clat_writer.Write(key, clat);
              }
            });

        num_task_submitted++;
        if (num_todo != -1 && num_task_submitted >= num_todo) break;
      }  // end utterance loop
    }    // end iterations loop

    cuda_pipeline.WaitForAllTasks();

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
