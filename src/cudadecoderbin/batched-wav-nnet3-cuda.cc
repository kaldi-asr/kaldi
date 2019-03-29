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

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
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
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
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

// using a macro here to avoid a ton of parameters in a function
// while also being able to reuse this in two spots
void FinishOneDecode(
    const BatchedThreadedNnet3CudaPipelineConfig &batched_decoder_config,
    const fst::SymbolTable *word_syms, const bool write_lattice,
    const int32 total_audio, const int32 count_per_iteration,
    BatchedThreadedNnet3CudaPipeline *cuda_pipeline,
    std::queue<std::pair<std::string, std::string>> *processed,
    CompactLatticeWriter *clat_writer, Timer *timer, int32 *current_count,
    int64 *num_frames, int32 *output_iter, double *tot_like) {
  std::string &utt = processed->front().first;
  std::string &key = processed->front().second;
  CompactLattice clat;
  bool valid;

  if (batched_decoder_config.determinize_lattice) {
    valid = cuda_pipeline->GetLattice(key, &clat);
  } else {
    Lattice lat;
    valid = cuda_pipeline->GetRawLattice(key, &lat);
    ConvertLattice(lat, &clat);
  }
  if (valid) {
    GetDiagnosticsAndPrintOutput(utt, word_syms, clat, num_frames, tot_like);
    if (write_lattice && key == utt) { /*only write output on first iteration*/
      nvtxRangePushA("Lattice Write");
      clat_writer->Write(utt, clat);
      nvtxRangePop();
    }
  }
  cuda_pipeline->CloseDecodeHandle(key);
  processed->pop();
  if (++(*current_count) ==
      count_per_iteration) { /*this utt is the last in an iter*/
    double total_time = timer->Elapsed();
    KALDI_VLOG(2) << "Iteration: " << *output_iter
                  << " ~Aggregate Total Time: " << total_time
                  << " Total Audio: " << total_audio * *output_iter
                  << " RealTimeX: " << *output_iter * total_audio / total_time;
    current_count = 0;
    (*output_iter)++;
  }
  }

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs "
        "are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: batched-wav-nnet3-cuda [options] <nnet3-in> <fst-in> "
        "<wav-rspecifier> <lattice-wspecifier>\n";

    std::string word_syms_rxfilename;

    bool write_lattice = true;
    int num_todo = -1;
    int iterations = 1;
    ParseOptions po(usage);
    int pipeline_length = 4000; // length of pipeline of outstanding requests,
                                // this is independent of queue lengths in
                                // decoder

    po.Register("write-lattice", &write_lattice,
                "Output lattice to a file. Setting to false is useful when "
                "benchmarking");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("file-limit", &num_todo,
                "Limits the number of files that are processed by this driver. "
                "After N files are processed the remaining files are ignored. "
                "Useful for profiling");
    po.Register("iterations", &iterations,
                "Number of times to decode the corpus. Output will be written "
                "only once.");

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

    CompactLatticeWriter clat_writer(clat_wspecifier);

    fst::Fst<fst::StdArc> *decode_fst =
        fst::ReadFstKaldiGeneric(fst_rxfilename);

    cuda_pipeline.Initialize(*decode_fst, am_nnet, trans_model);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
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

    int count_per_iteration = 0;
    int current_count = 0;
    int output_iter = 1;

    std::queue<std::pair<std::string, std::string>> processed;
    for (int iter = 0; iter < iterations; iter++) {
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);

      for (; !wav_reader.Done(); wav_reader.Next()) {
        nvtxRangePushA("Utterance Iteration");

        std::string utt = wav_reader.Key();
        std::string key = utt;
        if (iter > 0) {
          // make key unique for subsequent iterations
          key = key + "-" + std::to_string(iter);
        }
        const WaveData &wave_data = wav_reader.Value();

        if (iter == 0) {
          // calculating number of utterances per iteration
          count_per_iteration++;
          // calculating total audio time per iteration
          total_audio += wave_data.Duration();
        }

        cuda_pipeline.OpenDecodeHandle(key, wave_data);
        processed.push(pair<string, string>(utt, key));
        num_done++;

        while (processed.size() >= pipeline_length) {
          FinishOneDecode(batched_decoder_config, word_syms, write_lattice,
                          total_audio, count_per_iteration, &cuda_pipeline,
                          &processed, &clat_writer, &timer, &current_count,
                          &num_frames, &output_iter, &tot_like);
        }  // end while

        nvtxRangePop();
        if (num_todo != -1 && num_done >= num_todo)
          break;
      } // end utterance loop

    } // end iterations loop

    while (processed.size() > 0) {
      FinishOneDecode(batched_decoder_config, word_syms, write_lattice,
                      total_audio, count_per_iteration, &cuda_pipeline,
                      &processed, &clat_writer, &timer, &current_count,
                      &num_frames, &output_iter, &tot_like);
    } // end while

    KALDI_LOG << "Decoded " << num_done << " utterances, " << num_err
              << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";

    // number of seconds elapsed since the creation of timer
    double total_time = timer.Elapsed();
    nvtxRangePop();

    KALDI_LOG << "Overall: "
              << " Aggregate Total Time: " << total_time
              << " Total Audio: " << total_audio * iterations
              << " RealTimeX: " << total_audio * iterations / total_time;

    delete word_syms; // will delete if non-NULL.

    clat_writer.Close();

    cuda_pipeline.Finalize();
    cudaDeviceSynchronize();

    return 0;

    // return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
} // main()

#endif  // if HAVE_CUDA == 1
