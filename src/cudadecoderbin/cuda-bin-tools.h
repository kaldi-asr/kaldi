// cudadecoderbin/cuda-bin-tools.h
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

#ifndef KALDI_CUDADECODERBIN_CUDA_BIN_TOOLS_H_
#define KALDI_CUDADECODERBIN_CUDA_BIN_TOOLS_H_

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline2.h"
#include "cudadecoder/cuda-pipeline-common.h"
#include "fstext/fstext-lib.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace cuda_decoder {

// Print some statistics based on latencies stored in \p latencies.
inline void PrintLatencyStats(std::vector<double> &latencies) {
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

struct CudaOnlineBinaryOptions {
  bool write_lattice = false;
  int num_todo = -1;
  int niterations = 1;
  int num_streaming_channels = 2000;
  bool print_partial_hypotheses = false;
  bool print_hypotheses = false;
  bool print_endpoints = false;
  bool generate_lattice = false;
  std::string word_syms_rxfilename, nnet3_rxfilename, fst_rxfilename,
      wav_rspecifier, clat_wspecifier;
  std::string lattice_postprocessor_config_rxfilename;
  BatchedThreadedNnet3CudaOnlinePipelineConfig batched_decoder_config;
};

inline int SetUpAndReadCmdLineOptions(int argc, char *argv[],
                                      CudaOnlineBinaryOptions *opts_ptr) {
  CudaOnlineBinaryOptions &opts = *opts_ptr;
  const char *usage =
      "Reads in wav file(s) and simulates online decoding with neural nets\n"
      "(nnet3 setup).  Note: some configuration values and inputs are\n"
      "set via config files whose filenames are passed as options\n\n"
      "Usage: batched-wav-nnet3-cuda-online [options] <nnet3-in> <fst-in>"
      " <wav-rspecifier> <lattice-wspecifier>\n";

  ParseOptions po(usage);
  po.Register("print-hypotheses", &opts.print_hypotheses,
              "Prints the final hypotheses");
  po.Register("print-partial-hypotheses", &opts.print_partial_hypotheses,
              "Prints the partial hypotheses");
  po.Register("print-endpoints", &opts.print_endpoints,
              "Prints the detected endpoints");
  po.Register("word-symbol-table", &opts.word_syms_rxfilename,
              "Symbol table for words [for debug output]");
  po.Register("file-limit", &opts.num_todo,
              "Limits the number of files that are processed by this driver."
              " After N files are processed the remaining files are ignored."
              " Useful for profiling");
  po.Register("iterations", &opts.niterations,
              "Number of times to decode the corpus. Output will be written"
              " only once.");
  po.Register("num-parallel-streaming-channels", &opts.num_streaming_channels,
              "Number of channels streaming in parallel");
  po.Register("generate-lattice", &opts.generate_lattice,
              "Generate full lattices");
  po.Register("write-lattice", &opts.write_lattice, "Output lattice to a file");
  po.Register("lattice-postprocessor-rxfilename",
              &opts.lattice_postprocessor_config_rxfilename,
              "(optional) Config file for lattice postprocessor");

  CuDevice::RegisterDeviceOptions(&po);
  RegisterCuAllocatorOptions(&po);
  opts.batched_decoder_config.Register(&po);

  po.Read(argc, argv);

  if (po.NumArgs() != 4) {
    po.PrintUsage();
    return 1;
  }

  g_cuda_allocator.SetOptions(g_allocator_options);
  CuDevice::Instantiate().SelectGpuId("yes");
  CuDevice::Instantiate().AllowMultithreading();

  opts.batched_decoder_config.num_channels =
      std::max(opts.batched_decoder_config.num_channels,
               2 * opts.num_streaming_channels);

  opts.nnet3_rxfilename = po.GetArg(1);
  opts.fst_rxfilename = po.GetArg(2);
  opts.wav_rspecifier = po.GetArg(3);
  opts.clat_wspecifier = po.GetArg(4);

  if (opts.write_lattice) opts.generate_lattice = true;

  return 0;
}

inline void ReadModels(const CudaOnlineBinaryOptions &opts,
                       TransitionModel *trans_model,
                       nnet3::AmNnetSimple *am_nnet,
                       fst::Fst<fst::StdArc> **decode_fst,
                       fst::SymbolTable **word_syms) {
  // Read acoustic and transition models.
  bool binary;
  Input ki(opts.nnet3_rxfilename, &binary);
  trans_model->Read(ki.Stream(), binary);
  am_nnet->Read(ki.Stream(), binary);
  SetBatchnormTestMode(true, &(am_nnet->GetNnet()));
  SetDropoutTestMode(true, &(am_nnet->GetNnet()));
  nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet->GetNnet()));

  *decode_fst = fst::ReadFstKaldiGeneric(opts.fst_rxfilename);

  if (!opts.word_syms_rxfilename.empty()) {
    *word_syms = fst::SymbolTable::ReadText(opts.word_syms_rxfilename);
    if (*word_syms == nullptr) {
      KALDI_ERR << "Could not read symbol table from file "
                << opts.word_syms_rxfilename;
    }
  }
}

// Preload the whole test set to avoid mixing I/O into benchmark.
inline void ReadDataset(const CudaOnlineBinaryOptions &opts,
                        std::vector<std::shared_ptr<WaveData>> *all_wav,
                        std::vector<std::string> *all_wav_keys) {
  SequentialTableReader<WaveHolder> wav_reader(opts.wav_rspecifier);
  size_t file_count = 0;

  std::cout << "Loading eval dataset..." << std::flush;
  for (; !wav_reader.Done(); wav_reader.Next()) {
    if (file_count++ == opts.num_todo) break;
    std::string utt = wav_reader.Key();
    std::shared_ptr<WaveData> wave_data = std::make_shared<WaveData>();
    wave_data->Swap(&wav_reader.Value());
    all_wav->push_back(wave_data);
    all_wav_keys->push_back(utt);
    // total_audio += wave_data->Duration();
  }
  std::cout << "done" << std::endl;
}

inline void OpenOutputHandles(
    const std::string &output_wspecifier,
    std::unique_ptr<CompactLatticeWriter> *clat_writer,
    std::unique_ptr<Output> *ctm_writer) {
  WspecifierType ctm_wx_type =
      ClassifyWspecifier(output_wspecifier, NULL, NULL, NULL);

  if (ctm_wx_type == kNoWspecifier) {
    // No wspecifier, assume this is a .ctm file.
    ctm_writer->reset(new Output(output_wspecifier, /* binary= */ false));
  } else {
    // Lattice output.
    clat_writer->reset(new CompactLatticeWriter(output_wspecifier));
  }
}

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // KALDI_CUDADECODERBIN_CUDA_BIN_TOOLS_H_
