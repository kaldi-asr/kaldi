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

#include <iomanip>
#include <iostream>
#include <vector>
#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include "cudadecoder/batched-threaded-nnet3-cuda-pipeline2.h"
#include "cudadecoder/cuda-pipeline-common.h"
#include "fstext/fstext-lib.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

#ifndef KALDI_CUDA_DECODER_BIN_CUDA_BIN_TOOLS_H_
#define KALDI_CUDA_DECODER_BIN_CUDA_BIN_TOOLS_H_

#define KALDI_CUDA_DECODER_BIN_FLOAT_PRINT_PRECISION 2

namespace kaldi {
namespace cuda_decoder {

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
  BatchedThreadedNnet3CudaOnlinePipelineConfig batched_decoder_config;
};

int SetUpAndReadCmdLineOptions(int argc, char *argv[],
                               CudaOnlineBinaryOptions *opts_ptr) {
  CudaOnlineBinaryOptions &opts = *opts_ptr;
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
      "Usage: batched-wav-nnet3-cuda-online [options] "
      "<nnet3-in> "
      "<fst-in> "
      "<wav-rspecifier> <lattice-wspecifier>\n";

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
              "Limits the number of files that are processed by "
              "this driver. "
              "After N files are processed the remaining files "
              "are ignored. "
              "Useful for profiling");
  po.Register("iterations", &opts.niterations,
              "Number of times to decode the corpus. Output will "
              "be written "
              "only once.");
  po.Register("num-parallel-streaming-channels", &opts.num_streaming_channels,
              "Number of channels streaming in parallel");
  po.Register("generate-lattice", &opts.generate_lattice,
              "Generate full lattices");
  po.Register("write-lattice", &opts.write_lattice, "Output lattice to a file");

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

void ReadModels(const CudaOnlineBinaryOptions &opts,
                TransitionModel *trans_model, nnet3::AmNnetSimple *am_nnet,
                fst::Fst<fst::StdArc> **decode_fst,
                fst::SymbolTable **word_syms) {
  // read transition model and nnet
  bool binary;
  Input ki(opts.nnet3_rxfilename, &binary);
  trans_model->Read(ki.Stream(), binary);
  am_nnet->Read(ki.Stream(), binary);
  SetBatchnormTestMode(true, &(am_nnet->GetNnet()));
  SetDropoutTestMode(true, &(am_nnet->GetNnet()));
  nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet->GetNnet()));

  *decode_fst = fst::ReadFstKaldiGeneric(opts.fst_rxfilename);

  if (!opts.word_syms_rxfilename.empty()) {
    if (!(*word_syms = fst::SymbolTable::ReadText(opts.word_syms_rxfilename)))
      KALDI_ERR << "Could not read symbol "
                   "table from file "
                << opts.word_syms_rxfilename;
  }
}

void ReadDataset(const CudaOnlineBinaryOptions &opts,
                 std::vector<std::shared_ptr<WaveData>> *all_wav,
                 std::vector<std::string> *all_wav_keys) {
  // pre-loading data
  // we don't want to measure I/O
  SequentialTableReader<WaveHolder> wav_reader(opts.wav_rspecifier);
  size_t file_count = 0;
  {
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
}

// Reads all CTM outputs in results and merge them together
// into a single output. That output is then written as a CTM text format to
// ostream
void MergeSegmentsToCTMOutput(std::vector<CudaPipelineResult> &results,
                              const std::string &key, std::ostream &ostream,
                              fst::SymbolTable *word_syms = NULL) {
  size_t nresults = results.size();

  if (nresults == 0) {
    KALDI_WARN << "Utterance " << key << " has no results. Skipping";
    return;
  }

  bool all_results_valid = true;

  for (size_t iresult = 0; iresult < nresults; ++iresult)
    all_results_valid &= results[iresult].HasValidResult();

  if (!all_results_valid) {
    KALDI_WARN << "Utterance " << key
               << " has at least one segment with an error. Skipping";
    return;
  }

  ostream << std::fixed;
  ostream.precision(KALDI_CUDA_DECODER_BIN_FLOAT_PRINT_PRECISION);

  // opt: combine results into one here
  BaseFloat previous_segment_word_end = 0;
  for (size_t iresult = 0; iresult < nresults; ++iresult) {
    bool this_segment_first_word = true;
    bool is_last_segment = ((iresult + 1) == nresults);
    BaseFloat next_offset_seconds = FLT_MAX;
    if (!is_last_segment) {
      next_offset_seconds = results[iresult + 1].GetTimeOffsetSeconds();
    }

    auto &result = results[iresult];
    BaseFloat offset_seconds = result.GetTimeOffsetSeconds();
    auto &ctm = result.GetCTMResult();
    for (size_t iword = 0; iword < ctm.times_seconds.size(); ++iword) {
      BaseFloat word_from = offset_seconds + ctm.times_seconds[iword].first;
      BaseFloat word_to = offset_seconds + ctm.times_seconds[iword].second;

      // If beginning of this segment, only keep "new" words
      // i.e. the ones that were not already in previous segment
      if (this_segment_first_word) {
        if (word_from >= previous_segment_word_end) {
          // Found the first "new" word for this segment
          this_segment_first_word = false;
        } else
          continue;  // skipping this word
      }

      // If end of this segment, skip the words which are
      // overlapping two segments
      if (!is_last_segment) {
        if (word_from >= next_offset_seconds) break;  // done with this segment
      }

      previous_segment_word_end = word_to;

      ostream << key << " 1 " << word_from << ' ' << (word_to - word_from)
              << ' ';

      int32 word_id = ctm.words[iword];
      if (word_syms)
        ostream << word_syms->Find(word_id);
      else
        ostream << word_id;

      ostream << ' ' << ctm.conf[iword] << '\n';
    }
  }
}

void OpenOutputHandles(const std::string &output_wspecifier,
                       std::unique_ptr<CompactLatticeWriter> *clat_writer,
                       std::unique_ptr<Output> *ctm_writer) {
  WspecifierType ctm_wx_type;
  ctm_wx_type = ClassifyWspecifier(output_wspecifier, NULL, NULL, NULL);

  if (ctm_wx_type == kNoWspecifier) {
    // No Wspecifier, assume this is a .ctm file
    ctm_writer->reset(new Output(output_wspecifier,
                                 false));  // false == non-binary writing mode.
  } else {
    // Lattice output
    clat_writer->reset(new CompactLatticeWriter(output_wspecifier));
  }
}

// Write all lattices in results using clat_writer
// If print_offsets is true, will write each lattice
// under the key=[utterance_key]-[offset in seconds]
// prints_offsets should be true if results.size() > 1
void WriteLattices(std::vector<CudaPipelineResult> &results,
                   const std::string &key, bool print_offsets,
                   CompactLatticeWriter &clat_writer) {
  for (const CudaPipelineResult &result : results) {
    double offset = result.GetTimeOffsetSeconds();
    if (!result.HasValidResult()) {
      KALDI_WARN << "Utterance " << key << ": "
                 << " Segment with offset " << offset
                 << " is not valid. Skipping";
    }

    std::ostringstream key_with_offset;
    key_with_offset << key;
    if (print_offsets) key_with_offset << "-" << offset;
    clat_writer.Write(key_with_offset.str(), result.GetLatticeResult());
    if (!print_offsets) {
      if (results.size() > 1) {
        KALDI_WARN << "Utterance " << key
                   << " has multiple segments but only one is written to "
                      "output. Use print_offsets=true";
      }
      break;  // printing only one result if offsets are not used
    }
  }
}

// Read lattice postprocessor config, apply it,
// and assign it to the pipeline
void LoadAndSetLatticePostprocessor(
    const std::string &config_filename,
    BatchedThreadedNnet3CudaPipeline2 *cuda_pipeline) {
  ParseOptions po("");  // No usage, reading from a file
  LatticePostprocessorConfig pp_config;
  pp_config.Register(&po);
  po.ReadConfigFile(config_filename);
  auto lattice_postprocessor =
      std::make_shared<LatticePostprocessor>(pp_config);
  cuda_pipeline->SetLatticePostprocessor(lattice_postprocessor);
}

}  // namespace cuda_decoder
}  // namespace kaldi

#endif
